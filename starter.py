# The entire homework can be completed on this file. 
# It is derived from main_DebiasPL(_ZeroShot).py

# The command to run this code on 1 gpu is: 
# python starter.py
# add the tag --clip to load untrained CLIP (question 1), without the tag, the ZSL model is loaded (q.2)

# RANDOM ASIDE: You can also check out the huggingface imagenet data. 
# To access it, you need to create a free huggingface account
# then, in terminal type: huggingface-cli login
# follow the link to create a token (giving it read permissions to public gated repos) 
# copy the token to login
# navigate to: https://huggingface.co/datasets/ILSVRC/imagenet-1k
# click agree and access repository


import argparse
import builtins
import math
import os
import shutil
import time
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision import transforms

import data.datasets as datasets
import backbone as backbone_models
from models import get_fixmatch_model
from utils import utils, lr_schedule, get_norm, dist_utils
import data.transforms as data_transforms
# from engine import validate
# we are making a new validate function
from torch.utils.tensorboard import SummaryWriter

import clip

from tqdm import tqdm

from visualization import *


backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default = '', metavar='DIR',
                    help='path to dataset')
# OPTIONAL: you can modify to add your ImageNet100 path here
parser.add_argument('--arch', metavar='ARCH', default='FixMatch',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-c', '--clip', dest='clip', action='store_true',
                    help='use regular clip')
parser.add_argument('--pretrained', default='', 
                    type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--eman', action='store_true', default=False,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')

best_acc1 = 0




class NoTensorImageFolder(datasets.ImageFolder):
    def __init__(self, root):
        super().__init__(root)

    def __getitem__(self, index):
        original_image, original_label = super().__getitem__(index)
        return original_image, original_label


def main():
    args = parser.parse_args()
    print(args)

    if not args.clip:
        # create model
        print("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
        model_func = get_fixmatch_model(args.arch)
        norm = get_norm(args.norm)
        model = model_func(
            backbone_models.__dict__[args.backbone],
            eman=args.eman,
            momentum=args.ema_m,
            norm=norm
        )
        # print(model)
        # print("Total params: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    elif args.clip:
        # model is clip
        model, preprocess = clip.load("RN50")
        print('Using CLIP')


    if args.pretrained and not args.clip:
        # load trained weights 
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model from '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)    # set weights_only to False by yui, using PyTorch>=2.6
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if "module.main" in k:
                    new_key = k.replace("module.", "")
                    state_dict[new_key] = state_dict[k]
                del state_dict[k]
            model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))

    model.cuda() 
    try:
        preprocess.cuda() 
    except:
        pass

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    transform_val = data_transforms.get_transforms('DefaultVal')

    if not args.clip:
        # TODO: Edit validate_542_logit for q.2 
        # This is how you will generate the graphs you need
        validate_542_logit(transform_val, model, criterion, args)
        return
    
    elif args.clip:
        # TODO: Edit validate_542_clip for q.1 
        # This is how you will generate the graphs you need
        validate_542_clip(model, preprocess, args)
        return
    

def get_centroids(prob):
    # this is from the original code. It was used for CLDLoss
    # might be helpful for your implementation in part c
    # but you should not use it directly for getting centroids!
    N, D = prob.shape
    K = D
    cl = prob.argmin(dim=1).long().view(-1)  # -> class index
    Ncl = cl.view(cl.size(0), 1).expand(-1, D)
    unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
    labels_count_all = torch.ones([K]).long().cuda() # -> counts of each class
    labels_count_all[unique_labels[:,0]] = labels_count
    c = torch.zeros([K, D], dtype=prob.dtype).cuda().scatter_add_(0, Ncl, prob) # -> class centroids
    c = c / labels_count_all.float().unsqueeze(1)
    return cl, c

def CLDLoss(prob_s, prob_w, mask=None, weights=None):
    # this is from the original code, not used here
    cl_w, c_w = get_centroids(prob_w)
    affnity_s2w = torch.mm(prob_s, c_w.t())
    if mask is None:
        loss = F.cross_entropy(affnity_s2w.div(0.07), cl_w, weight=weights)
    else:
        loss = (F.cross_entropy(affnity_s2w.div(0.07), cl_w, reduction='none', weight=weights) * (1 - mask)).mean()
    return loss


def group_batch(batch):
    return {k: [v] for k, v in batch.items()}

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)
    
def validate_542_clip(model, preprocess, args): 
    # TODO: Implement the visualizations needed here 
    valdir = os.path.join(args.data, 'val')
    val_dataset = NoTensorImageFolder(valdir)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn
    ) 
    imagenet100_to_name = {0: 'robin', 1: 'Gila_monster', 2: 'hognose_snake', 3: 'garter_snake', 4: 'green_mamba', 5: 'garden_spider', 6: 'lorikeet', 7: 'goose', 8: 'rock_crab', 9: 'fiddler_crab', 
                          10: 'American_lobster', 11: 'little_blue_heron', 12: 'American_coot', 13: 'Chihuahua', 14: 'Shih-Tzu', 15: 'papillon', 16: 'toy_terrier', 17: 'Walker_hound', 18: 'English_foxhound', 19: 'borzoi', 
                          20: 'Saluki', 21: 'American_Staffordshire_terrier', 22: 'Chesapeake_Bay_retriever', 23: 'vizsla', 24: 'kuvasz', 25: 'komondor', 26: 'Rottweiler', 27: 'Doberman', 28: 'boxer', 29: 'Great_Dane', 
                          30: 'standard_poodle', 31: 'Mexican_hairless', 32: 'coyote', 33: 'African_hunting_dog', 34: 'red_fox', 35: 'tabby', 36: 'meerkat', 37: 'dung_beetle', 38: 'walking_stick', 39: 'leafhopper', 
                          40: 'hare', 41: 'wild_boar', 42: 'gibbon', 43: 'langur', 44: 'ambulance', 45: 'bannister', 46: 'bassinet', 47: 'boathouse', 48: 'bonnet', 49: 'bottlecap', 
                          50: 'car_wheel', 51: 'chime', 52: 'cinema', 53: 'cocktail_shaker', 54: 'computer_keyboard', 55: 'Dutch_oven', 56: 'football_helmet', 57: 'gasmask', 58: 'hard_disc', 59: 'harmonica', 
                          60: 'honeycomb', 61: 'iron', 62: 'jean', 63: 'lampshade', 64: 'laptop', 65: 'milk_can', 66: 'mixing_bowl', 67: 'modem', 68: 'moped', 69: 'mortarboard', 
                          70: 'mousetrap', 71: 'obelisk', 72: 'park_bench', 73: 'pedestal', 74: 'pickup', 75: 'pirate', 76: 'purse', 77: 'reel', 78: 'rocking_chair', 79: 'rotisserie', 
                          80: 'safety_pin', 81: 'sarong', 82: 'ski_mask', 83: 'slide_rule', 84: 'stretcher', 85: 'theater_curtain', 86: 'throne', 87: 'tile_roof', 88: 'tripod', 89: 'tub', 
                          90: 'vacuum', 91: 'window_screen', 92: 'wing', 93: 'head_cabbage', 94: 'cauliflower', 95: 'pineapple', 96: 'carbonara', 97: 'chocolate_sauce', 98: 'gyromitra', 99: 'stinkhorn'}

    names = list(imagenet100_to_name.values())
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in names])

    text_inputs = text_inputs.cuda()

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    start_time = time.time()
    with torch.no_grad():

        total_counts = [0 for _ in range(100)]
        all_preds, all_trues, logits_list = [], [], []

        for _, i in tqdm(enumerate(val_loader)): #dataset

            images_raw = i[0]
            target = torch.tensor(i[1])
            

            images = []
            for ii in range(len(images_raw)):
                images.append(preprocess(images_raw[ii]))

            images = torch.stack(images, dim=0)

            images = images.cuda()
            target = target.cuda() 

            image_features = model.encode_image(images)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            output = (100.0 * image_features @ text_features.T)
            similarity = output.softmax(dim=-1)

            pred = torch.argmax(similarity, dim=-1)

            ## record for plotting,
            ## Logit refers to the number before softmax.
            all_preds.extend(pred.cpu().tolist())       # extend to flatten the list of lists
            all_trues.extend(target.cpu().tolist())
            logits_list.append(output.cpu())            # keep logit similarity as tensors for centroid computation later

            for p in pred: 
                total_counts[p] += 1
            
            acc1, acc5 = utils.accuracy(similarity, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        logits_list = torch.cat(logits_list, dim=0)
    
    conf_mat = confusion_matrix(all_trues, all_preds, labels=list(range(100)))
    ranked = get_ranked_array(conf_mat)
    plot_precision_recall(conf_mat, ranked, 'result/imagenet100_clip_precision_recall.png')
    plot_confusion_matrix(conf_mat, ranked, names, 'result/imagenet100_clip_confmat.png')
    compute_centroid_similarity_clip(logits_list, torch.tensor(all_preds), ranked, names, 'result/imagenet100_clip_centroids.tex')
    
    print(total_counts)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))  
    print('total time: ', time.time()-start_time)
    
    return top1.avg



def validate_542_logit(transform_val, model, criterion, args): 
    # TODO: Implement the visualizations needed here 
    imagenet100_to_name = {0: 'robin', 1: 'Gila_monster', 2: 'hognose_snake', 3: 'garter_snake', 4: 'green_mamba', 5: 'garden_spider', 6: 'lorikeet', 7: 'goose', 8: 'rock_crab', 9: 'fiddler_crab', 
                          10: 'American_lobster', 11: 'little_blue_heron', 12: 'American_coot', 13: 'Chihuahua', 14: 'Shih-Tzu', 15: 'papillon', 16: 'toy_terrier', 17: 'Walker_hound', 18: 'English_foxhound', 19: 'borzoi', 
                          20: 'Saluki', 21: 'American_Staffordshire_terrier', 22: 'Chesapeake_Bay_retriever', 23: 'vizsla', 24: 'kuvasz', 25: 'komondor', 26: 'Rottweiler', 27: 'Doberman', 28: 'boxer', 29: 'Great_Dane', 
                          30: 'standard_poodle', 31: 'Mexican_hairless', 32: 'coyote', 33: 'African_hunting_dog', 34: 'red_fox', 35: 'tabby', 36: 'meerkat', 37: 'dung_beetle', 38: 'walking_stick', 39: 'leafhopper', 
                          40: 'hare', 41: 'wild_boar', 42: 'gibbon', 43: 'langur', 44: 'ambulance', 45: 'bannister', 46: 'bassinet', 47: 'boathouse', 48: 'bonnet', 49: 'bottlecap', 
                          50: 'car_wheel', 51: 'chime', 52: 'cinema', 53: 'cocktail_shaker', 54: 'computer_keyboard', 55: 'Dutch_oven', 56: 'football_helmet', 57: 'gasmask', 58: 'hard_disc', 59: 'harmonica', 
                          60: 'honeycomb', 61: 'iron', 62: 'jean', 63: 'lampshade', 64: 'laptop', 65: 'milk_can', 66: 'mixing_bowl', 67: 'modem', 68: 'moped', 69: 'mortarboard', 
                          70: 'mousetrap', 71: 'obelisk', 72: 'park_bench', 73: 'pedestal', 74: 'pickup', 75: 'pirate', 76: 'purse', 77: 'reel', 78: 'rocking_chair', 79: 'rotisserie', 
                          80: 'safety_pin', 81: 'sarong', 82: 'ski_mask', 83: 'slide_rule', 84: 'stretcher', 85: 'theater_curtain', 86: 'throne', 87: 'tile_roof', 88: 'tripod', 89: 'tub', 
                          90: 'vacuum', 91: 'window_screen', 92: 'wing', 93: 'head_cabbage', 94: 'cauliflower', 95: 'pineapple', 96: 'carbonara', 97: 'chocolate_sauce', 98: 'gyromitra', 99: 'stinkhorn'}

    names = list(imagenet100_to_name.values())

    imagenet100_to_1K = {0: 15, 1: 45, 2: 54, 3: 57, 4: 64, 5: 74, 6: 90, 7: 99, 8: 119, 9: 120, 10: 122, 11: 131,
                        12: 137, 13: 151, 14: 155, 15: 157, 16: 158, 17: 166, 18: 167, 19: 169, 20: 176, 21: 180, 
                        22: 209, 23: 211, 24: 222, 25: 228, 26: 234, 27: 236, 28: 242, 29: 246, 30: 267,
                        31: 268, 32: 272, 33: 275, 34: 277, 35: 281, 36: 299, 37: 305, 38: 313, 39: 317, 40: 331, 
                        41: 342, 42: 368, 43: 374, 44: 407, 45: 421, 46: 431, 47: 449, 48: 452, 49: 455, 50: 479,
                        51: 494, 52: 498, 53: 503, 54: 508, 55: 544, 56: 560, 57: 570, 58: 592, 59: 593,
                        60: 599, 61: 606, 62: 608, 63: 619, 64: 620, 65: 653, 66: 659, 67: 662, 68: 665, 69: 667, 
                        70: 674, 71: 682, 72: 703, 73: 708, 74: 717, 75: 724, 76: 748, 77: 758, 78: 765, 79: 766, 
                        80: 772, 81: 775, 82: 796, 83: 798, 84: 830, 85: 854, 86: 857, 87: 858, 88: 872,
                        89: 876, 90: 882, 91: 904, 92: 908, 93: 936, 94: 938, 95: 953, 96: 959, 97: 960, 98: 993, 99: 994} 

    keep_ind = list(imagenet100_to_1K.values())

    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        end = time.time()

        ## Optional: if want to use preloaded val dataset, each loop takes ~20-30sec 
        # in this case, overwrite like:

        valdir = os.path.join(args.data, 'val')
        val_dataset = datasets.ImageFolder(
                valdir, transform=transform_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True) 
        
        # then, enumerate val_loader instead of val_dataset
        # inside the loop, no longer transform, just get images and target


        total_counts = [0 for _ in range(100)]
        all_preds, all_trues, logits_list = [], [], []

        for _, i in tqdm(enumerate(val_loader)): #optional: _loader

            images = i[0]
            target = i[1]

            images = images.cuda() 
            target = target.cuda() 

            output = model(images)

            output = output[:, keep_ind]

            loss = criterion(output, target)
            probs = torch.softmax(output, dim=-1)

            pred = torch.argmax(output, dim=-1)

            ## record for plotting
            ## Logit refers to the number before softmax.
            all_preds.extend(pred.cpu().tolist())       # extend to flatten the list of lists
            all_trues.extend(target.cpu().tolist())
            logits_list.append(output.cpu())        # keep logit similarity as tensors for centroid computation later

            for p in pred: 
                total_counts[p] += 1


            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        logits_list = torch.cat(logits_list, dim=0)

    conf_mat = confusion_matrix(all_trues, all_preds, labels=list(range(100)))
    ranked_zsl = get_ranked_array(conf_mat, ranked_path='result/imagenet100_zsl_ranked.npy')   # load clip ranked
    ranked_clip = load_ranked_array()       # default loads clip ranked
    plot_precision_recall(conf_mat, ranked_clip, 'result/imagenet100_zsl_precision_recall.png')
    plot_confusion_matrix(conf_mat, ranked_clip, names, 'result/imagenet100_zsl_confmat.png')
    compute_centroid_similarity_zsl(logits_list, torch.tensor(all_preds), ranked_clip, ranked_zsl, names, 'result/imagenet100_zsl_centroids.tex')

    print(total_counts)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.4f}'
              .format(top1=top1, top5=top5, loss=losses))  
    print('total time: ', time.time()-start_time)
    
    return top1.avg



if __name__ == '__main__':
    main()
