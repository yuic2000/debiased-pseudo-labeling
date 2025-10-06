import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
import torch.nn.functional as F


def get_ranked_array(conf_mat, ranked_path='result/imagenet100_clip_ranked.npy'):    # default save clip ranked
    num_classes = conf_mat.shape[0]
    class_counts = conf_mat.sum(axis=0)     # num of predictions per class
    ranked = np.argsort(-class_counts)      # negate for descending order of # predictions

    # Save ranked array for later use
    np.save(ranked_path, ranked)
    print(f"Saved ranked array to: {ranked_path}")
    return ranked

def load_ranked_array(ranked_path='result/imagenet100_clip_ranked.npy'):  # default load clip ranked
    ranked = np.load(ranked_path)
    print(f"Loaded ranked array from: {ranked_path}")
    return ranked

# ---------- (a) Precision/Recall vs ranked classes ----------
def plot_precision_recall(conf_mat, ranked, save_path):
    num_classes = conf_mat.shape[0]
    class_counts = conf_mat.sum(axis=0)     # num of predictions per class
    precision = np.diag(conf_mat) / (conf_mat.sum(axis=0) + 1e-12)
    recall = np.diag(conf_mat) / (conf_mat.sum(axis=1) + 1e-12)
    x = np.arange(num_classes)              # x-axis: 0 (most) to 99 (least)

    # --- Polynomial fitting ---
    coeff_precision = np.polyfit(x, precision[ranked], deg=2)
    coeff_recall = np.polyfit(x, recall[ranked], deg=2)
    poly_precision = np.poly1d(coeff_precision)
    poly_recall = np.poly1d(coeff_recall)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Precision subplot
    ax0 = axes[0]
    ax2 = ax0.twinx()
    ax2.set_zorder(0)               # put ax2 behind ax0
    ax0.set_zorder(1)
    ax0.patch.set_visible(False)    # hide the 'canvas'

    ax2.bar(x, class_counts[ranked], color='lightgray')    # plot prediction bars first
    ax2.set_ylabel('# of Predictions', color='gray')
    ax2.tick_params(axis='y', colors='gray')    

    ax0.scatter(x, precision[ranked], color='royalblue', s=10)
    ax0.plot(x, poly_precision(x), color='royalblue', linewidth=1)
    ax0.set_ylabel('Precision', color='royalblue')
    ax0.tick_params(axis='y', colors='royalblue')

    # Recall subplot
    ax1 = axes[1]
    ax3 = ax1.twinx()
    ax3.set_zorder(0)               # put ax3 behind ax1
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)    # hide the 'canvas'

    ax3.bar(x, class_counts[ranked], color='lightgray')     # plot prediction bars first
    ax3.set_ylabel('# of Predictions', color='gray')
    ax3.tick_params(axis='y', colors='gray')

    ax1.scatter(x, recall[ranked], color='darkorange', s=10)
    ax1.plot(x, poly_recall(x), color='darkorange', linewidth=1)
    ax1.set_ylabel('Recall', color='darkorange')
    ax1.tick_params(axis='y', colors='darkorange')
        
    # --- Shared axes ---
    ax1.set_xlabel('Ranked Class Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"A: Saved plot file: {save_path}")


# ---------- (b) Confusion-matrix heatmap ----------
def plot_confusion_matrix(conf_mat, ranked, class_names, save_path):
    conf_sorted = conf_mat[ranked][:, ranked]
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_sorted, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='# Images')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    sorted_names = [class_names[i] for i in ranked]
    step = 1
    tick_indices = range(0, len(sorted_names), step)
    tick_labels = [sorted_names[i] for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=90, ha='right', fontsize=5)
    plt.yticks(tick_indices, tick_labels, fontsize=5)

    # Overlay numeric values
    max_val = conf_sorted.max()
    black_lower_bound = 15
    black_upper_bound = 35
    for i in range(conf_sorted.shape[0]):
        for j in range(conf_sorted.shape[1]):
            val = conf_sorted[i, j]
            if val >= 0:
                color = "black" if black_lower_bound < val < black_upper_bound else "white"
                plt.text(j, i, f"{val:.0f}", ha='center', va='center', color=color, fontsize=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"B: Saved plot file: {save_path}")

# ---------- (c) Centroid-similarity computation ----------
def compute_centroid_similarity_clip(logits, preds, ranked, names, save_path='centroid_clip.tex'):
    num_classes = logits.shape[1]
    centroids = []
    for k in range(num_classes):
        mask = (preds == k)
        if mask.sum() > 0:
            centroids.append(logits[mask].mean(0))
        else:
            centroids.append(torch.zeros(num_classes))
    centroids = torch.stack(centroids)
    centroids = F.normalize(centroids, dim=1)

    most_idx = ranked[0]
    least_idx = ranked[-1]

    sim_most = (centroids @ centroids[most_idx]).cpu().numpy()
    sim_least = (centroids @ centroids[least_idx]).cpu().numpy()

    top10_most = np.argsort(-sim_most)[1:11]
    top10_least = np.argsort(-sim_least)[1:11]

    most_name = names[most_idx]
    least_name = names[least_idx]

    with open(save_path, "w") as f:
        f.write("% ==========================================\n")
        f.write("% Submission: Part (c) - Centroid Similarity\n")
        f.write("% ==========================================\n\n")

        # --- class names ---
        f.write(f"\\textbf{{Most predicted class}}: {most_name} (index {most_idx})\\\\\n")
        f.write(f"\\textbf{{Least predicted class}}: {least_name} (index {least_idx})\\\\\n\n")

        # --- top-10 lists ---
        f.write("\\textbf{Top-10 similar classes for most predicted:}\\\\\n")
        f.write("\\textbf{Rank} & \\textbf{Class} & \\textbf{Cosine_Similarity}\\\\\n")
        for i, idx in enumerate(top10_most, 1):
            f.write(f"{i} & {names[idx]} & {sim_most[idx]:.4f} \\\\\n")
        f.write("\n")

        f.write("\\textbf{Top-10 similar classes for least predicted:}\\\\\n")
        f.write("\\textbf{Rank} & \\textbf{Class} & \\textbf{Cosine_Similarity}\\\\\n")
        for i, idx in enumerate(top10_least, 1):
            f.write(f"{i} & {names[idx]} & {sim_least[idx]:.4f} \\\\\n")

    print(f"C: Saved LaTeX file: {save_path}")

def compute_centroid_similarity_zsl(logits, preds, ranked_clip, ranked_zsl, names, save_path='centroid_zsl.tex'):
    num_classes = logits.shape[1]
    centroids = []
    for k in range(num_classes):
        mask = (preds == k)
        if mask.sum() > 0:
            centroids.append(logits[mask].mean(0))
        else:
            centroids.append(torch.zeros(num_classes))
    centroids = torch.stack(centroids)
    centroids = F.normalize(centroids, dim=1)

    most_idx_clip = ranked_clip[0]
    least_idx_clip = ranked_clip[-1]
    most_idx_zsl = ranked_zsl[0]
    least_idx_zsl = ranked_zsl[-1]

    sim_most_clip = (centroids @ centroids[most_idx_clip]).cpu().numpy()
    sim_least_clip = (centroids @ centroids[least_idx_clip]).cpu().numpy()
    sim_most_zsl = (centroids @ centroids[most_idx_zsl]).cpu().numpy()
    sim_least_zsl = (centroids @ centroids[least_idx_zsl]).cpu().numpy()

    top10_most_clip = np.argsort(-sim_most_clip)[1:11]
    top10_least_clip = np.argsort(-sim_least_clip)[1:11]
    top10_most_zsl = np.argsort(-sim_most_zsl)[1:11]
    top10_least_zsl = np.argsort(-sim_least_zsl)[1:11]

    most_name_clip = names[most_idx_clip]
    least_name_clip = names[least_idx_clip]
    most_name_zsl = names[most_idx_zsl]
    least_name_zsl = names[least_idx_zsl]

    with open(save_path, "w") as f:
        f.write("% ==========================================\n")
        f.write("% Submission: Part (c) - Centroid Similarity\n")
        f.write("% ==========================================\n\n")

        # --- class names ---
        f.write(f"\\textbf{{Most predicted class (CLIP)}}: {most_name_clip} (index {most_idx_clip})\\\\\n")
        f.write(f"\\textbf{{Least predicted class (CLIP)}}: {least_name_clip} (index {least_idx_clip})\\\\\n\n")
        f.write(f"\\textbf{{Most predicted class (ZSL)}}: {most_name_zsl} (index {most_idx_zsl})\\\\\n")
        f.write(f"\\textbf{{Least predicted class (ZSL)}}: {least_name_zsl} (index {least_idx_zsl})\\\\\n\n")

        # --- top-10 lists ---
        f.write("\\textbf{Top-10 similar classes for most predicted (CLIP):}\\\\\n")
        f.write("\\textbf{Rank} & \\textbf{Class} & \\textbf{Cosine_Similarity}\\\\\n")
        for i, idx in enumerate(top10_most_clip, 1):
            f.write(f"{i} & {names[idx]} & {sim_most_clip[idx]:.4f} \\\\\n")
        f.write("\n")

        f.write("\\textbf{Top-10 similar classes for least predicted (CLIP):}\\\\\n")
        f.write("\\textbf{Rank} & \\textbf{Class} & \\textbf{Cosine_Similarity}\\\\\n")
        for i, idx in enumerate(top10_least_clip, 1):
            f.write(f"{i} & {names[idx]} & {sim_least_clip[idx]:.4f} \\\\\n")
        f.write("\n")

        f.write("\\textbf{Top-10 similar classes for most predicted (ZSL):}\\\\\n")
        f.write("\\textbf{Rank} & \\textbf{Class} & \\textbf{Cosine_Similarity}\\\\\n")
        for i, idx in enumerate(top10_most_zsl, 1):
            f.write(f"{i} & {names[idx]} & {sim_most_zsl[idx]:.4f} \\\\\n")
        f.write("\n")

        f.write("\\textbf{Top-10 similar classes for least predicted (ZSL):}\\\\\n")
        f.write("\\textbf{Rank} & \\textbf{Class} & \\textbf{Cosine_Similarity}\\\\\n")
        for i, idx in enumerate(top10_least_zsl, 1):
            f.write(f"{i} & {names[idx]} & {sim_least_zsl[idx]:.4f} \\\\\n")
        f.write("\n")

    print(f"C: Saved LaTeX file: {save_path}")