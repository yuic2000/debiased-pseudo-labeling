#!/bin/bash
# The interpreter used to execute the script
#\#SBATCH" directives that convey submission options:
#SBATCH --job-name=dpl_clip
#SBATCH --mail-user=yuic@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4000m
#SBATCH --time=01:00:00
#SBATCH --account=cse598f25s014_class
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --gpus=1
#SBATCH --output=./log/dpl_clip.log

echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time is: $(date)"
echo "Directory is: $(pwd)"


# Activate your environment
source ~/.bashrc
conda activate debiaspl

python starter.py \
    --data /scratch/cse598f25s014_class_root/cse598f25s014_class/shared_data/imagenet-100 \
    --clip \
    > ./log/dpl_clip.txt

echo "Job finished with exit code $? at: $(date)"