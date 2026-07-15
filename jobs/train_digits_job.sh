#!/bin/bash
#SBATCH --job-name=train_digits
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/data/slurm_logs/slurm_job%j_train_digits.out
#SBATCH --error=/data/slurm_logs/slurm_job%j_train_digits.err

# Load environment
module load cuda/12.8
module load cudnn/9.9
module load mamba
micromamba activate columns_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run your code
cd /home/dasja/projects/ODE-Column
python -m scripts.digits.train_digits_cluster
