#!/bin/bash
#SBATCH --job-name=miss-alignment-worker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition={{partition}}
#SBATCH --output={{logs_dir}}/slurm-%j.out
#SBATCH --error={{logs_dir}}/slurm-%j.err

# Activate the miss-alignment conda environment.
# Adjust the path to match your installation.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate miss-alignment

{{command}}
