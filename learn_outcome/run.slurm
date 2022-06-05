#!/bin/bash
#SBATCH --job-name=run-pyramid
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="h.shavit@umail.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=150G
#SBATCH --time=0-23:00:00
#SBATCH --partition=gpu-medium
#SBATCH --ntasks=24
#SBATCH --gpus=1
#SBATCH --nodes=1

echo "[$SHELL] #### Starting GPU TensorFlow test"
echo "[$SHELL] This is $SLURM_JOB_USER and my first job has the ID $SLURM_JOB_ID"
source /home/${USER}/.bashrc
conda activate torch

python learn_bc.py

