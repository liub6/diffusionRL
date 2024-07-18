#!/bin/bash -l
#SBATCH --job-name=example   # Job name
#SBATCH --output=examplejob.o%j # Name of stdout output file
#SBATCH --error=examplejob.e%j  # Name of stderr error file
##SBATCH --nodes=1               # Total number of nodes 
##SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
##SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=30000M
#SBATCH --gres=gpu:1
#SBATCH --gpus=1

# module load scibuilder-spack-dev/2024-01
# module load gcc/13.2.0
# module load libx11/1.8.4

srun python ./diffusion/train_diffuser.py \
    --cond $1 \
    --minari $2 \
    --dataset $3 \
    --results_folder $4 \
    --save_samples $5 \
    --load_checkpoint $6


