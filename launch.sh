#!/bin/bash

#SBATCH -n 5
#SBATCH -c 4
#SBATCH --time=1:0:0
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --partition=debug

source ~/OptML/optML-project/bin/activate
pip install torch torchvision

module purge
module load gcc mvapich2 python

srun python train_mnist.py
deactivate