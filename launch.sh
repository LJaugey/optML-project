#!/bin/bash

#SBATCH -n 5
#SBATCH -c 4
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --partition=gpu

#source ~/OptML/optML-project/bin/activate
#pip install torch torchvision

#module purge
#module load gcc mvapich2 python

srun python train.py sgdm cifar10
sleep 10
srun python train.py adam cifar10
sleep 10
srun python train.py adgd cifar10

