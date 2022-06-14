#!/bin/bash

#SBATCH -n 5
#SBATCH -c 4
#SBATCH --time=1:30:0
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --partition=build



## standard run, default values
# time srun python train.py adgd mnist 0.02 1
# sleep 10
# time srun python train.py adgd fmnist 0.02 1
# sleep 10
# time srun python train.py adgd emnist 0.02 1

## run for amplifier = 0.02 damping = 2
time srun python train.py adgd cifar10 0.02 2
sleep 10
#time srun python train.py adgd fmnist 0.02 2
#sleep 10
#time srun python train.py adgd emnist 0.02 2

## run for amplifier = 0.1, damping = 1
time srun python train.py adgd cifar10 0.1 1
#sleep 10
#time srun python train.py adgd fmnist 0.1 1
#sleep 10
#time srun python train.py adgd emnist 0.1 1

