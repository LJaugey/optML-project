#!/bin/bash

#SBATCH -n 5
#SBATCH -c 4
#SBATCH --time=1:45:0
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --partition=build

## RUNS for the SGD with momentum method (lr = 0.01, beta = 0.9)

time srun python train.py sgdm mnist 0.0 0.0
sleep 10
time srun python train.py sgdm fmnist 0.0 0.0
sleep 10
time srun python train.py sgdm emnist 0.0 0.0
sleep 10
time srun python train.py sgdm cifar10 0.0 0.0


## RUNS for the Adam method (lr = 0.01)

time srun python train.py adam mnist 0.0 0.0
sleep 10
time srun python train.py adam fmnist 0.0 0.0
sleep 10
time srun python train.py adam emnist 0.0 0.0
sleep 10
time srun python train.py adam cifar10 0.0 0.0


## RUNS for the Adaptative SGD method

# standard run, default values
time srun python train.py adgd mnist 0.02 1
sleep 10
time srun python train.py adgd fmnist 0.02 1
sleep 10
time srun python train.py adgd emnist 0.02 1
sleep 10
time srun python train.py adgd cifar10 0.02 1

# run for amplifier = 0.02 damping = 2
time srun python train.py adgd mnist 0.02 2
sleep 10
time srun python train.py adgd fmnist 0.02 2
sleep 10
time srun python train.py adgd emnist 0.02 2
sleep 10
time srun python train.py adgd cifar10 0.02 2

# run for amplifier = 0.1, damping = 1
time srun python train.py adgd mnist 0.1 1
sleep 10
time srun python train.py adgd fmnist 0.1 1
sleep 10
time srun python train.py adgd emnist 0.1 1
sleep 10
time srun python train.py adgd cifar10 0.1 1

