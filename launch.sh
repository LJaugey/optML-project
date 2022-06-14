#!/bin/bash

#SBATCH -n 5
#SBATCH -c 4
#SBATCH --time=2:0:0
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --partition=build

#source ~/OptML/optML-project/bin/activate
#pip install torch torchvision

#module purge
#module load gcc mvapich2 python

# time srun python train.py sgdm fmnist
# sleep 10
# time srun python train.py adam fmnist
# sleep 10
# time srun python train.py adgd fmnist

# time srun python train.py sgdm emnist
# sleep 10

## standard run, default values
# time srun python train.py adgd mnist 0.02 1
# sleep 10
# time srun python train.py adgd fmnist 0.02 1
# sleep 10
# time srun python train.py adgd emnist 0.02 1

## run for amplifier = 0.02 damping = 2
# time srun python train.py adgd mnist 0.02 2
# sleep 10
# time srun python train.py adgd fmnist 0.02 2
# sleep 10
# time srun python train.py adgd emnist 0.02 2

## run for amplifier = 0.1, damping = 1
# time srun python train.py adgd mnist 0.1 1
# sleep 10
# time srun python train.py adgd fmnist 0.1 1
# sleep 10
# time srun python train.py adgd emnist 0.1 1

