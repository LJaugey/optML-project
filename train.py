import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys

from helpers import *
from lenet5 import LeNet5

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


opt = sys.argv[1]
#opt = 'sgdm'
#opt = 'adam'
#opt = 'adgd'

data_set = sys.argv[2]
#data_set = 'mnist'
#data_set = 'fmnist'
#data_set = 'emnist'
#data_set = 'cifar10'

lr_amplifier = float(sys.argv[3])
lr_damping = float(sys.argv[4])

output_folder = './results/'

batch_size = 512
num_workers = 0
n_epoch = 100


trainloader, testloader, num_classes = load_data(dataset=data_set, batch_size=batch_size, num_workers=num_workers)
N_train = len(trainloader.dataset) # number of training examples for mnist
print(N_train)
checkpoint = len(trainloader) // 3 + 1

# lr_amplifier = 0.02
# lr_damping = 1
weight_decay = 0

n_seeds_per_task = 1
max_seed = 424242
rng = np.random.default_rng(42+rank)
seeds = [rng.choice(max_seed, size=1, replace=False)[0] for _ in range(n_seeds_per_task)]

for r_, seed in enumerate(seeds):
    seed_everything(seed)
    net = LeNet5(dataset=data_set)
    net.to(device)
    net_name = "lenet5"

    if(opt=="sgdm"):
        losses, test_losses, acc, test_acc, it, it_test, lrs, grad_norms = run_sgdm(
                net=net, device=device, trainloader=trainloader, testloader=testloader,
                N_train=N_train, n_epoch=n_epoch, checkpoint=checkpoint, batch_size=batch_size)

        method = f"sgdm_{lr_amplifier}_{lr_damping}"


    if(opt=="adam"):
        losses, test_losses, acc, test_acc, it, it_test, lrs, grad_norms = run_adam(
                net=net, device=device, trainloader=trainloader, testloader=testloader,
                N_train=N_train, n_epoch=n_epoch, checkpoint=checkpoint, batch_size=batch_size)

        method = f"adam_{lr_amplifier}_{lr_damping}"


    if(opt=="adgd"):
        losses, test_losses, acc, test_acc, it, it_test, lrs, grad_norms = run_adgd(
                net=net, device=device, trainloader=trainloader, testloader=testloader,
                N_train=N_train, n_epoch=n_epoch, amplifier=lr_amplifier, damping=lr_damping,
                weight_decay=weight_decay, checkpoint=checkpoint, batch_size=batch_size,
                noisy_train_stat=False )

        method = f"/adgd_{lr_amplifier}_{lr_damping}"

    r = n_seeds_per_task*rank + r_

    experiment = net_name+"_seed_"+str(r)
    save_results(losses, test_losses, acc, test_acc, it, it_test, lrs=lrs, 
                grad_norms=grad_norms, method=method, experiment=experiment, folder=output_folder+data_set+method)

