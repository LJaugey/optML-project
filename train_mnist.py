import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from helpers import load_data, seed_everything, run_adgd, save_results
from lenet5 import LeNet5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_folder = './results/'

batch_size = 128
num_workers = 0
n_epoch = 5

trainloader, testloader, num_classes = load_data(dataset='emnist', batch_size=batch_size, num_workers=num_workers)
N_train = len(trainloader.dataset) # number of training examples for mnist
print(N_train)
checkpoint = len(trainloader) // 3 + 1

lr_amplifier = 0.02
lr_damping = 1
weight_decay = 0

n_seeds = 1
max_seed = 424242
rng = np.random.default_rng(42)
seeds = [rng.choice(max_seed, size=1, replace=False)[0] for _ in range(n_seeds)]

for r, seed in enumerate(seeds):
    seed_everything(seed)
    net = LeNet5(dataset='emnist')
    net.to(device)

    losses_adgd, test_losses_adgd, acc_adgd, test_acc_adgd, it_adgd, it_test_adgd, lrs_adgd, grad_norms_adgd = run_adgd(
            net=net, device=device, trainloader=trainloader, testloader=testloader, N_train=N_train, n_epoch=n_epoch, 
            amplifier=lr_amplifier, damping=lr_damping, weight_decay=weight_decay, 
            checkpoint=checkpoint, batch_size=batch_size, noisy_train_stat=False )

    method = f'adgd_{lr_amplifier}_{lr_damping}'
    experiment = 'emnist_lenet5'
    save_results(losses_adgd, test_losses_adgd, acc_adgd, test_acc_adgd, it_adgd, it_test_adgd, lrs=lrs_adgd, 
                grad_norms=grad_norms_adgd, method=method, experiment=experiment, folder=output_folder)

