import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from helpers_adam import load_data, seed_everything, run_adam, save_results
from lenet5 import LeNet5

# train_lenet with mnist, emnist, fashion-mnist, cifar10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_folder = './results_adam/'

batch_size = 512
num_workers = 0
n_epoch = 100

data_set = 'mnist'
#data_set = 'fmnist'
#data_set = 'emnist'
#data_set = 'cifar10'


trainloader, testloader, num_classes = load_data(dataset=data_set, batch_size=batch_size, num_workers=num_workers)
N_train = len(trainloader.dataset) # number of training examples for mnist
print(N_train)
checkpoint = len(trainloader) // 3 + 1

n_seeds = 5
max_seed = 424242
rng = np.random.default_rng(42)
seeds = [rng.choice(max_seed, size=1, replace=False)[0] for _ in range(n_seeds)]

for r, seed in enumerate(seeds):
    seed_everything(seed)
    print('==> Seed',{r+1},':')
    net = LeNet5(dataset=data_set)
    net.to(device)
    net_name = "lenet5"

    losses_adam, test_losses_adam, acc_adam, test_acc_adam, it_adam, it_test_adam, lrs_adam, grad_norms_adam = run_adam(
            net=net, device=device, trainloader=trainloader, testloader=testloader, N_train=N_train, n_epoch=n_epoch, 
            checkpoint=checkpoint, batch_size=batch_size)

    method = f"adam"

    experiment = net_name+"_seed_"+str(r)
    save_results(losses_adam, test_losses_adam, acc_adam, test_acc_adam, it_adam, it_test_adam, lrs=lrs_adam, 
                grad_norms=grad_norms_adam, method=method, experiment=experiment, folder=output_folder+data_set)

