import numpy as np
import os
import random
import torch
import torchvision

import torchvision.transforms as transforms

from pathlib import Path

def load_data(dataset='mnist', batch_size=128, num_workers=2):
    """
    Loads the required dataset
    :param dataset: Can be either 'mnist', 'fmnist', 'emnist' or 'cifar10'
    :param batch_size: The desired batch size
    :return: Tuple (train_loader, test_loader, num_classes)
    """
    print('==> Preparing data..')
    num_classes = 10
    if dataset == 'mnist':
        # classes = (0, 1, 2, ..., 9)
        train_set = torchvision.datasets.MNIST(root = './data/MNIST', train = True, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.MNIST(root = './data/MNIST', train = False, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
    elif dataset == 'fmnist':
        # classes = (0, 1, 2, ..., 9)
        train_set = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST', train = True, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST', train = False, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
    elif dataset == 'emnist':
        # classes = alphabet
        train_set = torchvision.datasets.EMNIST(root = './data/EMNIST',split='letters',  train = True, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.EMNIST(root = './data/EMNIST',split='letters', train = False, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        num_classes = 26
    elif dataset == 'cifar10':
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
                                                 transform=transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, 
                                                 transform=transforms.Compose([transforms.ToTensor()]))
    else:
        raise ValueError('Only mnist, fmnist, emnist, cifar10 are supported')

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes