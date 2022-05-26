import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from helpers import load_data

batch_size = 128
num_workers = 2

trainloader, testloader, num_classes = load_data(dataset='mnist', batch_size=batch_size, num_workers=num_workers)