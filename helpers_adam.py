import numpy as np
import os
import random
import copy
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam, SGD



from optimizer import Adsgd

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
        # label = (0, 1, 2, ..., 9)
        train_set = torchvision.datasets.MNIST(root = './data/MNIST', train = True, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.MNIST(root = './data/MNIST', train = False, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
    elif dataset == 'fmnist':
        # label = (0, 1, 2, ..., 9)
        train_set = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST', train = True, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST', train = False, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
    elif dataset == 'emnist':
        # classes = alphabet (1, ..., 26)
        train_set = torchvision.datasets.EMNIST(root = './data/EMNIST',split='letters', train = True, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.EMNIST(root = './data/EMNIST',split='letters', train = False, download = True,
                                                     transform = transforms.Compose([transforms.ToTensor()]))
        num_classes = 26
        train_set.targets -= 1 # changes the labels to be between 0 and 25
        test_set.targets -= 1
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

def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def accuracy_and_loss(net, dataloader, device, criterion):
    net.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1) # predicted contains the indices of the max, dim is the dimension to reduce
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)

    return correct / total, loss

def run_adgd(net, device, trainloader, testloader, N_train, n_epoch=2, amplifier=0.02, damping=1, weight_decay=0, eps=1e-8, checkpoint=125, batch_size=128, noisy_train_stat=True):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []
    
    prev_net = copy.deepcopy(net)
    prev_net.to(device)
    net.train()
    prev_net.train()
    lrs = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adsgd(net.parameters(), amplifier=amplifier, damping=damping, weight_decay=weight_decay, eps=eps)
    prev_optimizer = Adsgd(prev_net.parameters(), weight_decay=weight_decay)
            
    for epoch in range(n_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print(str(i/len(trainloader) * 100) + "%                   ", end='\r')
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            prev_optimizer.zero_grad(set_to_none=True)

            prev_outputs = prev_net(inputs)
            prev_loss = criterion(prev_outputs, labels)
            prev_loss.backward()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.compute_dif_norms(prev_optimizer)
            prev_net.load_state_dict(net.state_dict())
            optimizer.step()

            running_loss += loss.item()
            if (i % 10) == 0:
                if noisy_train_stat:
                    losses.append(loss.cpu().item())
                    it_train.append(epoch + i * batch_size / N_train)
                lrs.append(optimizer.param_groups[0]['lr'])

            if i % checkpoint == checkpoint - 1:
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                running_loss = 0.0
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                grad_norms.append(np.sum([p.grad.data.norm().item() for p in net.parameters()]))
                net.train()
                it_test.append(epoch + i * batch_size / N_train)
                
        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()

    del prev_net
    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(lrs), np.array(grad_norms))

def run_adam(net, device, trainloader, testloader, N_train, n_epoch=2, checkpoint=125, batch_size=128):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []
    
    net.train() # sets network to train mode
    lrs = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters())
            
    for epoch in range(n_epoch):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            #print(str(i/len(trainloader) * 100) + "%")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if (i % 10) == 0:
                lrs.append(optimizer.param_groups[0]['lr'])

            # the tesing is done at 1/3 of the way through the epoch
            if i % checkpoint == checkpoint - 1:
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                grad_norms.append(np.sum([p.grad.data.norm().item() for p in net.parameters()]))
                net.train()
                it_test.append(epoch + i * batch_size / N_train)
                
        it_train.append(epoch)
        train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
        train_acc.append(train_a)
        losses.append(train_l)
        print('Epoch',{epoch+1}, 'finished.')
        net.train()

    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(lrs), np.array(grad_norms))

def run_sgdm(net, device, trainloader, testloader, N_train, n_epoch=2, checkpoint=125, batch_size=128):
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []
    
    net.train() # sets network to train mode
    lrs = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), 0.1, 0.9) # standard values of lr = 0.1, beta = 0.9
            
    for epoch in range(n_epoch):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):
            print(str(i/len(trainloader) * 100) + "%")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if (i % 10) == 0:
                lrs.append(optimizer.param_groups[0]['lr'])

            # the tesing is done at 1/3 of the way through the epoch
            if i % checkpoint == checkpoint - 1:
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                grad_norms.append(np.sum([p.grad.data.norm().item() for p in net.parameters()]))
                net.train()
                it_test.append(epoch + i * batch_size / N_train)
                
        it_train.append(epoch)
        train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
        train_acc.append(train_a)
        losses.append(train_l)
        net.train()

    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(lrs), np.array(grad_norms))

def save_results(losses, test_losses, train_acc, test_acc, it_train, it_test, grad_norms, method='sgd', 
                 lrs=[], experiment='cifar10_resnet18', folder='./', to_save_extra=[], prefixes_extra=[]):
    path = f'./{folder}/{experiment}/'
    Path(path).mkdir(parents=True, exist_ok=True)
    to_save = [losses, test_losses, train_acc, test_acc, it_train, it_test, grad_norms, lrs] + to_save_extra
    prefixes = ['l', 'tl', 'a', 'ta', 'itr', 'ite', 'gn', 'lr'] + prefixes_extra
    for log, prefix in zip(to_save, prefixes):
        np.save(f'{path}/{method}_{prefix}.npy', log)