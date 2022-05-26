import torch
import torch.nn as nn
from dlc_practical_prologue import load_data
from helpers import Get_data_loaders, Train, Test

import Conv


torch.manual_seed(0)

reps = 10
batch_size = 50
epoch = 25
accuracies = torch.zeros(reps)


load_data()    # For downloading MNIST if it is not already here
print("\n\n")


print("\n=====Convolutional network=====\n")

print(sum(p.numel() for p in Conv.Net().parameters() if p.requires_grad), "parameters\n")

print("\nWithout data augmentation\n")


for i in range(reps):

    net = Conv.Net()

    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    Train(net, criterion, loader_train, optimizer, epoch)
    true, false = Test(net, loader_test)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n")



print("\nWith data augmentation\n")


for i in range(reps):

    net = Conv.Net()

    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    Train(net, criterion, loader_train, optimizer, epoch)
    true, false = Test(net, loader_test)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n\n\n\n")








print("\n=====Siamese network=====\n")


print(sum(p.numel() for p in Share.Net().parameters() if p.requires_grad), "parameters\n")

print("\nWithout data augmentation\n")



for i in range(reps):

    net = Share.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")
    

    loader_train, loader_test = Get_data_loaders(batch_size, augment=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    Train(net, criterion, loader_train, optimizer, epoch)
    true, false = Test(net, loader_test)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n")



print("\nWith data augmentation\n")


for i in range(reps):

    net = Share.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")
    

    loader_train, loader_test = Get_data_loaders(batch_size, augment=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    Train(net, criterion, loader_train, optimizer, epoch)
    true, false = Test(net, loader_test)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n\n\n\n")







print("\n=====Auxiliary network=====\n")


print(sum(p.numel() for p in Aux_.Net().parameters() if p.requires_grad), "parameters\n")

print("\nWithout data augmentation\n")


for i in range(reps):

    net = Aux_.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Note the separate training function
    Train(net, criterion, loader_train, optimizer, epoch, aux=True, alpha=3.5)
    true, false = Test(net, loader_test, aux=True)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n")



print("\nWith data augmentation\n")


for i in range(reps):

    net = Aux_.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Note the separate training function
    Train(net, criterion, loader_train, optimizer, epoch, aux=True, alpha=3.5)
    true, false = Test(net, loader_test, aux=True)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n")




print("\nWith data augmentation and alpha = 0\n")

for i in range(reps):

    net = Aux_.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Note the separate training function
    Train(net, criterion, loader_train, optimizer, epoch, aux=True, alpha=0)
    true, false = Test(net, loader_test, aux=True)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n\n\n\n")







print("\n=====Shared auxiliary network=====\n")


print(sum(p.numel() for p in Shared_aux.Net().parameters() if p.requires_grad), "parameters\n")

print("\nWithout data augmentation\n")


for i in range(reps):

    net = Shared_aux.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Note the separate training function
    Train(net, criterion, loader_train, optimizer, epoch, aux=True, alpha=2.5)
    true, false = Test(net, loader_test, aux=True)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%     (std: ", round(std.item() * 100, 2), "%)\n")



print("\nWith data augmentation\n")


for i in range(reps):

    net = Shared_aux.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Note the separate training function
    Train(net, criterion, loader_train, optimizer, epoch, aux=True, alpha=2.5)
    true, false = Test(net, loader_test, aux=True)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%      (std: ", round(std.item() * 100, 2), "%)\n")


print("\nWith data augmentation and alpha = 0\n")


for i in range(reps):

    net = Shared_aux.Net()
    
    
    print("repetition: ", i+1,"/",reps, end="\r")


    loader_train, loader_test = Get_data_loaders(batch_size, augment=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Note the separate training function
    Train(net, criterion, loader_train, optimizer, epoch, aux=True, alpha=0)
    true, false = Test(net, loader_test, aux=True)
    accuracies[i] = true/(true + false)

std, mean = torch.std_mean(accuracies)


print("Accuracy: ", round(mean.item() * 100, 2), "%      (std: ", round(std.item() * 100, 2), "%)\n\n\n\n")