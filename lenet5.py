from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    # modified LeNet5, Relu instead of sigmoid
    def __init__(self, cifar10=False):
        super(LeNet5, self).__init__()
        # input is 1 x 28 x 28 
        # if cifar10, input is 2 x 32 x 32
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=5, padding=2)
        if cifar10:
            self.conv1 = nn.Conv2d(in_channels = 3, out_channels=6, kernel_size=5, padding=2)
        self.r1 = nn.ReLU()
        # 6 x 28 x 28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 6 x 14 x 14
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=16, kernel_size=5)
        self.r2 = nn.ReLU()
        # 16 x 10 x 10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 16 x 5 x 5
        self.f = nn.Flatten()
        # 1 x 1 x 400
        self.lin1 = nn.Linear(in_features=400, out_features=120)
        if cifar10:
            self.lin1 = nn.Linear(in_features=576, out_features=120)
        self.r3 = nn.ReLU()
        # 1 x 1 x 120
        self.lin2 = nn.Linear(in_features=120, out_features=80)
        self.r4 = nn.ReLU()
        # 1 x 1 x 80
        self.lin3 = nn.Linear(in_features=80, out_features=10)

        self.layers = nn.Sequential(self.conv1, self.r1, 
                                    self.pool1, 
                                    self.conv2, self.r2,
                                    self.pool2,
                                    self.f,
                                    self.lin1, self.r3, 
                                    self.lin2, self.r4,
                                    self.lin3)
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out
