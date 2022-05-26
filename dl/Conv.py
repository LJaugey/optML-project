import torch.nn as nn

#=======================NETWORK=======================#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #layers
        self.C1 = nn.Conv2d(2, 32, 3)
        self.C2 = nn.Conv2d(32, 64, 3)
        self.C3 = nn.Conv2d(64, 128, 2)
        self.L1 = nn.Linear(128, 50)
        self.L2 = nn.Linear(50, 1)
        


        self.layers = nn.Sequential(
            self.C1, nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            self.C2, nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
            self.C3, nn.Dropout2d(0.2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            self.L1,
            nn.LeakyReLU(0.1),
            self.L2,
            nn.Sigmoid()
        )


        
        # weight initialization

        for l in self.layers:

            # Only initialize linear of convolutional layers' weights
            if(type(l) == nn.Linear or type(l) == nn.Conv2d):
                
                # sigmoid
                if(l == self.L2):

                    nn.init.xavier_uniform_(l.weight, gain=1)
                
                # leaky ReLU
                else:
                    nn.init.kaiming_uniform_(l.weight, a = 0.1, mode = 'fan_in', nonlinearity = 'leaky_relu')
                

                l.bias.data.fill_(0)


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        return x




