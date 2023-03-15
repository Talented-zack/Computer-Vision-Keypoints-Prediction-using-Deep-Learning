## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

#define resnet
class Residual(nn.Module):
    def __init__(self,input_channels, num_channels, use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size = 3, padding =1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 
                               kernel_size = 3, padding =1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                               kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return(F.relu(X))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    block = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.append(Residual(input_channels, num_channels, use_1x1conv = True, strides = 2))
        else:
            block.append(Residual(num_channels, num_channels))
    return(block)
                  
            

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn. BatchNorm2d(64)
        self.maxpool =  nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(64, 128, 2))
        self.layer3 = nn.Sequential(*resnet_block(128, 256, 2))
        self.layer4 = nn.Sequential(*resnet_block(256, 512, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,68*2)
        

        

        
    def forward(self, x):


        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(self.layer4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x
