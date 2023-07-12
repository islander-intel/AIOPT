import torch
from torch import nn
class LeNet(nn.Module):
    def __init__(self,numChannels = 1, classes=10): # numChannels can be 1 for gray or 3 for rgb
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=numChannels,out_channels=20,kernel_size=(5,5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.fc1 = nn.Sequential(
            nn.Linear(800,500),
            nn.SELU())
        self.fc2 = nn.Linear(in_features=500,out_features=classes)
        self.out = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()
    def forward(self,input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)