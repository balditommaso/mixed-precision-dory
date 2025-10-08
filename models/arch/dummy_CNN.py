from torch import nn, tensor
from typing import *


class DummyCNN(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2, 2)  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14 * 14, num_classes)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
        