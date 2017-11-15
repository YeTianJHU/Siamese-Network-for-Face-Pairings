import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import copy

class SIAMESE(nn.Module):
    def __init__(self):
        super(SIAMESE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = (5,5), stride = (1,1), padding = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2, stride=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5,5), stride = (1,1), padding = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size = 2, stride=(2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size = 2, stride=(2, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
            
        self.fc1 = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
        )
        
        self.fc2 = nn.Linear(2048, 1)
   
    
    def forward_once(self, x):
         output = self.conv1(x)
         output = self.conv2(output)
         output = self.conv3(output)
         output = self.conv4(output)
         output = output.view((output.data.size())[0], -1)
         output = self.fc1(output)
         return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        concated = torch.cat((output1, output2),1)
        output = self.fc2(concated)
        output = torch.sigmoid(output)
        return output

class SIAMESE2(nn.Module):
    def __init__(self):
        super(SIAMESE2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = (5,5), stride = (1,1), padding = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2, stride=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5,5), stride = (1,1), padding = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size = 2, stride=(2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size = 2, stride=(2, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
            
        self.fc1 = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
        )
            
    def forward_once(self, x):
         output = self.conv1(x)
         output = self.conv2(output)
         output = self.conv3(output)
         output = self.conv4(output)
         output = output.view((output.data.size())[0], -1)
         output = self.fc1(output)
         return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2