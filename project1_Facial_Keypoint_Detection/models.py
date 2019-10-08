## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, p=0.2):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 5) # 224x224x1 --> 220x220x32
        self.conv2 = nn.Conv2d(32, 64, 5) # 110x110x32 --> 106x106x64
        self.conv3 = nn.Conv2d(64, 128, 5) # 53x53x64 --> 49x49x128
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(24*24*128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 68*2)
        
        self.conv1_bn = nn.BatchNorm2d(32) # Use batch norm near the beginning of the network
        self.dropout = nn.Dropout(p=p) # Use dropout near the end of the network
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
                                                    # x --> 224x224x1
        x = F.relu(self.conv1_bn(self.conv1(x)))    # x --> 220x220x32
        x = self.pool(x)                            # x --> 110x110x32
        x = F.relu(self.conv2(x))                   # x --> 106x106x64
        x = self.pool(x)                            # x --> 53x53x64
        x = F.relu(self.conv3(x))                   # x --> 49x49x128
        x = self.pool(x)                            # x --> 24x24x128
        x = x.view(x.shape[0], -1)                  # x --> 73728
        x = F.relu(self.fc1(x))                     # x --> 1024
        x = F.relu(self.fc2(x))                     # x --> 512
        x = F.relu(self.fc3(x))                     # x --> 256
        x = self.dropout(x)
        x = self.fc4(x)                             # x --> 136
        
        # Since this is a regression problem, the output shouldn't be treated as a prob-distribution (softmax)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class DeepNet(nn.Module):

    def __init__(self, p=0.2):
        super(DeepNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 5) # 224x224x1 --> 220x220x32
        self.conv2 = nn.Conv2d(32, 64, 5) # 110x110x32 --> 106x106x64
        self.conv3 = nn.Conv2d(64, 128, 5) # 53x53x64 --> 49x49x128
        self.conv4 = nn.Conv2d(128, 256, 5) # 24x24x128 --> 20x20x256
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(10*10*256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 68*2)
        
        self.conv1_bn = nn.BatchNorm2d(32) # Use batch norm near the beginning of the network
        self.dropout = nn.Dropout(p=p) # Use dropout near the end of the network
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
                                                    # x --> 224x224x1
        x = F.relu(self.conv1_bn(self.conv1(x)))    # x --> 220x220x32
        x = self.pool(x)                            # x --> 110x110x32
        x = F.relu(self.conv2(x))                   # x --> 106x106x64
        x = self.pool(x)                            # x --> 53x53x64
        x = F.relu(self.conv3(x))                   # x --> 49x49x128
        x = self.pool(x)                            # x --> 24x24x128
        x = F.relu(self.conv4(x))                   # x --> 20x20x256
        x = self.pool(x)                            # x --> 10x10x256
        x = x.view(x.shape[0], -1)                  # x --> 25600
        x = F.relu(self.fc1(x))                     # x --> 1024
        x = F.relu(self.fc2(x))                     # x --> 512
        x = F.relu(self.fc3(x))                     # x --> 256
        x = self.dropout(x)
        x = self.fc4(x)                             # x --> 136
        
        # Since this is a regression problem, the output shouldn't be treated as a prob-distribution (softmax)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
