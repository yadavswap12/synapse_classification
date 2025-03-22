"""
-- Script to validate pytorch based model to classify synapses.
-- Model has 5 conv2d layers. 

Swapnil 4/23
"""

import os
import numpy as np
import pandas as pd
import urllib
import math

import random

import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics

import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim

# Setting the random seed for reproducible results
SEED = 123

tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# # Normalizing the image
# transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# Dataset path. 
dataset = datasets.ImageFolder("C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\data\\", transform=transform)
# dataset = datasets.ImageFolder('C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\\data\\')

# Get path for trained model.
num_epochs = 32
model_save_path = "C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\\trained_models\\pytorch_5_conv2d_num_epochs_{}.pt" .format(num_epochs)

# Get validation dataset.
test_split = 0.25
#Shuffle_dataset = True
#random_seed = 8
batch_size = 10

dataset_size = len(dataset)
indices  = list(range(dataset_size))
split = int(np.floor(test_split*dataset_size))

np.random.shuffle(indices)    # Shuffling the indices

test_indices = indices[:split]

# Creating data samplers and loaders
test_sampler = SubsetRandomSampler(test_indices)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


# Define classes.
classes = ['Ribeye', 'Synapsin_P21_647']

# Define a Convolutional Neural Network.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2, padding_mode='zeros')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)    
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2, padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, padding_mode='zeros')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, padding_mode='zeros')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, padding_mode='zeros')
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, padding_mode='zeros')

        self.fc1 = nn.Linear(in_features=256 * 2 * 2, out_features=512)    
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.fc4 = nn.Linear(in_features=10, out_features=2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
#        print(x.shape)

        x = x.view(-1, 256 * 2 * 2)    # PyTorch allows a tensor to be a View of an existing tensor
#        x = x.view(x.size(0), -1)    # flatten out a input for Dense Layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# # Load trained model using state_dict.
# net = Net(*args, **kwargs)
# net.load_state_dict(torch.load(model_save_path))
# net.eval()

# Model class must be defined somewhere
net = torch.load(model_save_path)
net.eval()

#Use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

class0_incorrect_images = []
class0_incorrect_lables = []
class0_incorrect_predictions = []

class1_incorrect_images = []
class1_incorrect_lables = []
class1_incorrect_predictions = []

# Get incorrect class predictions.
with torch.no_grad():
    for data in test_loader:
#        images, labels = data
        images, labels = data[0].to(device), data[1].to(device)


        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        
        
        # if (labels.tolist()[0] == 0) & (predicted.tolist()[0] != labels.tolist()[0]):
            # class0_incorrect_images.append(images)
            # class0_incorrect_lables.append(labels)
            # class0_incorrect_predictions.append(predicted)
            
        
        
        
        # Get incorrect prediction mask.
        class0_incorrect_mask = [True if (y==0 and x!=y) else False for x,y in zip(predicted, labels)]
        class1_incorrect_mask = [True if (y==1 and x!=y) else False for x,y in zip(predicted, labels)]        
        
        # Get images for incorrect prediction.        
        # class0_incorrect_images = [x if y==True for x, y in zip(images, class0_incorrect_mask)]
        class0_incorrect_images.extend([x for x, y in zip(images, class0_incorrect_mask) if y==True])
        class1_incorrect_images.extend([x for x, y in zip(images, class1_incorrect_mask) if y==True])        

        
        # Get labels for incorrect prediction.        
        # class0_incorrect_lables = [x if y==True for x, y in zip(labels, class0_incorrect_mask)]
        class0_incorrect_lables.extend([x for x, y in zip(labels, class0_incorrect_mask) if y==True])
        class1_incorrect_lables.extend([x for x, y in zip(labels, class1_incorrect_mask) if y==True])

        # Get prediction labels for incorrect predictions.        
        # class0_incorrect_predictions = [x if y==True for x, y in zip(predicted, class0_incorrect_mask)]          
        class0_incorrect_predictions = [x for x, y in zip(predicted, class0_incorrect_mask) if y==True]          
        class1_incorrect_predictions = [x for x, y in zip(predicted, class1_incorrect_mask) if y==True]
        
        
        
        
        
# Show sample images.
def imshow(img, label):
    # img = img / 2 + 0.5     # unnormalize the image (from (-1,1) to (0,1))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(label)
    plt.show()   

# Set number of images to look at.
num_img = 10

# print(len(class0_incorrect_images)) 
# print(type(class0_incorrect_images[0])) 
# print(class0_incorrect_images[0].size(0))                 
# print(class0_incorrect_lables[0].size(0))  
print(len(class0_incorrect_images))               
                

# for j in np.random.randint(low=0, high=len(class0_incorrect_images), size=num_img, dtype=int): 
    # imshow(class0_incorrect_images[j], classes[class0_incorrect_lables[j]])
    # # print('image {} has prediction {} and label {}' .format(j, classes[class0_incorrect_predictions[j]], classes[class0_incorrect_lables[j]]),"\n")
    # # print('image {} has prediction {} and label {}' .format(j, class0_incorrect_predictions[j], classes[class0_incorrect_lables[j]]),"\n")
    # print('image {} has label {}' .format(j, classes[class0_incorrect_lables[j]]),"\n")
    
    
            
            
            
    




