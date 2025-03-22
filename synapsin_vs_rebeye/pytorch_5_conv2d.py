"""
-- Script to train pytorch based model to classify synapses.
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

#import helper


# Setting the random seed for reproducible results
SEED = 123

tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# def main():

experiment_directory = "C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\\"
validation_directory = experiment_directory + "validation\\"

# Number of training epochs.
num_epochs = 64

experiment_name = "pytorch_5_conv2d_num_epochs_{}" .format(num_epochs)



# # Normalizing the image
# transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# Dataset path. 
dataset = datasets.ImageFolder("C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\\data\\", transform=transform)
# dataset = datasets.ImageFolder('C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\\data\\')


# Training and Testing datasets
test_split = 0.25
#Shuffle_dataset = True
#random_seed = 8
batch_size = 10

dataset_size = len(dataset)
indices  = list(range(dataset_size))
split = int(np.floor(test_split*dataset_size))

np.random.shuffle(indices)    # Shuffling the indices

train_indices, test_indices = indices[split:], indices[:split]

# Creating data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
# test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=2)

print("Images in Training set: {}\nImages in Testing set: {}".format(len(train_indices), len(test_indices)))
# print(len(train_loader.dataset))

#batch_size = 4
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Define classes.
classes = ['Ribeye', 'Synapsin_P21_647']

# # Show sample images.
# def imshow(img, label):
    # # img = img / 2 + 0.5     # unnormalize the image (from (-1,1) to (0,1))
    # npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.title(label)
    # plt.show()
    
# images, labels = next(iter(train_loader))    # data loader is a generator and to get data out of it, we need to loop through it or convert it to an iterator and call next()
# #helper.imshow(images[0], normalize=False)    

# images, labels = next(iter(train_loader))    # data loader is a generator and to get data out of it, we need to loop through it or convert it to an iterator and call next()
# #helper.imshow(images[0], normalize=False)

# # show images
# imshow(images[2], classes[labels[2]])


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

net = Net()

# Use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net.to(device)

print(device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training.

# Log the training history.
# Path to saved training history.
# Create new directory for training history if it does not exist.
if not os.path.exists(experiment_directory + "saved_training_history\\"):
    os.mkdir(experiment_directory + "saved_training_history\\")
 
history_directory = experiment_directory + "saved_training_history\\"
history_file = history_directory + experiment_name + "training_history.csv"

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
#        inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()    # item() method extracts the loss’s value as a Python float
        if i % 500 == 499:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, i + 1, running_loss / 500))
            
            # Save training history to file.
            with open(history_file, "a") as f_out:
                f_out.write("[{}, {}] loss: {}\n" .format(epoch + 1, i + 1, running_loss / 500))

            running_loss = 0.0
                

print('Finished Training')

# # Get model-save-path.
# model_save_path = "C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\\trained_models\\pytorch_5_conv2d_num_epochs_{}.pt" .format(num_epochs)


# Get model-save-path.
# model_save_path = "C:\\Users\\swapnil.yadav\\Research\\synapse_classification\\synapsin_vs_rebeye\\trained_models\\pytorch_5_conv2d_balanced_data_num_epochs_{}.pt" .format(num_epochs)
# model_save_path = experiment_directory + "trained_models\\pytorch_5_conv2d_balanced_data_num_epochs_{}.pt" .format(num_epochs)
model_save_path = experiment_directory + "trained_models\\" + experiment_name + ".pt" 




# # Save the trained model using state_dict.
# torch.save(net.state_dict(), model_save_path)

# Save the entire trained model.
torch.save(net, model_save_path)


# Validation.
images, labels = next(iter(test_loader))    # data loader is a generator and to get data out of it, we need to loop through it or convert it to an iterator and call next()
images, labels = images.to(device), labels.to(device)

#helper.imshow(images[0], normalize=False)

# # show images
# imshow(images[0], classes[labels[0]])

# Network outout on test images
outputs = net(images)

# Get maximum probability prediction
_, predicted = torch.max(outputs, 1)    # Returns a namedtuple (values, indices)

# # Compare prediction with image labels
# #for j in np.random.randint(0, high=len(test_loader), size=4, dtype=int):
# for j in np.random.randint(low=0, high=4, size=4, dtype=int): 
    # imshow(images[j], classes[labels[j]])
    # print('Diagnosis for image {}: {}' .format(j, classes[predicted[j]]),"\n")
    
# Training accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in train_loader:
#        images, labels = data
        images, labels = data[0].to(device), data[1].to(device)

        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the training images: %d %%' % (
    100 * correct / total))
    
# Save accuracy results to file.
with open(validation_directory + experiment_name + "_accuracy.csv","a") as f_out:
    f_out.write("Accuracy of the network on the training images: {}\n" .format(100 * correct / total))    

training_accuracy = correct / total

# Testing accuracy

predicted_test = []
labels_test = []

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
#        images, labels = data
        images, labels = data[0].to(device), data[1].to(device)

        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels.tolist())):
            predicted_test.append(predicted.tolist()[i])
            labels_test.append(labels.tolist()[i])        

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    
# Save accuracy results to file.
with open(validation_directory + experiment_name + "_accuracy.csv","a") as f_out:
    f_out.write("Accuracy of the network on the test images: {}\n" .format(100 * correct / total))    

testing_accuracy = correct / total



# Prepare to count predictions for each class.
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# Again no gradients needed.
with torch.no_grad():
    for data in test_loader:
#        images, labels = data
        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} % \n".format(classname,
                                                   accuracy))
                                                   
    # Save accuracy results to file.
    with open(validation_directory + experiment_name + "_accuracy.csv","a") as f_out:
        f_out.write("Accuracy for class {:5s} is: {:.1f} % \n".format(classname, accuracy))                                                         
                                                   
                                                   
result_cols = ['Model Name',
              'Training Accuracy',
              'Testing Accuracy',
              'Precision',
              'Recall',
              'F1-Score',
              'ROC AUC']
results = []



precision = metrics.precision_score(np.array(labels_test), np.array(predicted_test))
recall = metrics.recall_score(np.array(labels_test), np.array(predicted_test))
f1 = metrics.f1_score(np.array(labels_test), np.array(predicted_test))
auc = metrics.roc_auc_score(np.array(labels_test), np.array(predicted_test))    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.


# Add to results list
results.append(['Rebeye_vs_synapsin_pytorch_5_conv2d', training_accuracy, testing_accuracy, precision, recall, f1, auc])

df_results = pd.DataFrame(results, columns=result_cols)
print(df_results) 

# filepath = validation_directory + experiment_name + "_metric.csv"
# filepath.parent.mkdir(parents=True, exist_ok=True)  
# df_results.to_csv(filepath) 

if not os.path.exists(validation_directory):
    os.mkdir(validation_directory)
 
filepath = validation_directory + experiment_name + "_metric.csv"
df_results.to_csv(filepath) 

# if __name__ == '__main__':
    # main()    
                                                  







