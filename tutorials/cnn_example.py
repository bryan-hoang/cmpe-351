# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import transforms
import torch.nn.functional as F

# %% pycharm={"name": "#%%\n"}
# hyper parameters
batch_size = 32
num_classes = 10
learning_rate = 0.001
num_epochs = 2

# %% pycharm={"name": "#%%\n"}
# Use transforms.compose method to reformat images for modeling, we are getting a normalization from a range 0 to 1 to -1 to 1
all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.247, 0.243, 0.261])
                                     ])
# Create Training dataset
train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                             train = True,
                                             transform = all_transforms,
                                             download = True)

# Create Testing dataset
test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            transform = all_transforms,
                                            download=True)

# Create loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)

# check number of train/test data
print(len(train_loader))
print(len(test_loader))

# %% pycharm={"name": "#%%\n"}
# a small trial test for shape parameters
dataiter = iter(train_loader)
images, labels = dataiter.next()

conv1 = nn.Conv2d(3,6,5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6,16,5)
conv3 = nn.Conv2d(16,32,5)
print(images.shape)
trials = conv1(images)
print(trials.shape)
trials = pool(trials)
print(trials.shape)
trials = conv2(trials)
print(trials.shape)
trials = pool(trials)
print(trials.shape)


# %% pycharm={"name": "#%%\n"}
# Creating a CNN class
class MyCNN(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) #Size of kernels, i. e. of size 5x5
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(400, 128) #400 = 16 *5 *5
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    # Progresses data across layers
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.pool(out)

        out = self.relu(self.conv2(out))
        out = self.pool(out)

        out = out.reshape(out.size(0), -1)

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return out


# %% pycharm={"name": "#%%\n"}
model = MyCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

# %% pycharm={"name": "#%%\n"}
#training
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    #Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 500 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():4f}')

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))




# %% pycharm={"name": "#%%\n"}
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))

    for images, labels in train_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} train images: {} %'.format(len(train_loader), 100 * correct / total))


# %% pycharm={"name": "#%%\n"}
