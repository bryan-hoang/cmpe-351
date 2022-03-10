# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.9.10 ('cmpe-351-_rWzjxJw')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CMPE 351 Assignment 2
#
# We'll first import the necessary python packages to run the code in the
# notebook.
#

# %%
# Importing packages.
from os.path import dirname, join, realpath
from typing import Any, Callable, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import transforms


# %%
def is_interactive():
    """Determine if the current session is interactive."""
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    SCRIPT_DIR = dirname(realpath("__file__"))
else:
    SCRIPT_DIR = dirname(realpath(__file__))

DATA_DIR = join(SCRIPT_DIR, "data")
IMG_DIR = join(DATA_DIR, "img")

# %% [markdown]
#
# ## Part 1: Image Classification using CNN (50 points)

# %%
# Hyper parameters
EPOCH_COUNT = 2
CLASSES_COUNT = 13
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# %%
class FashionProductImageDataset(VisionDataset):
    """Fashion product images dataset."""

    classes = [
        "Topwear",
        "Bottomwear",
        "Innerwear",
        "Bags",
        "Watches",
        "Jewellery",
        "Eyewear",
        "Wallets",
        "Shoes",
        "Sandal",
        "Makeup",
        "Fragrance",
        "Others",
    ]

    target_encoder = LabelEncoder()

    def __init__(
        self,
        root: str,
        transform: Callable = None,
        target_transform: Callable = None,
        targets_file: str = None,
    ):
        """Construct FashionProductImageDataset.

        Args:
            root (string): Root directory of dataset where directory
                ``cifar-10-batches-py`` exists or will be saved to if download
                is set to True.
            transform (callable, optional): A function/transform that takes in
                an PIL image and returns a transformed version.
                E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
            targets_file (string): Path to the csv file with annotations.
        """
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )

        fashion_products_data_frame = pd.read_csv(targets_file, sep="\t")

        self.img_ids = fashion_products_data_frame[:]["imageid"]

        self.target_encoder.fit(FashionProductImageDataset.classes)
        self.targets = self.target_encoder.transform(
            fashion_products_data_frame["label"]
        )

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.img_ids)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get the image and target for the given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = (
            Image.open(
                join(self.root, f"{self.img_ids.iloc[index]}.jpg")
            ).convert("RGB"),
            self.targets[index],
        )

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


# %%
img_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# %%
train_dataset = FashionProductImageDataset(
    join(DATA_DIR, "img"),
    transform=img_transform,
    targets_file=join(DATA_DIR, "train.csv"),
)

test_dataset = FashionProductImageDataset(
    join(DATA_DIR, "img"),
    transform=img_transform,
    targets_file=join(DATA_DIR, "test.csv"),
)

# Create loader objects to facilitate processing
train_loader: DataLoader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)


test_loader: DataLoader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# Check number of train/test data
print(len(train_loader))
print(len(test_loader))


# %%
def trial_layers(train_loader):
    """Test the shape of the layers for a CNN."""
    dataiter = iter(train_loader)
    (images) = dataiter.next()

    conv1 = nn.Conv2d(3, 6, 5)
    pool = nn.MaxPool2d(2, 2)
    conv2 = nn.Conv2d(6, 16, 5)
    print(images.shape)
    trials = conv1(images)
    print(trials.shape)
    trials = pool(trials)
    print(trials.shape)
    trials = conv2(trials)
    print(trials.shape)
    trials = pool(trials)
    print(trials.shape)


trial_layers(train_loader)


# %%
# Creating a CNN class
class ConvolutionalNeuralNetwork(nn.Module):
    """CNN."""

    def __init__(self, num_classes):
        """Determine what layers and their order in CNN object."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Progresses data across layers."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# %%
model = ConvolutionalNeuralNetwork(CLASSES_COUNT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, weight_decay=0.005, momentum=0.9
)


# %%
def train_model(epoch_count, train_loader, model, criterion, optimizer):
    """Train a DL model."""
    n_total_steps = len(train_loader)

    for epoch in range(epoch_count):
        # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 500 == 0:
                print(
                    f"epoch {epoch+1}/{epoch_count},"
                    f" step {i+1}/{n_total_steps}, loss = {loss.item():4f}"
                )

        print(f"Epoch [{epoch + 1}/{epoch_count}], Loss: {loss.item():.4f}")


train_model(EPOCH_COUNT, train_loader, model, criterion, optimizer)


# %%
def evaluate_model(train_loader, test_loader, model):
    """Evaluate a DL model."""
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the {len(test_loader)} test images:"
            f" {100 * correct / total} %"
        )

        for images, labels in train_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the network on the {len(train_loader)} train images:"
            f" {100 * correct / total} %"
        )


evaluate_model(train_loader, test_loader, model)

# %% [markdown]
# ## Part 2: Improved Image Classification (50 points)

# %%
# Hyper parameters
EPOCH_COUNT = 2
CLASSES_COUNT = 13
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# %%
def trial_layers_v2(train_loader):
    """Test the shape of the layers for a CNN."""
    dataiter = iter(train_loader)
    (images) = dataiter.next()

    conv1 = nn.Conv2d(3, 6, 5)
    pool = nn.MaxPool2d(2, 2)
    conv2 = nn.Conv2d(6, 16, 5)
    print(images.shape)
    trials = conv1(images)
    print(trials.shape)
    trials = pool(trials)
    print(trials.shape)
    trials = conv2(trials)
    print(trials.shape)
    trials = pool(trials)
    print(trials.shape)


# %%
class ConvolutionalNeuralNetworkV2(nn.Module):
    """CNN."""

    def __init__(self, num_classes):
        """Determine what layers and their order in CNN object."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """Progresses data across layers."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# %%
model = ConvolutionalNeuralNetwork(CLASSES_COUNT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, weight_decay=0.005, momentum=0.9
)

train_model(EPOCH_COUNT, train_loader, model, criterion, optimizer)
# %%

evaluate_model(train_loader, test_loader, model)
