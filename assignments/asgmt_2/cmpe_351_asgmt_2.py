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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


# %%
def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


if is_interactive():
    script_dir = dirname(realpath("__file__"))
else:
    script_dir = dirname(realpath(__file__))

# %% [markdown]
#
# ## Part 1: Image Classification using CNN (50 points)

# %%
# Hyper parameters
epochs_count = 2
classes_count = 10
batch_size = 32
learning_rate = 0.001


# %%
class FashionProductImageDataset(Dataset):
    """Fashion product images dataset."""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fashion_products_data_frame = pd.read_csv(csv_file, sep="\t")
        self.img_dir = img_dir
        self.transform = transform

        self.image_names = self.fashion_products_data_frame[:]["imageid"]
        self.labels = self.fashion_products_data_frame["label"]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = io.imread(
            join(self.img_dir, f"{self.image_names.iloc[index]}.jpg")
        )

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        sample = {"image": image, "label": label}

        return sample


# %%
training_data_frame = FashionProductImageDataset(
    join(script_dir, "data/train.csv"), join(script_dir, "data/img/")
)

print(training_data_frame[1])
# test_data_frame = pd.read_csv(join(script_dir, "data/test.csv"))
# training_dataset = torchvision.datasets.MNIST(
#     root="./data", train=True, transform=transforms.ToTensor(), download=True
# )
