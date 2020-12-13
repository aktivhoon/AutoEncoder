# main.py
import os
import argparse
torch.backends.cudnn.benchmark = True
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import ConvAutoencoder

def arg_parse():
    desc = "AutoEncoder Practice"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")

    return parser.parse_args()

def __name__ == '__main__':
    arg = arg_parse()
    transform = transforms.ToTensor()

    # Download the training and test datasets
    train_data = datasets.CIFAR10(root='data', train=True, download=True)
    test_data = datasets.CIFAR10(root='data', train=False, download=True)

    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0)

    # Instantiate the model
    net = ConvAutoencoder()

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    net = nn.DataParallel(net).to(torch_device)

    model = AETrainer(net, torch_device, loss = criterion)

    model.train(train_loader)
