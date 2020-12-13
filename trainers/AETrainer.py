import os
from multiprocessing import Pool, Queue, Process

import scipy
import utils
import numpy as np

import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer

class AETrainer(BaseTrainer):
    def __init__(self, arg, G, torch_device, loss):
        super(AETrainer, self).__init__(arg, torch_device)
        self.loss = loss
        self.G = G
        self.optim = torch.optim.Adam(self.G.parameters(), lr = 0.001)

    def train(self, train_loader):
        print("\nStart Train")

        for epoch in range(self.start_epoch, self.epoch):
            train_loss = 0.0
            for data in train_loader:
                images, _ = data
                images = images.to(self.torch_device)
                self.optimizer.zero_grad()
                outputs = self.G(images)

                loss = self.loss(outputs, images)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss = train_loss/len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
