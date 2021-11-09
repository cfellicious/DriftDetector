"""
This file contains the Discriminator and Generator parts for training the GAN
"""

from torch import nn
from torch.nn import Sequential, Linear, ReLU, Module, BatchNorm1d, Dropout


class Generator(Module):
    def __init__(self, inp, out):
        super(Generator, self).__init__()
        self.net = Sequential(Linear(inp, 128), nn.ReLU(),
                              Linear(128, 128), nn.Sigmoid(),
                              Linear(128, out))

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(Module):
    def __init__(self, inp, out):
        super(Discriminator, self).__init__()
        self.net = Sequential(Linear(inp, 128), ReLU(inplace=True),
                              Linear(128, 256),
                              Linear(256, 512),
                              Dropout(inplace=True),
                              Linear(512, out), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x


class Network(Module):
    def __init__(self, inp, out):
        super(Network, self).__init__()
        self.net = Sequential(BatchNorm1d(num_features=inp),
                              Linear(inp, 128), ReLU(inplace=True),
                              Linear(128, 256),
                              Linear(256, 512),
                              Dropout(inplace=True),
                              Linear(512, out), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x
