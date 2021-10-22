"""
This file contains the Discriminator and Generator parts for training the GAN
"""

from torch import nn
from torch.nn import Sequential, Linear, ReLU, Module


class Generator(Module):
    def __init__(self, inp, sequence_length=2):
        """
        Initialization function of the Generator module
        :param inp: Input feature length
        :param sequence_length: Input sequence length. Input layer connections is input feature length * sequence length
        """
        super(Generator, self).__init__()
        self.net = Sequential(
            Linear(inp*sequence_length, 128),
            Linear(128, 4096), ReLU(inplace=True),
            Linear(4096, inp)
        )

    def forward(self, x_):
        output = self.net(x_.reshape(x_.shape[0], x_.shape[1] * x_.shape[2]))
        return output


class Discriminator(Module):
    def __init__(self, inp, final_layer_incoming_connections=512, reset_layers=False):
        """
        Initialization function of the Discriminator
        :param inp: Input feature length
        :param final_layer_incoming_connections: Number of connections into the final layer
        :param reset_layers: Resets the layers every time a retrain happens
        """
        super(Discriminator, self).__init__()
        self.input_connections = inp
        self.neuron_count = 2
        self.incoming_connections = final_layer_incoming_connections

        self.net = self.create_network()

        self.neurons = Linear(final_layer_incoming_connections, self.neuron_count)
        self.softmax = nn.Softmax(dim=1)
        self._reset_layers = reset_layers

    def forward(self, x_):
        result = self.net(x_)
        result = self.neurons(result)
        result = self.softmax(result)
        return result

    def update(self):
        """
        Resets the top layer and adds a neuron to the classification layer as an entirely new drift is detected
        :return:
        """
        if self._reset_layers:
            self.reset_layers()
        self.neuron_count += 1
        layer = Linear(self.incoming_connections, self.neuron_count)
        self.neurons = layer
        return

    def reset_top_layer(self):
        """
        Resets the top layer when a previous drift occurs
        :return:
        """
        if self._reset_layers:
            self.reset_layers()
        layer = Linear(self.incoming_connections, self.neuron_count)
        self.neurons = layer
        return

    def reset_layers(self):
        """
        Recreates the network
        :return:
        """
        self.net = self.create_network()

    def create_network(self):
        """
        Creates the network
        :return:
        """
        net = Sequential(
            Linear(self.input_connections, 1024), ReLU(inplace=True),
            Linear(1024, 1024), ReLU(inplace=True),
            Linear(1024, self.incoming_connections),
            nn.Sigmoid())
        return net
