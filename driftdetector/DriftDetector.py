"""
Main file for the drift detector class.
"""
from torch.autograd import Variable
from torch.optim import Adadelta
from torch import Tensor, tensor, nn
from torch.utils.data import DataLoader

from driftdetector.GAN import Generator, Discriminator, Network
import numpy as np
import torch
import random


class DriftDetector:
    def __init__(self, device="cpu", epochs=125, steps_generator=100, equalize=True, max_count=100,
                 shuffle_discriminator=True, shuffle_generator=False, batch_size=8, lr=0.001,  rho=0.9, eps=0.000001,
                 weight_decay=0.000000005, training_window_size=50, generator_batch_size=1, threshold=0.50,
                 sequence_length=1, final_layer_incoming_connections=512):
        """

        :param device: "cpu" or "cuda", device on which the model should be trained
        :param epochs: Number of epochs for training the models
        :param steps_generator: Number of steps for training the generator. Helps improve training the generator
        :param equalize: Equalize the number of instances for training from different drifts. If equalize is True,
        the number of input features for each unique drift will be the minimum of all the unique drift features count
        :param max_count: Number of features to be taken for each unique drift. Default is 100
        :param shuffle_discriminator: Whether the dataset for the discriminator should be shuffled, default=True
        :param shuffle_generator: Whether the dataset for the generator should be shuffled, default=False
        :param batch_size: Batch size for training
        :param lr: learning rate for the Adadelta optimizer
        :param rho: rho value for the adadelta optimizer
        :param eps: eps value for adadelta optimizer
        :param weight_decay: weight decay for adadelta optimizer. Default for the model is set to 0.0005
        :param training_window_size: Number of instances to be accumulated for training on a particular drift
        :param generator_batch_size: batch size for the generator
        :param threshold: Threshold for below it is considered a drift. Default is 0.5
        :param sequence_length: Number of sequences from the past to be concatenated for prediction
        on the current feature
        :param final_layer_incoming_connections: Number of input connection in the final layer
        """

        self.device = device
        self.epochs = epochs
        self.equalize = equalize
        self.max_count = max_count
        self.shuffle_discriminator = shuffle_discriminator
        self.shuffle_generator = shuffle_generator
        self.final_layer_incoming_connections = final_layer_incoming_connections

        self.batch_size = batch_size
        self.steps_generator = steps_generator
        self.generator_batch_size = generator_batch_size
        self.training_window_size = training_window_size
        self.sequence_length = sequence_length
        self.temporary_label = [0]
        self.threshold = threshold

        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

        self.drifts_detected = []
        self.drifts_indices = None
        self.drift_labels = []
        self.generator_label = 1
        self.max_idx = 0
        self.drift_training_data = []

        self.generator = None
        self.optimizer_generator = None
        self.loss_mse_generator = nn.CrossEntropyLoss
        self.loss_generator = nn.MSELoss()

        self.discriminator = None
        self.optimizer_discriminator = None
        self.loss_discriminator = nn.CrossEntropyLoss()

    def initialize_detection_model(self, features):
        """
        Initializes the generator and discriminator models with all additional variables
        :param features:
        :return:
        """

        # Create the generator and discriminator and move it to the CPU/GPU
        self.generator = Generator(inp=features.shape[1], out=features.shape[1])
        self.discriminator = Discriminator(inp=features.shape[1], out=1)
        self.generator = self.generator.to(device=self.device)
        self.discriminator = self.discriminator.to(device=self.device)

        # Create the optimizers for the models
        self.optimizer_generator = Adadelta(self.generator.parameters(),
                                            lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)
        self.optimizer_discriminator = Adadelta(self.discriminator.parameters(),
                                                lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)

        generator, discriminator = self.train_gan(features=features, discriminator=self.discriminator,
                                                  generator=self.generator)
        self.generator = generator
        self.discriminator = discriminator

    def detect_drifts(self, data):

        # Predict the results
        # 0 - No drift and 1 for drifted data
        result = self.discriminator(Tensor(data).to(torch.float).to(self.device))

        # No drift detected
        if np.mean(result.cpu().detach().numpy()) > self.threshold:
            return 0

        return 1

    def retrain_model(self, old_features, new_features):

        # Create the dataset by appending ones to the
        ones = np.ones(shape=(len(new_features), 1))
        zeros = np.zeros(shape=(len(old_features), 1))
        x_old = np.hstack((old_features, zeros))
        x_new = np.hstack((new_features, ones))
        training_set = np.vstack((x_old, x_new))

        network = Network(inp=old_features.shape[1], out=1)
        network = network.double()
        network = network.to(self.device)

        dl = DataLoader(training_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate)
        optimizer = Adadelta(network.parameters())
        loss = nn.BCELoss()

        # Train network
        for idx in range(self.epochs):
            for batch_x, batch_y in dl:
                network.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                out = network(batch_x)
                curr_loss = loss(out, batch_y)
                curr_loss.backward()
                optimizer.step()

        return network

    def train_gan(self, features, discriminator, generator):
        """
        Trains the Generative Adversarial Network
        :param features: features to be trained
        :param discriminator: Discriminator object that needs to be trained
        :param generator: Generator object that needs to be trained
        :return: Trained generator and discriminator
        """

        # Losses for the generator and discriminator
        loss_mse_generator = nn.MSELoss()
        loss_generator = nn.BCELoss()
        loss_discriminator = nn.MSELoss()

        # Create the optimizers for the models
        optimizer_generator = Adadelta(generator.parameters(),
                                       lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)
        optimizer_discriminator = Adadelta(discriminator.parameters(),
                                           lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)

        # Label vectors
        ones = Variable(torch.ones(self.batch_size)).float().to(self.device)
        zeros = Variable(torch.zeros(self.generator_batch_size, 1)).float().to(self.device)

        # Define the data loader for training
        real_data = DataLoader(features, batch_size=self.batch_size, shuffle=self.shuffle_discriminator,
                               collate_fn=self.collate)

        concatenated_dataset = self.concatenate_features(data=features)
        generator_data = DataLoader(concatenated_dataset, batch_size=self.generator_batch_size,
                                    shuffle=self.shuffle_generator, collate_fn=self.collate_generator)

        # This is the label for new drifts (any input other than the currently learned distributions)
        generator_label = zeros
        discriminator_labels = ones

        for epochs_trained in range(self.epochs):

            discriminator = self.train_discriminator(real_data=real_data, fake_data=generator_data,
                                                     discriminator=discriminator, generator=generator,
                                                     optimizer=optimizer_discriminator, loss_fn=loss_discriminator,
                                                     generator_labels=generator_label,
                                                     discriminator_labels=discriminator_labels)

            generator = self.train_generator(data_loader=generator_data, discriminator=discriminator,
                                             generator=generator,  optimizer=optimizer_generator,
                                             loss_fn=loss_generator, loss_mse=loss_mse_generator,
                                             steps=self.steps_generator, labels=generator_label)

        return generator, discriminator

    def train_discriminator(self, real_data, fake_data, discriminator, generator, optimizer, loss_fn,
                            generator_labels, discriminator_labels):
        # for idx in range(steps):
        for features in real_data:
            # Set the gradients as zero
            discriminator.zero_grad()
            optimizer.zero_grad()

            # Get the loss when the real data is compared to ones
            features = features.to(self.device).to(torch.float)
            labels = discriminator_labels.to(self.device)

            # Get the output for the real features
            output_discriminator = self.discriminator(features)

            # The real data is without any concept drift. Evaluate loss against zeros
            real_data_loss = loss_fn(output_discriminator.flatten(), labels[:output_discriminator.shape[0]])

            # Get the output from the generator for the generated data compared to ones which is drifted data
            generator_input = None
            for input_sequence, _ in fake_data:
                generator_input = input_sequence.to(self.device).to(torch.float)
                break
            generated_output = generator(generator_input)

            generated_output_discriminator = discriminator(generated_output)

            # Here instead of ones it should be the label of the drift category
            generated_data_loss = loss_fn(generated_output_discriminator, generator_labels)

            # Add the loss and compute back prop
            total_iter_loss = generated_data_loss + real_data_loss
            total_iter_loss.backward()

            # Update parameters
            optimizer.step()

        # return the trained discriminator
        return discriminator

    def train_discriminator_random_data(self, discriminator, loader, optimizer, loss_fn):

        for data, labels in loader:

            # Zero the gradients
            discriminator.zero_grad()
            optimizer.zero_grad()

            # Get the loss when the real data is compared to ones
            data = data.to(self.device).to(torch.float)
            labels = labels.to(self.device)

            # Get the output for the real features
            output_discriminator = self.discriminator(data)

            # Compute the loss and then the gradients.
            data_loss = loss_fn(output_discriminator, labels)
            data_loss.backward()

            # Update the gradients
            optimizer.step()

        return discriminator

    def train_generator(self, data_loader, discriminator, generator, optimizer, loss_fn, loss_mse, steps, labels):
        epoch_loss = 0
        for idx in range(steps):

            optimizer.zero_grad()
            generator.zero_grad()

            generated_input = None
            for generator_input, target in data_loader:
                generated_input = generator_input.to(torch.float).to(self.device)
                target = target.to(torch.float).to(self.device)
                break

            # Generating data for input to generator
            generated_output = generator(generated_input)

            # Compute loss based on whether discriminator can discriminate real data from generated data
            generated_training_discriminator_output = discriminator(generated_output)

            # Compute loss based on ideal target values
            loss_generated = loss_fn(torch.reshape(generated_training_discriminator_output,
                                                   shape=(generated_training_discriminator_output.shape[0], 1)), labels)

            loss_lstm = loss_mse(generated_output, target)

            total_generator_loss = loss_generated + loss_lstm

            # Back prop and parameter update
            total_generator_loss.backward()
            optimizer.step()
            epoch_loss += total_generator_loss.item()

        return generator

    def collate(self, batch):
        """
        Function for collating the batch to be used by the data loader. This function does not handle labels
        :param batch:
        :return:
        """
        # Stack each tensor variable
        x = torch.stack([tensor(x) for x in batch])
        # Return features
        return x

    def collate_generator(self, batch):
        """
        Function for collating the batch to be used by the data loader. This function does handle labels
        :param batch:
        :return:
        """

        # Stack each tensor variable
        feature_length = int(len(batch[0]) / (self.sequence_length + 1))
        # The last feature length corresponds to the feature we want to predict and
        # the last value is the label of the drift class
        x = torch.stack([torch.Tensor(np.reshape(x[:-feature_length], newshape=(self.sequence_length, feature_length)))
                         for x in batch])
        y = torch.stack([torch.tensor(x[-feature_length:]) for x in batch])
        # Return features and targets
        return x.to(torch.double), y

    def concatenate_features(self, data):

        idx = self.sequence_length
        modified_data = np.vstack((np.zeros((self.sequence_length - 1, len(data[idx]))), data))
        output = np.hstack((modified_data[idx - self.sequence_length:idx + 1, :].flatten()))
        idx += 1
        while idx < len(modified_data) - 1:
            output = np.vstack((output, np.hstack((modified_data[idx - self.sequence_length:idx + 1, :].flatten()))))
            idx += 1

        # The last value
        output = np.vstack((output, np.hstack((modified_data[idx - self.sequence_length:, :].flatten()))))
        output = np.vstack((output, np.hstack((modified_data[idx - self.sequence_length:idx, :].flatten(),
                                               modified_data[self.sequence_length - 1]))))
        return output
