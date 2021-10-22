"""
Main file for the drift detector class.
"""
from torch.autograd import Variable
from torch.optim import Adadelta
from torch import Tensor, tensor, nn
from torch.utils.data import DataLoader

from driftdetector.GAN import Generator, Discriminator
import numpy as np
import torch
import random


class DriftDetector:
    def __init__(self, device="cpu", epochs=125, steps_generator=100, equalize=True, max_count=100,
                 shuffle_discriminator=True, shuffle_generator=False, batch_size=8, lr=0.001,
                 rho=0.9, eps=0.000001, weight_decay=0.000000005, training_window_size=50, generator_batch_size=1,
                 sequence_length=2, final_layer_incoming_connections=512):
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
        self.generator = Generator(inp=features.shape[1], sequence_length=self.sequence_length)
        self.discriminator = Discriminator(inp=features.shape[1],
                                           final_layer_incoming_connections=self.final_layer_incoming_connections)
        self.generator = self.generator.to(device=self.device)
        self.discriminator = self.discriminator.to(device=self.device)

        # Create the optimizers for the models
        self.optimizer_generator = Adadelta(self.generator.parameters(),
                                            lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)
        self.optimizer_discriminator = Adadelta(self.discriminator.parameters(),
                                                lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)

        self.drifts_indices = [(0, self.training_window_size)]
        self.drift_training_data = np.reshape(features, (1, features.shape[0], features.shape[1]))

        training_dataset = self.create_training_dataset(self.drift_training_data, [0])
        generator, discriminator = self.train_gan(features=training_dataset, discriminator=self.discriminator,
                                                  generator=self.generator, max_label=self.generator_label)
        self.generator = generator
        self.discriminator = discriminator

    def detect_drifts(self, data):
        result = self.discriminator(Tensor(data).to(torch.float).to(self.device))
        prob, max_idx = torch.max(result, dim=1)
        max_idx = max_idx.cpu().detach().numpy()

        # No drift detected
        if np.all(max_idx != max_idx[0]) or max_idx[0] == 0:
            return 0

        self.max_idx = max_idx[0]
        return max_idx[0]

    def retrain_model(self, data, index):

        # Drift has been detected
        self.drifts_indices.append((index, index + self.training_window_size))

        # Add the data to the training dataset
        self.drift_training_data = np.vstack((self.drift_training_data,
                                              np.reshape(data, (1, data.shape[0], data.shape[1]))))

        # add the index of the previous drift if it was a recurring drift
        # If the previous drift is not 0, add that as the temporary label
        # Else append it as a new drift
        if self.temporary_label[0] != 0:
            self.drift_labels.append(self.temporary_label[0])

        else:
            self.drift_labels.append(self.generator_label)

        if self.max_idx != self.generator_label:
            # Increase the max_idx by 1 if it is above the previous drift
            if self.temporary_label[0] <= self.max_idx and self.temporary_label[0] != 0:
                self.max_idx += 1
            self.temporary_label = [self.max_idx]
            # We reset the top layer predictions because the drift order has changed and the network should be retrained
            self.discriminator.reset_top_layer()
            self.discriminator = self.discriminator.to(self.device)

        else:
            # If this is a new drift, label for the previous drift training dataset is the previous highest label
            # which is the generator label
            self.temporary_label = [0]
            self.discriminator.update()
            self.discriminator = self.discriminator.to(self.device)
            self.generator_label += 1

        # Move this to a new function that has the whole dataset available
        self.generator = Generator(inp=data.shape[1], sequence_length=self.sequence_length)
        self.generator = self.generator.to(device=self.device)

        self.generator.train()
        self.discriminator.train()

        training_dataset = self.create_training_dataset(dataset=self.drift_training_data,
                                                        drift_labels=self.drift_labels + self.temporary_label)

        generator, discriminator = self.train_gan(features=training_dataset,
                                                  discriminator=self.discriminator,
                                                  generator=self.generator, max_label=self.generator_label)
        generator.eval()
        discriminator.eval()
        self.generator = generator
        self.discriminator = discriminator

    def train_gan(self, features, discriminator, generator, max_label=1):
        """
        Trains the Generative Adversarial Network
        :param features: features to be trained
        :param discriminator: Discriminator object that needs to be trained
        :param generator: Generator object that needs to be trained
        :param max_label: Label for the hitherto unseen drifts
        :return: Trained generator and discriminator
        """

        # Losses for the generator and discriminator
        loss_mse_generator = nn.MSELoss()
        loss_generator = nn.CrossEntropyLoss()
        loss_discriminator = nn.CrossEntropyLoss()

        # Create the optimizers for the models
        optimizer_generator = Adadelta(generator.parameters(),
                                       lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)
        optimizer_discriminator = Adadelta(discriminator.parameters(),
                                           lr=self.lr, rho=self.rho, eps=self.eps, weight_decay=self.weight_decay)

        # Label vectors
        ones = Variable(torch.ones(self.generator_batch_size)).to(torch.long).to(self.device)

        # This data contains the current vector and next vector
        concatenated_data = self.concatenate_features(features)

        if self.equalize:
            features = self.equalize_classes(features)
            concatenated_data = self.equalize_classes(concatenated_data)

        # Define the data loader for training
        real_data = DataLoader(features, batch_size=self.batch_size, shuffle=self.shuffle_discriminator,
                               collate_fn=self.collate)
        generator_data = DataLoader(concatenated_data, batch_size=self.generator_batch_size,
                                    shuffle=self.shuffle_generator, collate_fn=self.collate_generator)

        # This is the label for new drifts (any input other than the currently learned distributions)
        generator_label = ones * max_label

        for epochs_trained in range(self.epochs):

            discriminator = self.train_discriminator(real_data=real_data, fake_data=generator_data,
                                                     discriminator=discriminator, generator=generator,
                                                     optimizer=optimizer_discriminator, loss_fn=loss_discriminator,
                                                     generator_labels=generator_label)

            generator = self.train_generator(data_loader=generator_data, discriminator=discriminator,
                                             generator=generator,  optimizer=optimizer_generator,
                                             loss_fn=loss_generator, loss_mse=loss_mse_generator,
                                             steps=self.steps_generator)

            continue

            # Train the discriminator on a dataset of mixture of both fake and real data
            random_data = np.reshape(concatenated_data[:, 0:self.sequence_length*(features.shape[1]-1)],
                                     (concatenated_data.shape[0], self.sequence_length, features.shape[1]-1))

            # Create the noisy data part
            noisy_data = np.random.rand(*random_data.shape)
            random_data = np.vstack((random_data, noisy_data))

            # Randomly sample 50% of both generated and noisy data
            len_indices = list(range(0, random_data.shape[0]))
            indices = random.sample(len_indices, int(random_data.shape[0] / 2))
            random_data = random_data[indices]

            random_data = torch.Tensor(random_data).to(torch.float).to(self.device)
            random_generated_data = generator(random_data)

            # Add the generator label for the generated data
            random_generated_data = random_generated_data.cpu().detach().numpy()
            random_generated_data = np.hstack((random_generated_data,
                                               np.ones((random_generated_data.shape[0], 1)) * self.generator_label))

            # Stack both generated data and noisy data
            training_data = np.vstack((features, random_generated_data))
            training_data_loader = DataLoader(training_data, batch_size=self.batch_size,
                                              shuffle=self.shuffle_discriminator, collate_fn=self.collate)

            discriminator = self.train_discriminator_random_data(discriminator=self.discriminator,
                                                                 loader=training_data_loader,
                                                                 optimizer=self.optimizer_discriminator,
                                                                 loss_fn=self.loss_discriminator)

        return generator, discriminator

    def equalize_classes(self, features):
        modified_dataset = None

        labels = features[:, -1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = min(min(counts), self.max_count)

        if min_count == max(counts) == self.max_count:
            return features

        for label, count in zip(unique_labels, counts):
            indices = np.where(features[:, -1] == label)[0]
            chosen_indices = np.random.choice(indices, min_count)
            if modified_dataset is None:
                modified_dataset = features[chosen_indices, :]
                continue
            modified_dataset = np.vstack((modified_dataset, features[chosen_indices, :]))
        return modified_dataset

    def concatenate_features(self, data):
        """
        Concatenates the data of sequence length into a single vector
        :param data: data to be concatenated
        :return: concatenated dataset
        """

        # Data is concatenated with the drift number as the label. So that has to be removed
        modified_data = data[:, :-1]

        idx = self.sequence_length
        modified_data = np.vstack((np.zeros((self.sequence_length - 1, len(modified_data[idx]))), modified_data))
        output = np.hstack((modified_data[idx - self.sequence_length:idx + 1, :].flatten(),
                            data[idx - self.sequence_length][-1]))
        idx += 1
        while idx < len(modified_data) - 1:
            output = np.vstack((output, np.hstack((modified_data[idx - self.sequence_length:idx + 1, :].flatten(),
                                                   data[idx - self.sequence_length][-1]))))
            idx += 1

        # The last value
        output = np.vstack((output, np.hstack((modified_data[idx - self.sequence_length:, :].flatten(), data[-1][-1]))))
        output = np.vstack((output, np.hstack((modified_data[idx - self.sequence_length:idx, :].flatten(),
                                               modified_data[self.sequence_length - 1],
                                               data[0][-1]))))
        return output

    def create_training_dataset(self, dataset, drift_labels):

        # If there is a periodicity, we switch all previous drifts to the same label
        modified_drift_labels = [x for x in drift_labels]
        # If there is no periodicity, we keep the modified drift labels, else we remove them and switch the previously
        # occurred drifts to 0 and decrement the labels of all the drifts that have a drift label greater than the
        # current drift label
        if drift_labels[-1] != 0:
            modified_drift_labels = []
            for label in drift_labels:
                if label == drift_labels[-1]:
                    modified_drift_labels.append(0)  # The current label
                elif label > drift_labels[-1]:
                    modified_drift_labels.append(label - 1)  # Decrease all labels that are greater than this
                else:
                    modified_drift_labels.append(label)

        training_dataset = np.hstack((dataset[0],
                                      np.ones((self.training_window_size, 1))
                                      * modified_drift_labels[0]))
        for idx in range(1, len(modified_drift_labels)):
            training_dataset = np.vstack((training_dataset,
                                          np.hstack((dataset[idx],
                                                     np.ones((self.training_window_size, 1)) *
                                                     modified_drift_labels[idx]))))

        return training_dataset

    def train_discriminator(self, real_data, fake_data, discriminator, generator, optimizer, loss_fn,
                            generator_labels):
        # for idx in range(steps):
        for features, labels in real_data:
            # Set the gradients as zero
            discriminator.zero_grad()
            optimizer.zero_grad()

            # Get the loss when the real data is compared to ones
            features = features.to(self.device).to(torch.float)
            labels = labels.to(self.device)

            # Get the output for the real features
            output_discriminator = self.discriminator(features)

            # The real data is without any concept drift. Evaluate loss against zeros
            real_data_loss = loss_fn(output_discriminator, labels)

            # Get the output from the generator for the generated data compared to ones which is drifted data
            generator_input = None
            for input_sequence, _, _ in fake_data:
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

    def train_generator(self, data_loader, discriminator, generator, optimizer, loss_fn, loss_mse, steps):
        epoch_loss = 0
        for idx in range(steps):

            optimizer.zero_grad()
            generator.zero_grad()

            generated_input = target = labels = None
            for generator_input, target, l in data_loader:
                generated_input = generator_input.to(torch.float).to(self.device)
                target = target.to(torch.float).to(self.device)
                labels = l.to(torch.long).to(self.device)
                # target = target.reshape((target.shape[0], target.shape[2]))
                break

            # Generating data for input to generator
            generated_output = generator(generated_input)

            # Compute loss based on whether discriminator can discriminate real data from generated data
            generated_training_discriminator_output = discriminator(generated_output)

            # Compute loss based on ideal target values
            loss_generated = loss_fn(generated_training_discriminator_output, labels)

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
        x = torch.stack([tensor(x[:-1]) for x in batch])
        y = Tensor([x[-1] for x in batch]).to(torch.long)
        # Return features and labels
        return x, y

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
        x = torch.stack([Tensor(np.reshape(x[:-feature_length - 1], newshape=(self.sequence_length, feature_length)))
                   for x in batch])
        y = torch.stack([tensor(x[-feature_length - 1:-1]) for x in batch])
        labels = torch.stack([tensor(x[-1]) for x in batch])
        # Return features and targets
        return x.to(torch.double), y, labels
