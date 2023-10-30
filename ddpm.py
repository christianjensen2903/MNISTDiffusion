from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class DDPM(ABC, nn.Module):
    def __init__(self, unet, T, device, n_classes, criterion, **kwargs):
        super(DDPM, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.T = T
        self.criterion = criterion

    @abstractmethod
    def forward(self, x, c):
        pass

    @abstractmethod
    def sample(self, n_sample, size):
        pass

    def get_ci(self, n_sample):
        c_i = torch.arange(0, 10).to(
            self.device
        )  # context cycles through the MNIST labels

        full_repeats = (
            n_sample // c_i.shape[0]
        )  # Number of times to repeat the entire label set
        remainder = (
            n_sample % c_i.shape[0]
        )  # Number of additional labels needed after repeating

        # Create repeated labels
        repeated_labels = c_i.repeat(full_repeats)

        # Take the remainder of the labels
        remaining_labels = c_i[:remainder]

        # Concatenate them together to get the desired length
        c_i = torch.cat([repeated_labels, remaining_labels])

        return c_i
