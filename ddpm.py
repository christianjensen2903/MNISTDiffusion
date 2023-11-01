from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class DDPM(ABC, nn.Module):
    def __init__(self, unet, T, device, criterion, **kwargs):
        super(DDPM, self).__init__()
        self.device = device
        self.T = T
        self.criterion = criterion

    @abstractmethod
    def forward(self, x, c):
        pass

    @abstractmethod
    def sample(self, n_sample, size):
        pass
