from abc import ABC, abstractmethod
import torch.nn as nn


class DDPM(ABC, nn.Module):
    def __init__(self, unet, T, device, **kwargs):
        super(DDPM, self).__init__()
        pass

    @abstractmethod
    def forward(self, x, c):
        pass

    @abstractmethod
    def sample(self, n_sample, size):
        pass
