import torch
import torch.nn as nn
from unet import ContextUnet
from ddpm import DDPM
from pixelate import Pixelate
from initializers import SampleInitializer
from utils import save_images
import math
import numpy as np


class LevelScheduler:
    def get_probabilities(self, number_of_levels, **kwargs):
        raise NotImplementedError("Must be implemented by subclasses.")


class ArithmeticScheduler(LevelScheduler):
    def get_probabilities(self, number_of_levels):
        prob_dist = [i for i in range(1, number_of_levels + 1)]
        return [p / sum(prob_dist) for p in prob_dist]


class PowerScheduler(LevelScheduler):
    def get_probabilities(self, number_of_levels, power=0.4):
        prob_dist = [i**power for i in range(1, number_of_levels + 1)]
        return [p / sum(prob_dist) for p in prob_dist]


class GeometricScheduler(LevelScheduler):
    def get_probabilities(self, number_of_levels, base=1.5):
        prob_dist = [base**i for i in range(number_of_levels)]
        return [p / sum(prob_dist) for p in prob_dist]


class ColdDDPM(DDPM):
    def __init__(
        self,
        unet: ContextUnet,
        T,
        device,
        n_classes,
        n_between: int,
        initializer: SampleInitializer,
        minimum_pixelation: int,
        level_scheduler: LevelScheduler = PowerScheduler(),
    ):
        super(ColdDDPM, self).__init__(
            unet=unet,
            T=T,
            device=device,
            n_classes=n_classes,
        )
        self.nn_model = unet.to(device)

        self.device = device
        self.loss_mse = nn.MSELoss()
        self.sample_initializer = initializer
        self.degredation = Pixelate(
            n_between=n_between,
            minimum_pixelation=minimum_pixelation,
        )
        self.T = T
        self.n_between = n_between
        self.min_size = minimum_pixelation
        self.scheduler = level_scheduler

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.T + 1, (x.shape[0],)).to(self.device)

        x_t = torch.cat(
            [self.degredation(x, t) for x, t in zip(x, _ts)], dim=0
        ).unsqueeze(1)

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, c, _ts / self.T)
        return self.loss_mse(x, pred)

    @torch.no_grad()
    def sample(self, n_sample, size):
        c_i = self.get_ci(n_sample)

        # Sample x_t for classes
        x_t = torch.cat(
            [self.sample_initializer.sample((1,) + size, c) for c in c_i]
        ).to(self.device)

        for t in range(self.T, 0, -1):
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample)

            x_0 = self.nn_model(x_t, c_i, t_is)
            x_0.clamp_(-1, 1)

            x_t = x_t - self.degredation(x_0, t) + self.degredation(x_0, (t - 1))

        return x_0, c_i

    def _sample_level(self, max_size, min_size):
        number_of_levels = int(math.log2(max_size) - math.log2(min_size)) + 1
        prob_dist = self.scheduler.get_probabilities(number_of_levels)
        return np.random.choice(number_of_levels, p=prob_dist)
