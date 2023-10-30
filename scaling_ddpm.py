import torch
import torch.nn as nn
from unet import UNetModel
from ddpm import DDPM
import numpy as np
import torch.nn.functional as F
from utils import scale_images, save_images
import math
from initializers import SampleInitializer
from pixelate import Pixelate


class LevelScheduler:
    def get_probabilities(self, number_of_levels, **kwargs):
        raise NotImplementedError("Must be implemented by subclasses.")


class ArithmeticScheduler(LevelScheduler):
    def get_probabilities(self, number_of_levels):
        prob_dist = [i for i in range(1, number_of_levels + 1)]
        return [p / sum(prob_dist) for p in prob_dist]


class PowerScheduler(LevelScheduler):
    def __init__(self, power=2):
        self.power = power

    def get_probabilities(self, number_of_levels):
        prob_dist = [i**self.power for i in range(1, number_of_levels + 1)]
        return [p / sum(prob_dist) for p in prob_dist]


class GeometricScheduler(LevelScheduler):
    def __init__(self, base=2):
        self.base = base

    def get_probabilities(self, number_of_levels):
        prob_dist = [self.base**i for i in range(number_of_levels)]
        return [p / sum(prob_dist) for p in prob_dist]


class UniformScheduler(LevelScheduler):
    def get_probabilities(self, number_of_levels):
        return [1 / number_of_levels for _ in range(number_of_levels)]


class ScalingDDPM(DDPM):
    def __init__(
        self,
        unet: UNetModel,
        T,
        device,
        n_classes,
        criterion,
        n_between: int,
        initializer: SampleInitializer,
        minimum_pixelation: int,
        positional_degree: int,
        level_scheduler: LevelScheduler = PowerScheduler(),
    ):
        super(ScalingDDPM, self).__init__(
            unet=unet,
            T=T,
            device=device,
            n_classes=n_classes,
            criterion=criterion,
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
        self.positional_degree = positional_degree
        self.scheduler = level_scheduler

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        initial_size = x.shape[-1]

        current_level = self._sample_level(initial_size, self.min_size * 2)

        # Calculate new size
        current_size = initial_size // (2**current_level)

        x = scale_images(x, to_size=current_size)

        _ts = torch.randint(1, self.n_between + 2, (x.shape[0],)).to(self.device)

        x_t_list = [self.degredation(x[i], _ts[i]) for i in range(x.shape[0])]
        x_t = torch.stack(x_t_list, dim=0)

        x_t_pos = self._add_positional_embedding(x_t)

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(
            x_t_pos, ((self.n_between + 1) * current_level + _ts) / self.T, c
        )

        return self.criterion(x, pred)

    @torch.no_grad()
    def sample(
        self, n_sample: int, size: tuple[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_i = self.get_ci(n_sample)

        # Sample x_t for classes
        x_t = torch.cat(
            [
                self.sample_initializer.sample((1, 1, self.min_size, self.min_size), c)
                for c in c_i
            ]
        ).to(self.device)

        x_t = scale_images(x_t, to_size=self.min_size * 2)

        current_size = self.min_size * 2

        t = self.T

        while current_size <= size[-1]:
            for relative_t in range(self.n_between + 1, 0, -1):
                t_is = torch.tensor([t]).to(self.device)
                t_is = t_is.repeat(n_sample)

                x_t_pos = self._add_positional_embedding(x_t)

                x_0 = self.nn_model(x_t_pos, t_is / self.T, c_i)

                x_0.clamp_(-1, 1)

                x_t = (
                    x_t
                    - self.degredation(x_0, (relative_t))
                    + self.degredation(x_0, (relative_t - 1))
                )

                if relative_t - 1 == 0:
                    current_size *= 2
                    x_t = scale_images(x_0, current_size)

                t -= 1

        return x_0, c_i

    def _add_positional_embedding(self, x):
        size = x.shape[-1]

        # Concatenate positional embedding
        x_matrix, y_matrix = self._get_pixel_coordinates(size)

        positional_embedding = self._get_positional_embedding(x_matrix, y_matrix).to(
            self.device
        )

        n_samples = x.shape[0]
        positional_embedding = positional_embedding.repeat(n_samples, 1, 1, 1)

        return torch.cat([x, positional_embedding], dim=1)

    def _get_positional_embedding(self, x_matrix, y_matrix):
        layers = []

        for d in range(self.positional_degree):
            freq = 2**d
            layers.append(torch.sin(freq * x_matrix))
            layers.append(torch.cos(freq * x_matrix))
            layers.append(torch.sin(freq * y_matrix))
            layers.append(torch.cos(freq * y_matrix))

        return torch.stack(layers, dim=0)

    def _get_pixel_coordinates(self, size):
        row = torch.arange(0, size) / (size - 1)
        x_matrix = torch.stack([row] * size)
        y_matrix = x_matrix.T
        return x_matrix, y_matrix

    def _sample_level(self, max_size, min_size):
        number_of_levels = int(math.log2(max_size) - math.log2(min_size)) + 1
        prob_dist = self.scheduler.get_probabilities(number_of_levels)
        return np.random.choice(number_of_levels, p=prob_dist)
