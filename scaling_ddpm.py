import torch
import torch.nn as nn
from unet import UNetModel
from ddpm import DDPM
import numpy as np
import torch.nn.functional as F
from utils import scale_images, save_images
import math
from initializers import SampleInitializer
from pixelate import Pixelate, round_up_to_nearest_multiple
import random


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
        self.minimum_pixelation = minimum_pixelation
        self.positional_degree = positional_degree
        self.scheduler = level_scheduler

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        initial_size = x.shape[-1]

        current_level = self._sample_level(initial_size, self.minimum_pixelation * 2)

        current_size = initial_size // (2**current_level)

        x_downscaled = scale_images(x, to_size=current_size)

        _ts = torch.randint(
            1, min(self.n_between + 2, current_size // 2), (x.shape[0],)
        )

        x_t = torch.stack(
            [self.degredation(x_downscaled[i], _ts[i]) for i in range(x.shape[0])],
            dim=0,
        )

        x_downscaled = torch.stack(
            [self.degredation(x_downscaled[i], _ts[i] - 1) for i in range(x.shape[0])],
            dim=0,
        )

        x_t_pos = self._add_positional_embedding(x_t)

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(
            x_t_pos,
            ((self.n_between + 1) * current_level + _ts).to(self.device) / self.T,
            c,
        )

        return self.criterion(x_downscaled, pred), pred, x_t, x_downscaled

    @torch.no_grad()
    def sample(
        self, n_sample: int, size: tuple[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c_i = self.get_ci(n_sample)

        # Sample x_t for classes
        x_t = torch.cat(
            [
                self.sample_initializer.sample(
                    (
                        1,
                        self.nn_model.out_channels,
                        self.minimum_pixelation,
                        self.minimum_pixelation,
                    ),
                    c,
                )
                for c in c_i
            ]
        ).to(self.device)

        current_size = self.minimum_pixelation
        image_size = self.minimum_pixelation * 2
        x_t = scale_images(x_t, to_size=image_size)

        t = self.T

        save_images(scale_images(x_t, size[-1]), f"debug/samples/{t}.png")

        while t > 0:
            t_is = torch.tensor([t]).to(self.device)
            t_is = t_is.repeat(n_sample)

            x_t_pos = self._add_positional_embedding(x_t)

            x_t = self.nn_model(x_t_pos, t_is / self.T, c_i)
            x_t.clamp_(-1, 1)

            t -= 1
            current_size += max(1, (image_size // 2) // (self.n_between + 1))
            next_size = round_up_to_nearest_multiple(current_size + 1)
            if next_size != image_size and t > 0:
                image_size = next_size
                x_t = scale_images(x_t, image_size)

            save_images(scale_images(x_t, size[-1]), f"debug/samples/{t}.png")

        return x_t, c_i

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

    def _sample_t(self, orig_image_size: int, n: int) -> np.ndarray:

        current_size = orig_image_size
        image_size = orig_image_size
        possible_ts: list[list[int]] = [[]]
        t = 0
        while image_size > self.minimum_pixelation:
            image_size = image_size - self.n_between
            new_size = max(
                self.minimum_pixelation, round_up_to_nearest_multiple(image_size)
            )

            t += 1
            possible_ts[-1].append(t)
            if new_size != current_size and new_size != self.minimum_pixelation:
                possible_ts.append([])
                current_size = new_size

        options = random.choices(possible_ts)[0]
        ts = np.random.choice(options, n)
        return ts

    def _sample_level(self, max_size, min_size):
        number_of_levels = int(math.log2(max_size) - math.log2(min_size)) + 1
        prob_dist = self.scheduler.get_probabilities(number_of_levels)
        return np.random.choice(number_of_levels, p=prob_dist)
