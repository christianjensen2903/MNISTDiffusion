import torch
import torch.nn as nn
from unet import ContextUnet
from abc import ABC, abstractmethod
from torchvision import transforms
from ddpm import DDPM
import numpy as np
import torch.nn.functional as F
from utils import upscale_images, save_images
import math


class Pixelate:
    def __init__(self, n_between: int = 1):
        self.n_between = n_between
        self.interpolation = transforms.InterpolationMode.NEAREST

    def calculate_T(self, image_size):
        """
        img0 -> img1/N -> img2/N -> .. -> img(N-1)/N -> img1 -> img(N+1)/N ->... imgK
        Where a fractional image denotes an interpolation between two images (imgA and img(A+1))
        """
        size = image_size
        count = 0
        while size > 4:
            count += 1
            size //= 2
        return count * self.n_between

    def set_to_random_grey(self, images: torch.Tensor, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)

        # Generate a different random grey value for each image in the batch
        # Only samples greys between 0.5 and 1.0
        if len(images.shape) == 4:
            random_greys = 0.5 + 0.5 * torch.rand(images.shape[0], 1, 1, 1).to(
                images.device
            )
        else:
            random_greys = 0.5 + 0.5 * torch.rand(images.shape[0], 1, 1).to(
                images.device
            )
        return images * 0 + random_greys

    def __call__(self, images: torch.Tensor, t: int, seed: int = None):
        """
        t = 1 -> no pixelation
        t = T -> full pixelations
        """

        if isinstance(t, torch.Tensor):
            t = t.item()
        image_size = images.shape[-1]

        from_index = t // (self.n_between + 1)
        current_level = (t - 1) // self.n_between  # Find out which segment t is in
        relative_t = (
            t - current_level * self.n_between
        )  # Position of t within that segment

        interpolation = -1 / (self.n_between - 1) * relative_t + self.n_between / (
            self.n_between - 1
        )

        from_size = image_size // (2 ** (from_index + 1))

        if from_size <= 4:
            from_images = self.set_to_random_grey(images, seed)
        else:
            from_transform = transforms.Compose(
                [
                    transforms.Resize(from_size, self.interpolation),
                    transforms.Resize(image_size, self.interpolation),
                ]
            )
            from_images = from_transform(images)

        to_size = image_size // (2 ** (from_index))

        to_transform = transforms.Compose(
            [
                transforms.Resize(to_size, self.interpolation),
                transforms.Resize(image_size, self.interpolation),
            ]
        )

        to_images = to_transform(images)

        return (1 - interpolation) * from_images + interpolation * to_images


def sample_level(max_size, min_size):
    number_of_levels = int(math.log2(max_size) - math.log2(min_size)) + 1

    # Create a probability distribution with higher probability for the last level.
    # This is just one way to do it. You can modify the probabilities as you see fit.
    prob_dist = [1] * (number_of_levels - 1) + [number_of_levels]

    # Normalize the probability distribution
    prob_dist = [p / sum(prob_dist) for p in prob_dist]

    return np.random.choice(number_of_levels, p=prob_dist)


def downscale_images(images: torch.Tensor, to_size: int) -> torch.Tensor:
    return F.interpolate(images, size=(to_size, to_size), mode="nearest")


class SampleInitializer(ABC):
    @abstractmethod
    def sample(
        self, n_sample: int, size: tuple[int, int], label: int | str
    ) -> torch.Tensor:
        pass


class RandomColorInitializer(SampleInitializer):
    def sample(self, n_sample, size, label):
        # Sample a random grey image between 0.5 and 1
        color = 0.5 + 0.5 * torch.rand(1)
        return torch.ones(n_sample, 1, size[-1], size[-1]) * color


class ScalingDDPM(DDPM):
    def __init__(
        self,
        unet: ContextUnet,
        T,
        device,
        n_between: int = 1,
    ):
        super(ScalingDDPM, self).__init__(
            unet=unet,
            T=T,
            device=device,
        )
        self.nn_model = unet.to(device)

        self.device = device
        self.loss_mse = nn.MSELoss()
        self.degredation = Pixelate(n_between=n_between)
        self.T = T
        self.sample_initializer = RandomColorInitializer()
        self.n_between = n_between
        self.min_size = 8

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        initial_size = x.shape[-1]

        current_level = sample_level(initial_size, self.min_size)

        # Calculate new size
        current_size = initial_size // (2**current_level)

        x = downscale_images(x, current_size)

        _ts = torch.randint(1, self.n_between + 1, (x.shape[0],)).to(self.device)

        x_t = torch.cat(
            [self.degredation(x, t) for x, t in zip(x, _ts)], dim=0
        ).unsqueeze(1)

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, c, (self.n_between * current_level + _ts) / self.T)

        return self.loss_mse(x, pred)

    def sample(self, n_sample, size):
        # Assuming context is required, initialize it here.
        c_t = torch.arange(0, 10).to(
            self.device
        )  # context cycles through the MNIST labels
        c_t = c_t.repeat(int(n_sample / c_t.shape[0]))

        # Sample x_t for classes
        x_t = torch.cat([self.sample_initializer.sample(1, (8, 8), c) for c in c_t]).to(
            self.device
        )

        # Sample random seed
        seed = torch.randint(0, 100000, (1,)).item()

        current_size = 8
        while current_size <= size[-1]:
            current_level = int(math.log2(size[-1]) - math.log2(current_size))
            for relative_t in range(self.n_between, 0, -1):
                t = self.n_between * current_level + relative_t
                t_is = torch.tensor([t]).to(self.device)
                t_is = t_is.repeat(n_sample)

                x_0 = self.nn_model(x_t, c_t, t_is / self.T)

                if relative_t - 1 > 0:
                    x_t = (
                        x_t
                        - self.degredation(x_0, relative_t, seed)
                        + self.degredation(x_0, (relative_t - 1), seed)
                    )
                else:
                    current_size *= 2
                    x_t = upscale_images(x_0, current_size)

        return x_0
