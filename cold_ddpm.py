import torch
import torch.nn as nn
from unet import UNetModel
from ddpm import DDPM
from pixelate import Pixelate
from initializers import SampleInitializer
from utils import save_images, scale_images
import math
import numpy as np


class ColdDDPM(DDPM):
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
    ):
        super(ColdDDPM, self).__init__(
            unet=unet,
            T=T,
            device=device,
            n_classes=n_classes,
            criterion=criterion,
        )
        self.nn_model = unet.to(device)

        self.device = device
        self.loss_mse = nn.L1Loss()
        self.sample_initializer = initializer
        self.degredation = Pixelate(
            n_between=n_between,
            minimum_pixelation=minimum_pixelation,
        )
        self.T = T
        self.n_between = n_between
        self.min_size = minimum_pixelation

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.T + 1, (x.shape[0],)).to(self.device)

        x_t_list = [self.degredation(x[i], _ts[i]) for i in range(x.shape[0])]
        x_t = torch.stack(x_t_list, dim=0)

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, _ts / self.T, c)
        return self.criterion(x, pred)

    @torch.no_grad()
    def sample(self, n_sample, size):
        c_i = self.get_ci(n_sample)

        # Sample x_t for classes
        x_t = torch.cat(
            [
                self.sample_initializer.sample(
                    (1, self.nn_model.out_channels, self.min_size, self.min_size), c
                )
                for c in c_i
            ]
        ).to(self.device)

        x_t = scale_images(x_t, to_size=size[-1])

        for t in range(self.T, 0, -1):
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample)

            x_0 = self.nn_model(x_t, t_is, c_i)
            x_0.clamp_(-1, 1)

            x_t = x_t - self.degredation(x_0, t) + self.degredation(x_0, (t - 1))

        return x_0, c_i

    def _sample_level(self, max_size, min_size):
        number_of_levels = int(math.log2(max_size) - math.log2(min_size)) + 1
        prob_dist = self.scheduler.get_probabilities(number_of_levels)
        return np.random.choice(number_of_levels, p=prob_dist)
