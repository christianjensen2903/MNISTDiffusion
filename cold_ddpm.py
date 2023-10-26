import torch
import torch.nn as nn
from unet import ContextUnet
from ddpm import DDPM
from pixelate import Pixelate
from initializers import SampleInitializer


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
            initializer,
            device,
            n_between=n_between,
            minimum_pixelation=minimum_pixelation,
        )
        self.T = T

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

        # Sample random seed
        seed = torch.randint(0, 100000, (1,)).item()

        for t in range(self.T, 0, -1):
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample)

            x_0 = self.nn_model(x_t, c_i, t_is)

            x_t = (
                x_t
                - self.degredation(x_0, t, seed)
                + self.degredation(x_0, (t - 1), seed)
            )

        return x_0
