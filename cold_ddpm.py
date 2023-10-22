import torch
import torch.nn as nn
from unet import ContextUnet
import torch.nn.functional as F
from utils import upscale_images
from gmm import sample_from_gmm_for_class
from abc import ABC, abstractmethod
from torchvision import transforms
from ddpm import DDPM


class Pixelate:
    def __init__(self, image_size, n_between: int = 1):
        """Sizes is a list of ints from smallest to largest"""
        self.sizes = self._calculate_sizes(image_size)
        self.transforms = [self.set_image_to_random_grey]
        interpolation = transforms.InterpolationMode.NEAREST
        for size in self.sizes:
            self.transforms.append(
                transforms.Compose(
                    [
                        transforms.Resize(size, interpolation),
                        transforms.Resize(self.sizes[-1], interpolation),
                    ]
                )
            )
        self.n_between = n_between
        self.T = self.calculate_T()

    def _calculate_sizes(self, image_size):
        # Double the size until we reach the image size
        result = []
        n = 8
        while n <= image_size:
            result.append(n)
            n *= 2
        return result

    def calculate_T(self):
        """
        img0 -> img1/N -> img2/N -> .. -> img(N-1)/N -> img1 -> img(N+1)/N ->... imgK
        Where a fractional image denotes a interpolation between two images (imgA and img(A+1))
        The number of images in the above becomes (excluding the original image):
        K * (N+1)
        """
        return len(self.sizes) * (self.n_between + 1)

    def set_image_to_random_grey(self, image: torch.Tensor):
        return image * 0 + torch.rand(1).to(image.device)

    def __call__(self, image: torch.Tensor, t: int):
        """
        t = 0 -> no pixelation
        t = T -> full pixelations
        """
        fromIndex = (self.T - t) // (self.n_between + 1)
        interpolation = ((self.T - t) % (self.n_between + 1)) / (self.n_between + 1)
        fromImage = self.transforms[fromIndex](image)
        if interpolation == 0:
            return fromImage
        else:
            toIndex = fromIndex + 1
            toImage = self.transforms[toIndex](image)
            return (1 - interpolation) * fromImage + interpolation * toImage


class SampleInitializer(ABC):
    @abstractmethod
    def sample(
        self, n_sample: int, size: tuple[int, int], label: int | str
    ) -> torch.Tensor:
        pass


class GMMInitializer(SampleInitializer):
    def __init__(self, gmms):
        self.gmms = gmms

    def sample(self, n_sample, size, label):
        samples = sample_from_gmm_for_class(self.gmms, label, n_samples=n_sample)
        sample_img_size = int(samples.shape[-1] ** 0.5)
        samples = samples.reshape(n_sample, 1, sample_img_size, sample_img_size)
        samples = torch.tensor(samples, dtype=torch.float32)
        return upscale_images(samples, size[-1])


class RandomColorInitializer(SampleInitializer):
    def sample(self, n_sample, size, label):
        # Sample a random grey image
        color = torch.rand(1)
        return torch.ones(n_sample, 1, size[-1], size[-1]) * color


class ColdDDPM(DDPM):
    def __init__(
        self,
        unet: ContextUnet,
        T,
        device,
    ):
        super(ColdDDPM, self).__init__(
            unet=unet,
            T=T,
            device=device,
        )
        self.nn_model = unet.to(device)

        self.device = device
        self.loss_mse = nn.MSELoss()
        self.degredation = Pixelate(16, 10)
        self.n_T = self.degredation.T
        self.sample_initializer = RandomColorInitializer()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)

        x_t = torch.cat(
            [self.degredation(x, t) for x, t in zip(x, _ts)], dim=0
        ).unsqueeze(1)

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, c, _ts / self.n_T)
        return self.loss_mse(x, pred)

    def sample(self, n_sample, size):
        # Assuming context is required, initialize it here.
        c_t = torch.arange(0, 10).to(
            self.device
        )  # context cycles through the MNIST labels
        c_t = c_t.repeat(int(n_sample / c_t.shape[0]))

        # Sample x_t for classes
        x_t = torch.cat([self.sample_initializer.sample(1, size, c) for c in c_t]).to(
            self.device
        )

        for t in range(self.n_T, 0, -1):
            t_is = torch.tensor([t / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample)

            x_0 = self.nn_model(x_t, c_t, t_is)

            x_t = x_t - self.degredation(x_0, t) + self.degredation(x_0, (t - 1))

        return x_0
