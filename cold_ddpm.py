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
        return count * (self.n_between + 1) - 1

    def set_to_random_grey(self, images: torch.Tensor, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)

        # Generate a different random grey value for each image in the batch
        random_greys = torch.rand(images.shape[0], 1, 1, 1).to(images.device)
        return images * 0 + random_greys

    def __call__(self, images: torch.Tensor, t: int, seed: int = None):
        """
        t = 0 -> no pixelation
        t = T -> full pixelations
        """

        if isinstance(t, torch.Tensor):
            t = t.item()
        image_size = images.shape[-1]

        from_index = t // (self.n_between + 1)
        interpolation = ((self.n_between - t) % (self.n_between + 1)) / (
            self.n_between + 1
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

        if interpolation == 0:
            return from_images
        else:
            to_size = image_size // (2 ** (from_index))

            to_transform = transforms.Compose(
                [
                    transforms.Resize(to_size, self.interpolation),
                    transforms.Resize(image_size, self.interpolation),
                ]
            )

            to_images = to_transform(images)

            return (1 - interpolation) * from_images + interpolation * to_images


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
        self.degredation = Pixelate(10)
        self.T = T
        self.sample_initializer = RandomColorInitializer()

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

        # Sample random seed
        seed = torch.randint(0, 100000, (1,)).item()

        for t in range(self.T, 0, -1):
            t_is = torch.tensor([t / self.T]).to(self.device)
            t_is = t_is.repeat(n_sample)

            x_0 = self.nn_model(x_t, c_t, t_is)

            x_t = (
                x_t
                - self.degredation(x_0, t, seed)
                + self.degredation(x_0, (t - 1), seed)
            )

        return x_0
