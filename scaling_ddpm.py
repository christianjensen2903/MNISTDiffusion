import torch
import torch.nn as nn
from unet import ContextUnet
from abc import ABC, abstractmethod
from torchvision import transforms
from ddpm import DDPM
import numpy as np
import torch.nn.functional as F
from utils import upscale_images, save_images


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
        while size >= 4:
            count += 1
            size //= 2
        return count * (self.n_between + 1) - 1

    def set_to_random_grey(self, images: torch.Tensor, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)

        # Generate a different random grey value for each image in the batch
        if len(images.shape) == 4:
            random_greys = torch.rand(images.shape[0], 1, 1, 1).to(images.device)
        else:
            random_greys = torch.rand(images.shape[0], 1, 1).to(images.device)
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

        if from_size < 4:
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


def random_downscale(tensor_img):
    # Ensure the image is square
    assert tensor_img.shape[-1] == tensor_img.shape[-2], "The image must be square!"

    # Get the original size (width or height)
    original_size = tensor_img.shape[-1]

    # Compute the maximum power of 2 for the image size
    max_power = int(np.log2(original_size))

    # Generate a list of possible sizes
    possible_sizes = [2**i for i in range(2, max_power + 1)][::-1]

    # Randomly select a size
    selected_size = np.random.choice(possible_sizes)

    # Downscale the image
    downscaled_tensor = F.interpolate(tensor_img, size=selected_size, mode="nearest")

    return downscaled_tensor


class SampleInitializer(ABC):
    @abstractmethod
    def sample(
        self, n_sample: int, size: tuple[int, int], label: int | str
    ) -> torch.Tensor:
        pass


class RandomColorInitializer(SampleInitializer):
    def sample(self, n_sample, size, label):
        # Sample a random grey image
        color = torch.rand(1)
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

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        # Calculate downscaling factor
        x = random_downscale(x)

        _ts = torch.randint(0, self.n_between, (x.shape[0],)).to(self.device)

        x_t = torch.cat(
            [self.degredation(x, t) for x, t in zip(x, _ts)], dim=0
        ).unsqueeze(1)

        # return MSE between added noise, and our predicted noise
        pred = self.nn_model(x_t, c, _ts / self.n_between)
        return self.loss_mse(x, pred), pred, x_t, x

    def sample(self, n_sample, size):
        # Assuming context is required, initialize it here.
        c_t = torch.arange(0, 10).to(
            self.device
        )  # context cycles through the MNIST labels
        c_t = c_t.repeat(int(n_sample / c_t.shape[0]))

        # Sample x_t for classes
        x_t = torch.cat([self.sample_initializer.sample(1, (4, 4), c) for c in c_t]).to(
            self.device
        )

        save_images(x_t, "debug/sample/start.png")

        # Sample random seed
        seed = torch.randint(0, 100000, (1,)).item()

        current_size = 4
        while current_size <= size[-1]:
            for t in range(self.n_between, 0, -1):
                t_is = torch.tensor([t / self.T]).to(self.device)
                t_is = t_is.repeat(n_sample)

                x_0 = self.nn_model(x_t, c_t, t_is)

                save_images(x_t, f"debug/sample/x_0_{current_size}_{t}.png")

                x_t = self.degredation(x_0, (t - 1), seed)

                save_images(x_t, f"debug/sample/x_t_{current_size}_{t}.png")

            current_size *= 2

            if current_size <= size[-1]:
                x_t = upscale_images(x_t, current_size)
                save_images(x_t, f"debug/sample/x_t_upscale_{current_size}.png")

        return x_0
