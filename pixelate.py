import torch
from utils import scale_images
import math


def round_up_to_nearest_multiple(x):
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


class Pixelate:
    def __init__(self, n_between: int = 1, minimum_pixelation: int = 8) -> None:
        assert n_between in [1, 3, 7, 15, 31]
        self.n_between = n_between
        self.minimum_pixelation = minimum_pixelation

    def calculate_T(self, image_size: int) -> int:
        """
        Calculate total number of steps based on image size and minimum pixelation.
        """
        steps = 0
        while image_size > self.minimum_pixelation:
            steps += min(self.n_between + 1, image_size // 2)
            image_size //= 2
        return steps

    def __call__(self, images: torch.Tensor, t: int) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            t = t.item()

        relative_t = t % (self.n_between + 1)
        image_size = images.shape[-1]
        step_size = max(1, (image_size // 2) // (self.n_between + 1))

        return scale_images(
            scale_images(images, image_size - step_size * relative_t), image_size
        )
