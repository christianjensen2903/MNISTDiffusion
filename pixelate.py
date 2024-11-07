import torch
from utils import scale_images
import math


def round_up_to_nearest_multiple(x):
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


class Pixelate:
    def __init__(self, max_step_size: int = 1, minimum_pixelation: int = 8) -> None:
        self.max_step_size = max_step_size
        self.minimum_pixelation = minimum_pixelation

    def calculate_T(self, image_size: int) -> int:
        """
        Calculate total number of steps based on image size and minimum pixelation.
        """
        steps = 0
        while image_size > self.minimum_pixelation:
            steps += 1
            image_size = image_size - self.max_step_size
        return steps

    def __call__(
        self, images: torch.Tensor, t: int, image_size: int | None = None
    ) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            t = t.item()

        orig_image_size = images.shape[-1]
        target_size = max(
            self.minimum_pixelation, orig_image_size - self.max_step_size * t
        )
        if image_size is None:
            image_size = max(
                round_up_to_nearest_multiple(max(target_size + 1, self.max_step_size)),
                orig_image_size,
            )

        return scale_images(
            scale_images(images, to_size=target_size), to_size=image_size
        )
