import torch
from utils import scale_images


class Pixelate:
    def __init__(self, minimum_pixelation: int = 8) -> None:
        self.minimum_pixelation = minimum_pixelation

    def calculate_T(self, image_size: int) -> int:
        """
        Calculate total number of steps based on image size and minimum pixelation.
        """
        steps = 0
        while image_size > self.minimum_pixelation:
            steps += 1
            image_size //= 2
        return steps

    @staticmethod
    def _scale_image(image: torch.Tensor, to_size: int) -> torch.Tensor:
        """Helper method to scale image to a given size and then back to original."""
        return scale_images(scale_images(image, to_size), image.shape[-1])

    def __call__(self, images: torch.Tensor) -> torch.Tensor:

        image_size = images.shape[-1]

        return self._scale_image(images, image_size // 2)
