import torch
from utils import scale_images


class Pixelate:
    def __init__(self, n_between: int = 1, minimum_pixelation: int = 8) -> None:
        self.n_between = n_between
        self.minimum_pixelation = minimum_pixelation

    def calculate_T(self, image_size: int) -> int:
        """
        Calculate total number of steps based on image size and minimum pixelation.
        """
        steps = 0
        while image_size > self.minimum_pixelation:
            steps += self.n_between + 1
            image_size //= 2
        return steps

    @staticmethod
    def _scale_image(image: torch.Tensor, to_size: int) -> torch.Tensor:
        """Helper method to scale image to a given size and then back to original."""
        return scale_images(scale_images(image, to_size), image.shape[-1])

    def __call__(self, images: torch.Tensor, t: int) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            t = t.item()

        image_size = images.shape[-1]
        step_size = self.n_between + 1

        # Directly pixelate image if not between two image sizes
        if t % step_size == 0:
            return self._scale_image(images, image_size // (2 ** (t // step_size)))

        # Calculate upper and lower image sizes
        upper_image_size = image_size // (2 ** (t // step_size))
        lower_image_size = image_size // (2 ** ((t // step_size) + 1))

        # Scale images to their respective sizes
        upper_image = self._scale_image(images, upper_image_size)
        lower_image = self._scale_image(images, lower_image_size)

        # Compute interpolation factor
        interpolation = (t % step_size) / step_size

        return (1 - interpolation) * upper_image + interpolation * lower_image
