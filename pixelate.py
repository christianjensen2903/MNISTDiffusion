import torch
from utils import scale_images


class Pixelate:
    def __init__(
        self,
        n_between: int = 1,
        minimum_pixelation: int = 8,
    ):
        self.n_between = n_between
        self.minimum_pixelation = minimum_pixelation

    def calculate_T(self, image_size):
        """Calculate total number of steps based on image size and minimum pixelation"""
        steps = 0
        size = image_size
        while size > self.minimum_pixelation:
            steps += self.n_between + 1
            size //= 2

        return steps

    def __call__(self, images: torch.Tensor, t: int):
        if isinstance(t, torch.Tensor):
            t = t.item()

        image_size = images.shape[-1]

        # If not between two image sizes
        if t % (self.n_between + 1) == 0:
            # Find pixelation level
            to_size = image_size // (2 ** (t // (self.n_between + 1)))

            # Pixelate image
            images = scale_images(images, to_size)
            return scale_images(images, image_size)
        else:
            # Find larger image size
            upper_image_size = image_size // (2 ** (t // (self.n_between + 1)))
            lower_image_size = image_size // (2 ** ((t // (self.n_between + 1)) + 1))
            upper_image = scale_images(images, upper_image_size)
            upper_image = scale_images(upper_image, image_size)

            lower_image = scale_images(images, lower_image_size)
            lower_image = scale_images(lower_image, image_size)

            # Find relative position between two image sizes
            relative_t = t % (self.n_between + 1)

            # Interpolate between two images
            interpolation = t / (self.n_between + 1)
            return (1 - interpolation) * upper_image + interpolation * lower_image
