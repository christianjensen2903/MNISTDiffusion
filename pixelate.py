import torch
from torchvision import transforms


class Pixelate:
    def __init__(
        self,
        n_between: int = 1,
        minimum_pixelation: int = 8,
    ):
        self.n_between = n_between
        self.interpolation = transforms.InterpolationMode.NEAREST
        self.minimum_pixelation = minimum_pixelation

    def calculate_T(self, image_size):
        """
        img0 -> img1/N -> img2/N -> .. -> img(N-1)/N -> img1 -> img(N+1)/N ->... imgK
        Where a fractional image denotes an interpolation between two images (imgA and img(A+1))
        """
        size = image_size
        count = 0
        while size > self.minimum_pixelation:
            count += 1
            size //= 2
        return count * self.n_between

    def __call__(self, images: torch.Tensor, t: int, seed: int = None):
        """
        t = 1 -> no pixelation
        t = T -> full pixelations
        """

        if isinstance(t, torch.Tensor):
            t = t.item()
        image_size = images.shape[-1]

        if seed is not None:
            torch.manual_seed(seed)

        from_index = t // (self.n_between + 1)
        current_level = (t - 1) // self.n_between  # Find out which segment t is in
        relative_t = (
            t - current_level * self.n_between
        )  # Position of t within that segment

        if self.n_between == 1:
            interpolation = 1
        else:
            interpolation = -1 / (self.n_between - 1) * relative_t + self.n_between / (
                self.n_between - 1
            )

        from_size = image_size // (2 ** (from_index + 1))

        from_transform = transforms.Compose(
            [
                transforms.Resize(from_size, self.interpolation),
                transforms.Resize(image_size, self.interpolation),
            ]
        )
        from_images = from_transform(images)

        to_size = image_size // (2 ** (from_index))

        to_transform = transforms.Compose(
            [
                transforms.Resize(to_size, self.interpolation),
                transforms.Resize(image_size, self.interpolation),
            ]
        )

        to_images = to_transform(images)

        return (1 - interpolation) * from_images + interpolation * to_images
