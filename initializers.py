import torch
from abc import ABC, abstractmethod
from gmm import sample_from_gmm_for_class, load_if_exists
from utils import scale_images


class SampleInitializer(ABC):
    @abstractmethod
    def sample(self, size: tuple, label: int | str) -> torch.Tensor:
        pass


class RandomColorInitializer(SampleInitializer):
    def sample(self, size, label):
        # Sample a random grey image between 0.5 and 1
        color = 0.5 + 0.5 * torch.rand(1)
        return torch.ones(size) * color


class GMMInitializer(SampleInitializer):
    def __init__(self, path: str = "gmm_model.pkl", image_size: int = 28, to_size=4):
        self.gmms = load_if_exists(path, image_size, to_size=to_size, n_components=10)

    def sample(self, size, label):
        n_sample = size[0]
        samples = sample_from_gmm_for_class(self.gmms, label, n_samples=n_sample)
        sample_img_size = int(samples.shape[-1] ** 0.5)
        samples = samples.reshape(n_sample, 1, sample_img_size, sample_img_size)
        samples = torch.tensor(samples, dtype=torch.float32)
        return scale_images(samples, size[-1])
