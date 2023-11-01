import torch
from abc import ABC, abstractmethod
from gmm import sample_from_gmm_for_class, train_gmm
from utils import scale_images
from torch.utils.data import DataLoader


class SampleInitializer(ABC):
    @abstractmethod
    def sample(self, size: tuple) -> torch.Tensor:
        pass


class RandomColorInitializer(SampleInitializer):
    def sample(self, size, label):
        # Sample a random grey image between 0.5 and 1
        color = 0.5 + 0.5 * torch.rand(1)
        return torch.ones(size) * color


class GMMInitializer(SampleInitializer):
    def __init__(
        self,
        dataloader: DataLoader,
        to_size: int,
        n_components: int,
    ):
        self.gmms = train_gmm(dataloader, n_components=n_components, to_size=to_size)

    def sample(self, size):
        n_sample = size[0]
        samples = sample_from_gmm_for_class(self.gmms, n_samples=n_sample)
        sample_img_size = int(samples.shape[-1] ** 0.5)
        samples = samples.reshape(n_sample, 1, sample_img_size, sample_img_size)
        samples = torch.tensor(samples, dtype=torch.float32)
        return scale_images(samples, size[-1])
