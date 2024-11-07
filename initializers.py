import torch
from abc import ABC, abstractmethod
from gmm import sample_from_gmm_for_class, train_gmm
from utils import scale_images
from torch.utils.data import DataLoader


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
    def __init__(
        self,
        dataloader: DataLoader,
        to_size: int,
        n_components: int,
    ):
        self.gmms = train_gmm(dataloader, n_components=n_components, to_size=to_size)

    def sample(self, size, label):
        n_sample = size[0]
        samples = sample_from_gmm_for_class(self.gmms, label, n_samples=n_sample)
        samples = samples.reshape(size)
        samples = torch.tensor(samples, dtype=torch.float32)
        return scale_images(samples, size[-1])


class RandomSampleInitializer(SampleInitializer):
    def __init__(
        self,
        dataloader: DataLoader,
        to_size: int,
        n_components: int,
    ):
        self.dataloader = dataloader
        self.to_size = to_size
        self.sample_lookup: dict[int, list[torch.Tensor]] = {}
        # Preload samples for each label
        for sample in self.dataloader:
            if sample[1][0] not in self.sample_lookup:
                self.sample_lookup[sample[1][0]] = []
            self.sample_lookup[sample[1][0]].append(
                scale_images(sample[0][0], self.to_size).unsqueeze(0)
            )

    def sample(self, size, label):
        # Sample a random sample from the lookup table
        return self.sample_lookup[label.item()][
            torch.randint(0, len(self.sample_lookup[label.item()]), (size[0],))
        ]
