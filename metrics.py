from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Any
from classification_utils import SimpleCNN, train_model, evalaute_model
from data_loader import create_mnist_dataloaders

DIMS = 2048


def calculate_fid(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: str,
    batch_size: int = 32,
) -> float:
    print("Calculating fid...")
    if target.shape[1] == 1:
        target = target.repeat(1, 3, 1, 1)
    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)

    fid_calculator = FrechetInceptionDistance(normalize=True)
    fid_calculator = fid_calculator.to(device)

    n = target.size(0)
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        target_batch = target[i:end_i]
        pred_batch = pred[i:end_i]
        fid_calculator.update(target_batch, real=True)
        fid_calculator.update(pred_batch, real=False)

    return fid_calculator.compute()


def calculate_ssim(pred, target: torch.Tensor, device: str) -> float:
    print("Calculating ssim...")
    ssim_calculator = StructuralSimilarityIndexMeasure().to(device)
    ssim_calculator.update(pred, target)
    return ssim_calculator.compute()


class SampleDataset(Dataset):
    def __init__(self, samples, labels):
        assert len(samples) == len(
            labels
        ), "samples and labels should have the same length"
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def calculate_cas(
    samples,
    labels,
    test_dataloader: DataLoader,
    device: str,
    batch_size: int = 32,
) -> float:
    print("Calculating CAS...")
    dataset = SampleDataset(samples, labels)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    image_size = samples.shape[-1]
    model = SimpleCNN(image_size).to(device)

    train_model(model, train_dataloader, device)
    accuracy = evalaute_model(model, test_dataloader, device)
    return accuracy


def main():
    # Train model based samples from training data
    # Calculate FID, SSIM, CAS

    n_samples = 2000
    batch_size = 32
    image_size = 32

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size, image_size=image_size
    )

    samples = []
    labels = []
    for batch in train_dataloader:
        x, y = batch
        samples.append(x)
        labels.append(y)

    samples = torch.cat(samples, dim=0)
    labels = torch.cat(labels, dim=0)

    samples = samples[:n_samples]
    labels = labels[:n_samples]

    # Sample some other images

    samples_ref = []
    labels_ref = []

    for batch in test_dataloader:
        x, y = batch
        samples_ref.append(x)
        labels_ref.append(y)

    samples_ref = torch.cat(samples_ref, dim=0)
    labels_ref = torch.cat(labels_ref, dim=0)

    samples_ref = samples_ref[:n_samples]
    labels_ref = labels_ref[:n_samples]

    fid = calculate_fid(samples, samples_ref, "cpu", batch_size=batch_size)
    ssim = calculate_ssim(samples, samples, "cpu")
    cas = calculate_cas(samples, labels, test_dataloader, "cpu")

    print(f"FID: {fid}")
    print(f"SSIM: {ssim}")
    print(f"CAS: {cas}")


if __name__ == "__main__":
    main()
