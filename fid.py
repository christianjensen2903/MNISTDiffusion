from pytorch_fid.fid_score import (
    calculate_fid_given_paths,
    compute_statistics_of_path,
    calculate_frechet_distance,
    InceptionV3,
)
from ddpm import DDPM
from utils import setup_device
from data_loader import create_mnist_dataloaders
import os
from torchvision.utils import save_image
import torch
from cold_ddpm import ColdDDPM
from initializers import RandomColorInitializer
from unet import ContextUnet
import json


DIMS = 2048


def calculate_fid(
    model: DDPM,
    device: str,
    batch_size: int = 32,
    count: int = 2048,
    image_size: int = 28,
) -> float:
    """
    Calculate the FID score of the model.
    """

    assert count >= DIMS, "count must be at least 2048"

    path = "fid/"

    mu_real, sigma_real = _get_real_statistics(path, image_size, batch_size, device)

    mu_fake, sigma_fake = _get_fake_statistics(
        path, count, image_size, batch_size, device, model
    )

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid


def _save_samples(path: str, samples: torch.Tensor) -> None:
    _ensure_directory_exists(path)
    _remove_saved_files(path)

    for i, sample in enumerate(samples):
        save_path = os.path.join(path, f"mnist_fake_{i + 1}.png")
        save_image(sample, save_path)


def _get_fake_statistics(
    path: str, count: int, image_size: int, batch_size: int, device: str, model: DDPM
) -> tuple:
    """
    Get the fake statistics of the model.
    """
    generated_samples = torch.tensor([])
    while len(generated_samples) < count:
        samples = model.sample(batch_size, (1, image_size, image_size))
        if device == "cuda":
            samples = samples.cuda().cpu()

        generated_samples = torch.cat((generated_samples, samples))
    generated_samples = generated_samples[:count]

    _save_samples(path + "fake", generated_samples)
    inception_model = _get_inception_model(device, DIMS)
    mu, sigma = compute_statistics_of_path(
        path + "fake", inception_model, batch_size, DIMS, device
    )
    return mu, sigma


def _get_real_statistics(
    path: str, image_size: int, batch_size: int, device: str
) -> tuple:
    """
    Get the real statistics of the dataset.
    """

    json_path = f"{path}real_stats_{image_size}.json"

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            stats = json.load(f)

        return stats["mu"], stats["sigma"]

    else:
        # Delete old files if they exist
        if os.path.exists(path + "real/"):
            _remove_saved_files(path + "real/")
        _save_real_samples(path + "real/", image_size, batch_size)

        # Calculate statistics
        inception_model = _get_inception_model(device, DIMS)
        mu, sigma = compute_statistics_of_path(
            path + "real/", inception_model, batch_size, DIMS, device
        )
        with open(json_path, "w") as f:
            json.dump({"mu": mu.tolist(), "sigma": sigma.tolist()}, f)

        return mu, sigma


def _get_inception_model(device: str, dims: int) -> torch.nn.Module:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    return InceptionV3([block_idx]).to(device)


def _ensure_directory_exists(path: str) -> None:
    """
    Make sure the directory exists.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _get_saved_image_files(path: str) -> list:
    """
    Returns a list of saved image files from the directory.
    """
    return [
        f
        for f in os.listdir(path)
        if f.startswith("mnist_real_") and os.path.isfile(os.path.join(path, f))
    ]


def _remove_saved_files(path: str) -> None:
    """
    Remove all saved image files.
    """
    saved_files = _get_saved_image_files(path)
    for f in saved_files:
        os.remove(os.path.join(path, f))


def _save_real_samples(path: str, image_size: int, batch_size: int) -> None:
    """
    Save new samples from the dataset.
    """
    _ensure_directory_exists(path)  # Ensure the directory exists before saving images

    _, dataloader = create_mnist_dataloaders(batch_size, image_size)
    for i, (images, _) in enumerate(dataloader):
        for j, image in enumerate(images):
            save_path = os.path.join(path, f"mnist_real_{batch_size * i + j}.png")
            save_image(image, save_path)


if __name__ == "__main__":
    device = setup_device()

    initializer = RandomColorInitializer()

    model = ColdDDPM(
        unet=ContextUnet(in_channels=1, n_feat=8, n_classes=10),
        T=1,
        device=device,
        n_classes=10,
        n_between=1,
        initializer=initializer,
        minimum_pixelation=8,
    )

    fid = calculate_fid(
        model=model,
        device=device,
        image_size=16,
    )

    print(f"FID: {fid}")
