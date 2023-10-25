from pytorch_fid.fid_score import calculate_fid_given_paths
from ddpm import DDPM
from utils import setup_device
from data_loader import create_mnist_dataloaders
import os
from torchvision.utils import save_image
import torch
from cold_ddpm import ColdDDPM
from initializers import RandomColorInitializer
from unet import ContextUnet


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

    assert count >= 2048, "count must be at least 2048"

    path = "fid/"
    _save_real_samples(path + "real/", count, image_size)

    generated_samples = model.sample(count, (1, image_size, image_size))
    _save_samples(path + "fake/", generated_samples)

    fid = calculate_fid_given_paths(
        [path + "real/", path + "fake/"], batch_size, device, 2048
    )
    return fid


def _save_samples(path: str, samples: torch.Tensor) -> None:
    _ensure_directory_exists(path)
    _remove_saved_files(path)

    for i, sample in enumerate(samples):
        save_path = os.path.join(path, f"mnist_fake_{i + 1}.png")
        save_image(sample, save_path)


def _save_real_samples(path: str, count: int, image_size: int) -> None:
    """
    Save real samples from the dataset if they don't exist or there isn't enough.
    If there aren't enough images, it deletes all existing and saves a new set.
    """
    _ensure_directory_exists(path)

    saved_files = _get_saved_image_files(path)
    num_saved = len(saved_files)

    if num_saved < count:
        _remove_saved_files(path)
        _save_new_samples(path, count, image_size)


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


def _save_new_samples(path: str, count: int, image_size: int) -> None:
    """
    Save new samples from the dataset.
    """
    dataloader, _ = create_mnist_dataloaders(count, image_size)
    for i, (images, _) in enumerate(dataloader):
        for j, image in enumerate(images):
            save_path = os.path.join(path, f"mnist_real_{j + 1}.png")
            save_image(image, save_path)
        break  # one batch is enough since batch size is set to the needed number of images


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
    )

    print(f"FID: {fid}")
