import torch
from torchvision.utils import save_image, make_grid
import os
import torch.nn.functional as F


def setup_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_images(samples: torch.Tensor, path: str) -> None:
    _create_folder_if_not_exist(path)
    grid = make_grid(samples, nrow=10)
    save_image(grid, path)


def save_model(model: torch.nn.Module, path: str) -> None:
    _create_folder_if_not_exist(path)
    torch.save(model.state_dict(), path)
    print("saved model at " + path)


def _create_folder_if_not_exist(path: str) -> None:
    directory = os.path.dirname(path)
    if not os.path.exists(directory) and directory != "":
        os.makedirs(directory)


def scale_images(images: torch.Tensor, to_size: int) -> torch.Tensor:
    squeeze = False
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
        squeeze = True

    images = F.interpolate(images, size=(to_size, to_size), mode="nearest")

    if squeeze:
        images = images.squeeze(0)

    return images
