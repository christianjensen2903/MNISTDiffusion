import torch
from torchvision.utils import save_image, make_grid
import os
import torch.nn.functional as F


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


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
