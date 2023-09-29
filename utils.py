import torch
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from args import ArgsModel
from torch.utils.data import DataLoader


def setup_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_images(x_gen: torch.Tensor, path: str) -> None:
    grid = make_grid(x_gen * -1 + 1, nrow=10)
    save_image(grid, path)


def save_final_model(model: torch.nn.Module, args: ArgsModel, ep: int) -> None:
    if args.save_model and ep == int(args.epochs - 1):
        path = args.save_dir + f"model_{ep}.pth"
        torch.save(model.state_dict(), path)
        print("saved model at " + path)
