import torch
from torchvision.utils import save_image
from torchvision.utils import save_image, make_grid
from args import ArgsModel
from torch.utils.data import DataLoader


def setup_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_real_samples(
    test_dataloader: DataLoader, x_gen: torch.Tensor, args: ArgsModel, device: str
) -> torch.Tensor:
    n_sample = x_gen.shape[0]
    x_real = torch.Tensor(x_gen.shape).to(device)

    for batch, (x, c) in enumerate(test_dataloader):
        x, c = x.to(device), c.to(device)
        for k in range(args.n_classes):
            for j in range(int(n_sample / args.n_classes)):
                try:
                    idx = torch.squeeze((c == k).nonzero())[j]
                except:
                    idx = 0
                x_real[k + (j * args.n_classes)] = x[idx]
        if batch * args.batch_size >= n_sample:
            break

    return x_real


def save_images(x_gen: torch.Tensor, path: str) -> None:
    grid = make_grid(x_gen * -1 + 1, nrow=10)
    save_image(grid, path)


def save_final_model(model: torch.nn.Module, args: ArgsModel, ep: int) -> None:
    if args.save_model and ep == int(args.epochs - 1):
        path = args.save_dir + f"model_{ep}.pth"
        torch.save(model.state_dict(), path)
        print("saved model at " + path)
