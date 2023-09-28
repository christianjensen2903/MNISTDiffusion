import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from model import DDPM
from unet import Unet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import wandb
from pydantic import BaseModel
from tqdm import tqdm
import os


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=2):
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = MNIST(
        root="./mnist_data", train=True, download=True, transform=preprocess
    )
    test_dataset = MNIST(
        root="./mnist_data", train=False, download=True, transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


class ArgsModel(BaseModel):
    batch_size: int = 256
    timesteps: int = 400
    n_feat = 128
    epochs: int = 15
    lr: float = 1e-4
    n_samples: int = 36  # Samples after every epoch trained
    log_freq: int = 10
    image_size: int = 28
    log_wandb: bool = False
    save_model = False
    save_dir = "./data/diffusion_outputs10/"


def init_wandb(args: ArgsModel) -> None:
    if args.log_wandb:
        wandb.login()
        wandb.init(
            project="speeding_up_diffusion",
            config=args.dict(),
            tags=["progressive_scaling", "mnist"],
        )


def setup_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model(
    model: torch.nn.Module, train_dataloader: DataLoader, args: ArgsModel, device: str
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        print(f"epoch {ep}")
        model.train()

        pbar = tqdm(train_dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            loss = model(x)
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        save_generated_samples(model, args, device, ep)
        if args.log_wandb:
            log_wandb(ep, loss_ema, args, ep)

    if args.save_model:
        save_final_model(model, args, ep)


def log_wandb(
    i: int,
    train_loss: float,
    args: ArgsModel,
    ep: int,
) -> None:
    wandb.log(
        {
            "epoch": i + 1,
            "train_loss": train_loss,
            f"sample": wandb.Image(args.save_dir + f"image_ep{ep}.png"),
        }
    )


def save_generated_samples(
    model: DDPM,
    args: ArgsModel,
    device: str,
    ep: int,
) -> None:
    model.eval()
    with torch.no_grad():
        n_sample = 16
        samples = model.sample(n_sample, (1, args.image_size, args.image_size), device)

        save_images(samples, args, ep)


def save_images(samples: torch.Tensor, args: ArgsModel, ep: int) -> None:
    grid = make_grid(samples, nrow=4)
    path = args.save_dir + f"image_ep{ep}.png"
    save_image(grid, path)
    print("saved image at " + path)


def save_final_model(model: torch.nn.Module, args: ArgsModel, ep: int) -> None:
    if args.save_model and ep == int(args.epochs - 1):
        path = args.save_dir + f"model_{ep}.pth"
        torch.save(model.state_dict(), path)
        print("saved model at " + path)


def main(args: ArgsModel):
    init_wandb(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = setup_device()

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.image_size
    )

    model = DDPM(
        eps_model=Unet(in_channels=1, n_feat=args.n_feat),
        betas=(1e-4, 0.02),
        n_T=args.timesteps,
    )
    model.to(device)

    train_model(model, train_dataloader, args, device)


if __name__ == "__main__":
    args = ArgsModel()
    main(args)
