import torch
from noise_ddpm import NoiseDDPM
from cold_ddpm import ColdDDPM
from ddpm import DDPM
from unet import ContextUnet
from torch.utils.data import DataLoader
from data_loader import create_mnist_dataloaders
import wandb
from pydantic import BaseModel
from tqdm import tqdm
from utils import save_images, save_model, setup_device
from enum import Enum


class ModelType(str, Enum):
    cold = "cold"
    noise = "noise"


class ArgsModel(BaseModel):
    batch_size: int = 64
    timesteps: int = 4
    n_feat = 32
    epochs: int = 50
    lr: float = 4e-4
    betas = (1e-4, 0.02)
    log_freq: int = 200
    image_size: int = 16
    n_classes: int = 10
    model_type: ModelType = ModelType.cold
    log_wandb: bool = False
    save_model = False
    save_dir = "./data/diffusion_outputs10/"
    models_dir = "./models/"


def init_wandb(args: ArgsModel) -> None:
    if args.log_wandb:
        wandb.login()
        wandb.init(
            project="speeding_up_diffusion",
            config=args.dict(),
            tags=["mnist", args.model_type.value],
        )


def train_model(
    model: DDPM, train_dataloader: DataLoader, args: ArgsModel, device: str
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        print(f"epoch {ep}")
        model.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = args.lr * (1 - ep / args.epochs)

        pbar = tqdm(train_dataloader)
        total_loss = 0

        for x, c in pbar:
            optim.zero_grad()
            x, c = x.to(device), c.to(device)

            loss = model(x, c)
            loss.backward()

            total_loss += loss.item()

            pbar.set_description(f"loss: {loss.item():.4f}")
            optim.step()

        save_generated_samples(model, args, ep)
        avg_loss = total_loss / len(train_dataloader)
        if args.log_wandb:
            log_wandb(ep, avg_loss, args)

        if args.save_model and ep == int(args.epochs - 1):
            save_model(model, args.save_dir + "model.pth")


def log_wandb(
    ep: int,
    train_loss: float,
    args: ArgsModel,
) -> None:
    wandb.log(
        {
            "epoch": ep + 1,
            "train_loss": train_loss,
            f"sample": wandb.Image(args.save_dir + f"image_ep{ep}.png"),
        }
    )


def save_generated_samples(
    model: DDPM,
    args: ArgsModel,
    ep: int,
) -> None:
    model.eval()
    with torch.no_grad():
        n_sample = 4 * args.n_classes
        x_gen = model.sample(n_sample, (1, args.image_size, args.image_size))

        save_images(x_gen, args.save_dir + f"image_ep{ep}.png")


def main(args: ArgsModel):
    init_wandb(args)

    device = setup_device()

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.image_size
    )

    if args.model_type == ModelType.noise:
        model = NoiseDDPM(
            unet=ContextUnet(
                in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes
            ),
            T=args.timesteps,
            device=device,
            betas=args.betas,
        )
    elif args.model_type == ModelType.cold:
        model = ColdDDPM(
            unet=ContextUnet(
                in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes
            ),
            T=args.timesteps,
            device=device,
        )
    model.to(device)

    train_model(model, train_dataloader, args, device)


if __name__ == "__main__":
    args = ArgsModel()
    main(args)
