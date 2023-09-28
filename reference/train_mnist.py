import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
import wandb
from pydantic import BaseModel


class ArgsModel(BaseModel):
    batch_size: int = 128
    ckpt: str | None = None  # Checkpoint path
    timesteps: int = 1000
    epochs: int = 30
    model_base_dim = 64
    lr: float = 0.001
    n_samples: int = 36  # Samples after every epoch trained
    model_ema_steps: int = 10
    model_ema_decay: float = 0.995
    log_freq: int = 10
    no_clip: bool = True
    image_size: int = 32
    dim_mults: list[int] = [2, 4]


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=2):
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size),
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


def main():
    args = ArgsModel()
    wandb.login()
    wandb.init(
        project="speeding_up_diffusion", config=args.dict(), tags=["reference", "mnist"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.image_size
    )
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=args.image_size,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=args.dim_mults,
    ).to(device)

    # torchvision ema setting
    # https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * len(train_dataloader),
        pct_start=0.25,
        anneal_strategy="cos",
    )
    loss_fn = nn.MSELoss(reduction="mean")

    # load checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    for i in range(args.epochs):
        model.train()

        total_loss = 0
        for j, (image, target) in enumerate(train_dataloader):
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            total_loss += loss.detach().cpu().item()

        avg_loss = total_loss / len(train_dataloader)
        print(
            "Epoch[{}/{}]:{:.5f}".format(
                i + 1,
                args.epochs,
                avg_loss,
            )
        )
        os.makedirs("results", exist_ok=True)
        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
        ckpt_name = f"results/epoch{i}.pt"
        torch.save(ckpt, ckpt_name)

        model_ema.eval()
        samples = model_ema.module.sampling(args.n_samples, device=device)

        save_image(
            samples,
            f"results/epoch_{i}.png",
            nrow=int(math.sqrt(args.n_samples)),
        )
        wandb.log(
            {
                "epoch": i + 1,
                "loss": avg_loss,
                "global_steps": global_steps,
                "lr": scheduler.get_last_lr()[0],
                f"sample": wandb.Image(f"results/epoch_{i}.png"),
            }
        )
    wandb.finish()


if __name__ == "__main__":
    main()