import torch
from model import DDPM
from unet import ContextUnet
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from data_loader import create_mnist_dataloaders
from utils import *
from args import ArgsModel


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

        # linear lrate decay
        optim.param_groups[0]["lr"] = args.lr * (1 - ep / args.epochs)

        pbar = tqdm(train_dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x, c = x.to(device), c.to(device)

            loss = model(x, c)
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        save_generated_samples(model, train_dataloader, args, device, ep)
        if args.log_wandb:
            log_wandb(ep, loss_ema, args, ep)

    if args.save_model:
        save_final_model(model, args, ep)


def main(args: ArgsModel):
    init_wandb(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = setup_device()

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.image_size
    )

    model = DDPM(
        nn_model=ContextUnet(
            in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes
        ),
        betas=(1e-4, 0.02),
        n_T=args.timesteps,
        device=device,
        drop_prob=0.1,
    )
    model.to(device)

    train_model(model, train_dataloader, args, device)


if __name__ == "__main__":
    args = ArgsModel()
    main(args)
