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
from simple_cnn import SimpleCNN
import time


def init_wandb(args: ArgsModel) -> None:
    if args.log_wandb:
        wandb.login()
        wandb.init(
            project="speeding_up_diffusion",
            config=args.dict(),
            tags=["reference", "mnist"],
        )


def train_model(
    model: DDPM,
    train_dataloader: DataLoader,
    args: ArgsModel,
    device: str,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    accumulated_time = 0.0  # Initialize the accumulated time

    for ep in range(args.epochs):
        print(f"epoch {ep}")
        model.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = args.lr * (1 - ep / args.epochs)

        pbar = tqdm(train_dataloader)
        loss_ema = None

        # Start the timer for the current epoch
        start_time = time.time()

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

        # End the timer for the current epoch and accumulate the time
        end_time = time.time()
        accumulated_time += end_time - start_time

        print(f"Training time up to epoch {ep}: {accumulated_time:.2f} seconds")

        total_samples = args.n_samples * args.n_classes
        with torch.no_grad():
            samples, labels = model.sample(
                total_samples,
                (1, args.image_size, args.image_size),
                device,
                guide_w=args.w,
            )

        accuracy = calculate_accuracy(samples, labels, device)

        save_images(samples, path=args.save_dir + f"image_ep{ep}.png")

        if args.log_wandb:
            wandb.log(
                {
                    "training_time": accumulated_time,
                    "train_loss": loss_ema,
                    "accuracy": accuracy,
                    f"sample": wandb.Image(args.save_dir + f"image_ep{ep}.png"),
                }
            )
    if args.save_model:
        save_final_model(model, args, ep)


def calculate_accuracy(x, c, device: str) -> float:
    """
    Evaluate the model's accuracy.
    """
    # Load mnist_classifier_weights.pth
    model = SimpleCNN(args.image_size).to(device)

    x, c = x.to(device), c.to(device)
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)
    total = c.size(0)
    correct = (predicted == c).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def main(args: ArgsModel):
    init_wandb(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = setup_device()

    train_dataloader, _ = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.image_size
    )

    model = DDPM(
        nn_model=ContextUnet(
            in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes
        ),
        betas=args.betas,
        n_T=args.timesteps,
        device=device,
        drop_prob=args.drop_prob,
    )
    model.to(device)

    train_model(model, train_dataloader, args, device)


if __name__ == "__main__":
    args = ArgsModel()
    main(args)
