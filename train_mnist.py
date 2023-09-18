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


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckpt", type=str, help="define checkpoint path", default="")
    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument(
        "--model_base_dim", type=int, help="base dim of Unet", default=64
    )
    parser.add_argument(
        "--timesteps", type=int, help="sampling steps of DDPM", default=1000
    )
    parser.add_argument(
        "--model_ema_steps", type=int, help="ema model evaluation interval", default=10
    )
    parser.add_argument(
        "--model_ema_decay", type=float, help="ema model decay", default=0.995
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=100,
    )
    parser.add_argument("--cpu", action="store_true", help="cpu training")

    args = parser.parse_args()

    return args


image_size = 32


def main(args):
    wandb.login()
    wandb.init(
        project="speeding_up_diffusion", config=args, tags=["progessive_scaling"]
    )

    device = "cpu" if args.cpu else "cuda"
    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=image_size
    )
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=image_size,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4, 8],
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
            random_int = torch.randint(0, args.timesteps, (1,))
            level = _calculate_level(random_int, args.timesteps, 3)

            image = _downscale(image, level)
            noise = torch.randn_like(image).to(device)
            image = image.to(device)

            # Expand it to have the same shape as the first dimension of x
            t = random_int.expand(image.shape[0]).to(image.device)

            pred = model(image, noise, t, level=level)
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

        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}

        os.makedirs("results", exist_ok=True)
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


def _calculate_level(t, timesteps, levels):
    # Divide the timesteps into levels equally
    level = (t) // (timesteps // levels)

    return level


def _calc_rescale_factor(level):
    rescale_factor = 2**level

    return rescale_factor


def _downscale(x, level):
    rescale_factor = _calc_rescale_factor(level)
    new_img_size = image_size // rescale_factor
    x = transforms.Resize(
        size=(new_img_size, new_img_size),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )(x)

    return x


if __name__ == "__main__":
    args = parse_args()
    main(args)
