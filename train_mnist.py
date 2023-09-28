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
from torchvision.models import inception_v3


def init_wandb(args: ArgsModel) -> None:
    if args.log_wandb:
        wandb.login()
        wandb.init(
            project="speeding_up_diffusion",
            config=args.dict(),
            tags=["progressive_scaling", "mnist"],
        )


def train_model(
    model: DDPM,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    args: ArgsModel,
    device: str,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    inception_model = inception_v3(pretrained=True, transform_input=True).to(device)
    inception_model.eval()

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

        total_samples = args.n_samples * args.n_classes
        with torch.no_grad():
            generated_samples, _ = model.sample(
                total_samples,
                (1, args.image_size, args.image_size),
                device,
                guide_w=args.w,
            )

        real_samples = get_real_samples(
            test_dataloader, generated_samples, args, device
        )

        fid = compute_fid(real_samples, generated_samples, inception_model)

        save_images(generated_samples, path=args.save_dir + f"image_ep{ep}.png")

        save_generated_samples(model, train_dataloader, args, device, ep)
        if args.log_wandb:
            wandb.log(
                {
                    "epoch": ep + 1,
                    "train_loss": loss_ema,
                    "fid": fid,
                    f"sample": wandb.Image(args.save_dir + f"image_ep{ep}.png"),
                }
            )

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

    train_model(model, train_dataloader, test_dataloader, args, device)


if __name__ == "__main__":
    args = ArgsModel()
    main(args)
