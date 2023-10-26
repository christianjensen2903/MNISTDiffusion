import torch
from noise_ddpm import NoiseDDPM
from cold_ddpm import ColdDDPM
from scaling_ddpm import ScalingDDPM
from pixelate import Pixelate
from initializers import RandomColorInitializer, GMMInitializer
from ddpm import DDPM
from unet import ContextUnet
from torch.utils.data import DataLoader
from data_loader import create_mnist_dataloaders
import wandb
from pydantic import BaseModel
from tqdm import tqdm
from utils import save_images, save_model, setup_device
from enum import Enum
from fid import calculate_fid
import time


class ModelType(str, Enum):
    cold = "cold"
    noise = "noise"
    scaling = "scaling"


class ArgsModel(BaseModel):
    batch_size: int = 64
    timesteps: int = 4
    n_between: int = 10
    minimum_pixelation: int = 4
    n_feat = 32
    epochs: int = 50
    lr: float = 4e-4
    betas = (1e-4, 0.02)
    log_freq: int = 200
    image_size: int = 16
    n_classes: int = 10
    model_type: ModelType = ModelType.cold
    log_wandb: bool = False
    calculate_fid: bool = False
    sweep_id: str = None
    save_model = False
    save_dir = "./data/diffusion_outputs10/"
    models_dir = "./models/"

    @classmethod
    def from_wandb_config(cls, wandb_config: dict) -> "ArgsModel":
        # Update default values with those from wandb.config
        updated_values = {**cls().dict(), **wandb_config}
        return cls(**updated_values)


def train_model(
    model: DDPM, train_dataloader: DataLoader, args: ArgsModel, device: str
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_time = 0

    for ep in range(args.epochs):
        start_time = time.time()

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

        end_time = time.time()

        total_time += end_time - start_time

        samples = generate_samples(model, args, device, 2048)

        save_images(
            samples[: (5 * args.n_classes)], args.save_dir + f"image_ep{ep}.png"
        )

        avg_loss = total_loss / len(train_dataloader)
        if args.calculate_fid:
            fid = calculate_fid(
                samples,
                device=device,
                batch_size=args.batch_size,
                image_size=args.image_size,
            )
        else:
            fid = 0

        print(f"EPOCH {ep + 1} | LOSS: {avg_loss:.4f} | FID: {fid:.4f}\n")

        if args.log_wandb:
            log_wandb(ep, avg_loss, fid, total_time, args)

        if args.save_model and ep == int(args.epochs - 1):
            save_model(model, args.save_dir + "model.pth")

    evaluate_model(model, args, device)


def generate_samples(
    model: DDPM, args: ArgsModel, device: str, count: int
) -> torch.Tensor:
    generated_samples = torch.tensor([])
    while len(generated_samples) < count:
        samples = model.sample(args.batch_size, (1, args.image_size, args.image_size))
        if device == "cuda":
            samples = samples.cuda().cpu()

        generated_samples = torch.cat((generated_samples, samples))
    generated_samples = generated_samples[:count]
    return generated_samples


def log_wandb(
    ep: int,
    train_loss: float,
    fid: float,
    total_time: float,
    args: ArgsModel,
) -> None:
    wandb.log(
        {
            "epoch": ep + 1,
            "train_loss": train_loss,
            "fid": fid,
            "total_time": total_time,
            f"sample": wandb.Image(args.save_dir + f"image_ep{ep}.png"),
        }
    )


def evaluate_model(
    model: DDPM,
    args: ArgsModel,
    device: str,
):
    print("\nEvaluating model...")

    start_time = time.time()

    samples = generate_samples(model, args, device, 10000)

    sampling_time = time.time() - start_time

    final_fid = calculate_fid(
        samples,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    print(f"Final FID: {final_fid:.4f}")
    print(f"Sampling time: {sampling_time:.4f}")

    if args.log_wandb:
        wandb.log({"sampling_time": sampling_time, "final_fid": final_fid})


def initialize_wandb(args: ArgsModel) -> ArgsModel:
    wandb.init(
        project="speeding_up_diffusion",
        config=args.dict() if args.sweep_id is None else None,
        tags=["mnist", args.model_type.value],
    )
    return args.from_wandb_config(wandb.config)


def main(args: ArgsModel):
    if args.log_wandb:
        args = initialize_wandb(args)

    device = setup_device()

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size=args.batch_size, image_size=args.image_size
    )

    initializer = GMMInitializer(
        image_size=args.image_size,
        to_size=args.minimum_pixelation,
        path=f"models/gmm_{args.minimum_pixelation}.pkl",
    )
    pixelate_T = Pixelate(args.n_between, args.minimum_pixelation).calculate_T(
        args.image_size
    )

    if args.model_type == ModelType.noise:
        model = NoiseDDPM(
            unet=ContextUnet(
                in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes
            ),
            T=args.timesteps,
            device=device,
            n_classes=args.n_classes,
            betas=args.betas,
        )
    elif args.model_type == ModelType.cold:
        model = ColdDDPM(
            unet=ContextUnet(
                in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes
            ),
            T=pixelate_T,
            device=device,
            n_classes=args.n_classes,
            n_between=args.n_between,
            initializer=initializer,
            minimum_pixelation=args.minimum_pixelation,
        )
    elif args.model_type == ModelType.scaling:
        model = ScalingDDPM(
            unet=ContextUnet(
                in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes
            ),
            T=pixelate_T,
            device=device,
            n_classes=args.n_classes,
            n_between=args.n_between,
            initializer=initializer,
            minimum_pixelation=args.minimum_pixelation,
        )
    model.to(device)

    train_model(model, train_dataloader, args, device)


if __name__ == "__main__":
    args = ArgsModel()

    if args.log_wandb:
        wandb.login()

    if args.sweep_id is not None and args.log_wandb:
        wandb.agent(args.sweep_id, main, count=20)
    else:
        main(args)
