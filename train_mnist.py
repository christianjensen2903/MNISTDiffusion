import torch
from noise_ddpm import NoiseDDPM
from cold_ddpm import ColdDDPM
from scaling_ddpm import (
    ScalingDDPM,
    PowerScheduler,
    ArithmeticScheduler,
    GeometricScheduler,
    UniformScheduler,
)
from pixelate import Pixelate
from initializers import GMMInitializer, RandomSampleInitializer
from ddpm import DDPM
from unet import UNetModel
from torch.utils.data import DataLoader
from data_loader import (
    create_mnist_dataloaders,
    create_cifar_dataloaders,
    create_celeba_dataloaders,
)
import wandb
from pydantic import BaseModel
from tqdm import tqdm
from utils import (
    save_images,
    save_model,
    setup_device,
    ExponentialMovingAverage,
    scale_images,
)
from enum import Enum
from metrics import calculate_fid, calculate_ssim, calculate_cas
import time


class ModelType(str, Enum):
    cold = "cold"
    noise = "noise"
    scaling = "scaling"


class Dataset(str, Enum):
    mnist = "mnist"
    cifar = "cifar"
    celeba = "celeba"


class LossType(str, Enum):
    l1 = "l1"
    l2 = "l2"


class ArgsModel(BaseModel):
    batch_size: int = 32
    timesteps: int = 1000
    n_between: int = 1
    minimum_pixelation: int = 2
    n_feat = 256
    num_res_blocks = 4
    num_heads = 1
    attention_resolutions = [2, 4]
    channel_mult = (1, 2, 4)
    epochs: int = 50
    lr: float = 1e-4
    positional_degree: int = 6
    betas = (1e-4, 0.02)
    log_freq: int = 200
    image_size: int = 32
    dropout: float = 0.3
    gmm_components: int = 10
    n_classes: int = 10
    model_ema_steps: int = 10
    model_ema_decay: float = 0.995
    model_type: ModelType = ModelType.scaling
    loss_type: LossType = LossType.l1
    dataset: Dataset = Dataset.cifar
    channels: int = 3
    level_scheduler: str = "power"
    power: float = 0
    log_wandb: bool = True
    calculate_metrics: bool = True
    sweep_id: str = None
    save_model = False
    save_dir = "./data/diffusion_outputs10/"
    models_dir = "./models/"
    gradient_accumulation_steps = 5

    class Config:
        use_enum_values = True

    @classmethod
    def from_wandb_config(cls, wandb_config: dict) -> "ArgsModel":
        # Update default values with those from wandb.config
        updated_values = {**cls().dict(), **wandb_config}
        return cls(**updated_values)


def train_model(
    model: DDPM,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    args: ArgsModel,
    device: str,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_time = 0

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    global_steps = 0
    for ep in range(args.epochs):
        start_time = time.time()

        print(f"epoch {ep}")
        model.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = args.lr * (1 - ep / args.epochs)

        pbar = tqdm(train_dataloader)
        total_loss = 0

        i = 0
        loss_per_size = {}
        accumulated_loss = 0.0
        gradient_accumulation_steps = 4  # New parameter for gradient accumulation steps

        for step, (x, c) in enumerate(pbar):
            x, c = x.to(device), c.to(device)

            loss, pred, x_t, x_downscaled = model(x, c)

            if x_downscaled.shape[-1] not in loss_per_size:
                loss_per_size[x_downscaled.shape[-1]] = []

            loss_per_size[x_downscaled.shape[-1]].append(loss.item())

            # Accumulate loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(
                train_dataloader
            ):
                optim.step()
                optim.zero_grad()

                if (global_steps % args.model_ema_steps) == 0:
                    model_ema.update_parameters(model)

                global_steps += 1

            # Save images for visualization
            if i < 5:
                size = x.shape[-1]
                save_images(
                    scale_images(x_downscaled, size), f"debug/x_downscaled_{i}.png"
                )
                save_images(scale_images(x_t, size), f"debug/x_t_{i}.png")
                save_images(scale_images(pred, size), f"debug/pred_{i}.png")
                save_images(x, f"debug/x_{i}.png")
            i += 1

            total_loss += (
                loss.item() * gradient_accumulation_steps
            )  # Accumulate for display purposes
            average_loss = accumulated_loss / i
            pbar.set_description(f"loss: {average_loss:.4f}")

        # Print average loss per size
        for size in sorted(loss_per_size.keys()):
            losses = loss_per_size[size]
            print(f"Average loss for size {size}: {sum(losses) / len(losses):.4f}")

        end_time = time.time()

        total_time += end_time - start_time

        visual_samples = 4 * args.n_classes

        samples, labels = generate_samples(
            model_ema.module,
            args,
            device,
            500 if args.calculate_metrics else visual_samples,
        )
        target = get_balanced_samples(train_dataloader, samples.shape[0]).to(device)

        save_images(samples[:40], f"debug/samples.png")
        save_images(target[:40], f"debug/target.png")

        save_images(samples[:visual_samples], args.save_dir + f"image_ep{ep}.png")

        avg_loss = total_loss / len(train_dataloader)
        if args.calculate_metrics:
            fid = calculate_fid(
                samples,
                target,
                device=device,
            )
            print(f"EPOCH {ep + 1} | LOSS: {avg_loss:.4f} | FID: {fid:.4f}\n")
        else:
            print(f"EPOCH {ep + 1} | LOSS: {avg_loss:.4f}\n")
            fid = None

        if args.log_wandb:
            log_wandb(ep, avg_loss, fid, total_time, args)

    if args.save_model:
        save_model(model, args.save_dir + "model.pth")
        if args.log_wandb:
            wandb.save(args.save_dir + "model.pth")

    evaluate_model(model, args, train_dataloader, test_dataloader, device)


def get_balanced_samples(dataloader: DataLoader, count: int) -> torch.Tensor:
    # Assuming that the dataset is labeled with classes 0, 1, ..., n-1
    # for a total of n classes
    n_classes = len(dataloader.dataset.classes)

    # Determine how many samples per class we want
    samples_per_class = count // n_classes

    # Dict to track how many samples we've taken from each class
    class_counts = {i: 0 for i in range(n_classes)}

    balanced_samples = []

    for x, y in dataloader:
        for i in range(len(y)):
            # Check if we've reached the desired number of samples for this class
            if class_counts[y[i].item()] < samples_per_class:
                balanced_samples.append(x[i])
                class_counts[y[i].item()] += 1

                # Check if we've reached the total count
                if len(balanced_samples) == count:
                    return torch.stack(balanced_samples)

    return torch.stack(balanced_samples)


def generate_samples(
    model: DDPM, args: ArgsModel, device: str, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generated_samples = torch.tensor([]).to(device)
    generated_labels = torch.tensor([], dtype=torch.long).to(
        device
    )  # assuming labels are of type long

    pbar = tqdm(total=count, desc="Generating Samples", unit="sample")

    while len(generated_samples) < count:
        samples, labels = model.sample(
            args.batch_size, (1, args.image_size, args.image_size)
        )
        generated_samples = torch.cat((generated_samples, samples))
        generated_labels = torch.cat((generated_labels, labels))

        pbar.update(min(args.batch_size, count - len(generated_samples)))

    generated_samples = generated_samples[:count]
    generated_labels = generated_labels[:count]

    pbar.close()

    return generated_samples, generated_labels


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
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: str,
):
    print("\nEvaluating model...")

    start_time = time.time()

    samples, labels = generate_samples(model, args, device, 2000)
    target = get_balanced_samples(train_dataloader, samples.shape[0]).to(device)

    sampling_time = time.time() - start_time

    final_fid = calculate_fid(
        samples,
        target,
        device=device,
    )

    final_cas = calculate_cas(
        samples,
        labels,
        test_dataloader,
        device,
    )

    print(f"Final FID: {final_fid:.4f}")
    print(f"Final CAS: {final_cas:.4f}")
    print(f"Sampling time: {sampling_time:.4f}")

    if args.log_wandb:
        wandb.log(
            {
                "sampling_time": sampling_time,
                "final_fid": final_fid,
                "final_cas": final_cas,
            }
        )


def initialize_wandb(args: ArgsModel) -> ArgsModel:
    wandb.init(
        project="speeding_up_diffusion",
        config=args.dict() if args.sweep_id is None else None,
        tags=[args.dataset.value, args.model_type.value],
    )
    return args.from_wandb_config(wandb.config)


def main():
    args = ArgsModel()

    if args.log_wandb:
        args = initialize_wandb(args)

    device = setup_device()

    if args.dataset == Dataset.mnist:
        train_dataloader, test_dataloader = create_mnist_dataloaders(
            batch_size=args.batch_size, image_size=args.image_size
        )
    elif args.dataset == Dataset.cifar:
        train_dataloader, test_dataloader = create_cifar_dataloaders(
            batch_size=args.batch_size, image_size=args.image_size
        )
    elif args.dataset == Dataset.celeba:
        train_dataloader, test_dataloader = create_celeba_dataloaders(
            batch_size=args.batch_size, image_size=args.image_size
        )

    if args.loss_type == LossType.l1:
        criterion = torch.nn.L1Loss()
    elif args.loss_type == LossType.l2:
        criterion = torch.nn.MSELoss()

    initializer = RandomSampleInitializer(
        train_dataloader,
        to_size=args.minimum_pixelation,
        n_components=args.gmm_components,
    )
    pixelate_T = Pixelate(args.n_between, args.minimum_pixelation).calculate_T(
        args.image_size
    )

    attention_ds = []
    for res in args.attention_resolutions:
        attention_ds.append(args.image_size // int(res))

    if args.model_type == ModelType.noise:
        model = NoiseDDPM(
            unet=UNetModel(
                in_channels=args.channels,
                model_channels=args.n_feat,
                out_channels=args.channels,
                num_res_blocks=args.num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                num_classes=args.n_classes,
                channel_mult=args.channel_mult,
                dropout=args.dropout,
                num_heads=args.num_heads,
            ),
            T=args.timesteps,
            device=device,
            criterion=criterion,
            n_classes=args.n_classes,
            betas=args.betas,
        )
    elif args.model_type == ModelType.cold:
        model = ColdDDPM(
            unet=UNetModel(
                in_channels=args.channels,
                model_channels=args.n_feat,
                out_channels=args.channels,
                num_res_blocks=args.num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                num_classes=args.n_classes,
                channel_mult=args.channel_mult,
                dropout=args.dropout,
                num_heads=args.num_heads,
            ),
            T=pixelate_T,
            device=device,
            criterion=criterion,
            n_classes=args.n_classes,
            max_step_size=args.n_between,
            initializer=initializer,
            minimum_pixelation=args.minimum_pixelation,
        )
    elif args.model_type == ModelType.scaling:
        if args.level_scheduler == "power":
            level_scheduler = PowerScheduler(power=args.power)
        elif args.level_scheduler == "arithmetic":
            level_scheduler = ArithmeticScheduler()
        elif args.level_scheduler == "geometric":
            level_scheduler = GeometricScheduler()
        elif args.level_scheduler == "uniform":
            level_scheduler = UniformScheduler()
        else:
            raise ValueError("Invalid level scheduler")

        model = ScalingDDPM(
            unet=UNetModel(
                in_channels=args.channels + args.positional_degree * 4,
                model_channels=args.n_feat,
                out_channels=args.channels,
                image_size=args.image_size,
                num_res_blocks=args.num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                num_classes=args.n_classes,
                channel_mult=args.channel_mult,
                dropout=args.dropout,
                num_heads=args.num_heads,
            ),
            T=pixelate_T,
            device=device,
            n_classes=args.n_classes,
            criterion=criterion,
            n_between=args.n_between,
            initializer=initializer,
            minimum_pixelation=args.minimum_pixelation,
            positional_degree=args.positional_degree,
            level_scheduler=level_scheduler,
        )
    model.to(device)

    train_model(model, train_dataloader, test_dataloader, args, device)


if __name__ == "__main__":
    args = ArgsModel()

    if args.log_wandb:
        wandb.login()

    if args.sweep_id is not None and args.log_wandb:
        wandb.agent(args.sweep_id, main, count=20)
    else:
        main()
