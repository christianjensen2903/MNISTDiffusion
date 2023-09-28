import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from model import DDPM
from unet import ContextUnet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
    dim_mults: list[int] = [2, 4]
    w = 0.5
    n_classes: int = 10
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
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args: ArgsModel,
    device: str,
    ep: int,
) -> None:
    model.eval()
    with torch.no_grad():
        n_sample = 4 * args.n_classes
        x_gen, x_gen_store = model.sample(n_sample, (1, 28, 28), device, guide_w=args.w)

        # append some real images at bottom, order by class also
        x_real = get_real_samples(test_dataloader, x_gen, args, device)

        save_images(x_gen, x_real, args, ep)

        if ep % 5 == 0 or ep == int(args.epochs - 1):
            create_gif(x_gen_store, args, n_sample, ep)


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


def save_images(
    x_gen: torch.Tensor, x_real: torch.Tensor, args: ArgsModel, ep: int
) -> None:
    x_all = torch.cat([x_gen, x_real])
    grid = make_grid(x_all * -1 + 1, nrow=10)
    path = args.save_dir + f"image_ep{ep}.png"
    save_image(grid, path)
    print("saved image at " + path)


def create_gif(
    x_gen_store: torch.Tensor, args: ArgsModel, n_sample: int, ep: int
) -> None:
    fig, axs = plt.subplots(
        nrows=int(n_sample / args.n_classes),
        ncols=args.n_classes,
        sharex=True,
        sharey=True,
        figsize=(8, 3),
    )

    def animate_diff(i: int) -> list:
        print(
            f"gif animating frame {i} of {x_gen_store.shape[0]}",
            end="\r",
        )
        plots = []
        for row in range(int(n_sample / args.n_classes)):
            for col in range(args.n_classes):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(
                    axs[row, col].imshow(
                        -x_gen_store[i, (row * args.n_classes) + col, 0],
                        cmap="gray",
                        vmin=(-x_gen_store[i]).min(),
                        vmax=(-x_gen_store[i]).max(),
                    )
                )
        return plots

    ani = FuncAnimation(
        fig,
        animate_diff,
        interval=200,
        blit=False,
        repeat=True,
        frames=x_gen_store.shape[0],
    )
    ani.save(
        args.save_dir + f"gif_ep{ep}.gif",
        dpi=100,
        writer=PillowWriter(fps=5),
    )
    print("saved image at " + args.save_dir + f"gif_ep{ep}.gif")
    plt.close(fig)


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
