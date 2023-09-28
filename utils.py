import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import wandb
from args import ArgsModel


def setup_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        x_gen, _ = model.sample(n_sample, (1, 28, 28), device, guide_w=args.w)

        # append some real images at bottom, order by class also
        x_real = get_real_samples(test_dataloader, x_gen, args, device)

        save_images(x_gen, x_real, args, ep)


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


def save_final_model(model: torch.nn.Module, args: ArgsModel, ep: int) -> None:
    if args.save_model and ep == int(args.epochs - 1):
        path = args.save_dir + f"model_{ep}.pth"
        torch.save(model.state_dict(), path)
        print("saved model at " + path)
