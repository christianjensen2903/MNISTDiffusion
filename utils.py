import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from args import ArgsModel
from scipy.linalg import sqrtm
import numpy as np


def setup_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_generated_samples(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    args: ArgsModel,
    device: str,
    ep: int,
) -> None:
    model.eval()
    with torch.no_grad():
        n_sample = args.n_samples * args.n_classes
        x_gen, _ = model.sample(
            n_sample, (1, args.image_size, args.image_size), device, guide_w=args.w
        )

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


def get_inception_features(samples, model):
    features = model(samples).detach()
    return features


def compute_fid(real_samples, generated_samples, inception_model):
    real_features = get_inception_features(real_samples, inception_model)
    fake_features = get_inception_features(generated_samples, inception_model)

    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def save_images(x_gen: torch.Tensor, path: str) -> None:
    grid = make_grid(x_gen * -1 + 1, nrow=10)
    save_image(grid, path)


def save_final_model(model: torch.nn.Module, args: ArgsModel, ep: int) -> None:
    if args.save_model and ep == int(args.epochs - 1):
        path = args.save_dir + f"model_{ep}.pth"
        torch.save(model.state_dict(), path)
        print("saved model at " + path)
