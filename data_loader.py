from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CelebA


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=2):
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = MNIST(
        root="./data/mnist", train=True, download=True, transform=preprocess
    )
    test_dataset = MNIST(
        root="./data/mnist", train=False, download=True, transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def create_cifar_dataloaders(batch_size, image_size=28, num_workers=2):
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=preprocess
    )
    test_dataset = CIFAR10(
        root="./data/cifar10", train=False, download=True, transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def create_celeba_dataloaders(batch_size, image_size=64, num_workers=2):
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )  # [0,1] to [-1,1]

    train_dataset = CelebA(
        root="./data/celeba", split="train", download=True, transform=preprocess
    )
    test_dataset = CelebA(
        root="./data/celeba", split="test", download=True, transform=preprocess
    )

    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    ), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
