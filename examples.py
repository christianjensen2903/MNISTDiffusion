from utils import save_images
import torch
from data_loader import create_mnist_dataloaders


def main():
    # Save a grid of 4 of each class
    batch_size = 32
    image_size = 32

    train_dataloader, test_dataloader = create_mnist_dataloaders(
        batch_size, image_size=image_size
    )

    samples = []
    labels = []
    for batch in train_dataloader:
        x, y = batch
        samples.append(x)
        labels.append(y)

    samples = torch.cat(samples, dim=0)
    labels = torch.cat(labels, dim=0)

    grid = []

    for i in range(10):
        x = samples[labels == i]
        grid.append(x[:4])

    # Make the rows to columns
    grid = torch.stack(grid, dim=1)
    grid = grid.view(-1, *grid.shape[2:])
    save_images(grid, "debug/grid.png")


if __name__ == "__main__":
    main()
