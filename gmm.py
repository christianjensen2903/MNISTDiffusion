import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import scale_images
import os
import torch
from torch.utils.data import DataLoader
from data_loader import create_mnist_dataloaders


def train_gmm(dataloader: DataLoader, to_size=4, n_components=10):
    # List to store pixelated images
    pixelated_images_list = []

    for images, labels in dataloader:
        # Scale the images to the desired size
        images = scale_images(images, to_size=to_size)

        # Convert the images to numpy
        images = images.detach().cpu().numpy()

        # Flatten each image in the batch and add to the list
        flattened_images = images.reshape(images.shape[0], -1)
        pixelated_images_list.append(flattened_images)

    # Convert list of arrays to single numpy array
    X_train = np.concatenate(pixelated_images_list, axis=0)

    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(X_train)

    return gmm


def save_gmm_model(gmms, filename="gmm_model.pkl"):
    """Save the GMM models to a file."""
    # Check if the directory exists
    if os.path.dirname(filename) != "":
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    joblib.dump(gmms, filename)


def load_gmm_model(filename="gmm_model.pkl"):
    """Load the GMM models from a file."""
    return joblib.load(filename)


def load_if_exists(
    dataloader: DataLoader,
    path="gmm_model.pkl",
    to_size=4,
    n_components=10,
):
    # If the GMM model is not trained, train it
    if not os.path.exists(path):
        gmms = train_gmm(dataloader, n_components=n_components, to_size=to_size)
        save_gmm_model(gmms, path)
    else:
        gmms = load_gmm_model(path)

    return gmms


def sample_from_gmm_for_class(gmm, n_samples=1):
    """Sample data from a trained GMM model for a specific class."""

    samples, _ = gmm.sample(n_samples)

    return samples


def display_samples(samples):
    """Display a set of samples."""
    # Prepare the figure
    fig, axes = plt.subplots(1, len(samples), figsize=(10, 2))

    image_size = int(np.sqrt(samples.shape[1]))

    for ax, sample in zip(axes, samples):
        # Reshape the sample to its original shape
        image = sample.reshape(image_size, image_size)

        ax.imshow(image, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    path = "models/gmm_model.pkl"
    dataloader, _ = create_mnist_dataloaders(batch_size=32)
    gmm = train_gmm(dataloader, to_size=4, n_components=10)
    save_gmm_model(gmm, path)
    gmm = load_gmm_model(path)

    # For instance, to sample from class 0
    samples = sample_from_gmm_for_class(gmm, n_samples=10)

    # Display the samples
    display_samples(samples)


if __name__ == "__main__":
    main()
