import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from data_loader import create_mnist_dataloaders
import numpy as np
import joblib
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import scale_images
import os
import torch


def train_gmm_on_pixelated_mnist(
    image_size=28, to_size=4, n_components=10, batch_size=64
):
    # Create dataloaders
    train_dataloader, _ = create_mnist_dataloaders(batch_size, image_size=image_size)

    # Dictionary to store pixelated images for each class
    pixelated_images_per_class = defaultdict(list)

    for images, labels in train_dataloader:
        for image, label in zip(images, labels):
            image = scale_images(image.unsqueeze(0), to_size=to_size)
            pixelated_images_per_class[int(label)].append(image.view(-1).numpy())

    # Dictionary to store trained GMM for each class
    gmms = {}

    # Train GMM for each class
    for label, pixelated_images in pixelated_images_per_class.items():
        print(f"Training GMM for class {label}")
        X_train = np.array(pixelated_images)
        gmm = GaussianMixture(n_components=n_components, covariance_type="full")
        gmm.fit(X_train)
        gmms[label] = gmm

    return gmms


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
    path="gmm_model.pkl", image_size: int = 28, to_size=4, n_components=10
):
    # If the GMM model is not trained, train it
    if not os.path.exists(path):
        gmms = train_gmm_on_pixelated_mnist(
            image_size=image_size, n_components=n_components, to_size=to_size
        )
        save_gmm_model(gmms, path)
    else:
        gmms = load_gmm_model(path)

    return gmms


def sample_from_gmm_for_class(gmms, label: int, n_samples=1):
    """Sample data from a trained GMM model for a specific class."""

    if isinstance(label, str):
        label = int(label)

    if isinstance(label, torch.Tensor):
        label = label.item()

    if label in gmms:
        samples, _ = gmms[label].sample(n_samples)

        return samples
    else:
        raise ValueError(f"No GMM trained for class {label}")


def display_samples(samples):
    """Display a set of samples."""
    # Prepare the figure
    fig, axes = plt.subplots(1, len(samples), figsize=(10, 2))

    image_size = int(np.sqrt(samples.shape[1]))

    for ax, sample in zip(axes, samples):
        # Reshape the sample to its original shape
        image = sample.reshape(image_size, image_size)

        # The images were normalized to [-1, 1], so denormalize them back to [0, 1]
        image = (image + 1) / 2.0

        ax.imshow(image, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    path = "models/gmm_model.pkl"
    gmms = train_gmm_on_pixelated_mnist(image_size=32, to_size=4, n_components=10)
    save_gmm_model(gmms, path)
    gmms = load_gmm_model(path)

    # For instance, to sample from class 0
    samples_for_class_0 = sample_from_gmm_for_class(gmms, label=1, n_samples=5)

    # Display the samples
    display_samples(samples_for_class_0)


if __name__ == "__main__":
    main()
