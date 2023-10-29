import torch
import torch.nn as nn
import torch.optim as optim
from simple_cnn import SimpleCNN
from torch.utils.data import DataLoader


def train_model(model: SimpleCNN, train_loader: DataLoader, device: str):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )


def evalaute_model(model: SimpleCNN, test_loader: DataLoader, device: str) -> float:
    # Test the model
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No gradients needed during evaluation
        correct = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy
