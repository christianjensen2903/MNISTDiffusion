import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import create_mnist_dataloaders
from args import ArgsModel
import torch.nn.functional as F

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = ArgsModel()


# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Compute the output size after convolutions and pooling operations
        # Assuming the input goes through two convs and two pooling operations
        self.output_size = input_size
        for _ in range(2):  # For the two conv + pool layers
            self.output_size = (
                self.output_size - 2 + 2 * 1
            ) // 2  # conv with padding=1 + pooling

        self.fc1_input_dim = 64 * self.output_size * self.output_size

        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, self.fc1_input_dim)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN(args.image_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the data
test_loader, train_loader = create_mnist_dataloaders(
    batch_size=args.batch_size, image_size=args.image_size, num_workers=0
)

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

    print(f"Accuracy of the model on the 10000 test images: {100 * correct / total}%")

# Save the model weights
torch.save(model.state_dict(), "mnist_classifier_weights.pth")
