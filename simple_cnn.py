import torch.nn as nn
import torch.nn.functional as F


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
