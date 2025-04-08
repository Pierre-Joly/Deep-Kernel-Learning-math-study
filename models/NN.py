import torch.nn as nn
import torch.nn.functional as F

class RotationRegressor(nn.Module):
    """
    Convolutional neural network for image-based rotation regression.

    Input shape:
        (batch_size, 1, 64, 64) grayscale image

    Output:
        Single continuous value per input (e.g., angle)

    """

    def __init__(self):
        super().__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Output: (B, 16, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # Output: (B, 16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # Output: (B, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # Output: (B, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # Output: (B, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # Output: (B, 64, 8, 8)
        )

        # Fully connected regression head
        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # Flattened input
            nn.ReLU(),
            nn.Linear(128, 1)            # Output: scalar angle
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 1, 64, 64)

        Returns:
            Tensor of shape (B, 1) with predicted rotation values
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.regressor(x)
        return x
