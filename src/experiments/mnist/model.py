import torch
from torch import nn


class MLP(nn.Module):
    """
    Simple 2-layer feedforward neural network for MNIST classification.

    Architecture:
        Input  →  Linear(input_dim, hidden_dim)
                → ReLU
                → Linear(hidden_dim, num_classes)
                → Output logits (before softmax)

    Notes:
        - The model is deliberately small, since its purpose is to allow
          fast gradient extraction for Fisher/empirical alignment analysis.
        - x is flattened automatically if received as (B, 1, 28, 28).
        - No batchnorm or dropout is used to avoid interfering with gradient
          statistics used by the alignment computations.
    """

    def __init__(self, input_dim=28 * 28, hidden_dim=4, num_classes=10):
        super().__init__()

        # First layer: flatten -> hidden space
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Nonlinearity (piecewise linear, stable for analysis)
        self.relu = nn.ReLU()

        # Output classifier
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): shape (B, 1, 28, 28) or (B, 784)

        Returns:
            Tensor: logits of shape (B, num_classes)
        """
        # Flatten images if needed
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        # Forward computation
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def build_mnist_model(device=None):
    """
    Factory function to construct an MLP model and move it to a device.

    Args:
        device (torch.device or None): If provided, model is moved to that device.

    Returns:
        nn.Module: Instantiated MLP model.
    """
    model = MLP()
    if device is not None:
        model = model.to(device)
    return model
