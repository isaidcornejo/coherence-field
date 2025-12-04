import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=4, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


def build_mnist_model(device=None):
    model = MLP()
    if device is not None:
        model = model.to(device)
    return model
