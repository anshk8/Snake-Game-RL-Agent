#QNetwork is the neural network for the DQNAgent
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Three-layer convolutional Q-network.
    
    For the default input size `(1, 20, 20)`, the spatial flow is:
    `(1, 20, 20) → conv(1→32, k=3, p=1) → (32, 20, 20)
                 → conv(32→64, k=3, p=1) → (64, 20, 20)
                 → conv(64→64, k=3, s=2) → (64,  9,  9)`
    
    There is no max-pooling layer in this architecture. The flattened input to
    the fully connected head is computed dynamically from `grid_size`; for a
    `20×20` grid it is `64×9×9 = 5,184`.
    """
    def __init__(self, grid_size=20, n_actions=3):
        super().__init__()

        self.grid_size = grid_size
        self.n_actions = n_actions

        # Spatial feature extractor — input shape: (batch, 1, grid_size, grid_size)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # → (B, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (B, 64, 20, 20)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),  # → (B, 64,  9,  9)
            nn.ReLU(),
        )

        # Dynamically compute flattened conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_size, grid_size)
            conv_out = self.conv(dummy).flatten(1).shape[1]

        # Q-value head
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        # x: (batch, grid_size²) — reshape to image tensor
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)  # returns Q(s, a) for all 3 actions