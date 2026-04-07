#QNetwork is the neural network for the DQNAgent
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    CNN with MaxPool — spatially compressed architecture.
    
    Spatial flow:
    (1, 20, 20) → conv → (32, 20, 20) → pool → (32, 10, 10)
               → conv → (64, 10, 10)  → pool → (64,  5,  5)
    
    FC input: 64×5×5 = 1,600  (was 25,600 — 16× smaller!)
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