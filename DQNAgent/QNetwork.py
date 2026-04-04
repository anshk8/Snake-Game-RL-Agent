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

    def __init__(self, grid_size=20, output_size=3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # → (32, 20, 20)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (32, 10, 10)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (64,  5,  5)
        )

        conv_out = 64 * 5 * 5   # = 1,600
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 20, 20) / 3.0   # ← normalize to [0, 1]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)