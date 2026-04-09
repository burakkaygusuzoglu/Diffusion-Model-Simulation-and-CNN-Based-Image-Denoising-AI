import torch
import torch.nn as nn

class SimpleDenoiseCNN(nn.Module):
    def __init__(self):
        super(SimpleDenoiseCNN, self).__init__()
        # Improved denoising architecture with residual-like structure
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)