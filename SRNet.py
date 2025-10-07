import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + F.relu(self.conv2(F.relu(self.conv1(x))))

class SRNet(nn.Module):
    def __init__(self, num_blocks=10, base_channels=64, scale=4):
        super().__init__()
        self.head = nn.Conv2d(3 + (scale ** 2) * 3, base_channels, 3, padding=1)  # concat LR + mapped HR
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.head(x))
        x = self.body(x)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        return torch.sigmoid(self.out(x))  # output in [0,1]