import torch
import torch.nn as nn
import torch.nn.functional as F

class FNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(6, 64, 3, padding=1)  # 2 frames (3x2 channels)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 2, 3, padding=1)  # (u, v) flow map

    def forward(self, prev_lr, curr_lr):
        x = torch.cat([prev_lr, curr_lr], dim=1)
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x = F.leaky_relu(self.deconv1(x3), 0.2)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        flow = torch.tanh(self.out(x))  # normalize flow to [-1, 1]
        return flow