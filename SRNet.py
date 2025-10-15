import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block for SRNet: two 3×3 convolutions with skip connection.
    Helps with gradient flow and allows the network to learn refinements.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # Residual connection: x + F(x)
        return x + F.relu(self.conv2(F.relu(self.conv1(x))))

class SRNet(nn.Module):
    """
    Super-Resolution Network for FRVSR.
    Takes concatenated current LR frame + space-to-depth mapped warped HR estimate.
    Outputs an HR estimate at scale×scale the input resolution.
    
    Architecture:
    - Head: projects input channels to base_channels
    - Body: stack of residual blocks for feature extraction
    - Up1/Up2: 2× upsampling each (total 4× for scale=4)
    - Out: final conv to RGB output
    """
    def __init__(self, num_blocks=10, base_channels=64, scale=4):
        super().__init__()
        self.scale = scale
        # Input channels: 3 (curr LR) + (scale^2)*3 (mapped warped HR from space-to-depth)
        # For scale=4: 3 + 16*3 = 51 channels
        in_channels = 3 + (scale ** 2) * 3
        
        # Head: project concatenated input to base_channels
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Body: stack of residual blocks for deep feature extraction
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        
        # Upsampling layers: 2× each (H -> 2H -> 4H for scale=4)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)  # H -> 2H
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)  # 2H -> 4H
        
        # Output: final conv to RGB
        self.out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x):
        """
        Forward pass through SRNet.
        
        Args:
            x: concatenated tensor [B, 3 + (scale^2)*3, H, W]
               - First 3 channels: current LR frame
               - Next (scale^2)*3 channels: warped previous HR mapped to LR space via space-to-depth
        
        Returns:
            HR estimate [B, 3, scale*H, scale*W] in [0, 1] range
        """
        # Validate input shape
        expected_channels = 3 + (self.scale ** 2) * 3
        assert x.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {x.shape[1]}"
        
        # Head: project to feature space
        x = F.relu(self.head(x))  # [B, base_channels, H, W]
        
        # Body: residual blocks for feature refinement
        x = self.body(x)  # [B, base_channels, H, W]
        
        # Upsampling: 2× each step
        x = F.relu(self.up1(x))  # [B, base_channels, 2H, 2W]
        x = F.relu(self.up2(x))  # [B, base_channels, 4H, 4W]
        
        # Output: project to RGB and normalize to [0, 1]
        return torch.sigmoid(self.out(x))  # [B, 3, 4H, 4W]