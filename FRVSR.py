import torch
import torch.nn as nn
import torch.nn.functional as F
from FNet import FNet
from SRNet import SRNet

def warp(img, flow):
    """Warp an image (B,C,H,W) using an optical flow (B,2,H,W)"""
    B, _, H, W = img.size()
    # Create normalized grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij"
    )
    grid = torch.stack((x, y), 2).unsqueeze(0).repeat(B, 1, 1, 1)
    flow = flow.permute(0, 2, 3, 1)
    new_grid = grid + flow  # add displacement
    warped = F.grid_sample(img, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped

def space_to_depth(x, scale):
    """Convert HR image to LR-space using space-to-depth"""
    B, C, H, W = x.size()
    x = x.view(B, C, H // scale, scale, W // scale, scale)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C * (scale ** 2), H // scale, W // scale)

class FRVSR(nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale
        self.fnet = FNet()
        self.srnet = SRNet(scale=scale)

    def forward(self, prev_lr, curr_lr, prev_est_hr):
        # 1. Estimate optical flow in LR space
        flow_lr = self.fnet(prev_lr, curr_lr)

        # 2. Upscale flow to HR
        flow_hr = F.interpolate(flow_lr, scale_factor=self.scale, mode='bilinear', align_corners=False)

        # 3. Warp previous HR estimate
        warped_prev_est = warp(prev_est_hr, flow_hr)

        # 4. Map warped HR to LR space
        mapped = space_to_depth(warped_prev_est, self.scale)

        # 5. Concatenate with current LR frame and run SRNet
        concat = torch.cat([curr_lr, mapped], dim=1)
        curr_est_hr = self.srnet(concat)

        return curr_est_hr, flow_lr