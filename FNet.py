import torch
import torch.nn as nn
import torch.nn.functional as F


class Correlation(nn.Module):
    """
    A simple correlation layer that computes the cost volume between two feature maps.
    For each displacement (dx, dy) in [-max_disp, max_disp], compute:
        corr(x, y, dx, dy) = sum_c a[c, y, x] * b[c, y+dy, x+dx]
    Returns a tensor of shape (B, (2*max_disp+1)^2, H, W)
    """

    def __init__(self, max_disp: int = 4):
        super().__init__()
        self.max_disp = int(max_disp)

    def forward(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fmap1.shape
        pad = self.max_disp
        fmap2_pad = F.pad(fmap2, (pad, pad, pad, pad))
        corrs = []
        for dy in range(-pad, pad + 1):
            y0 = dy + pad
            y1 = y0 + H
            for dx in range(-pad, pad + 1):
                x0 = dx + pad
                x1 = x0 + W
                shifted = fmap2_pad[:, :, y0:y1, x0:x1]
                corr = (fmap1 * shifted).sum(dim=1, keepdim=True)
                corrs.append(corr)
        return torch.cat(corrs, dim=1)


class FNet(nn.Module):
    """
    FlowNetC-style optical flow network using correlation volume.

    - Inputs: prev_lr, curr_lr in LR space, each (B,3,H,W)
    - Output: flow in normalized grid units (B,2,H,W), tanh-limited to [-1,1]
    - Design: Siamese encoders -> correlation volume -> redirected features -> encoder/decoder with multi-scale flow
    """

    def __init__(self, max_corr_disp: int = 4, leaky: float = 0.1):
        super().__init__()

        # Siamese feature extractors up to conv3
        self.conv1a = nn.Conv2d(3, 64, 7, stride=2, padding=3)  # H/2
        self.conv2a = nn.Conv2d(64, 128, 5, stride=2, padding=2)  # H/4
        self.conv3a = nn.Conv2d(128, 256, 5, stride=2, padding=2)  # H/8

        self.conv1b = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2b = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3b = nn.Conv2d(128, 256, 5, stride=2, padding=2)

        self.corr = Correlation(max_disp=max_corr_disp)
        corr_channels = (2 * max_corr_disp + 1) ** 2

        # Redirection conv to combine with correlation
        self.conv_redir = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        # Post-correlation encoder
        self.conv3_1 = nn.Conv2d(corr_channels + 64, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # H/16
        self.conv4_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)  # H/32
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)  # H/64
        self.conv6_1 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)

        # Decoder
        self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)

        # Flow predictors
        self.predict_flow6 = nn.Conv2d(1024, 2, 3, padding=1)
        self.predict_flow5 = nn.Conv2d(512, 2, 3, padding=1)
        self.predict_flow4 = nn.Conv2d(256, 2, 3, padding=1)
        self.predict_flow3 = nn.Conv2d(128, 2, 3, padding=1)
        self.predict_flow2 = nn.Conv2d(64, 2, 3, padding=1)

        # Flow upsamplers
        self.upsample_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
        self.upsample_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
        self.upsample_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
        self.upsample_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)

        # Iconvs
        self.iconv5 = nn.Conv2d(512 + 512 + 2, 512, 3, padding=1)
        self.iconv4 = nn.Conv2d(512 + 256 + 2, 256, 3, padding=1)
        self.iconv3 = nn.Conv2d(256 + 128 + 2, 128, 3, padding=1)
        self.iconv2 = nn.Conv2d(128 + 64 + 2, 64, 3, padding=1)

        self.act = nn.LeakyReLU(leaky, inplace=True)

    def encode_siamese(self, x: torch.Tensor, branch: str):
        if branch == 'a':
            c1 = self.act(self.conv1a(x))
            c2 = self.act(self.conv2a(c1))
            c3 = self.act(self.conv3a(c2))
        else:
            c1 = self.act(self.conv1b(x))
            c2 = self.act(self.conv2b(c1))
            c3 = self.act(self.conv3b(c2))
        return c1, c2, c3

    def forward(self, prev_lr: torch.Tensor, curr_lr: torch.Tensor) -> torch.Tensor:
        # Siamese encoders
        _, _, c3a = self.encode_siamese(prev_lr, 'a')
        _, _, c3b = self.encode_siamese(curr_lr, 'b')

        # Correlation volume at H/8 scale
        corr = self.corr(c3a, c3b)
        corr = self.act(corr)

        # Redirect and fuse
        redir = self.act(self.conv_redir(c3a))
        x = torch.cat([corr, redir], dim=1)

        # Deeper encoder
        c3_1 = self.act(self.conv3_1(x))
        c4 = self.act(self.conv4(c3_1))
        c4_1 = self.act(self.conv4_1(c4))
        c5 = self.act(self.conv5(c4_1))
        c5_1 = self.act(self.conv5_1(c5))
        c6 = self.act(self.conv6(c5_1))
        c6_1 = self.act(self.conv6_1(c6))

        # Decoder with multi-scale flow
        flow6 = self.predict_flow6(c6_1)
        up_flow6 = self.upsample_flow6_to_5(flow6)
        deconv5 = self.deconv5(c6_1)

        concat5 = torch.cat([c5_1, deconv5, up_flow6], dim=1)
        iconv5 = self.act(self.iconv5(concat5))
        flow5 = self.predict_flow5(iconv5)
        up_flow5 = self.upsample_flow5_to_4(flow5)
        deconv4 = self.deconv4(iconv5)

        concat4 = torch.cat([c4_1, deconv4, up_flow5], dim=1)
        iconv4 = self.act(self.iconv4(concat4))
        flow4 = self.predict_flow4(iconv4)
        up_flow4 = self.upsample_flow4_to_3(flow4)
        deconv3 = self.deconv3(iconv4)

        concat3 = torch.cat([c3_1, deconv3, up_flow4], dim=1)
        iconv3 = self.act(self.iconv3(concat3))
        flow3 = self.predict_flow3(iconv3)
        up_flow3 = self.upsample_flow3_to_2(flow3)
        deconv2 = self.deconv2(iconv3)

        # There is no encoder c2 in this branch; we upsample to H/4 and refine
        concat2 = torch.cat([deconv2, up_flow3], dim=1)
        iconv2 = self.act(self.iconv2(concat2))
        flow2 = self.predict_flow2(iconv2)  # H/4

        # Upsample to LR resolution (H, W)
        flow = F.interpolate(flow2, scale_factor=4, mode="bilinear", align_corners=True)

        # Normalize to [-1, 1]
        flow = torch.tanh(flow)
        return flow