import torch
import torch.nn as nn
import torch.nn.functional as F


class Correlation(nn.Module):
    """
    Correlation layer: computes cost volume between two feature maps.
    For each offset (dx, dy) in [-max_disp, max_disp], computes similarity between fmap1 and shifted fmap2.
    Output shape: (B, (2*max_disp+1)^2, H, W)
    This is the core of FlowNetC, allowing explicit matching over a local window.
    """

    def __init__(self, max_disp: int = 4):
        super().__init__()
        self.max_disp = int(max_disp)

    def forward(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = fmap1.shape
        pad = self.max_disp
        # Pad fmap2 so shifting doesn't shrink output
        fmap2_pad = F.pad(fmap2, (pad, pad, pad, pad))
        corrs = []
        # For each offset, compute similarity
        for dy in range(-pad, pad + 1):
            y0 = dy + pad
            y1 = y0 + H
            for dx in range(-pad, pad + 1):
                x0 = dx + pad
                x1 = x0 + W
                shifted = fmap2_pad[:, :, y0:y1, x0:x1]
                # Elementwise multiply and sum over channels
                corr = (fmap1 * shifted).sum(dim=1, keepdim=True)
                corrs.append(corr)
        # Stack all offsets into cost volume
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

        # Siamese feature extractors for each input frame (prev_lr and curr_lr)
        # Each branch: 3x3 input -> 64 channels (H/2) -> 128 (H/4) -> 256 (H/8)
        self.conv1a = nn.Conv2d(3, 64, 7, stride=2, padding=3)  # prev_lr branch
        self.conv2a = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3a = nn.Conv2d(128, 256, 5, stride=2, padding=2)

        self.conv1b = nn.Conv2d(3, 64, 7, stride=2, padding=3)  # curr_lr branch
        self.conv2b = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3b = nn.Conv2d(128, 256, 5, stride=2, padding=2)

        # Correlation layer: computes cost volume between c3a and c3b (H/8)
        self.corr = Correlation(max_disp=max_corr_disp)
        corr_channels = (2 * max_corr_disp + 1) ** 2  # e.g., 81 for max_disp=4

        # Redirection conv: compress c3a features to 64 channels for fusion
        self.conv_redir = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)

        # Post-correlation encoder: processes fused cost volume + redirected features
        self.conv3_1 = nn.Conv2d(corr_channels + 64, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # H/16
        self.conv4_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)  # H/32
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)  # H/64
        self.conv6_1 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)

        # Decoder: upsample features and refine flow at each scale
        self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)  # H/32
        self.deconv4 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)   # H/16
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)   # H/8
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)    # H/4

        # Flow predictors: regress flow at each scale
        self.predict_flow6 = nn.Conv2d(1024, 2, 3, padding=1)  # H/64
        self.predict_flow5 = nn.Conv2d(512, 2, 3, padding=1)   # H/32
        self.predict_flow4 = nn.Conv2d(256, 2, 3, padding=1)   # H/16
        self.predict_flow3 = nn.Conv2d(128, 2, 3, padding=1)   # H/8
        self.predict_flow2 = nn.Conv2d(64, 2, 3, padding=1)    # H/4

        # Flow upsamplers: learnable upsampling of flow to next finer scale
        self.upsample_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
        self.upsample_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
        self.upsample_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)
        self.upsample_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1)

        # Iconvs: fuse upsampled flow, skip features, and deconved features at each scale
        self.iconv5 = nn.Conv2d(512 + 512 + 2, 512, 3, padding=1)  # H/32
        self.iconv4 = nn.Conv2d(512 + 256 + 2, 256, 3, padding=1)  # H/16
        self.iconv3 = nn.Conv2d(256 + 128 + 2, 128, 3, padding=1)  # H/8
        self.iconv2 = nn.Conv2d(128 + 64 + 2, 64, 3, padding=1)    # H/4

        self.act = nn.LeakyReLU(leaky, inplace=True)

    def encode_siamese(self, x: torch.Tensor, branch: str):
        """
        Pass input through the corresponding encoder branch.
        Returns c1 (H/2), c2 (H/4), c3 (H/8) feature maps.
        """
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
        """
        Estimate optical flow between prev_lr and curr_lr (both LR frames).
        Returns flow in normalized grid units ([-1,1]) at LR resolution.
        """
        # 1. Siamese encoders: extract features from both frames
        _, c2a, c3a = self.encode_siamese(prev_lr, 'a')  # prev_lr features (H/8)
        _, _, c3b = self.encode_siamese(curr_lr, 'b')  # curr_lr features (H/8)

        # 2. Correlation volume: explicit matching over local window
        corr = self.corr(c3a, c3b)  # (B, corr_channels, H/8, W/8)
        corr = self.act(corr)

        # 3. Redirect and fuse: combine cost volume and compressed prev features
        redir = self.act(self.conv_redir(c3a))  # (B, 64, H/8, W/8)
        x = torch.cat([corr, redir], dim=1)     # (B, corr_channels+64, H/8, W/8)

        # 4. Deeper encoder: process fused features through deeper layers
        c3_1 = self.act(self.conv3_1(x))        # (B, 256, H/8, W/8)
        c4 = self.act(self.conv4(c3_1))         # (B, 512, H/16, W/16)
        c4_1 = self.act(self.conv4_1(c4))
        c5 = self.act(self.conv5(c4_1))         # (B, 512, H/32, W/32)
        c5_1 = self.act(self.conv5_1(c5))
        c6 = self.act(self.conv6(c5_1))         # (B, 1024, H/64, W/64)
        c6_1 = self.act(self.conv6_1(c6))

        # 5. Decoder: upsample and refine flow at each scale (coarse-to-fine)
        flow6 = self.predict_flow6(c6_1)        # (B,2,H/64,W/64)
        up_flow6 = self.upsample_flow6_to_5(flow6)  # (B,2,H/32,W/32)
        deconv5 = self.deconv5(c6_1)                # (B,512,H/32,W/32)

        # Resize to match c5_1 spatial dimensions
        if deconv5.shape[2:] != c5_1.shape[2:]:
            deconv5 = F.interpolate(deconv5, size=c5_1.shape[2:], mode='bilinear', align_corners=True)
        if up_flow6.shape[2:] != c5_1.shape[2:]:
            up_flow6 = F.interpolate(up_flow6, size=c5_1.shape[2:], mode='bilinear', align_corners=True)

        concat5 = torch.cat([c5_1, deconv5, up_flow6], dim=1)  # (B,1026,H/32,W/32)
        iconv5 = self.act(self.iconv5(concat5))
        flow5 = self.predict_flow5(iconv5)         # (B,2,H/32,W/32)
        up_flow5 = self.upsample_flow5_to_4(flow5) # (B,2,H/16,W/16)
        deconv4 = self.deconv4(iconv5)             # (B,256,H/16,W/16)

        # Resize to match c4_1 spatial dimensions
        if deconv4.shape[2:] != c4_1.shape[2:]:
            deconv4 = F.interpolate(deconv4, size=c4_1.shape[2:], mode='bilinear', align_corners=True)
        if up_flow5.shape[2:] != c4_1.shape[2:]:
            up_flow5 = F.interpolate(up_flow5, size=c4_1.shape[2:], mode='bilinear', align_corners=True)

        concat4 = torch.cat([c4_1, deconv4, up_flow5], dim=1)  # (B,770,H/16,W/16)
        iconv4 = self.act(self.iconv4(concat4))
        flow4 = self.predict_flow4(iconv4)         # (B,2,H/16,W/16)
        up_flow4 = self.upsample_flow4_to_3(flow4) # (B,2,H/8,W/8)
        deconv3 = self.deconv3(iconv4)             # (B,128,H/8,W/8)

        # Resize to match c3_1 spatial dimensions
        if deconv3.shape[2:] != c3_1.shape[2:]:
            deconv3 = F.interpolate(deconv3, size=c3_1.shape[2:], mode='bilinear', align_corners=True)
        if up_flow4.shape[2:] != c3_1.shape[2:]:
            up_flow4 = F.interpolate(up_flow4, size=c3_1.shape[2:], mode='bilinear', align_corners=True)

        concat3 = torch.cat([c3_1, deconv3, up_flow4], dim=1)  # (B,386,H/8,W/8)
        iconv3 = self.act(self.iconv3(concat3))
        flow3 = self.predict_flow3(iconv3)         # (B,2,H/8,W/8)
        up_flow3 = self.upsample_flow3_to_2(flow3) # (B,2,H/4,W/4)
        deconv2 = self.deconv2(iconv3)             # (B,64,H/4,W/4)

        # Resize to match c2a spatial dimensions
        if deconv2.shape[2:] != c2a.shape[2:]:
            deconv2 = F.interpolate(deconv2, size=c2a.shape[2:], mode='bilinear', align_corners=True)
        if up_flow3.shape[2:] != c2a.shape[2:]:
            up_flow3 = F.interpolate(up_flow3, size=c2a.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate encoder skip (c2a), deconv2, and upsampled flow
        concat2 = torch.cat([c2a, deconv2, up_flow3], dim=1)   # (B,194,H/4,W/4)
        iconv2 = self.act(self.iconv2(concat2))
        flow2 = self.predict_flow2(iconv2)         # (B,2,H/4,W/4)

        # 6. Final upsampling: bring flow to LR resolution (H,W)
        flow = F.interpolate(flow2, scale_factor=4, mode="bilinear", align_corners=True)  # (B,2,H,W)

        # 7. Normalize to [-1, 1] for grid_sample warping
        flow = torch.tanh(flow)
        return flow