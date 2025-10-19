import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log10
from tqdm import tqdm
from dataloader import FRVSRDatasetFast  # Use fast dataloader
from FRVSR import FRVSR, warp
from torch.amp import autocast, GradScaler

# PSNR Calculation
def calc_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse.item())

def compute_losses(est_hr, gt_hr, prev_lr, curr_lr, flow_lr):
    # L_sr: reconstruction loss
    l_sr = F.mse_loss(est_hr, gt_hr)

    # L_flow: auxiliary flow consistency loss
    warped_lr = warp(prev_lr, flow_lr)
    l_flow = F.mse_loss(warped_lr, curr_lr)

    return l_sr + l_flow, l_sr.item(), l_flow.item()

# Training Script
def train_frvsr(
    save_dir="checkpoints",
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    scale=4,
    fnet_variant="C",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)

    list_path = os.path.join("data", "sep_trainlist.txt")
    
    # Use fast dataloader with pre-processed LR images
    train_set = FRVSRDatasetFast(root_dir="data", list_path=list_path, scale=scale)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True, prefetch_factor=2, 
                             persistent_workers=True)

    model = FRVSR(scale=scale, fnet_variant=fnet_variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler('cuda')

    best_psnr = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0

        for batch_idx, (lr_seq, hr_seq) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            lr_seq = lr_seq.to(device, non_blocking=True)
            hr_seq = hr_seq.to(device, non_blocking=True)

            B, T, C, H, W = lr_seq.size()
            prev_est_hr = torch.zeros(B, C, H * scale, W * scale, device=device)

            # Accumulate loss across all temporal frames before backward
            total_loss = 0
            psnr_sum = 0

            for t in range(1, T):
                prev_lr, curr_lr = lr_seq[:, t - 1], lr_seq[:, t]
                gt_hr = hr_seq[:, t]

                # Use autocast for mixed precision
                with autocast('cuda'):
                    est_hr, flow_lr = model(prev_lr, curr_lr, prev_est_hr)
                    loss, _, _ = compute_losses(est_hr, gt_hr, prev_lr, curr_lr, flow_lr)
                
                # Accumulate loss (normalize by number of frames)
                total_loss += loss / (T - 1)
                
                with torch.no_grad():
                    psnr_sum += calc_psnr(est_hr, gt_hr)
                    prev_est_hr = est_hr.detach()

            # Single backward pass for entire sequence
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_psnr += psnr_sum / (T - 1)

        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.5f} | PSNR: {avg_psnr:.2f} dB")

        # Save checkpoint every 5 epochs instead of every epoch
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"frvsr_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

        # Optionally save a sample output
        if epoch == num_epochs - 1:
            save_image(est_hr[0].clamp(0, 1), os.path.join(save_dir, "sample_output.png"))

        # Keep track of best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(save_dir, "frvsr_best.pth"))

    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")

# Run training
if __name__ == "__main__":
    # Enable cudnn benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Check CUDA version and GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    train_frvsr(batch_size=16)  # Increase batch size
