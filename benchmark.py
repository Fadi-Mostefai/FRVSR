"""
Quick benchmark script to test training speed improvements
Run this to see actual iterations/second before full training
"""
import os
import time
import torch
from torch.utils.data import DataLoader
from dataloader import FRVSRDataset
from FRVSR import FRVSR
from main import compute_losses
from torch.amp import autocast, GradScaler

def benchmark(num_batches=20, batch_size=24):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {batch_size}")
    print(f"Testing {num_batches} batches...\n")
    
    # Setup
    list_path = os.path.join("data", "sep_trainlist.txt")
    train_set = FRVSRDataset(root_dir="data", list_path=list_path, scale=4)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True, prefetch_factor=4,
                             persistent_workers=True)
    
    model = FRVSR(scale=4, fnet_variant="C").to(device)
    
    # torch.compile is disabled due to Triton incompatibility on Windows
    # if hasattr(torch, 'compile'):
    #     print("Compiling model with torch.compile...")
    #     model = torch.compile(model, mode='max-autotune')
    #     print("Compilation done!\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda')
    
    model.train()
    
    # Warmup (for torch.compile and cuDNN)
    print("Warming up (5 iterations)...")
    for i, (lr_seq, hr_seq) in enumerate(train_loader):
        if i >= 5:
            break
        
        lr_seq = lr_seq.to(device, non_blocking=True)
        hr_seq = hr_seq.to(device, non_blocking=True)
        
        B, T, C, H, W = lr_seq.size()
        prev_est_hr = torch.zeros(B, C, H * 4, W * 4, device=device)
        
        total_loss = 0
        for t in range(1, T):
            with autocast('cuda'):
                est_hr, flow_lr = model(lr_seq[:, t-1], lr_seq[:, t], prev_est_hr)
                loss, _, _ = compute_losses(est_hr, hr_seq[:, t], lr_seq[:, t-1], 
                                           lr_seq[:, t], flow_lr)
            total_loss += loss / (T - 1)
            prev_est_hr = est_hr.detach()
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print("Warmup complete!\n")
    
    # Actual benchmark
    print(f"Running benchmark...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i, (lr_seq, hr_seq) in enumerate(train_loader):
        if i >= num_batches:
            break
        
        lr_seq = lr_seq.to(device, non_blocking=True)
        hr_seq = hr_seq.to(device, non_blocking=True)
        
        B, T, C, H, W = lr_seq.size()
        prev_est_hr = torch.zeros(B, C, H * 4, W * 4, device=device)
        
        total_loss = 0
        for t in range(1, T):
            with autocast('cuda'):
                est_hr, flow_lr = model(lr_seq[:, t-1], lr_seq[:, t], prev_est_hr)
                loss, _, _ = compute_losses(est_hr, hr_seq[:, t], lr_seq[:, t-1], 
                                           lr_seq[:, t], flow_lr)
            total_loss += loss / (T - 1)
            prev_est_hr = est_hr.detach()
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    # Results
    iterations_per_sec = num_batches / elapsed
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Time for {num_batches} batches: {elapsed:.2f} seconds")
    print(f"Iterations per second: {iterations_per_sec:.2f} it/s")
    print(f"Time per iteration: {elapsed/num_batches:.3f} seconds")
    print(f"\nGPU Memory Used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # Estimate training time
    total_iterations = len(train_loader) * 50  # 50 epochs
    estimated_hours = (total_iterations / iterations_per_sec) / 3600
    
    print(f"\n{'='*60}")
    print(f"TRAINING TIME ESTIMATE")
    print(f"{'='*60}")
    print(f"Total iterations: {total_iterations:,}")
    print(f"Estimated training time: {estimated_hours:.2f} hours ({estimated_hours/24:.2f} days)")
    print(f"{'='*60}")
    
    # Recommendations
    if iterations_per_sec < 10:
        print("\n⚠️  Speed is still slow. Recommendations:")
        print("   1. Check GPU utilization with: nvidia-smi -l 1")
        print("   2. Try increasing batch_size if GPU memory allows")
        print("   3. Ensure data is on SSD, not HDD")
    elif iterations_per_sec < 30:
        print("\n✓ Decent speed. Consider:")
        print("   1. Increasing batch_size if GPU memory allows")
        print("   2. Profile with torch.profiler to find remaining bottlenecks")
    else:
        print("\n✓✓✓ Excellent speed! GPU is well utilized.")

if __name__ == "__main__":
    benchmark()
