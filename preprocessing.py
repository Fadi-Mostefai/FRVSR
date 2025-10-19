"""
Preprocess Vimeo90K dataset: downsample HR frames to LR and save them.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def preprocess_sequences(root_dir, scale=4, list_file="sep_trainlist.txt"):
    """
    Downsample all sequences and save LR frames alongside HR frames.
    
    Structure:
        data/sequences/00001/0001/
            im1.png, im2.png, ... im7.png  (HR - already exist)
            im1_LRx4.png, im2_LRx4.png, ... im7_LRx4.png  (LR - will be created)
    """
    list_path = os.path.join(root_dir, list_file)
    
    with open(list_path, "r") as f:
        sequences = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Processing {len(sequences)} sequences with scale={scale}...")
    print(f"Reading from: {list_path}")
    
    skipped = 0
    processed = 0
    
    for seq_path in tqdm(sequences, desc="Preprocessing"):
        full_path = os.path.join(root_dir, "sequences", seq_path)
        
        for i in range(1, 8):
            hr_path = os.path.join(full_path, f"im{i}.png")
            lr_path = os.path.join(full_path, f"im{i}_LRx{scale}.png")
            
            # Skip if LR already exists
            if os.path.exists(lr_path):
                skipped += 1
                continue
            
            # Load HR image
            if not os.path.exists(hr_path):
                print(f"\nWarning: Missing {hr_path}")
                continue
            
            hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            
            if hr_img is None:
                print(f"\nWarning: Failed to read {hr_path}")
                continue
            
            # Downsample to LR
            H, W = hr_img.shape[:2]
            lr_img = cv2.resize(hr_img, (W // scale, H // scale), 
                               interpolation=cv2.INTER_CUBIC)
            
            # Save LR image
            cv2.imwrite(lr_path, lr_img)
            processed += 1
    
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped (already exist): {skipped} images")
    print(f"{'='*60}")

def preprocess_all(root_dir="data", scale=4):
    """Preprocess both train and test sets"""
    print("Preprocessing training set...")
    preprocess_sequences(root_dir, scale, "sep_trainlist.txt")
    
    print("\nPreprocessing test set...")
    preprocess_sequences(root_dir, scale, "sep_testlist.txt")

if __name__ == "__main__":
    preprocess_all(root_dir="data", scale=4)