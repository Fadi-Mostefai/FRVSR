import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class FRVSRDataset(Dataset):
    def __init__(self, root_dir, list_path, scale=4, transform=None):
        """
        Args:
            root_dir (str): path to the vimeo_septuplet directory (contains 'sequences')
            list_path (str): path to train/test list file (sep_trainlist.txt or sep_testlist.txt)
            scale (int): downsampling factor (e.g., 4 for 4× SR)
            transform: optional transform (e.g., ToTensor, normalization)
        """
        self.root_dir = root_dir
        self.scale = scale
        self.transform = transform

        # Load list of sequences (each line = '00001/0001', etc.)
        with open(list_path, "r") as f:
            self.sequences = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        full_path = os.path.join(self.root_dir, "sequences", seq_path)

        # Pre-allocate arrays for better performance
        frames = np.empty((7, 256, 448, 3), dtype=np.float32)  # Adjust size if needed
        
        for i in range(1, 8):
            img_path = os.path.join(full_path, f"im{i}.png")
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames[i-1] = img.astype(np.float32) / 255.0

        # More efficient downsampling
        H, W = frames.shape[1:3]
        lr_frames = np.empty((7, H // self.scale, W // self.scale, 3), dtype=np.float32)
        
        for i in range(7):
            lr_frames[i] = cv2.resize(frames[i], 
                                     (W // self.scale, H // self.scale), 
                                     interpolation=cv2.INTER_CUBIC)

        # Direct torch conversion (faster than Image transforms)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        lr_frames = torch.from_numpy(lr_frames).permute(0, 3, 1, 2).contiguous()

        return lr_frames, frames  # (LR_seq, HR_seq)

