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

        # Load all 7 frames
        frames = []
        for i in range(1, 8):
            img_path = os.path.join(full_path, f"im{i}.png")
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            frames.append(img)

        # Convert to numpy array of shape (7, H, W, 3)
        frames = np.stack(frames, axis=0)

        # Generate LR frames (downsampled)
        lr_frames = np.array([cv2.resize(f, 
                                         (f.shape[1] // self.scale, f.shape[0] // self.scale), 
                                         interpolation=cv2.INTER_CUBIC)
                              for f in frames])

        # Apply transforms
        if self.transform:
            frames = torch.stack([self.transform(Image.fromarray((f * 255).astype(np.uint8))) for f in frames])
            lr_frames = torch.stack([self.transform(Image.fromarray((lf * 255).astype(np.uint8))) for lf in lr_frames])
        else:
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
            lr_frames = torch.from_numpy(lr_frames).permute(0, 3, 1, 2)

        # Shape checks
        assert lr_frames.shape[1:] == (3, frames.shape[2] // self.scale, frames.shape[3] // self.scale), "LR shape mismatch"
        assert frames.shape[1:] == (3, frames.shape[2], frames.shape[3]), "HR shape mismatch"

        return lr_frames, frames  # (LR_seq, HR_seq)

