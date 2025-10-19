import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class FRVSRDatasetFast(Dataset):
    """
    Fast FRVSR dataset that loads pre-processed LR images instead of downsampling on-the-fly.
    
    Requires running preprocess_data.py first to generate LR images.
    """
    def __init__(self, root_dir, list_path, scale=4):
        """
        Args:
            root_dir (str): path to the vimeo_septuplet directory (contains 'sequences')
            list_path (str): path to train/test list file (sep_trainlist.txt or sep_testlist.txt)
            scale (int): downsampling factor (e.g., 4 for 4× SR)
        """
        self.root_dir = root_dir
        self.scale = scale

        # Load list of sequences
        with open(list_path, "r") as f:
            self.sequences = [line.strip() for line in f.readlines() if line.strip()]
        
        # Verify that preprocessed data exists for first sequence
        if len(self.sequences) > 0:
            test_path = os.path.join(self.root_dir, "sequences", self.sequences[0], 
                                    f"im1_LRx{scale}.png")
            if not os.path.exists(test_path):
                raise FileNotFoundError(
                    f"Preprocessed LR images not found at {test_path}\n"
                    f"Please run preprocess_data.py first!"
                )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        full_path = os.path.join(self.root_dir, "sequences", seq_path)

        # Pre-allocate tensors
        hr_frames = []
        lr_frames = []
        
        for i in range(1, 8):
            # Load HR image
            hr_path = os.path.join(full_path, f"im{i}.png")
            hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = hr_img.astype(np.float32) / 255.0
            
            # Load pre-processed LR image
            lr_path = os.path.join(full_path, f"im{i}_LRx{self.scale}.png")
            lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            lr_img = lr_img.astype(np.float32) / 255.0
            
            hr_frames.append(hr_img)
            lr_frames.append(lr_img)
        
        # Stack and convert to torch tensors
        hr_frames = np.stack(hr_frames, axis=0)  # (7, H, W, 3)
        lr_frames = np.stack(lr_frames, axis=0)  # (7, H/scale, W/scale, 3)
        
        # Convert to torch and permute to (T, C, H, W)
        hr_frames = torch.from_numpy(hr_frames).permute(0, 3, 1, 2).contiguous()
        lr_frames = torch.from_numpy(lr_frames).permute(0, 3, 1, 2).contiguous()

        return lr_frames, hr_frames

