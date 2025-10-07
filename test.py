import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0))
