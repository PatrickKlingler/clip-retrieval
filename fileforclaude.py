import torch
import torch.nn as nn

if __name__  == "__main__":
    grid_size = (7, 7)
    width = 768
    scale = 0.036

    print("Creating tensor...")
    t = scale * torch.randn(grid_size[0] * grid_size[1] + 1, width)
    print(f"Tensor shape: {t.shape}")

    print("Creating parameter...")
    p = nn.Parameter(t)
    print(f"Parameter shape: {p.shape}")
    print("Success!")