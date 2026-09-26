import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):

        # Compute mean
        mean = x.mean(dim=-1, keepdim=True)
        # Compute variance
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        # Apply scale and shift
        x_scaled_shifted = self.scale * x_normalized + self.shift

        return x_scaled_shifted