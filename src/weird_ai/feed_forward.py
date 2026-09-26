import torch
import torch.nn as nn

class GELU(nn.Module):

    def forward(self, x):

        return x * 0.5 * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        
    
    
class FeedForward(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)