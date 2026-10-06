from torch import nn as nn

from weird_ai.feed_forward import FeedForward
from weird_ai.layer_norm import LayerNorm
from weird_ai.attention import CausalAttention, SelfAttention

class TransformerBlock(nn.Module):
    
    # Create a TransformerBlock class, inheriting from nn.Module
    # using 
    #  - LayerNorm
    #  - SelfAttention from previous assignment
    #  - FeedForward
    #  - Residual connections
    def __init__(self, emb_dim, context_length, num_heads, dropout, qkv_bias, drop_rate_shortcut=0.1):
        super().__init__()
        self.layer_norm1 = LayerNorm(emb_dim)
            #self.self_attention = SelfAttention(emb_dim, emb_dim)
        self.self_attention = CausalAttention(emb_dim, emb_dim, context_length, dropout, qkv_bias)
        self.layer_norm2 = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate_shortcut)

    def forward(self, x):
        
        # Implement the forward pass of the TransformerBlock
        # using the components defined in the __init__ method
        # and applying residual connections appropriately.
        # Return the output of the TransformerBlock.
        x_norm1 = self.layer_norm1(x)
            # attention_output, attention_weights = self.self_attention(x_norm1)
        attention_output = self.self_attention(x_norm1)
        x_dropout =  self.drop_shortcut(attention_output)
        x_residual1 = x + x_dropout

        x_norm2 = self.layer_norm2(x_residual1)
        feed_forward_output = self.feed_forward(x_norm2)
        x_dropout2 = self.drop_shortcut(feed_forward_output)
        x_residual2 = x_residual1 + x_dropout2

        return x_residual2
