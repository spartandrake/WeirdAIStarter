import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    """
    A simple self-attention module without trainable query, key, and value projections.

    This version is primarily for learning.
    """

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute attention scores using matrix multiplication.
        # 2. Normalize scores with softmax.
        # 3. Compute context vectors as weighted sums of input vectors.
        query_index = 1
        query = x[query_index]
        attention_scores = torch.empty(x.shape[0])

        for index, token_embedding in enumerate(x):
            attention_scores[index] = torch.dot(query, token_embedding)
        attention_weights = torch.softmax(attention_scores, dim=0)

        context_vector = torch.zeros(query.shape)

        for index, token_embedding in enumerate(x):
            context_vector += attention_weights[index] * token_embedding

        all_attention_scores = x @ x.T
        all_attention_weights = torch.softmax(all_attention_scores, dim=-1)
        all_context_vectors = all_attention_weights @ x

        return all_context_vectors, all_attention_weights


class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute queries, keys, and values.
        # 2. Compute scaled attention scores.
        # 3. Apply softmax.
        # 4. Compute context vectors.
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        d_k = queries.shape[-1]
        attention_scores = queries @ keys.mT / d_k ** 0.5  
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vectors = attention_weights @ values

        return context_vectors, attention_weights

class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(self, embedding_dim, output_dim, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """

        # TODO:
        # 1. Compute keys, queries, and values.
        # 2. Compute scaled attention scores.
        # 3. Mask future tokens.
        # 4. Apply softmax.
        # 5. Apply dropout.
        # 6. Compute context vectors.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = query @ key.transpose(1,2) / (key.shape[-1] ** 0.5)
        num_tokens = x.shape[1]
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = attention_weights @ value

        return context_vectors