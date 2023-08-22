import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding1D(nn.Module):
    def __init__(self, embedding_size: int):
        """
        Positional Encoding layer: creates sinusoidal positional encoding for input sequence `x`

        Args:
            embedding_size (int): size of embedding dim
        """
        super().__init__()

        self.embedding_size = embedding_size

        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_size, 2).float() / embedding_size))
        self.register_buffer("inv_freq", inv_freq)

        # cache for storing encoding if already calculated
        self.cache = None

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise RuntimeError("Input should have dims (batch_size, seq_len, embedding_size)")

        if self.cache is not None and self.cache.ndim == x.ndim:
            return self.cache

        _, seq_len, _ = x.shape
        position = torch.arange(seq_len, device=x.device).float()
        pos_emb = torch.einsum("i,j->ij", position, self.inv_freq)
        
        pe = torch.zeros(seq_len, self.embedding_size, device=x.device)
        pe[:, 0::2] = torch.sin(pos_emb)
        pe[:, 1::2] = torch.cos(pos_emb)
        self.cache = pe

        return pe
