import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, embedding_dim: int, add_padding = None):
        super().__init__()

        self.embeddings = nn.Embedding(max_length + 1, embedding_dim, 0)
        self.add_padding = add_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1)

        y = x
        if self.add_padding:
            seq_len = x.max().item()

            row = torch.arange(1, seq_len + 1)
            y = row.unsqueeze(0) < x + 1
            y = row * y
            #(B, MaxSeq)

        return self.embeddings(y) # (B, MaxSeq, Embedding)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1)

        seq_len = x.max().item()

        row = torch.arange(1, seq_len + 1)
        y = row.unsqueeze(0) < x + 1
        y = (row * y).unsqueeze(2).repeat(1, 1, self.embedding_dim)

        angles = 1 / (10000 ** (2 * (y // 2) / self.embedding_dim))

        y = y * angles
        y [:, :, 0::2] = y [:, :, 0::2].sin()
        y [:, :, 1::2] = y [:, :, 1::2].cos()

        return y # (B, MaxSeq, Embedding)


class RotaryEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.inv_freq = 1 / (10000 ** (torch.arange(0, embedding_dim).float() / embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(B, 1)
        seq_len = x.max().item()

        row = torch.arange(1, seq_len + 1)
        y = row.unsqueeze(0) < x + 1
        y = (row * y).unsqueeze(2).repeat(1, 1, self.embedding_dim)

        y = y * self.inv_freq

        y[:, :, 0] = y[:, :, 0].sin()
        y[:, :, 1] = y[:, :, 1].cos()

        return y # (B, MaxSeq, Embedding)