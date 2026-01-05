import torch
import torch.nn as nn
from .embedding import RotaryEmbedding

class BasicAttention(nn.Module):
    def __init__(self, embedding_dim: int, query_dim: int, value_dim: int, device="cpu"):
        super().__init__()

        self.query = nn.Linear(embedding_dim, query_dim, device=device)
        self.key   = nn.Linear(embedding_dim, query_dim, device=device)
        self.value = nn.Linear(embedding_dim, value_dim, device=device)

        self.device = device

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        q = self.query(x)   # (B, SeqLen, Q)
        k = self.key(x)     # (B, SeqLen, Q)
        v = self.value(x)   # (B, SeqLen, V)

        weights = torch.matmul(q, k.transpose(-2, -1))  # (B, SeqLen, SeqLen)
        weights = weights / self.query.out_features

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)

            weights = weights.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(weights, dim=-1)  # (B, SeqLen, SeqLen)

        output = torch.matmul(weights, v)  # (B, SeqLen, V)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int, use_rope = True, device = "cpu"):
        super().__init__()

        self.query = nn.Linear(embedding_dim, attention_dim * num_heads, device=device)
        self.key = nn.Linear(embedding_dim, attention_dim * num_heads, device=device)
        self.value = nn.Linear(embedding_dim, attention_dim * num_heads, device=device)
        self.device = device

        self.pe = None

        if use_rope:
            self.pe = RotaryEmbedding(attention_dim, device)

        self.num_heads = num_heads
        self.attention_dim = attention_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (B, SeqLength, Embedding)

        B, SeqLen, _ = x.shape

        q = self.query(x) # (B, SeqLen, attention * heads)
        k = self.key(x) # (B, SeqLen, , attention * heads)
        v = self.value(x) # (B, SeqLen, , attention * heads)

        q = q.view(B, SeqLen, self.num_heads, self.attention_dim).transpose(1, 2) # (B, H, SeqLen, Attention)
        k = k.view(B, SeqLen, self.num_heads, self.attention_dim).transpose(1, 2) # (B, H, SeqLen, Attention)
        v = v.view(B, SeqLen, self.num_heads, self.attention_dim).transpose(1, 2) # (B, H, SeqLen, Attention)

        if self.pe is not None:
            emb = self.pe(torch.tensor([[SeqLen]], device=self.device))
            q += emb
            k += emb

        weights = torch.matmul(q, k.transpose(-2, -1)) # (B, H, SeqLen, SeqLen)
        weights /= self.attention_dim ** 0.5

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)

            weights = weights.masked_fill(mask == 0, float("-inf"))

        weights = weights.softmax(-1)
        # (B, H, SeqLen, SeqLen)

        y = torch.matmul(weights, v) # (B, H, SeqLen, Attention)

        y = y.transpose(1, 2).contiguous().flatten(2) #(B, SeqLen, Attention * H)

        return y # (B, SeqLen, Attention * H)
