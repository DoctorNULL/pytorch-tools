import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_num: int, embedding_dim: int, attention_dim: int, num_heads: int, out_dim: int, padding_index: int = None, device= "cuda"):
        super().__init__()

        # Learnable embeddings from input data
        self.embedding = nn.Embedding(embedding_num, embedding_dim, padding_index, device=device)
        self.attention = MultiHeadAttention(embedding_dim, attention_dim, num_heads, device=device)
        self.att_proj = nn.Linear(attention_dim * num_heads, embedding_dim, device=device)
        self.device = device

        self.out = nn.Linear(embedding_dim, out_dim, device=device)

        self.norm1 = nn.LayerNorm(embedding_dim, device= device)
        self.norm2 = nn.LayerNorm(embedding_dim, device= device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (B, SeqLen)

        emb = self.embedding(x) # Capture embeddings of input sequence
        # (B, SeqLen, Embedding)

        y = self.norm1(emb) # Pre norm

        y = self.attention(y, mask) # Capture attentions
        # (B, SeqLen, Attention * Head)

        y = self.att_proj(y)
        # (B, SeqLen, Embedding)

        y = y + emb # Residual Connection

        y = self.norm2(y)

        y = self.out(y)

        return y