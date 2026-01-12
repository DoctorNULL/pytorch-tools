import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerBase(nn.Module):
    def __init__(self,
                 embedding_num: int,
                 embedding_dim: int,
                 attention_dim: int,
                 num_heads: int,
                 out_dim: int,
                 padding_index: int = None,
                 device= "cuda"):
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

        res1 = y + emb # Residual Connection

        y = self.norm2(res1)
        # (B, SeqLen, Embedding)

        y = self.out(y)
        # (B, SeqLen, Embedding)

        return y + res1 # (B, SeqLen, Embedding)


class TransformerWithMoE(nn.Module):
    def __init__(self,
                 embedding_num: int,
                 embedding_dim: int,
                 attention_dim: int,
                 num_heads: int,
                 out_dim: int,
                 experts_num:int,
                 active_experts_num: int = 1,
                 padding_index: int = None,
                 device= "cuda"):
        super().__init__()

        # Learnable embeddings from input data
        self.embedding = nn.Embedding(embedding_num, embedding_dim, padding_index, device=device)
        self.attention = MultiHeadAttention(embedding_dim, attention_dim, num_heads, device=device)
        self.att_proj = nn.Linear(attention_dim * num_heads, embedding_dim, device=device)
        self.device = device

        self.router = nn.Linear(embedding_dim, experts_num, device=device)

        self.moe = nn.ModuleList([
            nn.Linear(embedding_dim, out_dim, device=device)
            for _ in range(experts_num)
        ])

        self.norm1 = nn.LayerNorm(embedding_dim, device= device)
        self.norm2 = nn.LayerNorm(embedding_dim, device= device)

        self.top_k = active_experts_num

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
        # (B, SeqLen, Embedding)

        routes = self.router(y).softmax(-1)
        # (B, SeqLen, experts_num)

        weights, idx = torch.topk(routes, self.top_k)

        out = torch.zeros(y.size(0), y.size(1), self.moe[0].out_features, device=self.device)

        for i in range(self.top_k): # Loop for each expert for all tokens
            expert_idx = idx[..., i] # Get Expert list for current iteration
            expert_weight = weights[..., i].unsqueeze(-1) # Get expert weight for current iteration

            expert_out = torch.zeros_like(out) # Buffer to store current iteration

            for e in range(len(self.moe)): # loop on all experts
                mask = (expert_idx == e).unsqueeze(-1)  # (B, SeqLen, 1) , Determine if this expert should be activated
                if mask.any(): # Search for experts
                    expert_out += self.moe[e](y) * mask.float() # Activate expert by passing all attentions and deleting unnecessary tokens


            out += expert_weight * expert_out # Adding all experts output from this iteration by their weights

        return out