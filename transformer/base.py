import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_num: int, embedding_dim: int, padding_index: int = None):
        super().__init__()

        # Learnable embeddings from input data
        self.embedding = nn.Embedding(embedding_num, embedding_dim, padding_index)