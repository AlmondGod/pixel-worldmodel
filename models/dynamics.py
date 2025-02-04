import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MaskGITDynamics(nn.Module):
    def __init__(
        self,
        n_codes=256,
        dim=256,
        n_layers=6,
        n_heads=4,
        max_seq_len=256  # 16x16 tokens
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_codes, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim*4,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        self.output = nn.Linear(dim, n_codes)
        
    def forward(self, tokens, mask=None):
        # tokens: [batch, seq_len]
        # mask: [batch, seq_len] boolean mask of tokens to predict
        
        x = self.token_embedding(tokens)
        x = x + self.position_embedding[:, :x.size(1)]
        
        # Apply transformer
        features = self.transformer(x)
        
        # Predict only masked tokens
        logits = self.output(features)
        
        if mask is not None:
            logits = logits[mask]
            
        return logits 