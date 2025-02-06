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
        max_seq_len=256,  # 16x16 tokens
        n_actions=8  # Number of possible actions
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_codes, dim)
        self.action_embedding = nn.Embedding(n_actions, dim)
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
        
    def forward(self, tokens, actions, mask=None):
        # tokens: [batch, seq_len]
        # actions: [batch] action indices
        # mask: [batch, seq_len] boolean mask of tokens to predict
        
        x = self.token_embedding(tokens)
        x = x + self.position_embedding[:, :x.size(1)]
        
        # Add action embeddings
        action_emb = self.action_embedding(actions)  # [batch, dim]
        x = x + action_emb.unsqueeze(1)  # Broadcast action embedding across sequence
        
        # Apply transformer
        features = self.transformer(x)
        
        # Predict only masked tokens
        logits = self.output(features)
        
        if mask is not None:
            logits = logits[mask]
            
        return logits 