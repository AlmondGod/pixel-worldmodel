import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MaskGITDynamics(nn.Module):
    def __init__(
        self,
        n_codes=32,     # Match VQVAE codebook size
        dim=256,        # Keep transformer dim for good feature learning
        n_layers=4,     # Reduced: simpler dynamics need fewer layers
        n_heads=4,      # Keep 4 heads for multi-scale attention
        max_seq_len=16,  # Changed to match sequence length
        n_actions=4     # Reduced to 4 actions (up/down for each paddle)
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(256, dim)  # Changed back to embedding
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
        
    def forward(self, tokens, actions):
        # tokens: [batch, seq_len] - token indices
        # actions: [batch] action indices
        
        # Embed tokens
        x = self.token_embedding(tokens.long())  # [batch, seq_len, dim]
        
        # Add positional embedding
        x = x + self.position_embedding[:, :x.size(1)]
        
        # Add action embeddings
        action_emb = self.action_embedding(actions)  # [batch, dim]
        x = x + action_emb.unsqueeze(1)  # [batch, seq_len, dim]
        
        # Apply transformer
        features = self.transformer(x)
        
        # Get logits
        logits = self.output(features)  # [batch, seq_len, n_codes]
        
        return logits 