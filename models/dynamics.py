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
        
        # Project token features to model dimension
        self.token_projection = nn.Linear(256, dim)  # Project 256-dim features to model dim
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
        # tokens: [batch, seq_len, 256] - token features from VQVAE
        # actions: [batch] action indices
        
        # Project token features to model dimension
        x = self.token_projection(tokens.float())  # [batch, seq_len, dim]
        
        # Add positional embedding, handling variable sequence lengths
        seq_len = x.size(1)
        if seq_len > self.position_embedding.size(1):
            # If sequence is too long, truncate
            x = x[:, :self.position_embedding.size(1)]
        else:
            # If sequence is shorter, use only needed positions
            x = x + self.position_embedding[:, :seq_len]
        
        # Add action embeddings
        action_emb = self.action_embedding(actions)  # [batch, dim]
        x = x + action_emb.unsqueeze(1)  # [batch, seq_len, dim]
        
        # Apply transformer
        features = self.transformer(x)
        
        # Get logits
        logits = self.output(features)  # [batch, seq_len, n_codes]
        
        return logits 