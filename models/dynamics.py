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
        max_seq_len=256,  # Changed to match number of patches (16x16)
        n_actions=4     # Reduced to 4 actions (up/down for each paddle)
    ):
        super().__init__()
        
        # Project token indices to model dimension
        self.token_embedding = nn.Embedding(n_codes, dim)
        self.action_embedding = nn.Embedding(n_actions, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim*4,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Output projection to logits over codebook
        self.output = nn.Linear(dim, n_codes)
        
    def forward(self, tokens, actions):
        # tokens: [batch, 256] - token indices for each patch
        # actions: [batch] action indices
        
        # Convert tokens to long integers
        tokens = tokens.long()
        actions = actions.long()
        
        # Embed tokens and add positional encoding
        x = self.token_embedding(tokens)  # [batch, 256, dim]
        x = x + self.position_embedding[:, :x.size(1)]
        
        # Add action embeddings to all positions
        action_emb = self.action_embedding(actions)  # [batch, dim]
        x = x + action_emb.unsqueeze(1)  # [batch, 256, dim]
        
        # Apply transformer
        features = self.transformer(x)
        
        # Get logits for next token prediction
        logits = self.output(features)  # [batch, 256, n_codes]
        
        return logits 