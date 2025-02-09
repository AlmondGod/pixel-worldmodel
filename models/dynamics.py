import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MaskGITDynamics(nn.Module):
    def __init__(
        self,
        n_codes=64,     # Match VQVAE codebook size
        dim=256,        # Keep transformer dim for good feature learning
        n_layers=4,     # Reduced: simpler dynamics need fewer layers
        n_heads=4,      # Keep 4 heads for multi-scale attention
        max_seq_len=256,  # Keep 256 (16x16 patches)
        n_actions=8     # Keep 8 actions (Pong needs few actions)
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_codes, dim)  # Embed from smaller codebook
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
        
        # Debug shapes
        print(f"\nDynamics model shapes:")
        print(f"  Input tokens: {tokens.shape}")
        print(f"  Input actions: {actions.shape}")
        
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch, seq_len, dim]
        print(f"  After token embedding: {x.shape}")
        
        # Add positional embedding
        x = x + self.position_embedding[:, :x.size(1)]
        print(f"  After position embedding: {x.shape}")
        
        # Add action embeddings
        action_emb = self.action_embedding(actions)  # [batch, dim]
        print(f"  Action embedding: {action_emb.shape}")
        
        # Broadcast action embedding across sequence dimension
        x = x + action_emb.unsqueeze(1)  # [batch, seq_len, dim]
        print(f"  After adding action: {x.shape}")
        
        # Apply transformer
        features = self.transformer(x)
        print(f"  After transformer: {features.shape}")
        
        # Get logits
        logits = self.output(features)
        print(f"  Output logits: {logits.shape}")
        
        return logits 