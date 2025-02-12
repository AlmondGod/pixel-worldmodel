import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MaskGITDynamics(nn.Module):
    def __init__(
        self,
        n_codes=16,     # Number of codebook vectors (reduced from 32 to match Genie)
        dim=256,        # Transformer dimension
        n_layers=4,     # Number of transformer layers
        n_heads=4,      # Number of attention heads
        n_patches=256,  # Number of patches (16x16)
        n_actions=4,    # Number of possible actions
        temperature=1.0,  # Temperature for sampling
        mask_ratio=0.75  # Default mask ratio during training (75%)
    ):
        super().__init__()
        
        # Project token indices to embedding dimension
        self.token_embedding = nn.Embedding(n_codes, dim)
        self.action_embedding = nn.Embedding(n_actions, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, n_patches, dim))
        
        # Special mask token embedding
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Temperature for sampling
        self.temperature = temperature
        self.mask_ratio = mask_ratio
        
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
        
    def forward(self, tokens, actions, mask_ratio=None):
        """
        Args:
            tokens: [batch, n_patches] token indices for a single frame
            actions: [batch] action indices
            mask_ratio: Optional override for mask ratio during training
        Returns:
            logits: [batch, n_patches, n_codes] prediction logits for next frame tokens
        """
        # Convert inputs to long integers
        tokens = tokens.long()
        actions = actions.long()
        
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch, n_patches, dim]
        
        # Add positional embeddings
        x = x + self.position_embedding
        
        # Apply masking during training
        if self.training:
            mask_ratio = mask_ratio or self.mask_ratio
            mask = torch.rand_like(tokens.float()) < mask_ratio
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)
        
        # Add action embeddings to all positions
        action_emb = self.action_embedding(actions)  # [batch, dim]
        x = x + action_emb.unsqueeze(1)  # [batch, n_patches, dim]
        
        # Apply transformer
        features = self.transformer(x)  # [batch, n_patches, dim]
        
        # Get logits for next token prediction
        logits = self.output(features)  # [batch, n_patches, n_codes]
        
        # Apply temperature scaling during inference
        if not self.training:
            logits = logits / self.temperature
        
        return logits
        
    def sample(self, logits):
        """Sample tokens from logits using temperature scaling"""
        probs = F.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.size(0), -1) 