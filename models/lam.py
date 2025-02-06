import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.vq_vae import STTransformerEncoder

class ActionVectorQuantizer(nn.Module):
    def __init__(self, n_codes=8, code_dim=256):  # |A| = 8 from paper
        super().__init__()
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.embedding.weight.data.uniform_(-1./n_codes, 1./n_codes)
        
    def forward(self, z):
        # Calculate distances
        d = torch.sum(z ** 2, dim=-1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=-1) - \
            2 * torch.matmul(z, self.embedding.weight.t())
            
        # Get nearest neighbor
        min_encoding_indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(min_encoding_indices)
        
        # Straight through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, min_encoding_indices

class LAMDecoder(nn.Module):
    def __init__(self, dim=256, n_heads=4, n_layers=4):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim*4,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.output = nn.Linear(dim, 64*64)  # Reconstruct frame
        
    def forward(self, x, actions):
        # Reshape if needed (b, f, n, d) -> (b*f, n, d)
        if len(x.shape) == 4:
            b, f, n, d = x.shape
            x = x.reshape(b*f, n, d)
            actions = actions.reshape(b*f, d)
        
        # Add action embeddings
        x = x + actions.unsqueeze(1)
        x = self.transformer(x)
        return self.output(x.mean(dim=1))  # Average over sequence dimension

class LAM(nn.Module):
    def __init__(self, dim=256, n_heads=4, n_layers=4):
        super().__init__()
        self.encoder = STTransformerEncoder(dim, n_heads, n_layers)
        self.action_proj = nn.Linear(dim, dim)
        self.quantizer = ActionVectorQuantizer(n_codes=8, code_dim=dim)
        self.decoder = LAMDecoder(dim, n_heads, n_layers)
        
    def forward(self, frames, next_frames):
        """
        Args:
            frames: [batch, time, height, width] Previous frames
            next_frames: [batch, height, width] Next frame to predict
        Returns:
            reconstructed: Reconstructed next frame
            actions: Quantized actions
            indices: Action indices
        """
        # Encode all frames
        b, t, h, w = frames.shape
        all_frames = torch.cat([frames, next_frames.unsqueeze(1)], dim=1)
        features = self.encoder(all_frames)  # [b*f, n, d]
        
        # Reshape features back to include frame dimension
        features = features.reshape(b, t+1, -1, features.size(-1))  # [b, f, n, d]
        
        # Project to action space
        actions_continuous = self.action_proj(features[:, :-1].mean(dim=2))  # [b, t, d]
        
        # Quantize actions
        actions_quantized, indices = self.quantizer(actions_continuous)
        
        # Decode for training
        reconstructed = self.decoder(features[:, :-1], actions_quantized)  # Use all but last frame features
        
        return reconstructed, actions_quantized, indices
    
    def infer_actions(self, frames, next_frames):
        """Get only the quantized actions for a sequence."""
        with torch.no_grad():
            all_frames = torch.cat([frames, next_frames.unsqueeze(1)], dim=1)
            features = self.encoder(all_frames)
            actions_continuous = self.action_proj(features[:, :-1])
            _, indices = self.quantizer(actions_continuous)
        return indices 