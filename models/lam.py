import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.vq_vae import STTransformerEncoder

class ActionVectorQuantizer(nn.Module):
    def __init__(self, n_codes=4, code_dim=256):  # Reduced to 4 actions for Pong (up/down for each paddle)
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
        
        # Straight through estimator with gradient support
        z_q = z + (z_q.detach() - z.detach())  # Modified to preserve gradients
        
        return z_q, min_encoding_indices

class LAMDecoder(nn.Module):
    def __init__(self, dim=256, n_heads=4, n_layers=4, threshold=0.5):
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
        self.threshold = threshold
        
    def forward(self, x, actions):
        # x shape: [b, f, n, d] or [b*f, n, d]
        # actions shape: [b, d]
        
        original_batch_size = actions.size(0)
        
        if len(x.shape) == 4:
            b, f, n, d = x.shape
            x = x.reshape(b*f, n, d)
            # Expand actions to match number of frames
            actions = actions.unsqueeze(1).expand(b, f, d).reshape(b*f, d)
        
        # Add action embeddings
        x = x + actions.unsqueeze(1)
        x = self.transformer(x)
        logits = self.output(x.mean(dim=1))  # Average over sequence dimension
        
        # Reshape to match target frame dimensions [b, h, w]
        logits = logits.reshape(original_batch_size, 64, 64)
        
        # Always output binary values
        output = (torch.sigmoid(logits) > self.threshold).float()
        
        return output

class LAM(nn.Module):
    def __init__(self, dim=256, n_heads=4, n_layers=4):  # Keep same transformer params
        super().__init__()
        self.encoder = STTransformerEncoder(dim, n_heads, n_layers)
        self.action_proj = nn.Linear(dim, dim)
        self.quantizer = ActionVectorQuantizer(n_codes=4, code_dim=dim)  # Reduced to 4 actions
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
        features = self.encoder(all_frames)  # [b*(t+1), n, d]
        
        # Reshape features back to include frame dimension
        features = features.reshape(b, t+1, -1, features.size(-1))  # [b, t+1, n, d]
        
        # Project to action space (average over patches)
        actions_continuous = self.action_proj(features[:, :-1].mean(dim=2))  # [b, t, d]
        
        # Quantize actions
        actions_quantized, indices = self.quantizer(actions_continuous.reshape(-1, actions_continuous.size(-1)))
        actions_quantized = actions_quantized.reshape(b, t, -1)
        
        # Decode for training (use last frame's actions)
        # Only pass the batch's worth of features and actions
        reconstructed = self.decoder(
            features[:, -2:-1],  # Use only the last frame's features [b, 1, n, d]
            actions_quantized[:, -1]  # Use only the last frame's actions [b, d]
        )
        
        return reconstructed, actions_quantized, indices
    
    def infer_actions(self, frames, next_frames):
        """Get only the quantized actions for a sequence.
        Args:
            frames: [batch, time, height, width] Previous frames
            next_frames: [batch, time, height, width] Next frames
        Returns:
            indices: Action indices
        """
        with torch.no_grad():
            # No need to unsqueeze next_frames since it's already [B, T, H, W]
            all_frames = torch.cat([frames, next_frames], dim=1)
            features = self.encoder(all_frames)
            features = features.reshape(frames.size(0), -1, features.size(1), features.size(2))
            actions_continuous = self.action_proj(features[:, :-1].mean(dim=2))
            _, indices = self.quantizer(actions_continuous.reshape(-1, actions_continuous.size(-1)))
        return indices 