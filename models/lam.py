import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.vq_vae import STTransformerEncoder

class ActionVectorQuantizer(nn.Module):
    def __init__(self, n_codes=4, code_dim=256, temperature=1.0):  # Added temperature
        super().__init__()
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.embedding.weight.data.uniform_(-1./n_codes, 1./n_codes)
        self.temperature = temperature
        
    def forward(self, z):
        # Calculate distances with temperature scaling
        d = torch.sum(z ** 2, dim=-1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=-1) - \
            2 * torch.matmul(z, self.embedding.weight.t())
        d = d / self.temperature  # Scale distances by temperature
            
        # Get nearest neighbor with Gumbel noise for exploration
        if self.training:
            # Add Gumbel noise during training for exploration
            noise = -torch.log(-torch.log(torch.rand_like(d) + 1e-10) + 1e-10)
            d = d + noise * 0.1  # Scale noise to not dominate early training
        
        min_encoding_indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(min_encoding_indices)
        
        # Modified straight-through estimator with stronger gradients
        z_q = z + (z_q - z).detach()  # Allow gradients through z
        
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
        
        if self.training:
            # During training, return logits for BCE loss
            return logits
        else:
            # During inference, always return binary values
            binary_output = (torch.sigmoid(logits) > self.threshold).float()
            # Add assertion to guarantee binary output
            assert torch.all(torch.logical_or(binary_output == 0, binary_output == 1)), "Output must be binary (0 or 1)"
            return binary_output

class LAM(nn.Module):
    def __init__(self, dim=256, n_heads=4, n_layers=4):
        super().__init__()
        self.encoder = STTransformerEncoder(dim, n_heads, n_layers)
        self.action_proj = nn.Linear(dim, dim)
        self.quantizer = ActionVectorQuantizer(n_codes=4, code_dim=dim, temperature=2.0)  # Higher temperature
        self.decoder = LAMDecoder(dim, n_heads, n_layers)
        
        # Add action history tracking for diversity enforcement
        register_buffer = getattr(self, 'register_buffer', None)
        if register_buffer is not None:
            self.register_buffer('action_history', torch.zeros(4))
        else:
            self.action_history = torch.zeros(4)
        
    def get_diversity_loss(self, indices):
        """Calculate diversity loss with historical context"""
        # Update action history with exponential moving average
        current_dist = torch.bincount(indices, minlength=4).float()
        current_dist = current_dist / (current_dist.sum() + 1e-10)
        self.action_history = 0.99 * self.action_history + 0.01 * current_dist
        
        # Calculate entropy of current batch
        action_probs = current_dist
        action_entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
        
        # KL divergence from uniform
        uniform_probs = torch.ones_like(action_probs) / 4.0
        kl_div = F.kl_div(
            torch.log(action_probs + 1e-10),
            uniform_probs,
            reduction='sum'
        )
        
        # Historical imbalance penalty
        historical_imbalance = torch.max(self.action_history) - torch.min(self.action_history)
        
        # Combined diversity loss
        diversity_loss = (
            10.0 * kl_div +                    # Increased from 2.0
            20.0 * (1.386 - action_entropy) +  # Increased from 5.0
            15.0 * historical_imbalance        # Increased from 3.0
        )
        
        return diversity_loss, action_entropy
    
    def forward(self, frames, next_frames):
        frames = frames.float().requires_grad_(True)
        next_frames = next_frames.float().requires_grad_(True)
        
        b, t, h, w = frames.shape
        all_frames = torch.cat([frames, next_frames.unsqueeze(1)], dim=1)
        features = self.encoder(all_frames)
        features = features.reshape(b, t+1, -1, features.size(-1))
        
        actions_continuous = self.action_proj(features[:, :-1].mean(dim=2))
        actions_quantized, indices = self.quantizer(actions_continuous.reshape(-1, actions_continuous.size(-1)))
        actions_quantized = actions_quantized.reshape(b, t, -1)
        
        # Get diversity loss before reconstruction
        diversity_loss, action_entropy = self.get_diversity_loss(indices)
        
        reconstructed = self.decoder(
            features[:, -2:-1],
            actions_quantized[:, -1]
        )
        
        # Store diversity metrics for logging
        self.current_entropy = action_entropy
        self.current_diversity_loss = diversity_loss
        
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