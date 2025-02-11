import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.vq_vae import STTransformerEncoder

class ActionVectorQuantizer(nn.Module):
    def __init__(self, n_codes=4, code_dim=256, temperature=1.0):
        super().__init__()
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.embedding.weight.data.uniform_(-1./n_codes, 1./n_codes)
        self.temperature = temperature
        self.training_steps = 0
        self.warmup_steps = 2000  # Increased warmup period
        self.force_uniform_prob = 1.0  # Start with 100% forced uniform
        
    def forward(self, z):
        self.training_steps += 1
        batch_size = z.size(0)
        
        # Calculate probability of forcing uniform sampling
        if self.training:
            # Linearly decrease force_uniform_prob from 1.0 to 0.2 over warmup period
            self.force_uniform_prob = max(0.2, 1.0 - (self.training_steps / self.warmup_steps))
            
            # Force uniform sampling with current probability
            if torch.rand(1).item() < self.force_uniform_prob:
                # Ensure perfectly uniform distribution
                repeats = batch_size // 4 * 4
                remainder = batch_size % 4
                uniform_indices = torch.repeat_interleave(torch.arange(4), repeats//4)
                if remainder > 0:
                    uniform_indices = torch.cat([uniform_indices, torch.randint(0, 4, (remainder,))])
                uniform_indices = uniform_indices[torch.randperm(batch_size)]
                uniform_indices = uniform_indices.to(z.device)
                z_q = self.embedding(uniform_indices)
                # Use straight-through gradient
                z_q = z + (z_q - z).detach()
                return z_q, uniform_indices
        
        # Calculate distances
        d = torch.sum(z ** 2, dim=-1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=-1) - \
            2 * torch.matmul(z, self.embedding.weight.t())
        
        # Add strong Gumbel noise during training
        if self.training:
            noise = -torch.log(-torch.log(torch.rand_like(d) + 1e-10) + 1e-10)
            noise_scale = 2.0  # Increased noise for more exploration
            d = d / self.temperature + noise * noise_scale
        
        # Get nearest neighbor
        min_encoding_indices = torch.argmin(d, dim=-1)
        
        # Apply hard constraints during training
        if self.training:
            # Count actions in batch
            action_counts = torch.bincount(min_encoding_indices, minlength=4)
            max_count = batch_size // 2  # No action can take more than 50% of batch
            
            # If any action exceeds max_count, randomly reassign excess to underused actions
            for action in range(4):
                if action_counts[action] > max_count:
                    excess_mask = (min_encoding_indices == action)
                    excess_indices = torch.where(excess_mask)[0]
                    n_excess = action_counts[action] - max_count
                    
                    # Find underused actions
                    underused = torch.where(action_counts < max_count)[0]
                    if len(underused) > 0:
                        # Randomly select indices to reassign
                        to_reassign = excess_indices[torch.randperm(len(excess_indices))[:n_excess]]
                        # Assign to random underused actions
                        new_actions = underused[torch.randint(0, len(underused), (n_excess,))]
                        min_encoding_indices[to_reassign] = new_actions
        
        z_q = self.embedding(min_encoding_indices)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
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
        
        # KL divergence from uniform with extreme scaling
        uniform_probs = torch.ones_like(action_probs) / 4.0
        kl_div = F.kl_div(
            torch.log(action_probs + 1e-10),
            uniform_probs,
            reduction='sum'
        )
        
        # Historical imbalance penalty with super-exponential scaling
        historical_imbalance = torch.max(self.action_history) - torch.min(self.action_history)
        historical_penalty = torch.exp(10.0 * historical_imbalance)  # More aggressive exponential
        
        # Entropy penalty with super-exponential scaling
        target_entropy = torch.log(torch.tensor(4.0))  # Maximum entropy for 4 actions
        entropy_gap = torch.max(target_entropy - action_entropy, torch.tensor(0.0))
        entropy_penalty = torch.exp(20.0 * entropy_gap)  # Much more aggressive
        
        # Add minimum entropy constraint
        if action_entropy < 1.0:  # If entropy drops below ~75% of maximum
            entropy_penalty = entropy_penalty * 100.0  # Massive penalty boost
        
        # Combined diversity loss with extreme scaling
        diversity_loss = (
            100.0 * kl_div +                # 2x larger KL penalty
            entropy_penalty +                # Super-exponential entropy penalty
            50.0 * historical_penalty        # 2.5x larger historical penalty
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