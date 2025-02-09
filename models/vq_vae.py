import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class STTransformerEncoder(nn.Module):
    def __init__(self, dim=256, n_heads=4, n_layers=4, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        
        # Project patches to embedding dimension
        self.patch_embedding = nn.Linear(patch_size * patch_size, dim)
        
        # Position embedding for patches (16x16 patches for 64x64 image with patch_size=4)
        self.pos_embedding = nn.Parameter(torch.randn(1, (64//patch_size)**2, dim))
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim*4,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        # x shape: [batch, frames, height, width]
        b, f, h, w = x.shape
        
        # Patch embedding
        x = rearrange(x, 'b f (h p1) (w p2) -> b f (h w) (p1 p2)', 
                     p1=self.patch_size, p2=self.patch_size)
        
        # Project patches to embedding dimension
        x = self.patch_embedding(x)  # [b, f, h*w/p^2, dim]
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Process each frame independently
        x = rearrange(x, 'b f n d -> (b f) n d')
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, n_codes=64, code_dim=16, commitment_weight=0.5, decay=0.99, epsilon=1e-5, temp=1.0):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.epsilon = epsilon
        self.temp = temp
        
        # Initialize embedding with Xavier/Glorot initialization
        self.embedding = nn.Embedding(n_codes, code_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Initialize EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(n_codes))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
    def _update_codebook(self, encodings, z):
        """Update codebook using EMA updates"""
        # Flatten batch and token dimensions
        encodings_one_hot = F.one_hot(encodings, self.n_codes).float()
        encodings_one_hot = encodings_one_hot.reshape(-1, self.n_codes)  # [batch*tokens, n_codes]
        z = z.reshape(-1, z.size(-1))  # [batch*tokens, code_dim]
        
        # Calculate new cluster sizes
        cluster_size = encodings_one_hot.sum(0)
        
        # EMA update for cluster sizes with minimum cluster size
        self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                            (1 - self.decay) * cluster_size
        
        # Laplace smoothing with minimum cluster size
        n = self.ema_cluster_size.sum()
        cluster_size = ((self.ema_cluster_size + self.epsilon) / \
                    (n + self.n_codes * self.epsilon) * n).clamp(min=0.01)
        
        # Calculate new embeddings
        dw = torch.matmul(encodings_one_hot.t(), z)
        
        # EMA update for embeddings
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
        
        # Normalize embeddings
        self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
        
    def forward(self, z):
        # z shape: [batch, tokens, code_dim]
        
        # Calculate distances with temperature scaling
        d = torch.sum(z ** 2, dim=-1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=-1) - \
            2 * torch.matmul(z, self.embedding.weight.t())
        d = d / self.temp  # Apply temperature scaling
            
        # Get nearest neighbor
        encoding_indices = torch.argmin(d, dim=-1)
        z_q = self.embedding(encoding_indices)
        
        # Update codebook in training mode
        if self.training:
            self._update_codebook(encoding_indices, z)
        
        # Compute losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        # Straight through estimator
        z_q = z + (z_q - z).detach()
        
        # Calculate perplexity (measure of codebook usage) with numerical stability
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.n_codes).float()
            avg_probs = encodings.sum(0) / encodings.sum()  # Calculate average usage of each code
            # Add small epsilon to avoid log(0)
            avg_probs = torch.clamp(avg_probs, min=1e-10)
            # Normalize to ensure sum to 1
            avg_probs = avg_probs / avg_probs.sum()
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs)))
        
        return z_q, encoding_indices, commitment_loss * self.commitment_weight + codebook_loss, perplexity

class VQVAE(nn.Module):
    def __init__(
        self,
        dim=256,            # Keep transformer dim for good feature extraction
        n_heads=4,          # Keep 4 heads for multi-scale feature learning
        n_layers=4,         # Keep 4 layers for hierarchical features
        patch_size=4,       # 4x4 patches (good balance for Pong)
        n_codes=16,         # Small codebook for binary game
        code_dim=16,        # Compact embeddings for simple patterns
        commitment_weight=0.5  # Increased commitment for better codebook usage
    ):
        super().__init__()
        
        self.encoder = STTransformerEncoder(dim, n_heads, n_layers, patch_size)
        self.pre_quantize = nn.Linear(dim, code_dim)
        self.quantizer = VectorQuantizer(
            n_codes=n_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            temp=0.5  # Lower temperature for sharper code assignments
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, dim),
            nn.GELU(),  # Keep GELU for better gradient flow
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=n_heads,
                    dim_feedforward=dim*4,
                    batch_first=True,
                    dropout=0.1  # Keep dropout for regularization
                ),
                num_layers=n_layers
            ),
            nn.Linear(dim, patch_size * patch_size)
        )
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)  # [b*f, n, dim]
        z = self.pre_quantize(z)  # [b*f, n, code_dim]
        
        # Quantize
        z_q, indices, vq_loss, perplexity = self.quantizer(z)
        
        # Decode
        recon = self.decoder(z_q)
        recon = rearrange(recon, '(b f) (h w) (p1 p2) -> b f (h p1) (w p2)',
                         f=x.size(1), h=16, w=16, p1=4, p2=4)
        
        return recon, indices, vq_loss, perplexity 