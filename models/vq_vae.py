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
    def __init__(self, n_codes=256, code_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(n_codes, code_dim)
        self.embedding.weight.data.uniform_(-1./n_codes, 1./n_codes)
        
    def forward(self, z):
        # z shape: [batch, tokens, code_dim]
        
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

class VQVAE(nn.Module):
    def __init__(
        self,
        dim=256,
        n_heads=4,
        n_layers=4,
        patch_size=4,
        n_codes=256,
        code_dim=16
    ):
        super().__init__()
        
        self.encoder = STTransformerEncoder(dim, n_heads, n_layers, patch_size)
        self.pre_quantize = nn.Linear(dim, code_dim)
        self.quantizer = VectorQuantizer(n_codes, code_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=n_heads,
                    dim_feedforward=dim*4,
                    batch_first=True
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
        z_q, indices = self.quantizer(z)
        
        # Decode
        recon = self.decoder(z_q)
        recon = rearrange(recon, '(b f) (h w) (p1 p2) -> b f (h p1) (w p2)',
                         f=x.size(1), h=16, w=16, p1=4, p2=4)
        
        return recon, indices 