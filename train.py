import torch
from torch.utils.data import DataLoader
from models.vq_vae import VQVAE
from models.dynamics import MaskGITDynamics
from video_dataset import VideoFrameDataset, convert_video_to_training_data
from torch.nn import functional as F
from pathlib import Path
import argparse
import numpy as np

EPOCHS = 50
SAVE_DIR = Path("saved_models")

# write a parse args to take in data path
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_data_hdf5.h5")
    args = parser.parse_args()
    return args

def train_vqvae(model, dataloader, optimizer, epochs=EPOCHS, device="cuda", verbose=False):
    """
    Train VQVAE model with codebook usage monitoring
    Args:
        verbose: If True, print loss for every batch (for debugging)
    """
    model.train()
    for epoch in range(epochs):
        total_recon_loss = 0
        total_vq_loss = 0
        avg_perplexity = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon, _, vq_loss, perplexity = model(batch)
            
            # Compute reconstruction loss
            recon_loss = F.mse_loss(recon, batch)
            
            # Total loss is reconstruction loss plus VQ losses
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            avg_perplexity += perplexity.item()
            n_batches += 1
            
            if verbose:
                print(f"Batch losses - Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f}, "
                      f"Perplexity: {perplexity.item():.1f}")
        
        # Print epoch statistics
        print(f"Epoch {epoch}")
        print(f"  Reconstruction Loss: {total_recon_loss/n_batches:.4f}")
        print(f"  VQ Loss: {total_vq_loss/n_batches:.4f}")
        print(f"  Average Perplexity: {avg_perplexity/n_batches:.1f}")
        
        # Print codebook usage statistics
        if hasattr(model.quantizer, 'code_usage'):
            code_usage = model.quantizer.code_usage.cpu().numpy()
            active_codes = (code_usage > 0).sum()
            print(f"  Active Codes: {active_codes}/{model.quantizer.n_codes}")
            if verbose:
                print("  Most used codes:", np.argsort(-code_usage)[:10])
                print("  Usage values:", np.sort(code_usage)[-10:])

def train_dynamics(model, vqvae, lam, dataloader, optimizer, epochs=EPOCHS, device="cuda"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get tokens from VQVAE
            with torch.no_grad():
                _, tokens, _, _ = vqvae(batch)  # Updated to handle new return values
                # Split into previous and next frames
                prev_frames = batch[:, :-1]  # [B, T-1, H, W]
                next_frames = batch[:, 1:]   # [B, T-1, H, W]
                actions = lam.infer_actions(prev_frames, next_frames)
                
                # Reshape actions to match batch size
                actions = actions.reshape(batch.size(0), -1)  # [B, (T-1)]
                actions = actions[:, 0]  # Take first action for each sequence [B]
                
                # Ensure tokens and actions have same batch dimension
                tokens = tokens[:actions.size(0)]
            
            # Create random masks
            mask_ratio = torch.rand(1).item() * 0.5 + 0.5
            mask = torch.rand_like(tokens[:, :-1].float()) < mask_ratio
            
            # Predict next tokens
            logits = model(tokens[:, :-1], actions)
            
            # Apply mask to both predictions and targets
            target_tokens = tokens[:, 1:][mask]
            pred_tokens = logits[mask]
            
            loss = F.cross_entropy(pred_tokens, target_tokens)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        print(f"Epoch {epoch}, Loss: {total_loss/n_batches:.4f}")

def train_lam(model, dataloader, optimizer, epochs=EPOCHS, device="cuda"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Split into previous frames and next frame
            prev_frames = batch[:, :-1]  # [B, T-1, H, W]
            next_frame = batch[:, -1]    # [B, H, W]
            
            # Forward pass
            reconstructed, _, _ = model(prev_frames, next_frame)
            
            # Reconstruction loss
            loss = F.mse_loss(reconstructed, next_frame)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        print(f"Epoch {epoch}, Loss: {total_loss/n_batches:.4f}")

def main():
    args = parse_args()
    # Create save directory
    SAVE_DIR.mkdir(exist_ok=True)

    data_path = args.data_path
    
    # First, convert video to training data if it doesn't exist
    if not Path(data_path).exists():
        print("Converting video to training data...")
        convert_video_to_training_data(
            video_path="pong.mp4",  # Update this path
            output_path=data_path,
            target_size=(64, 64),
            n_colors=2,
            sequence_length=16,
            stride=8,
            source_fps=30.0,
            target_fps=10.0
        )
    
    # Load dataset
    dataset = VideoFrameDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Initialize models with updated parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = VQVAE(n_codes=64, commitment_weight=1.0).to(device)  # Reduced codebook size
    dynamics = MaskGITDynamics().to(device)
    
    # Training VQVAE with monitoring
    print("Training VQVAE...")
    vqvae_optim = torch.optim.AdamW(vqvae.parameters(), lr=3e-4, betas=(0.9, 0.9))
    train_vqvae(vqvae, dataloader, vqvae_optim, verbose=True)
    torch.save(vqvae.state_dict(), SAVE_DIR / "vqvae.pth")
    
    # Training Dynamics
    print("Training Dynamics...")
    dynamics_optim = torch.optim.AdamW(dynamics.parameters(), lr=3e-4, betas=(0.9, 0.9))
    train_dynamics(dynamics, vqvae, dynamics_optim)
    torch.save(dynamics.state_dict(), SAVE_DIR / "dynamics.pth")

if __name__ == "__main__":
    main() 