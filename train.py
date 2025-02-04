import torch
from torch.utils.data import DataLoader
from models.vq_vae import VQVAE
from models.dynamics import MaskGITDynamics
from video_dataset import VideoFrameDataset, convert_video_to_training_data
from torch.nn import functional as F
from pathlib import Path
import argparse

EPOCHS = 50
SAVE_DIR = Path("saved_models")

# write a parse args to take in data path
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_data_hdf5.h5")
    args = parser.parse_args()
    return args

def train_vqvae(model, dataloader, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

def train_dynamics(model, vqvae, lam, dataloader, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Get tokens from VQVAE
            with torch.no_grad():
                _, tokens = vqvae(batch)
                # Get actions between frames
                prev_frames = batch[:, :-1]
                next_frames = batch[:, 1:]
                actions = lam.infer_actions(prev_frames, next_frames)
            
            # Create random masks
            mask_ratio = torch.rand(1).item() * 0.5 + 0.5
            mask = torch.rand_like(tokens.float()) < mask_ratio
            
            # Predict next tokens
            logits = model(tokens[:, :-1], actions, mask)  # Use actions as conditioning
            loss = F.cross_entropy(logits, tokens[:, 1:][mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

def train_lam(model, dataloader, optimizer, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
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
            
        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

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
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = VQVAE().to(device)
    dynamics = MaskGITDynamics().to(device)
    
    # Training VQVAE
    print("Training VQVAE...")
    vqvae_optim = torch.optim.AdamW(vqvae.parameters(), lr=3e-4, betas=(0.9, 0.9))
    train_vqvae(vqvae, dataloader, vqvae_optim)
    torch.save(vqvae.state_dict(), SAVE_DIR / "vqvae.pth")
    
    # Training Dynamics
    print("Training Dynamics...")
    dynamics_optim = torch.optim.AdamW(dynamics.parameters(), lr=3e-4, betas=(0.9, 0.9))
    train_dynamics(dynamics, vqvae, dynamics_optim)
    torch.save(dynamics.state_dict(), SAVE_DIR / "dynamics.pth")

if __name__ == "__main__":
    main() 