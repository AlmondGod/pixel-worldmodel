import torch
from torch.utils.data import DataLoader
from models.vq_vae import VQVAE
from models.dynamics import MaskGITDynamics
from models.lam import LAM
from video_dataset import VideoFrameDataset, convert_video_to_training_data
from torch.nn import functional as F
from pathlib import Path
import argparse
import numpy as np
import gc
from datetime import datetime
from test_reconstruction import test_vqvae_reconstruction
from train import *

EPOCHS = 4
SAVE_DIR = Path("saved_models")
BATCH_SIZE = 8  # Reduced from 32
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced from 16
CHECKPOINT_EVERY = 1

def main():
    args = parse_args()
    # Create save directory with timestamp
    SAVE_DIR.mkdir(exist_ok=True)
    save_dir = get_timestamped_dir()
    save_dir.mkdir(exist_ok=True)

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
    dataloader = DataLoader(dataset, 
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=0,  # No multiprocessing to reduce memory
                          pin_memory=False,  # Disable pinned memory
                          persistent_workers=False)  # Disable persistent workers
    
    # Initialize models with updated parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print available GPU memory
    if device == "cuda":
        print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    # Initialize models with default parameters
    vqvae = VQVAE().to(device)  # Use default parameters optimized for binary Pong
    lam = LAM().to(device)
    dynamics = MaskGITDynamics().to(device)

    vqvae_path = Path("/mnt/base/pixel-worldmodel/saved_models/20250210_025105/vqvae/checkpoint_epoch_1.pth")
    
    # Training VQVAE with monitoring
    print("\nTraining VQVAE...")
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
    
    # Clear GPU memory before training LAM
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Training LAM
    print("\nTraining LAM...")
    lam_optim = torch.optim.AdamW(lam.parameters(), lr=3e-4, betas=(0.9, 0.9))
    train_lam(lam, dataloader, lam_optim, save_dir)
    # lam.load_state_dict(torch.load("/mnt/base/pixel-worldmodel/saved_models/20250210_032822/lam/checkpoint_epoch_4.pth", map_location=device, weights_only=True))
    
    # Clear GPU memory before training dynamics
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Training Dynamics
    print("\nTraining Dynamics...")
    dynamics_optim = torch.optim.AdamW(dynamics.parameters(), lr=3e-4, betas=(0.9, 0.9))
    train_dynamics(dynamics, vqvae, lam, dataloader, dynamics_optim, save_dir)

if __name__ == "__main__":
    main() 