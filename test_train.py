import torch
from torch.utils.data import DataLoader
from models.vq_vae import VQVAE
from models.lam import LAM
from models.dynamics import MaskGITDynamics
from video_dataset import VideoFrameDataset, convert_video_to_training_data
from train import train_vqvae, train_lam, train_dynamics
from inference import WorldModelInference
from pathlib import Path
import argparse
import torch.nn.functional as F

SAVE_DIR = Path("saved_models")
BATCH_SIZE = 4   # Small batch size for testing

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="test_data.h5")
    parser.add_argument("--video_path", type=str, default="pong.mp4")
    parser.add_argument("--train_epochs", type=int, default=1)
    args = parser.parse_args()
    
    # Create save directory
    SAVE_DIR.mkdir(exist_ok=True)
    
    # Check if video exists
    if not Path("pong.mp4").exists():
        raise FileNotFoundError("pong.mp4 not found! Please place a video file named 'pong.mp4' in the current directory.")
    
    # Create test data if it doesn't exist
    if not Path(args.data_path).exists():
        print("Creating test dataset...")
        convert_video_to_training_data(
            video_path="pong.mp4",
            output_path=args.data_path,
            target_size=(64, 64),
            n_colors=2,
            sequence_length=16,
            stride=8,
            source_fps=30.0,
            target_fps=10.0
        )
    
    # Verify H5 file was created and has data
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Failed to create {args.data_path}")
        
    # Load dataset with verification
    dataset = VideoFrameDataset(args.data_path)
    if len(dataset) == 0:
        # Print H5 file structure for debugging
        import h5py
        with h5py.File(args.data_path, 'r') as f:
            print("\nH5 file contents:")
            print("Attributes:", dict(f.attrs))
            print("Groups:", list(f.keys()))
            for group in f:
                print(f"Sequences in {group}:", len(f[group]))
        raise ValueError(f"Dataset is empty! No sequences were created in {args.data_path}")
    
    print(f"Successfully loaded dataset with {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)
    
    # Get a batch for testing
    test_batch = next(iter(dataloader))
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = VQVAE().to(device)
    lam = LAM().to(device)
    dynamics = MaskGITDynamics().to(device)
    
    if args.train_epochs > 0:
        print(f"\nQuick training for {args.train_epochs} epochs...")
        
        print("\nTraining VQVAE...")
        vqvae_optim = torch.optim.AdamW(vqvae.parameters(), lr=3e-4)
        train_vqvae(vqvae, dataloader, vqvae_optim, epochs=args.train_epochs, device=device, verbose=True)
        torch.save(vqvae.state_dict(), SAVE_DIR / "vqvae.pth")
        
        print("\nTraining LAM...")
        lam_optim = torch.optim.AdamW(lam.parameters(), lr=3e-4)
        train_lam(lam, dataloader, lam_optim, epochs=args.train_epochs, device=device)
        torch.save(lam.state_dict(), SAVE_DIR / "lam.pth")
        
        print("\nTraining Dynamics...")
        dynamics_optim = torch.optim.AdamW(dynamics.parameters(), lr=3e-4)
        train_dynamics(dynamics, vqvae, lam, dataloader, dynamics_optim, epochs=args.train_epochs, device=device)
        torch.save(dynamics.state_dict(), SAVE_DIR / "dynamics.pth")
    
    # Test inference using WorldModelInference
    print("\nTesting inference pipeline...")
    inference = WorldModelInference(
        vqvae_path=SAVE_DIR / "vqvae.pth",
        lam_path=SAVE_DIR / "lam.pth",
        dynamics_path=SAVE_DIR / "dynamics.pth",
        device=device
    )
    
    # Move test batch to device for inference
    test_batch = test_batch.to(device)
    
    # Test both interactive and autonomous modes with small number of steps
    print("\nTesting interactive mode (2 steps)...")
    inference.run_interactive(test_batch.cpu().numpy(), n_steps=2)
    
    print("\nTesting autonomous mode (2 steps)...")
    inference.run_autonomous(test_batch.cpu().numpy(), n_steps=2)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 