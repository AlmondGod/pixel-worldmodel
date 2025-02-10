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

EPOCHS = 4
SAVE_DIR = Path("saved_models")
BATCH_SIZE = 8  # Reduced from 32
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced from 16
CHECKPOINT_EVERY = 1

# write a parse args to take in data path
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_data_hdf5.h5")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    args = parser.parse_args()
    return args

def get_timestamped_filename(base_name):
    """Generate a filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.pth"

def get_timestamped_dir():
    """Generate a timestamped directory path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SAVE_DIR / timestamp

def save_checkpoint(model, model_name, epoch, save_dir):
    """Save a model checkpoint"""
    checkpoint_dir = save_dir / model_name
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved {model_name} checkpoint at epoch {epoch}")

def train_vqvae(model, dataloader, optimizer, save_dir=SAVE_DIR, scheduler=None, epochs=EPOCHS, device="cuda", verbose=False):
    """
    Train VQVAE model with codebook usage monitoring and gradient accumulation
    """
    model.train()
    
    for epoch in range(epochs):
        total_recon_loss = 0
        total_vq_loss = 0
        avg_perplexity = 0
        total_accuracy = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Verify data is binary
            assert torch.all(torch.logical_or(batch == 0, batch == 1)), "Input data must be binary (0 or 1)"
            
            # Move batch to GPU and clear cache
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits, indices, vq_loss, perplexity = model(batch)
            
            # Calculate positive weight for BCE loss based on black/white ratio
            n_white = batch.sum()
            n_black = batch.numel() - n_white
            pos_weight = (n_black / n_white).clamp(min=1.0, max=10.0)  # Clamp to prevent extreme values
            
            # Binary cross entropy with logits and positive weight
            recon_loss = F.binary_cross_entropy_with_logits(
                logits, 
                batch,
                pos_weight=torch.tensor([pos_weight], device=device)
            )
            
            # Print statistics about black/white ratio and pos_weight
            if verbose and batch_idx % 10 == 0:
                white_ratio = (n_white / batch.numel()).item()
                print(f"  White pixel ratio: {white_ratio:.3f}")
                print(f"  Positive weight: {pos_weight:.3f}")
            
            # Total loss is reconstruction loss plus VQ losses
            loss = recon_loss + vq_loss
            
            # Normalize loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Calculate binary accuracy
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predictions == batch).float().mean()
                
                # Calculate separate accuracies for white and black pixels
                white_accuracy = (predictions[batch == 1] == 1).float().mean() if n_white > 0 else torch.tensor(0.0)
                black_accuracy = (predictions[batch == 0] == 0).float().mean() if n_black > 0 else torch.tensor(0.0)
                
                total_accuracy += accuracy.item()
            
            # Track metrics before clearing memory
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            avg_perplexity += perplexity.item()
            n_batches += 1
            
            # Step optimizer after accumulating gradients
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
            
            if verbose and batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}")
                print(f"  Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f}")
                print(f"  Perplexity: {perplexity.item():.1f}")
                print(f"  Overall Accuracy: {accuracy.item():.4f}")
                print(f"  White Pixel Accuracy: {white_accuracy.item():.4f}")
                print(f"  Black Pixel Accuracy: {black_accuracy.item():.4f}")
                if scheduler is not None:
                    print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                
                # Memory stats
                if device == "cuda":
                    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")
            
            # Clear all intermediate tensors
            del logits, indices, vq_loss, perplexity, recon_loss, loss, batch
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        
        # Print epoch statistics
        print(f"\nEpoch {epoch}")
        print(f"  Reconstruction Loss: {total_recon_loss/n_batches:.4f}")
        print(f"  VQ Loss: {total_vq_loss/n_batches:.4f}")
        print(f"  Average Perplexity: {avg_perplexity/n_batches:.1f}")
        print(f"  Average Accuracy: {total_accuracy/n_batches:.4f}")
        
        # Save checkpoint and test reconstruction
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == epochs - 1:
            save_checkpoint(model, "vqvae", epoch + 1, save_dir)
            model.eval()
            test_vqvae_reconstruction(model, dataloader, device, save_dir)
            model.train()

def train_dynamics(model, vqvae, lam, dataloader, optimizer, save_dir, epochs=EPOCHS, device="cuda"):
    model.train()
    vqvae.eval()  # Ensure VQVAE is in eval mode
    lam.eval()    # Ensure LAM is in eval mode
    
    # Track best model
    best_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
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
            
            # Create random masks with higher masking rate for binary data
            mask_ratio = torch.rand(1).item() * 0.3 + 0.7  # 70-100% masking rate
            mask = torch.rand_like(tokens[:, :-1].float()) < mask_ratio
            
            # Predict next tokens
            logits = model(tokens[:, :-1], actions)
            
            # Apply mask to both predictions and targets
            target_tokens = tokens[:, 1:][mask]
            pred_tokens = logits[mask]
            
            # Calculate cross entropy loss
            loss = F.cross_entropy(pred_tokens, target_tokens)
            loss.backward()
            
            # Calculate accuracy
            with torch.no_grad():
                pred_indices = torch.argmax(pred_tokens, dim=-1)
                accuracy = (pred_indices == target_tokens).float().mean()
                total_accuracy += accuracy.item()
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Print batch statistics
            if n_batches % 10 == 0:
                print(f"Batch {n_batches}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Token Accuracy: {accuracy.item():.4f}")
                print(f"  Mask Ratio: {mask_ratio:.2f}")
        
        print(f"\nEpoch {epoch}")
        print(f"  Average Loss: {total_loss/n_batches:.4f}")
        print(f"  Average Token Accuracy: {total_accuracy/n_batches:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating dynamics model...")
        model.eval()
        accuracy, iou, white_acc, black_acc = test_dynamics(
            vqvae, model, lam, dataloader, device, save_dir
        )
        model.train()
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            save_checkpoint(model, "dynamics_best", epoch + 1, save_dir)
        
        # Save regular checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == epochs - 1:
            save_checkpoint(model, "dynamics", epoch + 1, save_dir)
            
        print(f"\nBest model so far from epoch {best_epoch} with accuracy {best_accuracy:.4f}")

def train_lam(model, dataloader, optimizer, save_dir, epochs=EPOCHS, device="cuda"):
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
        
        # Save checkpoint every CHECKPOINT_EVERY epochs and at the final epoch
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == epochs - 1:
            save_checkpoint(model, "lam", epoch + 1, save_dir)

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
    
    # Training VQVAE with monitoring
    print("\nTraining VQVAE...")
    vqvae_optim = torch.optim.AdamW(vqvae.parameters(), lr=1e-4)
    train_vqvae(vqvae, dataloader, vqvae_optim, save_dir, verbose=True)
    
    # Clear GPU memory before training LAM
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Training LAM
    print("\nTraining LAM...")
    lam_optim = torch.optim.AdamW(lam.parameters(), lr=3e-4, betas=(0.9, 0.9))
    train_lam(lam, dataloader, lam_optim, save_dir)
    
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