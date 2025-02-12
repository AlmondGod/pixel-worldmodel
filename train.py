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
from test_reconstruction import test_vqvae_reconstruction, test_dynamics

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

    accuracy, iou, white_acc, black_acc = test_dynamics(
            vqvae, model, lam, dataloader, device, save_dir
        )
        
    for epoch in range(epochs):

        
        total_loss = 0
        total_token_accuracy = 0
        total_mask_ratio = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get tokens from VQVAE
            with torch.no_grad():
                _, tokens, _, _ = vqvae(batch)  # [B, T, H*W/16] tokens
                
                # For each frame pair (t, t+1), get the action that transforms t into t+1
                # We only need one action per sequence
                prev_frame = batch[:, 0:1]  # [B, 1, H, W] - first frame
                next_frame = batch[:, 1:2]  # [B, 1, H, W] - second frame
                actions = lam.infer_actions(prev_frame, next_frame)  # [B] - one action per sequence
                
                # Reshape tokens to [B, T, N] where N is number of patches
                B = batch.size(0)
                T = batch.size(1)
                tokens = tokens.reshape(B, T, -1)  # [B, T, N]
                
                # Debug prints for shapes
                if n_batches % 10 == 0:
                    print("\nLAM Action Debug:")
                    print(f"  Batch shape: {batch.shape}")
                    print(f"  Prev frame shape: {prev_frame.shape}")
                    print(f"  Next frame shape: {next_frame.shape}")
                    print(f"  Actions shape: {actions.shape}")
                    print(f"  Action values: {actions.unique().tolist()}")
                    print(f"  Action distribution: {torch.bincount(actions, minlength=4)}")
                    print(f"  Token shape: {tokens.shape}")
            
            # Create random masks with higher masking rate for binary data
            mask_ratio = torch.rand(1).item() * 0.3 + 0.7  # 70-100% masking rate
            mask = torch.rand(tokens[:, 0].shape, device=tokens.device) < mask_ratio  # Create mask for all patches of first frame
            
            # Predict next tokens using current tokens and action
            logits = model(tokens[:, 0], actions)  # Use first frame tokens to predict second frame
            
            # Get target tokens (second frame)
            target_tokens = tokens[:, 1]  # [B, N]
            pred_tokens = logits  # [B, N, n_codes]
            
            # Apply mask to both predictions and targets
            target_tokens = target_tokens[mask]  # [M, N] where M is number of masked tokens
            pred_tokens = pred_tokens[mask]  # [M, N, n_codes]
            
            # Reshape for cross entropy
            target_tokens = target_tokens.reshape(-1).long()  # [M*N]
            pred_tokens = pred_tokens.reshape(-1, pred_tokens.size(-1))  # [M*N, n_codes]
            
            # Calculate cross entropy loss on masked positions only
            loss = F.cross_entropy(pred_tokens, target_tokens)
            
            # Add prediction diversity loss to encourage balanced token usage (only on masked positions)
            pred_probs = torch.softmax(pred_tokens, dim=-1).mean(dim=0)
            pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-10)).sum()
            pred_diversity_loss = -0.5 * pred_entropy
            
            # Add action diversity loss
            action_probs = torch.bincount(actions, minlength=4).float()
            action_probs = action_probs / action_probs.sum()
            action_entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum()
            action_diversity_loss = -2.0 * (action_entropy - torch.log(torch.tensor(4.0)).to(device))
            
            # Calculate frame differences
            last_prev_frames = prev_frame.float()
            next_frames_last = next_frame.float()
            frame_diffs = next_frames_last.float() - last_prev_frames.float()
            frame_diffs = frame_diffs.reshape(batch.size(0), -1)
            frame_diffs = F.normalize(frame_diffs, dim=1, p=2)  # L2 normalize
            
            # Dynamic loss weighting based on action entropy
            # If entropy is very low, increase diversity weight
            entropy_factor = torch.exp(-8.0 * action_entropy)  # Much steeper exponential
            diversity_weight = 100.0 * entropy_factor  # 10x larger base weight
            
            # If entropy is critically low, make diversity loss completely dominate
            if action_entropy < 0.2:  # About 15% of maximum entropy
                diversity_weight = diversity_weight * 10.0
            
            # Combine losses with dynamic weights
            total_loss = (
                0.1 * loss +           # Reduce reconstruction weight
                0.5 * pred_entropy +         # Reduce prediction entropy weight
                diversity_weight * action_diversity_loss  # Let diversity dominate
            )
            
            # Add gradient penalty to prevent collapse
            if total_loss < 0.1:
                total_loss = total_loss + 0.1 * torch.sum(torch.abs(torch.sigmoid(logits)))
            
            loss = total_loss
            loss.backward()
            
            # Calculate accuracy
            with torch.no_grad():
                pred_indices = torch.argmax(pred_tokens, dim=-1)
                token_accuracy = (pred_indices == target_tokens).float().mean()
                total_token_accuracy += token_accuracy.item()
                
                # Track unique tokens and predictions
                n_unique_targets = len(torch.unique(target_tokens))
                n_unique_preds = len(torch.unique(pred_indices))
                
                # Track action entropy
                action_entropy_val = action_entropy.item()
            
            optimizer.step()
            
            total_loss += loss.item()
            total_mask_ratio += mask_ratio
            n_batches += 1
            
            # Print batch statistics
            if n_batches % 10 == 0:
                print(f"\nBatch {n_batches}/{len(dataloader)}")
                print(f"  CE Loss: {loss.item():.4f}")
                print(f"  Action Diversity Loss: {action_diversity_loss.item():.4f}")
                print(f"  Pred Diversity Loss: {pred_diversity_loss.item():.4f}")
                print(f"  InfoNCE Loss: {pred_entropy.item():.4f}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Token Accuracy: {token_accuracy.item():.4f}")
                print(f"  Action Entropy: {action_entropy_val:.4f}")
                print(f"  Mask Ratio: {mask_ratio:.2f}")
                print(f"  Unique Target Tokens: {n_unique_targets}")
                print(f"  Unique Predicted Tokens: {n_unique_preds}")
                print(f"  Action Distribution: {torch.bincount(actions, minlength=4)}")
        
        print(f"\nEpoch {epoch}")
        print(f"  Average Loss: {total_loss/n_batches:.4f}")
        print(f"  Average Token Accuracy: {total_token_accuracy/n_batches:.4f}")
        print(f"  Average Mask Ratio: {total_mask_ratio/n_batches:.2f}")
        
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
        total_accuracy = 0
        total_white_accuracy = 0
        total_black_accuracy = 0
        total_action_entropy = 0
        n_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Split into previous frames and next frame
            prev_frames = batch[:, :-1]  # [B, T-1, H, W]
            next_frame = batch[:, -1]    # [B, H, W]
            
            # Forward pass
            logits, actions_quantized, indices = model(prev_frames, next_frame)
            
            # Binary cross entropy loss with logits
            recon_loss = F.binary_cross_entropy_with_logits(logits, next_frame)
            
            # Get diversity loss from model
            diversity_loss = model.current_diversity_loss
            action_entropy = model.current_entropy
            
            # InfoNCE loss to maximize mutual information between actions and transitions
            batch_size = prev_frames.size(0)
            
            # Get frame transitions with stronger normalization
            last_prev_frames = prev_frames[:, -1]
            frame_diffs = next_frame.float() - last_prev_frames.float()
            frame_diffs = frame_diffs.reshape(batch_size, -1)
            frame_diffs = F.normalize(frame_diffs, dim=1, p=2)  # L2 normalize
            
            # Calculate similarity matrix between transitions
            similarities = torch.matmul(frame_diffs, frame_diffs.t())
            
            # Get last action for each sequence
            last_actions = indices.reshape(-1)[-batch_size:]
            
            # Calculate positive and negative masks for InfoNCE with stronger temperature
            pos_mask = (last_actions.unsqueeze(0) == last_actions.unsqueeze(1)).float()
            neg_mask = 1 - pos_mask
            
            # InfoNCE loss with sharper temperature
            temperature = 0.1  # Sharper temperature for better discrimination
            exp_similarities = torch.exp(similarities / temperature)
            
            # Exclude self-similarities
            self_mask = torch.eye(batch_size, device=device)
            exp_similarities = exp_similarities * (1 - self_mask)
            
            # Calculate InfoNCE loss with stability improvements
            pos_term = (exp_similarities * pos_mask).sum(1) + 1e-6
            neg_term = (exp_similarities * neg_mask).sum(1) + 1e-6
            infonce_loss = -torch.log(pos_term / (pos_term + neg_term + 1e-6)).mean()
            
            # Dynamic loss weighting based on action entropy
            # If entropy is very low, increase diversity weight
            entropy_factor = torch.exp(-8.0 * action_entropy)  # Much steeper exponential
            diversity_weight = 100.0 * entropy_factor  # 10x larger base weight
            
            # If entropy is critically low, make diversity loss completely dominate
            if action_entropy < 0.2:  # About 15% of maximum entropy
                diversity_weight = diversity_weight * 10.0
            
            # Combine losses with dynamic weights
            total_loss = (
                0.1 * recon_loss +           # Reduce reconstruction weight
                0.5 * infonce_loss +         # Reduce InfoNCE weight
                diversity_weight * diversity_loss  # Let diversity dominate
            )
            
            # Add gradient penalty to prevent collapse
            if total_loss < 0.1:
                total_loss = total_loss + 0.1 * torch.sum(torch.abs(torch.sigmoid(logits)))
            
            loss = total_loss
            loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Calculate accuracies
            with torch.no_grad():
                predictions = (torch.sigmoid(logits) > 0.5).float()
                accuracy = (predictions == next_frame).float().mean()
                
                white_mask = next_frame == 1
                black_mask = next_frame == 0
                white_accuracy = (predictions[white_mask] == 1).float().mean() if white_mask.any() else torch.tensor(0.0)
                black_accuracy = (predictions[black_mask] == 0).float().mean() if black_mask.any() else torch.tensor(0.0)
                
                total_accuracy += accuracy.item()
                total_white_accuracy += white_accuracy.item()
                total_black_accuracy += black_accuracy.item()
                total_action_entropy += action_entropy.item()
            
            total_loss += loss.item()
            n_batches += 1
            
            if n_batches % 10 == 0:
                print(f"\nBatch {n_batches}/{len(dataloader)}")
                print(f"  Recon Loss: {recon_loss.item():.4f}")
                print(f"  InfoNCE Loss: {infonce_loss.item():.4f}")
                print(f"  Diversity Loss: {diversity_loss.item():.4f}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Action Entropy: {action_entropy.item():.4f}")
                print(f"  Diversity Weight: {diversity_weight.item():.4f}")
                print(f"  Overall Accuracy: {accuracy.item():.4f}")
                print(f"  White Pixel Accuracy: {white_accuracy.item():.4f}")
                print(f"  Black Pixel Accuracy: {black_accuracy.item():.4f}")
                print(f"  Action Distribution: {torch.bincount(indices, minlength=4)}")
                print(f"  Unique Actions: {len(torch.unique(indices))}")
                print(f"  Action History: {model.action_history}")
                print(f"  Frame Similarities:")
                print(f"    Mean: {similarities.mean():.4f}")
                print(f"    Max: {similarities.max():.4f}")
                print(f"    Min: {similarities.min():.4f}")
                print(f"  Mean Frame Difference: {frame_diffs.abs().mean().item():.4f}")
        
        # Print epoch statistics
        print(f"\nEpoch {epoch}")
        print(f"  Average Loss: {total_loss/n_batches:.4f}")
        print(f"  Average Accuracy: {total_accuracy/n_batches:.4f}")
        print(f"  Average White Accuracy: {total_white_accuracy/n_batches:.4f}")
        print(f"  Average Black Accuracy: {total_black_accuracy/n_batches:.4f}")
        print(f"  Average Action Entropy: {total_action_entropy/n_batches:.4f}")
        
        # Save checkpoint
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