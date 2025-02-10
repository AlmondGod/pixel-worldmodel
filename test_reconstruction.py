import torch
import numpy as np
from models.vq_vae import VQVAE
from models.dynamics import MaskGITDynamics
from models.lam import LAM
from video_dataset import VideoFrameDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def plot_comparison(original, reconstructed, title, save_path):
    """Plot original vs reconstructed frames side by side"""
    n_frames = min(8, original.shape[0])  # Show up to 8 frames
    plt.figure(figsize=(20, 4))
    
    for i in range(n_frames):
        # Original
        plt.subplot(2, n_frames, i + 1)
        plt.imshow(1 - original[i], cmap='binary', vmin=0, vmax=1)  # Invert colors with 1 - original
        plt.axis('off')
        if i == 0:
            plt.title('Original')
            
        # Reconstructed
        plt.subplot(2, n_frames, n_frames + i + 1)
        plt.imshow(1 - reconstructed[i], cmap='binary', vmin=0, vmax=1)  # Invert colors with 1 - reconstructed
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(original, reconstructed):
    """Calculate binary accuracy and IoU for binary images"""
    # Ensure inputs are binary
    original = original.astype(bool)
    reconstructed = reconstructed.astype(bool)
    
    # Calculate accuracy
    accuracy = np.mean(original == reconstructed)
    
    # Calculate IoU (Intersection over Union)
    intersection = np.logical_and(original, reconstructed).sum()
    union = np.logical_or(original, reconstructed).sum()
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return accuracy, iou

def test_vqvae_reconstruction(model, dataloader, device, save_dir):
    """Test VQVAE reconstruction quality"""
    model.eval()
    total_accuracy = 0
    total_iou = 0
    n_sequences = 0
    
    print("\nTesting VQVAE reconstruction...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Test on 5 sequences
                break
                
            batch = batch.to(device)
            
            # Get reconstruction
            recon, _, vq_loss, perplexity = model(batch)
            
            # Convert to numpy for visualization
            original = batch.cpu().numpy()
            reconstructed = recon.cpu().numpy()

            print(f"reconstructed: {reconstructed}")
            
            # Calculate metrics
            accuracy, iou = calculate_metrics(original, reconstructed)
            total_accuracy += accuracy
            total_iou += iou
            n_sequences += 1
            
            # Plot and save comparison
            plot_comparison(
                original[0], reconstructed[0],
                f'VQVAE Reconstruction (Accuracy: {accuracy:.4f}, IoU: {iou:.4f})',
                save_dir / f'vqvae_reconstruction_{batch_idx}.png'
            )
            
            print(f"Sequence {batch_idx}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  IoU: {iou:.4f}")
            print(f"  VQ Loss: {vq_loss.item():.4f}")
            print(f"  Codebook Perplexity: {perplexity.item():.2f}")
    
    avg_accuracy = total_accuracy / n_sequences
    avg_iou = total_iou / n_sequences
    print(f"\nAverage metrics across {n_sequences} sequences:")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  IoU: {avg_iou:.4f}")

def test_dynamics_prediction(vqvae, dynamics, lam, dataloader, device, save_dir):
    """Test dynamics model prediction quality"""
    vqvae.eval()
    dynamics.eval()
    lam.eval()
    
    total_accuracy = 0
    total_iou = 0
    n_sequences = 0
    
    print("\nTesting dynamics model prediction...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # Test on 5 sequences
                break
                
            batch = batch.to(device)
            
            # Split into previous and next frames
            prev_frames = batch[:, :-1]  # [B, T-1, H, W]
            next_frames = batch[:, 1:]   # [B, T-1, H, W]
            
            # Debug shapes
            print(f"\nBatch {batch_idx} shapes:")
            print(f"  batch: {batch.shape}")
            print(f"  prev_frames: {prev_frames.shape}")
            print(f"  next_frames: {next_frames.shape}")
            
            # Get actions using LAM
            actions = lam.infer_actions(prev_frames, next_frames)
            print(f"  actions after LAM: {actions.shape}")
            actions = actions.reshape(batch.size(0), -1)  # [B, (T-1)]
            print(f"  actions after reshape: {actions.shape}")
            actions = actions[:, 0]  # Take first action for each sequence [B]
            print(f"  actions final: {actions.shape}")
            
            # Get tokens from VQVAE for the last frame only
            _, last_frame_tokens, _, _ = vqvae(batch[:, -1:])  # Get tokens for last frame
            print(f"  last frame tokens: {last_frame_tokens.shape}")
            
            # Predict next tokens using only the last frame tokens
            logits = dynamics(last_frame_tokens, actions)
            print(f"  logits: {logits.shape}")
            next_tokens = torch.argmax(logits, dim=-1)
            print(f"  next tokens: {next_tokens.shape}")
            
            # Get embeddings from quantizer
            z_q = vqvae.quantizer.embedding(next_tokens)
            print(f"  z_q: {z_q.shape}")
            
            # Decode the quantized embeddings
            predicted = vqvae.decoder(z_q)
            print(f"  raw predicted: {predicted.shape}")
            
            # Reshape predictions to match frame dimensions
            predicted = predicted.reshape(batch.size(0), 1, 64, 64)  # [B, 1, H, W]
            print(f"  final predicted: {predicted.shape}")
            
            # Take only the first next frame for comparison
            next_frame = next_frames[:, 0]  # [B, H, W]
            
            # Convert to numpy for visualization
            original = next_frame.cpu().numpy()
            predicted = predicted.squeeze(1).cpu().numpy()
            
            # Calculate metrics
            accuracy, iou = calculate_metrics(original, predicted)
            total_accuracy += accuracy
            total_iou += iou
            n_sequences += 1
            
            # Plot and save comparison
            plot_comparison(
                original[0:1], predicted[0:1],
                f'Dynamics Prediction (Accuracy: {accuracy:.4f}, IoU: {iou:.4f})',
                save_dir / f'dynamics_prediction_{batch_idx}.png'
            )
            
            print(f"Sequence {batch_idx}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  IoU: {iou:.4f}")
    
    avg_accuracy = total_accuracy / n_sequences
    avg_iou = total_iou / n_sequences
    print(f"\nAverage metrics across {n_sequences} sequences:")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  IoU: {avg_iou:.4f}")

def main():
    # Create save directory for visualizations
    save_dir = Path("test_results")
    save_dir.mkdir(exist_ok=True)
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize VQVAE with matching architecture and tracking disabled
    vqvae = VQVAE().to(device)
    vqvae.quantizer.track_usage = False  # Disable usage tracking
    
    # Initialize dynamics model with matching codebook size
    dynamics = MaskGITDynamics().to(device)
    lam = LAM().to(device)
    
    # Load weights
    models_dir = Path("saved_models")
    
    # Print available saved models
    print("\nAvailable saved models:")
    for model_path in models_dir.glob("*.pth"):
        print(f"  {model_path}")
    
    # try:
        # Load the timestamped models instead of the generic ones
        # vqvae.load_state_dict(torch.load("/Users/almondgod/Repositories/pixel-worldmodel/saved_models/28081027/vqvae_20250209_080614.pth", map_location=device))
        # dynamics.load_state_dict(torch.load("/Users/almondgod/Repositories/pixel-worldmodel/saved_models/28081027/dynamics_20250209_081759.pth", map_location=device))
        # lam.load_state_dict(torch.load("/Users/almondgod/Repositories/pixel-worldmodel/saved_models/28081027/lam_20250209_081027.pth", map_location=device))
    # except Exception as e:
    #     print(f"\nError loading models: {str(e)}")
    #     print("\nTip: Make sure you've run training first and have model files in saved_models/")
    #     return
    
    # Load dataset
    try:
        dataset = VideoFrameDataset("test_data.h5")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")
        print("\nTip: Make sure you've run training first to generate the dataset")
        return
    
    # Test VQVAE reconstruction
    test_vqvae_reconstruction(vqvae, dataloader, device, save_dir)
    
    # Test dynamics prediction
    test_dynamics_prediction(vqvae, dynamics, lam, dataloader, device, save_dir)

if __name__ == "__main__":
    main() 