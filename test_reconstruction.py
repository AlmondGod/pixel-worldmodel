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
    
    # Verify inputs are binary
    assert np.all(np.logical_or(original == 0, original == 1)), "Original must be binary"
    assert np.all(np.logical_or(reconstructed == 0, reconstructed == 1)), "Reconstructed must be binary"
    
    for i in range(n_frames):
        # Original
        plt.subplot(2, n_frames, i + 1)
        plt.imshow(1 - original[i], cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
            
        # Reconstructed
        plt.subplot(2, n_frames, n_frames + i + 1)
        plt.imshow(1 - reconstructed[i], cmap='binary', vmin=0, vmax=1, interpolation='nearest')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
            
            # Get reconstruction (model outputs binary values in eval mode)
            recon, _, vq_loss, perplexity = model(batch)
            
            # Verify outputs are binary
            assert torch.all(torch.logical_or(recon == 0, recon == 1)), "Model outputs must be binary (0 or 1)"
            
            # Convert to numpy for visualization
            original = batch.cpu().numpy()
            reconstructed = recon.cpu().numpy()
            
            # Verify numpy arrays are binary
            assert np.all(np.logical_or(original == 0, original == 1)), "Original must be binary"
            assert np.all(np.logical_or(reconstructed == 0, reconstructed == 1)), "Reconstructed must be binary"
            
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
            
            # Debug prints
            print(f"  Original unique values: {np.unique(original)}")
            print(f"  Reconstructed unique values: {np.unique(reconstructed)}")
    
    avg_accuracy = total_accuracy / n_sequences
    avg_iou = total_iou / n_sequences
    print(f"\nAverage metrics across {n_sequences} sequences:")
    print(f"  Accuracy: {avg_accuracy:.4f}")
    print(f"  IoU: {avg_iou:.4f}")

def test_dynamics(vqvae, dynamics, lam, dataloader, device, save_dir, n_test_sequences=5):
    """Test dynamics model's ability to predict next frames."""
    vqvae.eval()
    dynamics.eval()
    lam.eval()
    
    total_accuracy = 0
    total_iou = 0
    total_white_accuracy = 0
    total_black_accuracy = 0
    n_sequences = 0
    
    print("\nTesting dynamics model prediction...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_test_sequences:
                break
                
            batch = batch.to(device)
            
            # Get initial frame and actual next frame
            initial_frame = batch[:, 0:1]  # [B, 1, H, W]
            actual_next_frame = batch[:, 1:2]  # [B, 1, H, W]
            
            print("\nInput debug:")
            print(f"  Initial frame unique values: {torch.unique(initial_frame).tolist()}")
            print(f"  Initial frame white ratio: {(initial_frame == 1).float().mean().item():.4f}")
            print(f"  Next frame unique values: {torch.unique(actual_next_frame).tolist()}")
            print(f"  Next frame white ratio: {(actual_next_frame == 1).float().mean().item():.4f}")
            
            # Get tokens for initial frame
            _, initial_tokens, _, _ = vqvae(initial_frame)
            print("\nVQVAE debug:")
            print(f"  Initial tokens unique values: {torch.unique(initial_tokens).tolist()}")
            print(f"  Initial tokens shape: {initial_tokens.shape}")
            
            # Get action using LAM
            action = lam.infer_actions(initial_frame, actual_next_frame)
            action = action.reshape(-1)[0]  # Take first action
            print(f"  Selected action: {action.item()}")
            
            # Reshape initial tokens to [B, 1, N] and then pad to [B, 16, N]
            initial_tokens = initial_tokens.unsqueeze(1)  # [B, 1, N]
            B, _, N = initial_tokens.shape
            padded_tokens = torch.zeros(B, 16, N, device=initial_tokens.device)
            padded_tokens[:, 0:1] = initial_tokens  # Only put tokens in first position
            
            print("\nDynamics input debug:")
            print(f"  Padded tokens unique values: {torch.unique(padded_tokens).tolist()}")
            print(f"  Padded tokens shape: {padded_tokens.shape}")
            
            # Predict next tokens
            logits = dynamics(padded_tokens, action.unsqueeze(0))
            print("\nDynamics output debug:")
            print(f"  Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
            print(f"  Logits mean/std: {logits.mean().item():.4f}/{logits.std().item():.4f}")
            
            # Get next token predictions for all positions
            next_tokens = torch.argmax(logits, dim=-1)  # [B, seq_len]
            print(f"  Predicted tokens unique values: {torch.unique(next_tokens).tolist()}")
            
            # Take only the first predicted token for each sequence
            next_tokens = next_tokens[:, 0]  # [B]
            print(f"  Selected tokens unique values: {torch.unique(next_tokens).tolist()}")
            
            # Convert to VQVAE patch tokens (256 patches for 16x16 grid)
            # Ensure token indices are within valid range
            next_tokens = next_tokens.clamp(0, vqvae.quantizer.n_codes - 1)  # Clamp to valid range
            next_tokens = next_tokens.unsqueeze(-1).repeat(1, 256)  # [B, 256]
            print("\nDecoder input debug:")
            print(f"  Expanded tokens unique values: {torch.unique(next_tokens).tolist()}")
            
            # Get embeddings from the quantizer
            z_q = vqvae.quantizer.embedding(next_tokens)  # [B, 256, code_dim]
            print(f"  Embeddings min/max: {z_q.min().item():.4f}/{z_q.max().item():.4f}")
            print(f"  Embeddings mean/std: {z_q.mean().item():.4f}/{z_q.std().item():.4f}")
            
            # Reshape for decoder - keep it in the format [batch*patches, code_dim]
            z_q = z_q.reshape(-1, z_q.size(-1))  # [B*256, code_dim]
            
            # Decode to get predicted frame
            logits = vqvae.decoder(z_q)  # Get logits first
            print("\nDecoder output debug:")
            print(f"  Decoder logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
            print(f"  Decoder logits mean/std: {logits.mean().item():.4f}/{logits.std().item():.4f}")
            
            # Reshape logits and apply sigmoid + threshold
            logits = logits.reshape(B, 16, 16, 16)  # [B, H/4, W/4, 16]
            logits = logits.reshape(B, 16, 16, 4, 4)  # [B, H/4, W/4, p, p]
            logits = logits.permute(0, 1, 3, 2, 4)  # [B, H/4, p, W/4, p]
            logits = logits.reshape(B, 64, 64)  # [B, H, W]
            
            # Apply sigmoid and threshold with a small margin for numerical stability
            predicted_frame = torch.sigmoid(logits)
            print("\nFinal output debug:")
            print(f"  Sigmoid output min/max: {predicted_frame.min().item():.4f}/{predicted_frame.max().item():.4f}")
            print(f"  Sigmoid output mean/std: {predicted_frame.mean().item():.4f}/{predicted_frame.std().item():.4f}")
            
            predicted_frame = (predicted_frame > 0.5).float()  # Convert to binary (0 or 1)
            predicted_frame = predicted_frame.unsqueeze(1)  # [B, 1, H, W]
            print(f"  Final output unique values: {torch.unique(predicted_frame).tolist()}")
            print(f"  Final output white ratio: {(predicted_frame == 1).float().mean().item():.4f}")
            
            # Double-check binary output
            assert torch.all(torch.logical_or(predicted_frame == 0, predicted_frame == 1)), "Predicted frame must be binary"
            
            # Convert to numpy for metrics
            actual = actual_next_frame.cpu().numpy()
            predicted = predicted_frame.cpu().numpy()
            
            # Verify numpy arrays are binary
            assert np.all(np.logical_or(actual == 0, actual == 1)), "Actual frame must be binary"
            assert np.all(np.logical_or(predicted == 0, predicted == 1)), "Predicted frame must be binary"
            
            # Calculate metrics
            accuracy, iou = calculate_metrics(actual.squeeze(), predicted.squeeze())
            
            # Calculate separate accuracies for white and black pixels
            n_white = actual.sum()
            n_black = actual.size - n_white
            white_accuracy = np.mean(predicted[actual == 1] == 1) if n_white > 0 else 0
            black_accuracy = np.mean(predicted[actual == 0] == 0) if n_black > 0 else 0
            
            total_accuracy += accuracy
            total_iou += iou
            total_white_accuracy += white_accuracy
            total_black_accuracy += black_accuracy
            n_sequences += 1
            
            # Plot comparison
            plot_comparison(
                actual.squeeze(),
                predicted.squeeze(),
                f'Dynamics Prediction (Acc: {accuracy:.3f}, IoU: {iou:.3f})',
                save_dir / f'dynamics_prediction_{batch_idx}.png'
            )
            
            print(f"\nSequence {batch_idx}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  IoU: {iou:.4f}")
            print(f"  White Pixel Accuracy: {white_accuracy:.4f}")
            print(f"  Black Pixel Accuracy: {black_accuracy:.4f}")
    
    # Calculate averages
    avg_accuracy = total_accuracy / n_sequences
    avg_iou = total_iou / n_sequences
    avg_white_accuracy = total_white_accuracy / n_sequences
    avg_black_accuracy = total_black_accuracy / n_sequences
    
    print(f"\nAverage metrics across {n_sequences} sequences:")
    print(f"  Overall Accuracy: {avg_accuracy:.4f}")
    print(f"  IoU: {avg_iou:.4f}")
    print(f"  White Pixel Accuracy: {avg_white_accuracy:.4f}")
    print(f"  Black Pixel Accuracy: {avg_black_accuracy:.4f}")
    
    return avg_accuracy, avg_iou, avg_white_accuracy, avg_black_accuracy

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
    test_dynamics(vqvae, dynamics, lam, dataloader, device, save_dir)

if __name__ == "__main__":
    main() 