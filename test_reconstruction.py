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
            
            # Get tokens for initial frame
            _, initial_tokens, _, _ = vqvae(initial_frame)
            
            # Get action using LAM
            action = lam.infer_actions(initial_frame, actual_next_frame)
            action = action.reshape(-1)[0]  # Take first action
            
            # Predict next tokens
            logits = dynamics(initial_tokens, action.unsqueeze(0))
            next_tokens = torch.argmax(logits, dim=-1)
            
            # Decode predicted tokens
            z_q = vqvae.quantizer.embedding(next_tokens)
            predicted_frame = vqvae.decoder(z_q)
            predicted_frame = predicted_frame.reshape(batch.size(0), 1, 64, 64)
            
            # Convert to binary
            predicted_frame = (torch.sigmoid(predicted_frame) > 0.5).float()
            
            # Convert to numpy for metrics
            actual = actual_next_frame.cpu().numpy()
            predicted = predicted_frame.cpu().numpy()
            
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
    
    try:
        # Load the timestamped models instead of the generic ones
        vqvae.load_state_dict(torch.load("/Users/almondgod/Repositories/pixel-worldmodel/saved_models/complete_1/:mnt:base:pixel-worldmodel:saved_models:20250210_025105:vqvae:checkpoint_epoch_1.pth", map_location=device))
        dynamics.load_state_dict(torch.load("/Users/almondgod/Repositories/pixel-worldmodel/saved_models/complete_1/dynamics", map_location=device))
        lam.load_state_dict(torch.load("/Users/almondgod/Repositories/pixel-worldmodel/saved_models/complete_1/lam", map_location=device))
    except Exception as e:
        print(f"\nError loading models: {str(e)}")
        print("\nTip: Make sure you've run training first and have model files in saved_models/")
        return
    
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