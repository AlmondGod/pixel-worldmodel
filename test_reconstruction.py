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
        plt.imshow(original[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')
            
        # Reconstructed
        plt.subplot(2, n_frames, n_frames + i + 1)
        plt.imshow(reconstructed[i], cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(original, reconstructed):
    """Calculate MSE and PSNR"""
    mse = np.mean((original - reconstructed) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))  # Assuming max pixel value is 1
    return mse, psnr

def test_vqvae_reconstruction(model, dataloader, device, save_dir):
    """Test VQVAE reconstruction quality"""
    model.eval()
    total_mse = 0
    total_psnr = 0
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
            mse, psnr = calculate_metrics(original, reconstructed)
            total_mse += mse
            total_psnr += psnr
            n_sequences += 1
            
            # Plot and save comparison
            plot_comparison(
                original[0], reconstructed[0],
                f'VQVAE Reconstruction (MSE: {mse:.4f}, PSNR: {psnr:.2f}dB)',
                save_dir / f'vqvae_reconstruction_{batch_idx}.png'
            )
            
            print(f"Sequence {batch_idx}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  PSNR: {psnr:.2f}dB")
            print(f"  VQ Loss: {vq_loss.item():.4f}")
            print(f"  Codebook Perplexity: {perplexity.item():.2f}")
    
    avg_mse = total_mse / n_sequences
    avg_psnr = total_psnr / n_sequences
    print(f"\nAverage metrics across {n_sequences} sequences:")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  PSNR: {avg_psnr:.2f}dB")

def test_dynamics_prediction(vqvae, dynamics, lam, dataloader, device, save_dir):
    """Test dynamics model prediction quality"""
    vqvae.eval()
    dynamics.eval()
    lam.eval()
    
    total_mse = 0
    total_psnr = 0
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
            mse, psnr = calculate_metrics(original, predicted)
            total_mse += mse
            total_psnr += psnr
            n_sequences += 1
            
            # Plot and save comparison
            plot_comparison(
                original[0:1], predicted[0:1],
                f'Dynamics Prediction (MSE: {mse:.4f}, PSNR: {psnr:.2f}dB)',
                save_dir / f'dynamics_prediction_{batch_idx}.png'
            )
            
            print(f"Sequence {batch_idx}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  PSNR: {psnr:.2f}dB")
    
    avg_mse = total_mse / n_sequences
    avg_psnr = total_psnr / n_sequences
    print(f"\nAverage metrics across {n_sequences} sequences:")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  PSNR: {avg_psnr:.2f}dB")

def main():
    # Create save directory for visualizations
    save_dir = Path("test_results")
    save_dir.mkdir(exist_ok=True)
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize VQVAE with matching architecture and tracking disabled
    vqvae = VQVAE(
        dim=256,
        n_heads=4,
        n_layers=4,
        patch_size=4,
        n_codes=16,  # Match the saved model's codebook size
        code_dim=16,
        commitment_weight=1.0
    ).to(device)
    vqvae.quantizer.track_usage = False  # Disable usage tracking
    
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
        vqvae.load_state_dict(torch.load("/mnt/base/pixel-worldmodel/saved_models/vqvae_20250209_202437.pth", map_location=device))
        dynamics.load_state_dict(torch.load("/mnt/base/pixel-worldmodel/saved_models/dynamics_20250209_203449.pth", map_location=device))
        lam.load_state_dict(torch.load("/mnt/base/pixel-worldmodel/saved_models/lam_20250209_202901.pth", map_location=device))
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
    test_dynamics_prediction(vqvae, dynamics, lam, dataloader, device, save_dir)

if __name__ == "__main__":
    main() 