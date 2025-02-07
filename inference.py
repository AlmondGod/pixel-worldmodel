import torch
import numpy as np
from models.vq_vae import VQVAE
from models.lam import LAM
from models.dynamics import MaskGITDynamics
from video_player import VideoPlayer
import argparse
from pathlib import Path
from video_dataset import VideoFrameDataset, convert_video_to_training_data
import cv2
from einops import rearrange
import torch.nn.functional as F

class WorldModelInference:
    def __init__(
        self,
        vqvae_path: str,
        lam_path: str,
        dynamics_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize models for inference."""
        self.device = device
        
        # Load models
        self.vqvae = VQVAE().to(device)
        self.lam = LAM().to(device)
        self.dynamics = MaskGITDynamics().to(device)
        
        # Load weights
        self.vqvae.load_state_dict(torch.load(vqvae_path, map_location=device, weights_only=True))
        self.lam.load_state_dict(torch.load(lam_path, map_location=device, weights_only=True))
        self.dynamics.load_state_dict(torch.load(dynamics_path, map_location=device, weights_only=True))
        
        # Set to eval mode
        self.vqvae.eval()
        self.lam.eval()
        self.dynamics.eval()
        
        # Initialize video player
        self.player = VideoPlayer(display_size=512)
        
    def predict_action(self, frames):
        """Predict action from sequence of frames."""
        with torch.no_grad():
            frames_tensor = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
            # Split into previous frames and next frame
            prev_frames = frames_tensor[:, :-1]  # [1, T-1, H, W]
            next_frame = frames_tensor[:, -1]    # [1, H, W]
            
            # Get reconstructed frame and actions
            _, actions_quantized, indices = self.lam(prev_frames, next_frame)
            # Take the last action
            action = indices.reshape(-1)[-1]  # Take last predicted action
        return action.item()
    
    def debug_network_outputs(self, step, tokens, logits, next_tokens, decoded):
        """Track and compare network outputs across steps."""
        if not hasattr(self, 'debug_history'):
            self.debug_history = []
            
        # Store current outputs
        current_outputs = {
            'step': step,
            'token_hash': hash(tokens.cpu().numpy().tobytes()),
            'logits_stats': {
                'min': logits.min().item(),
                'max': logits.max().item(),
                'mean': logits.mean().item(),
                'std': logits.std().item(),
                'unique': len(torch.unique(logits))
            },
            'next_token_stats': {
                'unique': len(torch.unique(next_tokens)),
                'values': torch.unique(next_tokens).cpu().tolist()
            },
            'decoded_stats': {
                'min': decoded.min().item(),
                'max': decoded.max().item(),
                'mean': decoded.mean().item(),
                'std': decoded.std().item()
            }
        }
        
        self.debug_history.append(current_outputs)
        
        # Compare with previous step if available
        if len(self.debug_history) > 1:
            prev = self.debug_history[-2]
            curr = self.debug_history[-1]
            
            print("\n[Network Output Comparison]")
            print(f"Step {curr['step']} vs {prev['step']}:")
            
            # Check if tokens are identical
            tokens_identical = curr['token_hash'] == prev['token_hash']
            print(f"Tokens identical: {tokens_identical}")
            
            # Compare logits statistics
            print("\nLogits stats changes:")
            for key in ['min', 'max', 'mean', 'std', 'unique']:
                diff = curr['logits_stats'][key] - prev['logits_stats'][key]
                print(f"  {key}: {diff:+.6f}")
            
            # Compare next token distributions
            print("\nNext token changes:")
            print(f"  Unique tokens: {curr['next_token_stats']['unique']} vs {prev['next_token_stats']['unique']}")
            print(f"  Current values: {curr['next_token_stats']['values'][:5]}...")
            print(f"  Previous values: {prev['next_token_stats']['values'][:5]}...")
            
            # Compare decoded output statistics
            print("\nDecoded output changes:")
            for key in ['min', 'max', 'mean', 'std']:
                diff = curr['decoded_stats'][key] - prev['decoded_stats'][key]
                print(f"  {key}: {diff:+.6f}")
    
    def generate_next_frame(self, frames, action, step=0):
        """Generate next frame given current frames and action."""
        with torch.no_grad():
            # Convert frames to tensor: shape becomes (1, T, H, W)
            frames_tensor = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
            print("\n[generate_next_frame] frames_tensor shape:", frames_tensor.shape)
            
            # Run VQVAE forward to get tokens
            recon, tokens, _, _ = self.vqvae(frames_tensor)  # Properly unpack all 4 return values
            print("[generate_next_frame] VQVAE tokens shape:", tokens.shape)
            
            # Print frame change detection
            print(f"[generate_next_frame] Input frames hash: {hash(frames.tobytes())}")
            
            # We want to generate the next frame from the last frame's tokens:
            last_frame_tokens = tokens[-1:].clone()  # shape (1, num_patches)
            print("[generate_next_frame] last_frame_tokens shape:", last_frame_tokens.shape)
            
            # Predict next tokens with dynamics using only the last frame tokens:
            action_tensor = torch.tensor([action]).to(self.device)
            logits = self.dynamics(last_frame_tokens, action_tensor)
            print("[generate_next_frame] Dynamics logits shape:", logits.shape)
            
            # Sample next tokens with temperature
            temperature = 0.8  # Reduced temperature for less randomness
            probs = F.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.size(0), -1)
            print("[generate_next_frame] Sampled next_tokens shape:", next_tokens.shape)
            print(f"[generate_next_frame] Unique tokens in sample: {len(torch.unique(next_tokens))}")
            
            # Get embeddings from quantizer: expected shape (1, num_patches, embedding_dim)
            z_q = self.vqvae.quantizer.embedding(next_tokens)
            print("[generate_next_frame] z_q initial shape:", z_q.shape)
            
            # Now decode the quantized embeddings:
            decoded = self.vqvae.decoder(z_q)
            print("[generate_next_frame] Decoded raw shape:", decoded.shape)
            
            # Debug raw tensor values before any reshaping
            print("\n[RAW TENSOR DEBUG]")
            raw_decoded = decoded.cpu().numpy()
            print("Raw decoded tensor stats:")
            print(f"Shape: {raw_decoded.shape}")
            print(f"Global min/max: {raw_decoded.min():.3f}/{raw_decoded.max():.3f}")

                        
            # Check for pattern repetition in raw tensor
            print("\nChecking for pattern repetition...")
            if len(raw_decoded.shape) >= 2:
                # Take first few values from different regions
                n_samples = 4
                step = raw_decoded.shape[1] // n_samples
                samples = []
                for i in range(n_samples):
                    idx = i * step
                    region = raw_decoded[0, idx:idx+4]  # Take 4 values from each region
                    samples.append(region)
                print("\nSampling values from different regions:")
                for i, sample in enumerate(samples):
                    print(f"Region {i}: {sample}")
                
                # Check if consecutive patches are identical
                print("\nChecking consecutive patches:")
                for i in range(min(4, raw_decoded.shape[1]-1)):
                    patch1 = raw_decoded[0, i]
                    patch2 = raw_decoded[0, i+1]
                    is_identical = np.allclose(patch1, patch2, rtol=1e-5)
                    print(f"Patches {i} and {i+1} identical: {is_identical}")
            
            # Debug network outputs
            self.debug_network_outputs(step, last_frame_tokens, logits, next_tokens, decoded)
            
            # Rearrange patches into image
            try:
                # The decoder outputs (batch, n_patches, patch_values)
                # where n_patches = 16x16 = 256 (64x64 image with 4x4 patches)
                # and patch_values = 16 (4x4 patch flattened)
                
                # First unflatten the patches
                decoded = decoded.reshape(1, 16, 16, 4, 4)  # [batch, h_patches, w_patches, patch_h, patch_w]
                print("\n[RESHAPE DEBUG]")
                print(f"After unflatten patches: {decoded.shape}")
                
                # Now rearrange to form final image
                decoded_image = decoded.permute(0, 1, 3, 2, 4).reshape(1, 1, 64, 64)
                print(f"After final reshape: {decoded_image.shape}")
                
                # Verify patch independence
                patches = decoded_image[0, 0].reshape(16, 4, 16, 4).permute(0, 2, 1, 3).reshape(256, 16)
                n_unique_patches = len(torch.unique(patches, dim=0))
                print(f"Number of unique 4x4 patches: {n_unique_patches}")
                
                print("[generate_next_frame] Reshaped decoded shape:", decoded_image.shape)
            except Exception as e:
                print("[generate_next_frame] Error during reshape:", e)
                decoded_image = decoded.reshape(1, 1, 64, 64)  # fallback
            
            print("[generate_next_frame] Decoded image shape after rearrange:", decoded_image.shape)
            
            # Extract next frame (we generated one frame)
            next_frame = decoded_image[0, 0].cpu().numpy()

            #print frame
            print(f"Next frame: {next_frame}")
            
            # Debug final frame stats before normalization
            print("[generate_next_frame] Final frame stats before normalization:")
            print(f"  Shape: {next_frame.shape}")
            print(f"  min: {next_frame.min():.3f}, max: {next_frame.max():.3f}, mean: {next_frame.mean():.3f}")
            print(f"  Unique values: {len(np.unique(next_frame))}")
            print(f"  Frame hash: {hash(next_frame.tobytes())}")
            
            # Normalize to [0, 1]
            next_frame = (next_frame - next_frame.min()) / (next_frame.max() - next_frame.min() + 1e-8)
            
            print("[generate_next_frame] Final frame stats after normalization:")
            print(f"  min: {next_frame.min():.3f}, max: {next_frame.max():.3f}, mean: {next_frame.mean():.3f}")
            print(f"  Frame hash: {hash(next_frame.tobytes())}")
            
            return next_frame
    
    def run_interactive(self, initial_frames, n_steps=100):
        """Run interactive inference where user can input actions."""
        current_frames = initial_frames.copy()
        
        print("\nControls:")
        print("0-7: Select action")
        print("a: Auto (use predicted actions)")
        print("q: Quit")
        
        auto_mode = False
        
        try:
            for step in range(n_steps):
                print(f"\nStep {step}")
                # Display current frame and get key
                key = self.player.display_frame(current_frames[-1], wait_time=100)
                print(f"Key pressed: {key}")
                
                # Process key input
                if auto_mode:
                    action = self.predict_action(current_frames)
                    print(f"Auto mode action: {action}")
                else:
                    if key == ord('q'):
                        break
                    elif key == ord('a'):
                        auto_mode = True
                        action = self.predict_action(current_frames)
                        print("Switching to auto mode")
                    elif ord('0') <= key <= ord('7'):
                        action = key - ord('0')
                        print(f"Manual action: {action}")
                    else:
                        continue
                
                # Generate next frame
                next_frame = self.generate_next_frame(current_frames, action, step)
                print(f"Next frame min/max: {next_frame.min():.2f}/{next_frame.max():.2f}")
                
                # Update frame buffer
                current_frames = np.roll(current_frames, -1, axis=0)
                current_frames[-1] = next_frame
        finally:
            self.player.cleanup()
    
    def run_autonomous(self, initial_frames, n_steps=100):
        """Run autonomous inference using predicted actions."""
        current_frames = initial_frames.copy()
        
        try:
            for step in range(n_steps):
                print(f"\nStep {step}")
                
                # Debug current frame state
                print("Current frame debug:")
                print(f"Shape: {current_frames[-1].shape}")
                print(f"Stats - min: {current_frames[-1].min():.3f}, max: {current_frames[-1].max():.3f}")
                print(f"Mean/std: {current_frames[-1].mean():.3f}/{current_frames[-1].std():.3f}")
                print(f"Unique values: {len(np.unique(current_frames[-1]))}")
                
                # Display current frame with longer delay
                key = self.player.display_frame(current_frames[-1], wait_time=200)
                
                # Predict action
                action = self.predict_action(current_frames)
                print(f"Predicted action: {action}")
                
                # Generate next frame
                next_frame = self.generate_next_frame(current_frames, action, step)
                
                # Save debug images periodically
                if step % 5 == 0:
                    debug_frame = self.player._prepare_frame(next_frame)
                    cv2.imwrite(f'debug_frame_step_{step}.png', debug_frame)
                
                # Update frame buffer
                current_frames = np.roll(current_frames, -1, axis=0)
                current_frames[-1] = next_frame
        finally:
            self.player.cleanup()

def main():
    parser = argparse.ArgumentParser(description='World Model Inference')
    parser.add_argument('--vqvae', type=str, default='saved_models/vqvae1.pth', help='Path to VQVAE weights')
    parser.add_argument('--lam', type=str, default='saved_models/lam1.pth', help='Path to LAM weights')
    parser.add_argument('--dynamics', type=str, default='saved_models/dynamics1.pth', help='Path to Dynamics weights')
    parser.add_argument('--video', type=str, default='pong.mp4', help='Path to initial video')
    parser.add_argument('--mode', type=str, choices=['interactive', 'autonomous'], 
                       default='autonomous', help='Inference mode')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to generate')
    
    args = parser.parse_args()
    
    # Check if model files exist
    for path in [args.vqvae, args.lam, args.dynamics]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}. Did you run train.py first?")
    
    # Initialize inference
    inference = WorldModelInference(
        vqvae_path=args.vqvae,
        lam_path=args.lam,
        dynamics_path=args.dynamics
    )
    
    # Convert video to h5 format if needed
    h5_data = "training_data_hdf5.h5"
    if not Path(h5_data).exists():
        convert_video_to_training_data(
            video_path=args.video,
            output_path=h5_data
        )
    
    # Load initial frames
    dataset = VideoFrameDataset(h5_data)
    initial_frames = dataset[0].numpy()
    
    try:
        # Run inference
        if args.mode == 'interactive':
            inference.run_interactive(initial_frames, args.steps)
        else:
            inference.run_autonomous(initial_frames, args.steps)
    finally:
        # Clean up temporary file
        if Path(h5_data).exists():
            Path(h5_data).unlink()

if __name__ == "__main__":
    main() 