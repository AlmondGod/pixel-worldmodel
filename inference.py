import torch
import numpy as np
from models.vq_vae import VQVAE
from models.lam import LAM
from models.dynamics import MaskGITDynamics
from video_player import VideoPlayer
import argparse
from pathlib import Path
from video_dataset import VideoFrameDataset, convert_video_to_training_data

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
        self.vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        self.lam.load_state_dict(torch.load(lam_path, map_location=device))
        self.dynamics.load_state_dict(torch.load(dynamics_path, map_location=device))
        
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
            action_logits = self.lam(frames_tensor)
            action = torch.argmax(action_logits, dim=-1)
        return action.item()
    
    def generate_next_frame(self, frames, action):
        """Generate next frame given current frames and action."""
        with torch.no_grad():
            # Convert frames to tensor
            frames_tensor = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
            
            # Get tokens from VQVAE
            _, tokens = self.vqvae(frames_tensor)
            
            # Create action tensor
            action_tensor = torch.tensor([action]).to(self.device)
            
            # Predict next tokens
            logits = self.dynamics(tokens, action_tensor)
            next_tokens = torch.argmax(logits, dim=-1)
            
            # Decode tokens back to frame
            next_frame = self.vqvae.decode(next_tokens)
            
        return next_frame.squeeze().cpu().numpy()
    
    def run_interactive(self, initial_frames, n_steps=100):
        """Run interactive inference where user can input actions."""
        current_frames = initial_frames.copy()
        
        print("\nControls:")
        print("0-7: Select action")
        print("a: Auto (use predicted actions)")
        print("q: Quit")
        
        auto_mode = False
        
        for _ in range(n_steps):
            # Display current frame
            self.player.play_sequence([current_frames[-1]], delay=0)
            
            if auto_mode:
                action = self.predict_action(current_frames)
            else:
                # Get action from user
                key = input("Enter action (0-7) or command (a/q): ")
                if key == 'q':
                    break
                elif key == 'a':
                    auto_mode = True
                    action = self.predict_action(current_frames)
                else:
                    try:
                        action = int(key)
                        if not 0 <= action <= 7:
                            print("Invalid action! Using predicted action instead.")
                            action = self.predict_action(current_frames)
                    except ValueError:
                        print("Invalid input! Using predicted action instead.")
                        action = self.predict_action(current_frames)
            
            # Generate next frame
            next_frame = self.generate_next_frame(current_frames, action)
            
            # Update frame buffer
            current_frames = np.roll(current_frames, -1, axis=0)
            current_frames[-1] = next_frame
    
    def run_autonomous(self, initial_frames, n_steps=100):
        """Run autonomous inference using predicted actions."""
        current_frames = initial_frames.copy()
        
        for _ in range(n_steps):
            # Predict action
            action = self.predict_action(current_frames)
            
            # Generate next frame
            next_frame = self.generate_next_frame(current_frames, action)
            
            # Display frame
            self.player.play_sequence([next_frame], delay=100)
            
            # Update frame buffer
            current_frames = np.roll(current_frames, -1, axis=0)
            current_frames[-1] = next_frame

def main():
    parser = argparse.ArgumentParser(description='World Model Inference')
    parser.add_argument('--vqvae', type=str, default='saved_models/vqvae.pth', help='Path to VQVAE weights')
    parser.add_argument('--lam', type=str, default='saved_models/lam.pth', help='Path to LAM weights')
    parser.add_argument('--dynamics', type=str, default='saved_models/dynamics.pth', help='Path to Dynamics weights')
    parser.add_argument('--video', type=str, required=True, help='Path to initial video')
    parser.add_argument('--mode', type=str, choices=['interactive', 'autonomous'], 
                       default='interactive', help='Inference mode')
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