import cv2
import numpy as np
from typing import List

"""
Video player for displaying processed frame sequences.
Handles conversion of quantized frames back to displayable format.
Provides interactive playback with adjustable frame delay.
"""

class VideoPlayer:
    def __init__(self, display_size: int = 512):
        """
        Initialize the video player.
        
        Args:
            display_size: Size of the display window (square)
        """
        self.display_size = display_size
        self.window_name = 'Video'

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert quantized frame to displayable RGB image."""
        # Ensure frame is 2D (height x width)
        if len(frame.shape) > 2:
            frame = frame.reshape(64, 64)  # Assuming 64x64 is our target size
            
        # Normalize to 0-255 range
        frame_normalized = (frame * (255 / (frame.max() or 1))).astype(np.uint8)
        
        # Resize for display
        frame_large = cv2.resize(frame_normalized, (self.display_size, self.display_size), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_large, cv2.COLOR_GRAY2BGR)
        return frame_rgb

    def display_frame(self, frame: np.ndarray, wait_time: int = 1) -> int:
        """Display a single frame and return the key pressed."""
        # Debug prints
        print(f"\nDisplay debug:")
        print(f"Input frame shape: {frame.shape}")
        print(f"Frame min/max: {frame.min():.3f}/{frame.max():.3f}")
        print(f"Frame mean/std: {frame.mean():.3f}/{frame.std():.3f}")
        
        display_frame = self._prepare_frame(frame)
        print(f"Prepared frame shape: {display_frame.shape}")
        print(f"Prepared frame min/max: {display_frame.min()}/{display_frame.max()}")
        
        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(wait_time) & 0xFF
        return key

    def cleanup(self):
        """Clean up OpenCV windows."""
        cv2.destroyAllWindows()

    def play_sequence(self, frames: List[np.ndarray], delay: int = 100):
        """
        Display a sequence of frames.
        
        Args:
            frames: List of quantized frames
            delay: Delay between frames in milliseconds
        """
        for frame in frames:
            display_frame = self._prepare_frame(frame)
            
            cv2.imshow('Video', display_frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows() 