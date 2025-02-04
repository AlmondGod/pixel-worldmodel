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

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert quantized frame to displayable RGB image."""
        # Normalize to 0-255 range
        frame_normalized = (frame * (255 / (frame.max() or 1))).astype(np.uint8)
        
        # Resize for display
        frame_large = cv2.resize(frame_normalized, (self.display_size, self.display_size), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_large, cv2.COLOR_GRAY2BGR)
        return frame_rgb

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