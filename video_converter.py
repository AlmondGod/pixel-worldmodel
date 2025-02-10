import cv2
import numpy as np
from typing import List, Tuple

"""
Video frame converter that processes raw video frames.
Handles grayscale conversion, resizing, and color quantization.
Uses k-means clustering to convert frames to a specified number of colors.
"""

class VideoConverter:
    def __init__(self, target_size: Tuple[int, int] = (64, 64), n_colors: int = 2):
        """
        Initialize the video converter.
        
        Args:
            target_size: Tuple of (width, height) for output frames
            n_colors: Number of colors to quantize to (default 2 for black/white)
        """
        self.target_size = target_size
        self.n_colors = n_colors

    def convert_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert a single frame to binary (black and white)"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] range
        normalized = resized.astype(float) / 255.0
        
        # Apply Otsu's thresholding for optimal binary separation
        if self.n_colors == 2:  # Binary mode
            threshold, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return (binary > 0).astype(float)  # Convert to binary float (0.0 or 1.0)
        else:
            # For non-binary cases, use kmeans as before
            flat_frame = normalized.reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(flat_frame, self.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Sort centers and relabel accordingly
            sorted_indices = np.argsort(centers.flatten())
            relabel_map = np.zeros_like(sorted_indices)
            relabel_map[sorted_indices] = np.arange(len(sorted_indices))
            labels = relabel_map[labels.flatten().astype(int)].reshape(labels.shape)
            
            # Normalize to [0, 1]
            quantized = labels.reshape(self.target_size[1], self.target_size[0]).astype(float)
            quantized = quantized / (self.n_colors - 1)
            return quantized

    def get_frame_sequence(self, video_path: str, n_frames: int = 16) -> List[np.ndarray]:
        """Extract and convert a sequence of frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        while len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 2 == 0:  # Skip every other frame
                converted = self.convert_frame(frame)
                frames.append(converted)
            
            frame_count += 1
            
        cap.release()
        return frames 