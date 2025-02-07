import numpy as np
from video_player import VideoPlayer
import time

def test_display():
    player = VideoPlayer(display_size=512)
    
    # Create a test pattern that changes over time
    for i in range(10):
        # Create a frame with stripes that move
        frame = np.zeros((64, 64))
        frame[:, i::10] = 1.0
        
        print(f"\nDisplaying frame {i}")
        key = player.display_frame(frame, wait_time=500)  # 500ms delay
        print(f"Key pressed: {key}")
        
        if key == ord('q'):
            break
    
    player.cleanup()

if __name__ == "__main__":
    test_display() 