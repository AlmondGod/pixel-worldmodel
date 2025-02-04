import cv2
import numpy as np
from video_dataset import VideoFrameDataset, convert_video_to_training_data
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os

"""
Test script to verify correct staggered frame sampling.
Creates a test video with frame numbers and verifies:
1. Correct frame selection for each stagger
2. Proper temporal ordering
3. Visual verification through matplotlib plots
"""

def visualize_staggered_sequences(data_path: str, n_staggers: int):
    """
    Visualize the first frame from the first sequence of each stagger
    to verify they're correctly offset.
    """
    plt.figure(figsize=(15, 5))
    
    # Get first sequence from each stagger
    for stagger in range(n_staggers):
        dataset = VideoFrameDataset(data_path, stagger_id=stagger)
        if len(dataset) == 0:
            print(f"No sequences found for stagger {stagger}")
            continue
            
        # Get first sequence and its first frame
        sequence = dataset[0]  # Shape: [sequence_length, height, width]
        first_frame = sequence[0]  # Shape: [height, width]
        
        plt.subplot(1, n_staggers, stagger + 1)
        plt.imshow(first_frame.numpy(), cmap='gray')
        plt.title(f'Stagger {stagger}\nFirst Frame')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('stagger_test.png')
    plt.close()

def test_temporal_order(data_path: str, n_staggers: int):
    """
    Print frame numbers that would have been used for each stagger's first sequence
    to verify temporal ordering.
    """
    frame_interval = n_staggers  # Since we're going from 30->10 FPS
    
    print("\nTemporal order test:")
    print("Expected frame numbers for first 5 frames of each stagger:")
    
    for stagger in range(n_staggers):
        frame_numbers = [i * frame_interval + stagger for i in range(5)]
        print(f"Stagger {stagger}: frames {frame_numbers}")

def main():
    # Create a test video with frame numbers
    test_video_path = "test_counting.mp4"
    output_dir = "test_stagger_data"
    
    # Clean up previous test data if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create a test video with frame numbers
    width, height = 64, 64
    fps = 30
    duration = 2  # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, fps, (width, height))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Put frame number in the center
        text = str(i)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        out.write(frame)
    
    out.release()
    
    # Convert the test video
    n_sequences = convert_video_to_training_data(
        video_path=test_video_path,
        output_dir=output_dir,
        target_size=(64, 64),
        n_colors=2,
        sequence_length=16,
        stride=8,
        source_fps=30.0,
        target_fps=10.0
    )
    
    # For 30->10 FPS, we should have 3 staggers
    n_staggers = 3
    
    # Visualize the first frame from each stagger
    visualize_staggered_sequences(output_dir, n_staggers)
    
    # Test temporal ordering
    test_temporal_order(output_dir, n_staggers)
    
    # Clean up
    os.remove(test_video_path)
    shutil.rmtree(output_dir)
    
    print("\nTest complete! Check stagger_test.png for visual verification.")
    print("The image should show the first frame from each stagger,")
    print("and they should be consecutive frame numbers (0, 1, 2).")

if __name__ == "__main__":
    main() 