"""
Basic test script for the video processing pipeline.
Demonstrates conversion and playback of a single video file.
Tests basic functionality of converter and player classes.
"""
from video_converter import VideoConverter
from video_player import VideoPlayer

def main():
    # Initialize converter and player
    converter = VideoConverter(target_size=(64, 64), n_colors=2)
    player = VideoPlayer(display_size=512)
    
    # Convert 16 frames from video
    video_path = "pong.mp4"  # Update this to your video path
    frames = converter.get_frame_sequence(video_path, n_frames=16)
    
    # Play the converted frames
    player.play_sequence(frames, delay=100)

if __name__ == "__main__":
    main() 