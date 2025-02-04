# Video Processing Pipeline for World Model Training

Implemented based on the Genie paper (this is a nano version)

A collection of tools for processing video data into training sequences for world models.

## Components

### Core Modules
- `video_converter.py`: Converts raw video frames to quantized format
  - Resizes frames to target resolution
  - Converts to grayscale
  - Quantizes colors using k-means clustering
  - Ensures consistent color mapping between frames

- `video_player.py`: Displays processed frame sequences
  - Converts quantized frames back to displayable format
  - Provides interactive playback
  - Supports adjustable display size and frame delay

- `video_dataset.py`: Processes videos into training data
  - Implements staggered frame sampling for maximum data utilization
  - Converts 30 FPS video to multiple 10 FPS sequences
  - Provides PyTorch Dataset class for training
  - Supports loading full or stagger-specific sequences

### Test Scripts
- `test_video.py`: Basic pipeline test
  - Tests video conversion and playback
  - Verifies basic functionality

- `test_staggering.py`: Validates staggered sampling
  - Creates test video with frame numbers
  - Verifies correct frame selection
  - Provides visual verification
  - Tests temporal ordering

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run test_video.py:
```bash
python test_video.py
```

3. Run test_staggering.py:
```bash
python test_staggering.py
```

4. Run test_dataset.py:
```bash
python test_dataset.py
```

5. Run test_video.py:
```bash
python test_video.py
```