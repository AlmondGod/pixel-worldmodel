import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from video_converter import VideoConverter
from pathlib import Path
import os
import cv2
import h5py
from typing import Optional

"""
Dataset processing pipeline for converting videos to training data.
Implements staggered frame sampling to maximize data usage.
Provides PyTorch Dataset class for loading processed sequences.
Supports multiple staggered sequences from single source video.
"""

class VideoFrameDataset(Dataset):
    def __init__(self, data_path: str, stagger_id: Optional[int] = None):
        """
        Dataset for loading preprocessed video frame sequences from HDF5.
        
        Args:
            data_path: Path to the HDF5 file
            stagger_id: If provided, only load sequences from this stagger
        """
        self.data_path = data_path
        self.stagger_id = stagger_id
        
        # Open HDF5 file in read mode
        self.h5_file = h5py.File(data_path, 'r')
        
        # Get sequence information
        if stagger_id is not None:
            self.sequences = self.h5_file[f'stagger_{stagger_id}']
        else:
            self.sequences = []
            for i in range(len(self.h5_file.attrs['sequence_counts'])):
                self.sequences.extend(self.h5_file[f'stagger_{i}'].values())
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx][:]  # Load from HDF5
        return torch.from_numpy(sequence).float()
    
    def __del__(self):
        self.h5_file.close()

def convert_video_to_training_data(
    video_path: str,
    output_path: str,
    target_size: tuple = (64, 64),
    n_colors: int = 2,
    sequence_length: int = 16,
    stride: int = 8,
    source_fps: float = 30.0,
    target_fps: float = 10.0,
    compression: Optional[str] = 'lzf'  # or 'gzip' for better compression
):
    """Convert video to training data and save as HDF5."""
    converter = VideoConverter(target_size=target_size, n_colors=n_colors)
    
    # Calculate staggers
    frame_interval = int(round(source_fps / target_fps))
    n_staggers = frame_interval
    
    # Open HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['target_size'] = target_size
        f.attrs['n_colors'] = n_colors
        f.attrs['sequence_length'] = sequence_length
        f.attrs['source_fps'] = source_fps
        f.attrs['target_fps'] = target_fps
        
        # Create groups for each stagger
        stagger_groups = [f.create_group(f'stagger_{i}') for i in range(n_staggers)]
        sequence_counts = [0 for _ in range(n_staggers)]
        frame_buffers = [[] for _ in range(n_staggers)]
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            stagger_idx = frame_count % frame_interval
            converted = converter.convert_frame(frame)
            frame_buffers[stagger_idx].append(converted)
            
            # Save sequence when buffer is full
            if len(frame_buffers[stagger_idx]) >= sequence_length:
                sequence = np.stack(frame_buffers[stagger_idx][:sequence_length])
                
                # Create dataset for this sequence
                stagger_groups[stagger_idx].create_dataset(
                    f'sequence_{sequence_counts[stagger_idx]}',
                    data=sequence,
                    compression=compression
                )
                
                frame_buffers[stagger_idx] = frame_buffers[stagger_idx][stride:]
                sequence_counts[stagger_idx] += 1
            
            frame_count += 1
            
        cap.release()
        
        # Store final sequence counts
        f.attrs['sequence_counts'] = sequence_counts

def main():
    # Example usage
    video_path = "pong.mp4"
    output_path = "training_data_hdf5.h5"
    
    # Convert video to sequences
    n_sequences = convert_video_to_training_data(
        video_path=video_path,
        output_path=output_path,
        target_size=(64, 64),
        n_colors=2,
        sequence_length=16,
        stride=8,
        source_fps=30.0,
        target_fps=10.0
    )
    
    # Test loading the dataset - both full and staggered
    full_dataset = VideoFrameDataset(output_path)
    stagger0_dataset = VideoFrameDataset(output_path, stagger_id=0)
    
    # Create dataloaders
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
    stagger0_loader = DataLoader(stagger0_dataset, batch_size=32, shuffle=True)
    
    # Print sample information
    sample_batch = next(iter(full_loader))
    print(f"\nFull dataset:")
    print(f"Sample batch shape: {sample_batch.shape}")
    print(f"Total sequences: {len(full_dataset)}")
    print(f"Data type: {sample_batch.dtype}")
    print(f"Value range: [{sample_batch.min()}, {sample_batch.max()}]")
    
    print(f"\nStagger 0 dataset:")
    print(f"Total sequences: {len(stagger0_dataset)}")

if __name__ == "__main__":
    main() 