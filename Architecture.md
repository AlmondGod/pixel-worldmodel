# World Model Architecture (B&W Small-Scale Version)

## Overview

- Input Resolution: 64x64 pixels
- Color Format: Binary (0/1 black and white)
- Source Video: 30 FPS, downsampled to 10 FPS in 3 staggered ways
- Sequence Length: 16 frames
- Dataset Size: 1 hour = 108,000 frames = 6,750 sequences of 16 frames

## Dataset Generation
From 30 FPS source video, create 3 staggered sets of 10 FPS sequences:
1. Frames [0, 3, 6, 9, ...] 
2. Frames [1, 4, 7, 10, ...]
3. Frames [2, 5, 8, 11, ...]

## Dataset Split
Total sequences: 6,750
- Training: 6,000 sequences
- Validation: 450 sequences
- Test: 300 sequences
Note: Ensure splits don't contain overlapping time periods

## 1. Video Tokenizer (VQ-VAE with ST-Transformer)

### Architecture:
- **Encoder**:
 - Patch Size: 4x4 (resulting in 16x16 patches)
 - Layers: 4 
 - Model Dimension: 256
 - Attention Heads: 4
 - Type: ST-Transformer (Spatial-Temporal)

- **Decoder**:
 - Layers: 4
 - Model Dimension: 256
 - Attention Heads: 4

- **Codebook**:
 - Number of Codes: 32 (reduced due to binary data)
 - Embedding Size: 16
 - Latent Dimension: 16

### Training Details:
- Optimizer: AdamW
- Learning Rate: 3e-4 → 3e-4 (cosine decay)
- Betas: (0.9, 0.9)
- Weight Decay: 1e-4
- Warmup Steps: 2k (adjusted for larger dataset)
- Training Steps: 50k (adjusted for larger dataset)

## 2. Latent Action Model (LAM)

### Architecture:
- **Encoder**:
 - Patch Size: 8x8
 - Layers: 4
 - Model Dimension: 256
 - Attention Heads: 4
 - Type: ST-Transformer

- **Decoder**:
 - Layers: 4
 - Model Dimension: 256
 - Attention Heads: 4

- **Action Codebook**:
 - Number of Actions: 8 (for human playability)
 - Embedding Size: 16
 - Latent Dimension: 16

## 3. Dynamics Model (MaskGIT Transformer)

### Architecture:
- Type: Decoder-only MaskGIT transformer
- Layers: 6
- Model Dimension: 256
- Number of Layers: 4

### Training Details:
- Temperature: 1.0
- MaskGIT Steps: 10
- Random masking rate: 0.5 to 1.0 (Bernoulli distribution)
- Actions treated as additive embeddings

## Training Process

1. Train video tokenizer first
  - Train on all frames from training set
  - Validate on validation set sequences

2. Freeze tokenizer and co-train LAM and dynamics model:
  - LAM trained on raw binary pixels
  - Dynamics model trained on tokenized frames
  - Both use 16-frame sequences

## Memory and Batch Size

- Batch size: 128 sequences (increased due to binary data)
- Sequence length: 16 frames
- Total images per batch: 2048
- Should fit in 16GB memory due to:
 - Binary pixel values (1 bit per pixel)
 - 64x64 resolution
 - Reduced model dimensions

## Inference Pipeline

1. Initial frame x₁ (64x64 binary) is tokenized to z₁
2. User selects discrete action a₁ (0-7)
3. Action is embedded via VQ codebook
4. Dynamics model predicts next frame tokens
5. Tokens are decoded to binary pixel space
6. Process repeats for next frame

### Inference Parameters:
- MaskGIT sampling steps: 10
- Temperature: 1.0
- Uses random sampling