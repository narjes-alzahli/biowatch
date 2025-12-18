# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Train the Model

```bash
# Train with default settings (RGB + Thermal, feature fusion)
python models/train.py

# Train with early fusion (simpler, faster)
python models/train.py --fusion-method early

# Train only on RGB images
python models/train.py --use-rgb --no-use-thermal

# Train only on thermal images
python models/train.py --no-use-rgb --use-thermal

# Train only on paired RGB+thermal images
python models/train.py --require-both
```

### 2. Model Architecture Options

**Fusion Methods:**
- `early`: Concatenate RGB and thermal at input (4 channels) - simplest
- `feature`: Process separately, fuse at feature level with attention - recommended
- `late`: Process separately, fuse at detection head - most flexible

**Backbones:**
- `resnet50`: Faster, less memory
- `resnet101`: More accurate, more memory

### 3. Training Configuration

Key parameters:
- `--batch-size`: Batch size (default: 16)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--input-size`: Image size as "height width" (default: 640 640)

### 4. Checkpoints

Checkpoints are saved in `checkpoints/`:
- `best_model_epoch_N.pth`: Best model based on validation loss
- `checkpoint_epoch_N.pth`: Periodic checkpoints (every 10 epochs)
- `final_model.pth`: Final model after training

Resume training:
```bash
python models/train.py --resume checkpoints/best_model_epoch_10.pth
```

## Model Capabilities

The unified model can handle:

1. **RGB-only images** (54,545 images in your dataset)
2. **Thermal-only images** (49,783 images in your dataset)
3. **RGB+thermal paired images** (from LLVIP, Caltech Aerial)

The model automatically:
- Detects which modalities are available
- Processes them accordingly
- Fuses features when both are available
- Falls back to single modality when only one is available

## Next Steps

1. **Complete the detection head**: Currently returns features as placeholder
2. **Implement loss function**: Add proper detection loss
3. **Add evaluation**: Implement mAP calculation
4. **Create inference script**: For running on new images

## Troubleshooting

**Out of memory?**
- Reduce `--batch-size` (try 8 or 4)
- Use smaller `--input-size` (try 512 512)
- Use `resnet50` instead of `resnet101`

**Training too slow?**
- Use `--fusion-method early` (simpler)
- Increase `--batch-size` if memory allows
- Use fewer workers in DataLoader

**Want to use only paired data?**
- Use `--require-both` flag
- This will only use images where both RGB and thermal are available
