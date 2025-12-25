# BioWatch Multi-Modal Object Detection Model

⚠️ **NOTE: This README describes an older architecture (Faster R-CNN-based).**

**Current implementation uses YOLOv8 with 6-channel early fusion. See main `README.md` for current training instructions.**

## Architecture (Legacy)

The model supports three fusion strategies:

1. **Early Fusion**: Concatenate RGB and thermal channels at the input (4 channels total)
2. **Feature Fusion**: Process RGB and thermal separately, then fuse at feature level using attention
3. **Late Fusion**: Process separately and fuse at the detection head (not yet fully implemented)

## Features

- ✅ Supports RGB-only images
- ✅ Supports thermal-only images  
- ✅ Supports RGB+thermal paired images (when available)
- ✅ Flexible fusion strategies
- ✅ COCO format annotations
- ✅ Data augmentation
- ✅ Checkpoint saving/loading

## Usage

### Training

```bash
# Basic training with default settings
python models/train.py

# Train with specific fusion method
python models/train.py --fusion-method feature --backbone resnet50

# Train only on RGB images
python models/train.py --use-rgb --no-use-thermal

# Train only on paired RGB+thermal images
python models/train.py --require-both

# Resume from checkpoint
python models/train.py --resume checkpoints/best_model_epoch_10.pth
```

### Configuration

Edit `models/config.py` or use command-line arguments to customize:
- Backbone architecture (resnet50, resnet101)
- Fusion method (early, late, feature)
- Batch size, learning rate, epochs
- Input image size
- Modality settings

## Dataset Structure

The model expects:
- `dataset/annotations.json` - COCO format annotations
- `dataset/rgb/{category}/{image}.jpg` - RGB images
- `dataset/thermal/{category}/{image}.jpg` - Thermal images

## Model Architecture

```
Input (RGB and/or Thermal)
    ↓
Backbone (ResNet50/101)
    ↓
Feature Extraction
    ↓
[Fusion Module] (if both modalities available)
    ↓
Detection Head (Faster R-CNN)
    ↓
Output (boxes, labels, scores)
```

## Next Steps

1. **Complete Detection Head**: Integrate with torchvision's Faster R-CNN or implement custom detection head
2. **Loss Function**: Implement proper detection loss (classification + bbox regression)
3. **Evaluation Metrics**: Add mAP calculation
4. **Inference Script**: Create script for running inference on new images
5. **Visualization**: Add visualization tools for predictions

## Notes

- The current implementation is a framework/skeleton
- Detection head integration is pending (currently returns features as placeholder)
- Loss computation needs to be implemented
- Model can handle missing modalities gracefully
