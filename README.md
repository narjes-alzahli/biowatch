# BioWatch - Multi-Modal Object Detection (RGB + Thermal)

YOLOv8-based object detection system for detecting humans, animals, and vehicles using RGB and/or thermal cameras.

## Quick Start

### 1. Prepare Dataset
```bash
python scripts/prepare_yolo_dataset.py
```
Creates 6-channel `.npz` files with proper preprocessing from RGB and thermal images.

### 2. Train Model
```bash
python train_yolo.py
```
Or use the batch file:
```bash
run_training_with_log.bat
```

### 3. Evaluate Model
```bash
python scripts/evaluate_field_kpi.py --model saved_models/biowatch_yolov8n_epoch100_best.pt
```

## Directory Structure

```
biowatch/
├── dataset/                    # Original source data (RGB + thermal images)
│   ├── rgb/                   # RGB images by category
│   ├── thermal/               # Thermal images by category
│   └── annotations.json       # COCO format annotations
│
├── dataset_yolo_6ch/          # Prepared dataset for training (6-channel format)
│   ├── images/
│   │   ├── train/            # Training images (.jpg + .npz files)
│   │   └── val/              # Validation images (.jpg + .npz files)
│   └── labels/
│       ├── train/            # Training labels (.txt)
│       └── val/              # Validation labels (.txt)
│
├── saved_models/              # Saved trained models
│   └── biowatch_yolov8n_epoch100_best.pt
│
├── runs/detect/               # Training output (logs, plots, checkpoints)
│   └── biowatch_yolov8n/
│       ├── weights/          # best.pt, last.pt
│       └── results.csv
│
├── train_yolo.py              # Main training script
├── utils/                     # Preprocessing utilities
│   └── image_preprocessing.py
└── scripts/                   # Utility scripts
    ├── prepare_yolo_dataset.py
    └── evaluate_field_kpi.py
```

## Data Preparation

### What It Does
- Loads RGB and thermal images from `dataset/`
- Applies proper preprocessing (normalization, CLAHE for thermal)
- Creates 6-channel images (RGB 3 channels + Thermal 3 channels)
- Saves as compressed `.npz` files in `dataset_yolo_6ch/`

### Running Preparation
```bash
python scripts/prepare_yolo_dataset.py
```

**Output:** `dataset_yolo_6ch/` directory with:
- `.npz` files: 6-channel images (normalized float32 [0,1])
- `.jpg` files: Visual representation (first 3 channels)
- `.txt` files: YOLO format labels

**Note:** The script automatically resumes if interrupted - it skips already processed files.

## Training

### Configuration
- **Model:** YOLOv8n (nano) - optimized for edge deployment
- **Input:** 6 channels (RGB 3 + Thermal 3)
- **Classes:** Human, Animal, Vehicle
- **Epochs:** 100
- **Batch Size:** 64
- **Image Size:** 640x640
- **Class Weights:** Automatically applied (Human: 2.71, Animal: 1.42, Vehicle: 1.00)

### Training
```bash
python train_yolo.py
```

**Output:**
- Best model: `runs/detect/biowatch_yolov8n/weights/best.pt`
- Last model: `runs/detect/biowatch_yolov8n/weights/last.pt`
- Results: `runs/detect/biowatch_yolov8n/results.csv`

### Save Model
After training, copy best model to `saved_models/`:
```bash
copy runs\detect\biowatch_yolov8n\weights\best.pt saved_models\biowatch_yolov8n_epoch100_best.pt
```

## Image Preprocessing

The preprocessing module (`utils/image_preprocessing.py`) ensures:
- **RGB:** Normalized to [0, 1] range
- **Thermal:** CLAHE enhancement + normalization to [0, 1] range
- **6-Channel Creation:** Handles all modalities (RGB-only, thermal-only, or both)

## Evaluation Metrics

### Standard Metrics
- **mAP50-95:** Mean Average Precision at IoU 0.5-0.95 (primary metric)
- **mAP50:** Mean Average Precision at IoU 0.5
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)

### KPI Metrics (Field Testing)
- **Detection Accuracy:** TP / (TP + FP + FN) (target: ≥85%)
- **False Alarm Rate:** FP / Total Non-Threat Scenarios (target: <10%)

## Additional Documentation

- `dataset_info.md` - Detailed dataset statistics and sources
