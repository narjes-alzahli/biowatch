#!/usr/bin/env python3
"""
Prepare dataset for YOLO training by creating 6-channel images (RGB+thermal).
Creates the directory structure YOLO expects: images/train and images/val
Uses proper normalization and preprocessing for both RGB and thermal images.
"""

import json
import shutil
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.image_preprocessing import create_6channel_image

def create_6ch_image(rgb_path, thermal_path, output_path):
    """
    Create 6-channel image using proper preprocessing and normalization.
    Saves as compressed numpy array (.npz).
    """
    # Convert to Path objects if needed
    rgb_path = Path(rgb_path) if rgb_path else None
    thermal_path = Path(thermal_path) if thermal_path else None
    output_path = Path(output_path)
    
    # Skip if neither exists
    if (rgb_path is None or not rgb_path.exists()) and (thermal_path is None or not thermal_path.exists()):
        return False
    
    try:
        # Determine modality
        if rgb_path and rgb_path.exists() and thermal_path and thermal_path.exists():
            modality = 'both'
        elif rgb_path and rgb_path.exists():
            modality = 'rgb'
        elif thermal_path and thermal_path.exists():
            modality = 'thermal'
        else:
            return False
        
        # Create 6-channel image with proper preprocessing
        img_6ch, detected_modality = create_6channel_image(
            rgb_path=rgb_path,
            thermal_path=thermal_path,
            modality=modality
        )
        
        # Save as compressed numpy array (.npz) - normalized float32 [0, 1]
        npz_path = output_path.with_suffix('.npz')
        # Skip if already exists (allows resuming)
        if not npz_path.exists():
            np.savez_compressed(str(npz_path), img=img_6ch)
        
        # Also save a visual representation (first 3 channels) for compatibility
        # Only save .jpg if it doesn't exist (skip if already processed, saves time)
        if not output_path.exists():
            # Convert from normalized [0,1] to uint8 [0,255] for saving
            visual_rgb = (img_6ch[:, :, :3] * 255.0).astype(np.uint8)
            Image.fromarray(visual_rgb).save(output_path)
        
        return True
    except Exception as e:
        print(f"Error processing {output_path}: {e}")
        return False

def prepare_yolo_structure(dataset_root, annotations_file, output_dir):
    """Prepare dataset in YOLO format."""
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    annotations_file = Path(annotations_file)
    
    # Create output directories
    train_img_dir = output_dir / 'images' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    train_label_dir = output_dir / 'labels' / 'train'
    val_label_dir = output_dir / 'labels' / 'val'
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process images
    print("Processing images and creating 6-channel images...")
    train_count = 0
    val_count = 0
    skipped_count = 0
    
    for img_info in tqdm(coco_data['images'], desc="Preparing dataset"):
        img_id = img_info['id']
        file_name = img_info['file_name']
        split = img_info.get('split', 'train')
        
        # Determine directories
        if split == 'train':
            img_dir = train_img_dir
            label_dir = train_label_dir
        elif split == 'val':
            img_dir = val_img_dir
            label_dir = val_label_dir
        else:
            continue
        
        # Find RGB and thermal paths
        rgb_path = None
        thermal_path = None
        
        if 'rgb/' in file_name:
            rgb_path = dataset_root / file_name
            # Try to find corresponding thermal
            thermal_name = file_name.replace('rgb/', 'thermal/')
            thermal_path = dataset_root / thermal_name
        elif 'thermal/' in file_name:
            thermal_path = dataset_root / file_name
            # Try to find corresponding RGB
            rgb_name = file_name.replace('thermal/', 'rgb/')
            rgb_path = dataset_root / rgb_name
        else:
            # Try to infer from path
            img_path = dataset_root / file_name
            if img_path.exists():
                if 'rgb' in str(file_name):
                    rgb_path = img_path
                elif 'thermal' in str(file_name):
                    thermal_path = img_path
        
        # Skip if neither exists
        if (rgb_path is None or not rgb_path.exists()) and (thermal_path is None or not thermal_path.exists()):
            skipped_count += 1
            continue
        
        # Create 6-channel image
        output_img_path = img_dir / Path(file_name).name
        npz_path = output_img_path.with_suffix('.npz')
        
        # Skip if already processed (allows resuming) - check BEFORE processing
        if npz_path.exists():
            skipped_count += 1
            if split == 'train':
                train_count += 1
            else:
                val_count += 1
        else:
            # Process image only if not already done
            if not create_6ch_image(rgb_path, thermal_path, output_img_path):
                skipped_count += 1
                continue
            
            if split == 'train':
                train_count += 1
            else:
                val_count += 1
        
        # Copy/create label file (always do this, even if image was already processed)
        label_file_name = Path(file_name).stem + '.txt'
        label_file = label_dir / label_file_name
        
        # Get annotations
        anns = img_to_anns.get(img_id, [])
        if not anns:
            # Create empty label file if it doesn't exist
            if not label_file.exists():
                label_file.touch()
            continue
        
        # Write YOLO format labels
        img_width = img_info['width']
        img_height = img_info['height']
        
        with open(label_file, 'w') as f:
            for ann in anns:
                # COCO bbox: [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Convert to center, normalized
                x_center = (x + w / 2.0) / img_width
                y_center = (y + h / 2.0) / img_height
                width_norm = w / img_width
                height_norm = h / img_height
                
                # Class ID (0-indexed)
                class_id = ann['category_id'] - 1
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    
    print(f"\n✅ Dataset prepared!")
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output: {output_dir}")
    print(f"\nNote: 6-channel images saved as .npy files alongside .jpg images")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset'),
                       help='Root directory of dataset')
    parser.add_argument('--annotations', type=Path, default=Path('dataset/annotations.json'),
                       help='Path to annotations JSON')
    parser.add_argument('--output', type=Path, default=Path('dataset_yolo_6ch'),
                       help='Output directory for YOLO format dataset with 6-channel images')
    
    args = parser.parse_args()
    
    prepare_yolo_structure(args.dataset_root, args.annotations, args.output)
    print(f"\n✅ Next step: Update biowatch.yaml to point to: {args.output}")

