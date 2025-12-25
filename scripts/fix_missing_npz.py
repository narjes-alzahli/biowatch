#!/usr/bin/env python3
"""
Fix missing .npz files for validation images that have labels but no .npz files.
This script identifies which val labels don't have corresponding .npz files
and creates them by processing the source images.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.image_preprocessing import create_6channel_image

def find_image_paths(file_name, dataset_root):
    """
    Find RGB and thermal paths for an image file name.
    Uses the same logic as prepare_yolo_dataset.py.
    Returns (rgb_path, thermal_path) tuple.
    """
    dataset_root = Path(dataset_root)
    rgb_path = None
    thermal_path = None
    
    # Use same logic as prepare_yolo_dataset.py
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
    
    return rgb_path, thermal_path

def create_missing_npz_files(dataset_root, annotations_file, output_dir):
    """
    Find val labels without .npz files and create the missing .npz files.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    annotations_file = Path(annotations_file)
    
    val_img_dir = output_dir / 'images' / 'val'
    val_label_dir = output_dir / 'labels' / 'val'
    
    # Load annotations to get image info
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to image_info mapping
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Find all val labels
    val_labels = list(val_label_dir.glob('*.txt'))
    print(f"Found {len(val_labels):,} validation label files")
    
    # Find labels without corresponding .npz files
    missing_npz = []
    for label_file in val_labels:
        label_stem = label_file.stem
        npz_path = val_img_dir / f"{label_stem}.npz"
        
        if not npz_path.exists():
            missing_npz.append(label_file)
    
    print(f"\nFound {len(missing_npz):,} labels without .npz files")
    
    if len(missing_npz) == 0:
        print("✅ All labels have corresponding .npz files!")
        return
    
    print(f"\nProcessing {len(missing_npz):,} missing .npz files...")
    
    # Find the corresponding image info from annotations
    # We need to match label filename to image file_name in annotations
    processed = 0
    failed = 0
    
    for label_file in tqdm(missing_npz, desc="Creating missing .npz"):
        label_stem = label_file.stem
        
        # Try to find the image in annotations by matching filename
        # The label filename should match the image file_name (without extension)
        matched_img_info = None
        for img_info in coco_data['images']:
            img_file_name = Path(img_info['file_name']).stem
            if img_file_name == label_stem:
                matched_img_info = img_info
                break
        
        if not matched_img_info:
            print(f"\n⚠️  Warning: Could not find image info for {label_stem}")
            failed += 1
            continue
        
        # Get the file_name from annotations (includes directory path like 'rgb/human/image.jpg')
        file_name = matched_img_info['file_name']
        
        # Find RGB and thermal paths
        rgb_path, thermal_path = find_image_paths(file_name, dataset_root)
        
        # Determine the output path (use the label filename)
        output_img_path = val_img_dir / f"{label_stem}.jpg"
        npz_path = val_img_dir / f"{label_stem}.npz"
        
        # Create 6-channel image
        try:
            import numpy as np
            from PIL import Image
            
            # Determine modality
            if rgb_path and rgb_path.exists() and thermal_path and thermal_path.exists():
                modality = 'both'
            elif rgb_path and rgb_path.exists():
                modality = 'rgb'
            elif thermal_path and thermal_path.exists():
                modality = 'thermal'
            else:
                print(f"\n⚠️  Warning: No source image found for {label_stem}")
                failed += 1
                continue
            
            # Create 6-channel image with proper preprocessing
            img_6ch, detected_modality = create_6channel_image(
                rgb_path=rgb_path,
                thermal_path=thermal_path,
                modality=modality
            )
            
            # Save as compressed numpy array (.npz) - normalized float32 [0, 1]
            np.savez_compressed(str(npz_path), img=img_6ch)
            
            # Also save a visual representation (first 3 channels) for compatibility
            if not output_img_path.exists():
                visual_rgb = (img_6ch[:, :, :3] * 255.0).astype(np.uint8)
                Image.fromarray(visual_rgb).save(output_img_path)
            
            processed += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {label_stem}: {e}")
            failed += 1
            continue
    
    print(f"\n✅ Processing complete!")
    print(f"  Processed: {processed:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Total: {len(missing_npz):,}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix missing .npz files for validation images')
    parser.add_argument('--dataset-root', type=str, default='dataset',
                        help='Root directory of the dataset (contains rgb/ and thermal/ subdirectories)')
    parser.add_argument('--annotations', type=str, default='dataset/annotations.json',
                        help='Path to COCO format annotations JSON file')
    parser.add_argument('--output', type=str, default='dataset_yolo_6ch',
                        help='Output directory (should be dataset_yolo_6ch)')
    
    args = parser.parse_args()
    
    create_missing_npz_files(args.dataset_root, args.annotations, args.output)

