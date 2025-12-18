#!/usr/bin/env python3
"""
Build HIT-UAV dataset - complete pipeline.

This script processes HIT-UAV dataset with quality filters:
1. Extracts and processes the zip file
2. Filters by minimum bbox size (20px for aerial/trap cameras)
3. Maps categories: Person → Human, Car/Bicycle/OtherVehicle → Vehicle
4. Skips DontCare category only
5. Outputs to dataset/thermal/

Usage:
    python3 scripts/build_hit_uav.py
"""

import sys
import shutil
from pathlib import Path
from PIL import Image

from utils import extract_zip, AnnotationCollector


def parse_yolo_label(label_file, image_width, image_height, min_bbox_size=20):
    """Parse YOLO format label file and return valid bounding boxes in COCO format."""
    valid_boxes = []
    
    if not label_file.exists():
        return valid_boxes
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                
                bbox_width = width_norm * image_width
                bbox_height = height_norm * image_height
                min_dimension = min(bbox_width, bbox_height)
                
                if min_dimension >= min_bbox_size:
                    if class_id == 4:  # Skip DontCare (4) only, include Bicycle (2) in vehicles
                        continue
                    # Convert YOLO (center, normalized) to COCO (top-left, absolute)
                    # YOLO: (x_center_norm, y_center_norm, width_norm, height_norm) where values are 0-1
                    # COCO: [x, y, width, height] where x,y are top-left corner in absolute pixels
                    x = (x_center - width_norm / 2) * image_width
                    y = (y_center - height_norm / 2) * image_height
                    # Ensure non-negative coordinates (handle edge cases)
                    x = max(0, x)
                    y = max(0, y)
                    valid_boxes.append({
                        'class_id': class_id,
                        'bbox': [x, y, bbox_width, bbox_height],  # COCO format: [x, y, width, height]
                        'area': bbox_width * bbox_height
                    })
            except (ValueError, IndexError):
                continue
    
    return valid_boxes


def process_hit_uav(zip_path, output_dir, min_bbox_size=20, ann_collector=None):
    """Process HIT-UAV dataset and extract filtered images."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    
    temp_dir = output_dir.parent / 'temp_hit_uav_extract'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        extract_zip(zip_path, temp_dir)
        
        dataset_dir = temp_dir / 'hit-uav'
        if not dataset_dir.exists():
            possible_paths = list(temp_dir.glob('*hit*'))
            if possible_paths:
                dataset_dir = possible_paths[0]
            else:
                raise FileNotFoundError(f"Could not find dataset directory in {temp_dir}")
        
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError("images/ or labels/ directory not found in dataset")
        
        output_human = output_dir / 'thermal' / 'human'
        output_vehicle = output_dir / 'thermal' / 'vehicle'
        output_human.mkdir(parents=True, exist_ok=True)
        output_vehicle.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'human': {'copied': 0, 'filtered': 0},
            'vehicle': {'copied': 0, 'filtered': 0},
            'too_small': 0
        }
        
        for split in ['train', 'val', 'test']:
            split_images_dir = images_dir / split
            split_labels_dir = labels_dir / split
            
            if not split_images_dir.exists():
                print(f"  Skipping {split} (images directory not found)")
                continue
            
            print(f"\nProcessing {split} split...")
            image_files = list(split_images_dir.glob('*.jpg'))
            print(f"  Found {len(image_files)} images")
            
            for img_file in image_files:
                label_file = split_labels_dir / (img_file.stem + '.txt')
                
                try:
                    img = Image.open(img_file)
                    img_width, img_height = img.size
                except Exception as e:
                    print(f"  Warning: Could not open {img_file.name}: {e}")
                    continue
                
                valid_boxes = parse_yolo_label(label_file, img_width, img_height, min_bbox_size)
                
                if not valid_boxes:
                    stats['too_small'] += 1
                    continue
                
                classes_in_image = set(box['class_id'] for box in valid_boxes)
                has_human = 0 in classes_in_image
                has_vehicle = 1 in classes_in_image or 2 in classes_in_image or 3 in classes_in_image  # Include bicycle (2) in vehicles
                
                if has_human:
                    dest = output_human / img_file.name
                    shutil.copy2(img_file, dest)
                    stats['human']['copied'] += 1
                    stats['human']['filtered'] += sum(1 for b in valid_boxes if b['class_id'] == 0)
                    
                    # Add to annotations
                    if ann_collector:
                        human_bboxes = [{'category': 'human', **b} for b in valid_boxes if b['class_id'] == 0]
                        rel_path = Path('thermal') / 'human' / dest.name
                        ann_collector.add_image(rel_path, 'thermal', 'human', img_width, img_height, human_bboxes)
                
                if has_vehicle:
                    dest = output_vehicle / img_file.name
                    shutil.copy2(img_file, dest)
                    stats['vehicle']['copied'] += 1
                    stats['vehicle']['filtered'] += sum(1 for b in valid_boxes if b['class_id'] in [1, 2, 3])  # Include bicycle (2)
                    
                    # Add to annotations
                    if ann_collector:
                        vehicle_bboxes = [{'category': 'vehicle', **b} for b in valid_boxes if b['class_id'] in [1, 2, 3]]  # Include bicycle (2)
                        rel_path = Path('thermal') / 'vehicle' / dest.name
                        ann_collector.add_image(rel_path, 'thermal', 'vehicle', img_width, img_height, vehicle_bboxes)
        
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"\nHuman category:")
        print(f"  Images with valid human boxes: {stats['human']['copied']}")
        print(f"  Total human boxes (≥{min_bbox_size}px): {stats['human']['filtered']}")
        
        print(f"\nVehicle category:")
        print(f"  Images with valid vehicle boxes: {stats['vehicle']['copied']}")
        print(f"  Total vehicle boxes (≥{min_bbox_size}px): {stats['vehicle']['filtered']}")
        
        print(f"\nFiltered out:")
        print(f"  Images with only small boxes (<{min_bbox_size}px): {stats['too_small']}")
        print("=" * 60)
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary extraction directory.")


def main():
    """Build HIT-UAV dataset with complete pipeline."""
    print("=" * 70)
    print("HIT-UAV Dataset Builder")
    print("=" * 70)
    
    zip_file = 'zips/hit_uav.zip'
    output_dir = Path('dataset')
    min_bbox_size = 20  # Supports aerial/trap cameras
    
    if not Path(zip_file).exists():
        print(f"ERROR: Zip file not found: {zip_file}")
        return 1
    
    print(f"\nConfiguration:")
    print(f"  Zip file: {zip_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Min bbox size: {min_bbox_size}x{min_bbox_size} pixels")
    print("=" * 70)
    
    # Initialize annotation collector
    ann_collector = AnnotationCollector(output_dir / 'annotations.json')
    
    print("\nProcessing HIT-UAV dataset...")
    process_hit_uav(zip_file, output_dir, min_bbox_size, ann_collector)
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    
    human_count = len(list((output_dir / 'thermal' / 'human').glob('*.jpg'))) if (output_dir / 'thermal' / 'human').exists() else 0
    vehicle_count = len(list((output_dir / 'thermal' / 'vehicle').glob('*.jpg'))) if (output_dir / 'thermal' / 'vehicle').exists() else 0
    
    print(f"\nFinal counts:")
    print(f"  Human: {human_count} images")
    print(f"  Vehicle: {vehicle_count} images")
    print(f"  Total: {human_count + vehicle_count} images")
    print(f"\nOutput location: {output_dir}/thermal/human/ and {output_dir}/thermal/vehicle/")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
