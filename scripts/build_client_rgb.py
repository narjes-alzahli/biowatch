#!/usr/bin/env python3
"""
Build Client RGB dataset - process client-provided RGB images with annotations.

This script:
1. Extracts client_rgb.zip
2. Loads COCO format annotations
3. Copies images to dataset/rgb/human/ or dataset/rgb/animal/
4. Adds annotations to dataset/annotations.json

Usage:
    python3 scripts/build_client_rgb.py
"""

import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

from utils import extract_zip, AnnotationCollector


def process_client_rgb(zip_path, output_dir, min_bbox_size=20, ann_collector=None):
    """Process Client RGB dataset."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    
    extract_dir = Path("temp_extract") / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    extract_zip(zip_path, extract_dir)
    
    # Find annotation file
    annotation_file = extract_dir / 'annotations' / 'instances_default.json'
    if not annotation_file.exists():
        print(f"Error: Annotation file not found: {annotation_file}")
        return
    
    # Load COCO annotations
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    images_dict = {img['id']: img for img in coco_data.get('images', [])}
    annotations_list = coco_data.get('annotations', [])
    
    print(f"  Found {len(images_dict)} images")
    print(f"  Found {len(annotations_list)} annotations")
    print(f"  Categories: {list(categories.values())}")
    
    # Group annotations by image and category
    image_annotations = defaultdict(lambda: defaultdict(list))
    
    for ann in annotations_list:
        image_id = ann['image_id']
        category_id = ann['category_id']
        category_name = categories.get(category_id, 'unknown')
        
        # Filter by min bbox size
        bbox = ann['bbox']  # COCO format: [x, y, w, h]
        if len(bbox) >= 4:
            w, h = bbox[2], bbox[3]
            if w >= min_bbox_size and h >= min_bbox_size:
                image_annotations[image_id][category_name].append(bbox)
    
    # Find images directory
    images_dir = extract_dir / 'images'
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    stats = {'human': 0, 'animal': 0, 'skipped': 0}
    
    # Process each image
    for image_id, img_info in images_dict.items():
        filename = img_info['file_name']
        img_path = images_dir / filename
        
        if not img_path.exists():
            stats['skipped'] += 1
            continue
        
        # Check which categories this image has
        has_animal = 'animal' in image_annotations[image_id]
        has_human = 'human' in image_annotations[image_id]
        
        if not has_animal and not has_human:
            stats['skipped'] += 1
            continue
        
        width = img_info['width']
        height = img_info['height']
        
        # If image has both categories, copy to both folders and add annotations for both
        # This is standard practice - same image can appear in multiple category folders
        categories_to_process = []
        if has_human:
            categories_to_process.append('human')
        if has_animal:
            categories_to_process.append('animal')
        
        # Copy image to each category folder and add annotations
        for category in categories_to_process:
            bboxes = image_annotations[image_id][category]
            if not bboxes:
                continue
            
            # Copy image to output
            output_path = output_dir / category / filename
            
            # Handle duplicates
            if output_path.exists():
                stem = output_path.stem
                suffix = output_path.suffix
                counter = 1
                while output_path.exists():
                    output_path = output_dir / category / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, output_path)
            stats[category] += 1
            
            # Add annotations for this category
            if ann_collector:
                # Convert bboxes to annotation format
                annotation_bboxes = [
                    {
                        'category': category,
                        'bbox': bbox,  # Already COCO format [x, y, w, h]
                        'area': bbox[2] * bbox[3]
                    }
                    for bbox in bboxes
                ]
                
                rel_path = Path('rgb') / category / output_path.name
                ann_collector.add_image(rel_path, 'rgb', category, width, height, annotation_bboxes)
    
    print(f"\n✓ Human: {stats['human']}, Animal: {stats['animal']}, Skipped: {stats['skipped']}")
    
    # Clean up
    shutil.rmtree(extract_dir.parent, ignore_errors=True)


def main():
    """Build Client RGB dataset."""
    print("=" * 70)
    print("Client RGB Dataset Builder")
    print("=" * 70)
    
    zip_file = Path('zips/client_rgb.zip')
    output_dir = Path('dataset/rgb')
    min_bbox_size = 20  # Supports aerial/trap cameras
    
    if not zip_file.exists():
        print(f"ERROR: Zip file not found: {zip_file}")
        return 1
    
    print(f"\nConfiguration:")
    print(f"  Zip file: {zip_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Min bbox size: {min_bbox_size}x{min_bbox_size} pixels")
    print("=" * 70)
    
    # Initialize annotation collector
    ann_collector = AnnotationCollector(Path('dataset') / 'annotations.json')
    
    print("\nProcessing Client RGB dataset...")
    process_client_rgb(zip_file, output_dir, min_bbox_size, ann_collector)
    
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    
    human_count = len(list((output_dir / 'human').glob('*.jpg'))) if (output_dir / 'human').exists() else 0
    animal_count = len(list((output_dir / 'animal').glob('*.jpg'))) if (output_dir / 'animal').exists() else 0
    
    print(f"\nFinal counts:")
    print(f"  Human: {human_count} images")
    print(f"  Animal: {animal_count} images")
    print(f"  Total: {human_count + animal_count} images")
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
