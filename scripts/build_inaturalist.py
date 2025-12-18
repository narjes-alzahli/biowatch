#!/usr/bin/env python3
"""
Build iNaturalist dataset - Mammalia only.

This script:
1. Extracts JSON annotations from inaturalisti_val.json.tar.gz
2. Filters for Mammalia class (mammals only)
3. Extracts corresponding images from inaturalist_val.tar.gz
4. Copies images to dataset/rgb/animal/
5. Limits number of images to what's needed (configurable)

Usage:
    python3 scripts/build_inaturalist.py [--max-images N]
"""

import sys
import json
import tarfile
import shutil
from pathlib import Path
from collections import defaultdict

from utils import AnnotationCollector

def extract_tar(tar_path, extract_to):
    """Extract tar.gz file to directory."""
    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


def load_annotations(json_path):
    """Load and parse COCO-format JSON annotations."""
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data.get('images', []))} images, {len(data.get('categories', []))} categories")
    return data


def get_mammalia_categories(categories):
    """Filter categories to only Mammalia (mammals)."""
    mammalia_categories = {}
    
    for cat in categories:
        # Check if it's Animalia and Mammalia
        if cat.get('kingdom') == 'Animalia' and cat.get('class') == 'Mammalia':
            mammalia_categories[cat['id']] = cat
    
    print(f"✓ Found {len(mammalia_categories)} Mammalia categories")
    return mammalia_categories


def get_mammalia_images(annotations_data, mammalia_categories):
    """Get all images that belong to Mammalia categories."""
    mammalia_category_ids = set(mammalia_categories.keys())
    
    # Map image_id to category_id via annotations
    image_to_category = {}
    for ann in annotations_data.get('annotations', []):
        if ann['category_id'] in mammalia_category_ids:
            image_to_category[ann['image_id']] = ann['category_id']
    
    # Get image info for mammalia images
    mammalia_images = {}
    for img in annotations_data.get('images', []):
        if img['id'] in image_to_category:
            mammalia_images[img['id']] = {
                'file_name': img['file_name'],
                'category_id': image_to_category[img['id']],
                'width': img.get('width', 0),
                'height': img.get('height', 0)
            }
    
    print(f"✓ Found {len(mammalia_images)} Mammalia images")
    return mammalia_images


def process_inaturalist(json_tar_path, images_tar_path, output_dir, max_images=None, ann_collector=None, min_bbox_size=20):
    """Process iNaturalist dataset - extract Mammalia images only."""
    json_tar_path = Path(json_tar_path)
    images_tar_path = Path(images_tar_path)
    output_dir = Path(output_dir)
    
    # Temporary extraction directories
    temp_dir = output_dir.parent / 'temp_inaturalist_extract'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Extract JSON annotations
        print(f"\n[Step 1/3] Extracting JSON annotations...")
        extract_tar(json_tar_path, temp_dir)
        
        # Find val.json file
        json_file = None
        for f in temp_dir.rglob('val.json'):
            json_file = f
            break
        
        if not json_file:
            print(f"Error: Could not find val.json in extracted files")
            return None
        
        # Step 2: Load and filter annotations
        print(f"\n[Step 2/3] Processing annotations...")
        annotations_data = load_annotations(json_file)
        
        mammalia_categories = get_mammalia_categories(annotations_data.get('categories', []))
        mammalia_images = get_mammalia_images(annotations_data, mammalia_categories)
        
        if not mammalia_images:
            print("No Mammalia images found!")
            return None
        
        # Limit number of images if specified
        if max_images and len(mammalia_images) > max_images:
            # Take first N images (or could randomize)
            image_items = list(mammalia_images.items())[:max_images]
            mammalia_images = dict(image_items)
            print(f"✓ Limited to {max_images} images")
        
        # Step 3: Extract and copy images
        print(f"\n[Step 3/3] Extracting and copying images...")
        output_animal = output_dir / 'rgb' / 'animal'
        output_animal.mkdir(parents=True, exist_ok=True)
        
        # Extract images tar
        images_extract_dir = temp_dir / 'images_extract'
        images_extract_dir.mkdir(exist_ok=True)
        extract_tar(images_tar_path, images_extract_dir)
        
        # Find val directory
        val_dir = None
        for d in images_extract_dir.rglob('val'):
            if d.is_dir():
                val_dir = d
                break
        
        if not val_dir:
            print(f"Error: Could not find val/ directory in extracted images")
            return None
        
        # Copy images
        stats = {'copied': 0, 'not_found': 0}
        
        for img_id, img_info in mammalia_images.items():
            file_name = img_info['file_name']
            # Remove 'val/' prefix if present
            if file_name.startswith('val/'):
                file_name = file_name[4:]
            
            source_path = val_dir / file_name
            
            if not source_path.exists():
                stats['not_found'] += 1
                continue
            
            # Copy to output (keep UUID filename)
            dest = output_animal / source_path.name
            
            # Handle duplicates
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = output_animal / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(source_path, dest)
            stats['copied'] += 1
            
            # Add to annotations
            if ann_collector:
                # Get annotations for this image from COCO data
                img_anns = []
                for ann in annotations_data.get('annotations', []):
                    if ann['image_id'] == img_id:
                        # iNaturalist uses COCO format, bbox is already [x, y, w, h]
                        if 'bbox' not in ann:
                            continue  # Skip annotations without bbox
                        bbox = ann['bbox']
                        if len(bbox) < 4:
                            continue  # Skip invalid bboxes
                        # Filter by minimum bbox size (20px for aerial/trap cameras)
                        if min(bbox[2], bbox[3]) < min_bbox_size:
                            continue
                        img_anns.append({
                            'category': 'animal',  # All mammalia mapped to animal
                            'bbox': bbox,
                            'area': ann.get('area', bbox[2] * bbox[3])
                        })
                
                if img_anns:
                    # Get image dimensions
                    from PIL import Image
                    with Image.open(source_path) as img:
                        width, height = img.size
                    
                    rel_path = Path('rgb') / 'animal' / dest.name
                    ann_collector.add_image(rel_path, 'rgb', 'animal', width, height, img_anns)
            
            if stats['copied'] % 100 == 0:
                print(f"  Copied {stats['copied']} images...")
        
        return stats
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary extraction directory.")


def main():
    """Build iNaturalist dataset - Mammalia only."""
    print("=" * 70)
    print("iNaturalist Dataset Builder (Mammalia Only)")
    print("=" * 70)
    
    json_tar = Path('zips/inaturalisti_val.json.tar.gz')
    images_tar = Path('zips/inaturalist_val.tar.gz')
    output_dir = Path('dataset')
    
    # Check current animal count to determine how many we need
    current_animal_count = 0
    animal_dir = output_dir / 'rgb' / 'animal'
    if animal_dir.exists():
        current_animal_count = len(list(animal_dir.glob('*.JPG'))) + len(list(animal_dir.glob('*.jpg')))
    
    # Default: add up to 2000 more images (or use --max-images to override)
    import argparse
    parser = argparse.ArgumentParser(description='Build iNaturalist dataset - Mammalia only')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to extract (default: all available)')
    parser.add_argument('--target-count', type=int, default=None,
                       help='Target total animal count (will extract enough to reach this)')
    parser.add_argument('--min-bbox-size', type=int, default=20,
                       help='Minimum bounding box size in pixels (default: 20, supports aerial/trap cameras)')
    
    args = parser.parse_args()
    
    max_images = args.max_images
    
    # If target_count is specified, calculate how many we need
    if args.target_count:
        needed = max(0, args.target_count - current_animal_count)
        if needed > 0:
            max_images = needed
            print(f"\nCurrent animal count: {current_animal_count}")
            print(f"Target count: {args.target_count}")
            print(f"Will extract up to {needed} images")
        else:
            print(f"\nAlready have {current_animal_count} images, target is {args.target_count}")
            print("No additional images needed.")
            return 0
    
    if not json_tar.exists():
        print(f"ERROR: JSON tar file not found: {json_tar}")
        return 1
    
    if not images_tar.exists():
        print(f"ERROR: Images tar file not found: {images_tar}")
        return 1
    
    print(f"\nConfiguration:")
    print(f"  JSON tar: {json_tar}")
    print(f"  Images tar: {images_tar}")
    print(f"  Output directory: {output_dir}")
    print(f"  Filter: Mammalia (mammals) only")
    print(f"  Min bbox size: {args.min_bbox_size}x{args.min_bbox_size} pixels")
    if max_images:
        print(f"  Max images: {max_images}")
    else:
        print(f"  Max images: All available")
    print("=" * 70)
    
    # Initialize annotation collector
    ann_collector = AnnotationCollector(output_dir / 'annotations.json')
    
    stats = process_inaturalist(json_tar, images_tar, output_dir, max_images, ann_collector, args.min_bbox_size)
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    if not stats:
        return 1
    
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    
    print(f"\nResults:")
    print(f"  Images copied: {stats['copied']}")
    print(f"  Images not found: {stats['not_found']}")
    
    final_count = len(list((output_dir / 'rgb' / 'animal').glob('*.jpg'))) + len(list((output_dir / 'rgb' / 'animal').glob('*.JPG')))
    print(f"\nTotal animal images in dataset: {final_count}")
    print(f"Output location: {output_dir}/rgb/animal/")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

