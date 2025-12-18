#!/usr/bin/env python3
"""
Build PASCAL VOC 2012 dataset - complete pipeline.

This script processes PASCAL VOC 2012 dataset with quality filters:
1. Extracts zip file
2. Parses XML annotations to extract bounding boxes
3. Filters by category (person, animals, vehicles)
4. Applies quality filters (difficult, truncated, min bbox size 50px)
5. Excludes boat and aeroplane from vehicles
6. Copies images to dataset/rgb/{human,animal,vehicle}/

Usage:
    python3 scripts/build_pascal_voc.py [--max-images-per-category N]
"""

import sys
import xml.etree.ElementTree as ET
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

from utils import extract_zip, AnnotationCollector


# Category mappings
ANIMAL_CATEGORIES = {'cat', 'dog', 'bird', 'cow', 'horse', 'sheep'}
VEHICLE_CATEGORIES = {'car', 'bus', 'bicycle', 'motorbike', 'train'}  # Excluding boat and aeroplane
HUMAN_CATEGORIES = {'person'}


def parse_xml_annotation(xml_path):
    """
    Parse PASCAL VOC XML annotation file.
    
    Returns:
        dict with 'objects' list containing {name, bbox, difficult, truncated}
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text) if size is not None else 0
        img_height = int(size.find('height').text) if size is not None else 0
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            truncated = int(obj.find('truncated').text) if obj.find('truncated') is not None else 0
            
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin
                
                objects.append({
                    'name': name,
                    'bbox': (xmin, ymin, bbox_w, bbox_h),
                    'difficult': difficult,
                    'truncated': truncated,
                    'min_dim': min(bbox_w, bbox_h)
                })
        
        return {
            'width': img_width,
            'height': img_height,
            'objects': objects
        }
    except Exception as e:
        print(f"  Warning: Could not parse {xml_path.name}: {e}")
        return {'width': 0, 'height': 0, 'objects': []}


def get_valid_objects(annotation, min_bbox_size=20, filter_difficult=True, filter_truncated=False):
    """
    Filter objects based on quality criteria.
    
    Returns:
        dict with 'human', 'animal', 'vehicle' keys, each containing list of valid objects
    """
    valid = {
        'human': [],
        'animal': [],
        'vehicle': []
    }
    
    for obj in annotation['objects']:
        # Filter by difficulty
        if filter_difficult and obj['difficult'] == 1:
            continue
        
        # Filter by truncated
        if filter_truncated and obj['truncated'] == 1:
            continue
        
        # Filter by minimum size
        if obj['min_dim'] < min_bbox_size:
            continue
        
        # Categorize
        name = obj['name'].lower()
        if name in HUMAN_CATEGORIES:
            valid['human'].append(obj)
        elif name in ANIMAL_CATEGORIES:
            valid['animal'].append(obj)
        elif name in VEHICLE_CATEGORIES:
            valid['vehicle'].append(obj)
    
    return valid


def process_pascal_voc(zip_path, output_dir, min_bbox_size=20, filter_difficult=True, 
                       filter_truncated=False, max_images_per_category=None):
    """Process PASCAL VOC 2012 dataset."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    
    # Temporary extraction directory
    extract_dir = output_dir.parent / 'temp_pascal_extract'
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract zip
        print(f"\n[Step 1/3] Extracting PASCAL VOC 2012 zip...")
        extract_zip(zip_path, extract_dir)
        
        # Find dataset directories (nested structure)
        train_val_dir = None
        test_dir = None
        
        # Look for the actual dataset directory (may be nested)
        for d in extract_dir.rglob('VOC2012_train_val'):
            if d.is_dir() and (d / 'Annotations').exists():
                train_val_dir = d
                break
            # Also check nested structure
            for subd in d.rglob('VOC2012_train_val'):
                if subd.is_dir() and (subd / 'Annotations').exists():
                    train_val_dir = subd
                    break
        
        for d in extract_dir.rglob('VOC2012_test'):
            if d.is_dir() and (d / 'Annotations').exists():
                test_dir = d
                break
            for subd in d.rglob('VOC2012_test'):
                if subd.is_dir() and (subd / 'Annotations').exists():
                    test_dir = subd
                    break
        
        if not train_val_dir:
            print(f"Error: Could not find VOC2012_train_val directory with Annotations")
            # Try to find any directory with Annotations
            possible_dirs = list(extract_dir.rglob('Annotations'))
            if possible_dirs:
                train_val_dir = possible_dirs[0].parent
                print(f"  Found alternative: {train_val_dir}")
            else:
                return None
        
        # Create output directories
        output_human = output_dir / 'rgb' / 'human'
        output_animal = output_dir / 'rgb' / 'animal'
        output_vehicle = output_dir / 'rgb' / 'vehicle'
        
        for d in [output_human, output_animal, output_vehicle]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize annotation collector
        ann_collector = AnnotationCollector(output_dir / 'annotations.json')
        
        stats = {
            'human': {'total': 0, 'copied': 0, 'filtered': 0, 'annotated': 0},
            'animal': {'total': 0, 'copied': 0, 'filtered': 0, 'annotated': 0},
            'vehicle': {'total': 0, 'copied': 0, 'filtered': 0, 'annotated': 0}
        }
        
        # Process train+val set
        print(f"\n[Step 2/3] Processing annotations...")
        annotations_dir = train_val_dir / 'Annotations'
        images_dir = train_val_dir / 'JPEGImages'
        
        if not annotations_dir.exists() or not images_dir.exists():
            print(f"Error: Annotations or JPEGImages directory not found")
            return None
        
        xml_files = list(annotations_dir.glob('*.xml'))
        print(f"  Found {len(xml_files)} annotation files")
        
        # Track images per category to limit if needed
        images_by_category = {
            'human': set(),
            'animal': set(),
            'vehicle': set()
        }
        
        print(f"\n[Step 3/3] Processing images...")
        for xml_file in xml_files:
            annotation = parse_xml_annotation(xml_file)
            valid_objects = get_valid_objects(annotation, min_bbox_size, filter_difficult, filter_truncated)
            
            # Get corresponding image
            image_name = xml_file.stem + '.jpg'
            image_path = images_dir / image_name
            
            if not image_path.exists():
                continue
            
            # Copy image to appropriate categories
            for category in ['human', 'animal', 'vehicle']:
                if valid_objects[category]:
                    # Check if we've reached the limit for this category
                    if max_images_per_category and len(images_by_category[category]) >= max_images_per_category:
                        continue
                    
                    stats[category]['total'] += 1
                    images_by_category[category].add(image_name)
                    
                    dest = output_dir / 'rgb' / category / image_name
                    
                    # Handle duplicates - find available filename
                    if dest.exists():
                        stem = dest.stem
                        suffix = dest.suffix
                        counter = 1
                        while dest.exists():
                            dest = output_dir / 'rgb' / category / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    # Copy image
                    shutil.copy2(image_path, dest)
                    stats[category]['copied'] += 1
                    
                    # Add to annotations (convert bbox format to COCO: [x, y, w, h])
                    bboxes = []
                    for obj in valid_objects[category]:
                        xmin, ymin, bbox_w, bbox_h = obj['bbox']
                        bboxes.append({
                            'category': category,
                            'bbox': [xmin, ymin, bbox_w, bbox_h],  # COCO format
                            'area': bbox_w * bbox_h
                        })
                    
                    # Get image dimensions
                    from PIL import Image
                    with Image.open(image_path) as img:
                        width, height = img.size
                    
                    # Add to annotation collector (use final dest.name which may have _N suffix)
                    # IMPORTANT: Always add annotation after copying to ensure consistency
                    rel_path = Path('rgb') / category / dest.name
                    try:
                        ann_collector.add_image(rel_path, 'rgb', category, width, height, bboxes)
                        stats[category]['annotated'] = stats[category].get('annotated', 0) + 1
                    except Exception as e:
                        print(f"  Warning: Failed to add annotation for {dest.name}: {e}")
                    
                    stats[category]['filtered'] += len(valid_objects[category])
            
            if (stats['human']['total'] + stats['animal']['total'] + stats['vehicle']['total']) % 500 == 0:
                print(f"  Processed {stats['human']['total'] + stats['animal']['total'] + stats['vehicle']['total']} images...")
        
        # Save annotations
        print(f"\nSaving annotations...")
        ann_collector.save()
        
        # Verification step: Ensure all copied images have annotations
        print(f"\nVerifying annotations...")
        for category in ['human', 'animal', 'vehicle']:
            category_dir = output_dir / 'rgb' / category
            if category_dir.exists():
                # Get all PASCAL files (start with 20XX_)
                pascal_files = [f for f in category_dir.glob('20*.jpg')]
                annotated_files = {Path(img['file_name']).name for img in ann_collector.images 
                                  if f'rgb/{category}' in img['file_name']}
                missing = [f for f in pascal_files if f.name not in annotated_files]
                
                if missing:
                    print(f"  Warning: {len(missing)} {category} images missing annotations, fixing...")
                    # Re-process missing files
                    fixed_count = 0
                    for img_file in missing:
                        # Extract base name (remove _N suffix if present)
                        base_name = img_file.stem
                        if '_' in base_name:
                            parts = base_name.split('_')
                            if len(parts) >= 3 and parts[-1].isdigit():
                                base_name = '_'.join(parts[:-1])
                        
                        xml_file = annotations_dir / f"{base_name}.xml"
                        if not xml_file.exists():
                            continue
                        
                        annotation = parse_xml_annotation(xml_file)
                        if not annotation:
                            continue
                        
                        valid_objects = get_valid_objects(annotation, min_bbox_size, filter_difficult, filter_truncated)
                        if not valid_objects[category]:
                            continue
                        
                        # Convert to COCO format bboxes
                        bboxes = []
                        for obj in valid_objects[category]:
                            xmin, ymin, bbox_w, bbox_h = obj['bbox']
                            bboxes.append({
                                'category': category,
                                'bbox': [xmin, ymin, bbox_w, bbox_h],
                                'area': bbox_w * bbox_h
                            })
                        
                        # Get image dimensions
                        from PIL import Image
                        try:
                            with Image.open(img_file) as img:
                                width, height = img.size
                        except:
                            continue
                        
                        # Add annotation
                        rel_path = Path('rgb') / category / img_file.name
                        ann_collector.add_image(rel_path, 'rgb', category, width, height, bboxes)
                        fixed_count += 1
                    
                    if fixed_count > 0:
                        print(f"  ✓ Fixed {fixed_count} missing {category} annotations")
                        # Re-save annotations
                        ann_collector.save()
                else:
                    print(f"  ✓ All {len(pascal_files)} {category} images have annotations")
        
        return stats
        
    finally:
        # Cleanup
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            print(f"\nCleaned up temporary extraction directory.")


def main():
    """Build PASCAL VOC 2012 dataset with complete pipeline."""
    print("=" * 70)
    print("PASCAL VOC 2012 Dataset Builder")
    print("=" * 70)
    
    zip_file = Path('zips/PASCAL_VOC_2012.zip')
    output_dir = Path('dataset')
    
    filter_difficult = True  # Exclude: ambiguous/hard cases that confuse training
    filter_truncated = False  # Include: partial objects are common in real scenarios (aerial/trap cameras)
    
    import argparse
    parser = argparse.ArgumentParser(description='Build PASCAL VOC 2012 dataset')
    parser.add_argument('--max-images-per-category', type=int, default=None,
                       help='Maximum images per category (default: all available)')
    parser.add_argument('--min-bbox-size', type=int, default=20,
                       help='Minimum bounding box size in pixels (default: 20, supports aerial/trap cameras)')
    
    args = parser.parse_args()
    min_bbox_size = args.min_bbox_size
    
    if not zip_file.exists():
        print(f"ERROR: Zip file not found: {zip_file}")
        return 1
    
    print(f"\nConfiguration:")
    print(f"  Zip file: {zip_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  Min bbox size: {min_bbox_size}x{min_bbox_size} pixels")
    print(f"  Filter difficult objects: {filter_difficult} (exclude ambiguous cases)")
    print(f"  Filter truncated objects: {filter_truncated} (include partial objects for real-world scenarios)")
    print(f"  Animal categories: {', '.join(sorted(ANIMAL_CATEGORIES))}")
    print(f"  Vehicle categories: {', '.join(sorted(VEHICLE_CATEGORIES))} (excluding boat, aeroplane)")
    if args.max_images_per_category:
        print(f"  Max images per category: {args.max_images_per_category}")
    print("=" * 70)
    
    stats = process_pascal_voc(
        zip_file, output_dir, min_bbox_size, filter_difficult, 
        filter_truncated, args.max_images_per_category
    )
    
    if not stats:
        return 1
    
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    
    print(f"\nHuman category:")
    print(f"  Images with valid human boxes: {stats['human']['copied']}")
    print(f"  Total human boxes (≥{min_bbox_size}px): {stats['human']['filtered']}")
    
    print(f"\nAnimal category:")
    print(f"  Images with valid animal boxes: {stats['animal']['copied']}")
    print(f"  Total animal boxes (≥{min_bbox_size}px): {stats['animal']['filtered']}")
    
    print(f"\nVehicle category:")
    print(f"  Images with valid vehicle boxes: {stats['vehicle']['copied']}")
    print(f"  Total vehicle boxes (≥{min_bbox_size}px): {stats['vehicle']['filtered']}")
    
    human_count = len(list((output_dir / 'rgb' / 'human').glob('*.jpg'))) if (output_dir / 'rgb' / 'human').exists() else 0
    animal_count = len(list((output_dir / 'rgb' / 'animal').glob('*.jpg'))) + len(list((output_dir / 'rgb' / 'animal').glob('*.JPG'))) if (output_dir / 'rgb' / 'animal').exists() else 0
    vehicle_count = len(list((output_dir / 'rgb' / 'vehicle').glob('*.jpg'))) if (output_dir / 'rgb' / 'vehicle').exists() else 0
    
    print(f"\nFinal counts in dataset:")
    print(f"  Human: {human_count} images")
    print(f"  Animal: {animal_count} images")
    print(f"  Vehicle: {vehicle_count} images")
    print(f"  Total: {human_count + animal_count + vehicle_count} images")
    print(f"\nOutput location: {output_dir}/rgb/human/, {output_dir}/rgb/animal/, {output_dir}/rgb/vehicle/")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

