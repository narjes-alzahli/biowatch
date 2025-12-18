#!/usr/bin/env python3
"""
Build LLVIP (Low-Light Visible-Infrared Paired) dataset.

This script processes LLVIP dataset with quality filters:
1. Extracts zip file
2. Parses XML annotations (PASCAL VOC format)
3. Filters by category (person → human)
4. Applies quality filters (min bbox size 20px for aerial/trap cameras)
5. Filters out images that might contain cars (optional, requires manual review or detection model)
6. Processes both visible (RGB) and infrared (thermal) images
7. Copies images to dataset/rgb/ and dataset/thermal/

Note: LLVIP annotations only include "person" category. Some images may contain
cars in the background (not annotated). Use --filter-cars to exclude images
that might contain cars (requires manual review or external detection model).

Usage:
    python3 scripts/build_llvip.py [--filter-cars] [--min-bbox-size N]
"""

import sys
import xml.etree.ElementTree as ET
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

from utils import extract_zip, AnnotationCollector


def parse_xml_annotation(xml_path, min_bbox_size=20):
    """
    Parse PASCAL VOC XML annotation file.
    
    Returns:
        dict with 'objects' list containing {name, bbox, width, height}
        and 'image_size' tuple (width, height)
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
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            
            name = name_elem.text.lower()
            
            # Only process person/human
            if name != 'person':
                continue
            
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            
            try:
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                width = xmax - xmin
                height = ymax - ymin
                
                # Filter by minimum bbox size
                if min(width, height) < min_bbox_size:
                    continue
                
                objects.append({
                    'name': name,
                    'bbox': (xmin, ymin, xmax, ymax),
                    'width': width,
                    'height': height
                })
            except (ValueError, AttributeError):
                continue
        
        return {
            'objects': objects,
            'image_size': (img_width, img_height),
            'filename': root.find('filename').text if root.find('filename') is not None else None
        }
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None


def has_cars_in_image(image_path, detection_model=None):
    """
    Check if image contains cars/vehicles.
    
    Args:
        image_path: Path to image file
        detection_model: Optional detection model (YOLO, etc.) to detect cars
    
    Returns:
        bool: True if image likely contains cars
    """
    # For now, return False (no car detection)
    # TODO: Add car detection model if needed
    # This could use YOLO or similar to detect cars in images
    return False


def process_llvip(extracted_dir, output_base, min_bbox_size=20, filter_cars=False, max_images_per_category=None, ann_collector=None):
    """
    Process LLVIP dataset.
    
    Args:
        extracted_dir: Base extracted directory
        output_base: Base output directory (dataset/)
        min_bbox_size: Minimum bounding box size
        filter_cars: Whether to filter out images with cars
        max_images_per_category: Maximum images per category (None = no limit)
    """
    annotations_dir = extracted_dir / 'LLVIP' / 'Annotations'
    visible_dir = extracted_dir / 'LLVIP' / 'visible' / 'test'
    infrared_dir = extracted_dir / 'LLVIP' / 'infrared' / 'test'
    
    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return
    
    print("Processing LLVIP dataset...")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Visible images: {visible_dir}")
    print(f"  Infrared images: {infrared_dir}")
    
    # Get all annotation files
    xml_files = list(annotations_dir.glob('*.xml'))
    print(f"  Total annotation files: {len(xml_files)}")
    
    # Process annotations
    valid_images = []
    images_with_cars = []
    
    for xml_file in xml_files:
        annotation = parse_xml_annotation(xml_file, min_bbox_size)
        if annotation is None or len(annotation['objects']) == 0:
            continue
        
        image_id = xml_file.stem
        
        # Check if image files exist
        visible_path = visible_dir / f"{image_id}.jpg"
        infrared_path = infrared_dir / f"{image_id}.jpg"
        
        if not visible_path.exists() or not infrared_path.exists():
            continue
        
        # Check for cars if filtering enabled
        if filter_cars:
            if has_cars_in_image(visible_path):
                images_with_cars.append(image_id)
                continue
        
        valid_images.append({
            'id': image_id,
            'visible_path': visible_path,
            'infrared_path': infrared_path,
            'objects': annotation['objects'],
            'num_objects': len(annotation['objects'])
        })
    
    print(f"  Valid images with person annotations: {len(valid_images)}")
    if filter_cars:
        print(f"  Images filtered out (contain cars): {len(images_with_cars)}")
    
    # Apply max limit if specified
    if max_images_per_category and len(valid_images) > max_images_per_category:
        # Sort by number of objects (prefer images with more people)
        valid_images.sort(key=lambda x: x['num_objects'], reverse=True)
        valid_images = valid_images[:max_images_per_category]
        print(f"  Limited to {max_images_per_category} images")
    
    # Copy images
    rgb_output = output_base / 'rgb' / 'human'
    thermal_output = output_base / 'thermal' / 'human'
    
    rgb_output.mkdir(parents=True, exist_ok=True)
    thermal_output.mkdir(parents=True, exist_ok=True)
    
    copied_rgb = 0
    copied_thermal = 0
    
    for img_info in valid_images:
        # Convert bboxes to COCO format [x, y, w, h]
        bboxes = []
        for obj in img_info['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']
            bboxes.append({
                'category': 'human',  # LLVIP only has person
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO format
                'area': (xmax - xmin) * (ymax - ymin)
            })
        
        # Get image dimensions
        from PIL import Image
        with Image.open(img_info['visible_path']) as img:
            width, height = img.size
        
        # Copy visible (RGB) image
        rgb_dest = rgb_output / f"llvip_{img_info['id']}.jpg"
        if not rgb_dest.exists():
            shutil.copy2(img_info['visible_path'], rgb_dest)
            copied_rgb += 1
            
            # Add to annotations
            if ann_collector:
                rel_path = Path('rgb') / 'human' / rgb_dest.name
                ann_collector.add_image(rel_path, 'rgb', 'human', width, height, bboxes)
        
        # Copy infrared (thermal) image
        thermal_dest = thermal_output / f"llvip_{img_info['id']}.jpg"
        if not thermal_dest.exists():
            shutil.copy2(img_info['infrared_path'], thermal_dest)
            copied_thermal += 1
            
            # Add to annotations
            if ann_collector:
                rel_path = Path('thermal') / 'human' / thermal_dest.name
                ann_collector.add_image(rel_path, 'thermal', 'human', width, height, bboxes)
    
    print(f"\nCopied images:")
    print(f"  RGB human: {copied_rgb}")
    print(f"  Thermal human: {copied_thermal}")
    
    return {
        'rgb_human': copied_rgb,
        'thermal_human': copied_thermal,
        'filtered_cars': len(images_with_cars) if filter_cars else 0
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build LLVIP dataset')
    parser.add_argument('--filter-cars', action='store_true',
                       help='Filter out images that contain cars (requires detection model)')
    parser.add_argument('--min-bbox-size', type=int, default=20,
                       help='Minimum bounding box size in pixels (default: 20, supports aerial/trap cameras)')
    parser.add_argument('--max-images-per-category', type=int, default=None,
                       help='Maximum images per category (default: no limit)')
    
    args = parser.parse_args()
    
    # Paths
    workspace = Path(__file__).parent.parent
    zip_path = workspace / 'zips' / 'LLVIP.zip'
    extract_dir = workspace / 'temp_llvip_extract'
    output_dir = workspace / 'dataset'
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found")
        sys.exit(1)
    
    # Extract zip
    print("=" * 60)
    print("LLVIP Dataset Builder")
    print("=" * 60)
    extract_zip(zip_path, extract_dir)
    
    # Initialize annotation collector
    ann_collector = AnnotationCollector(output_dir / 'annotations.json')
    
    # Process dataset
    results = process_llvip(
        extract_dir, output_dir,
        min_bbox_size=args.min_bbox_size,
        filter_cars=args.filter_cars,
        max_images_per_category=args.max_images_per_category,
        ann_collector=ann_collector
    )
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    # Cleanup
    print(f"\nCleaning up temporary extraction directory...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    print("✓ Done!")
    
    if results:
        print(f"\nSummary:")
        print(f"  RGB Human: {results['rgb_human']} images")
        print(f"  Thermal Human: {results['thermal_human']} images")
        if results['filtered_cars'] > 0:
            print(f"  Filtered (cars): {results['filtered_cars']} images")


if __name__ == '__main__':
    main()

