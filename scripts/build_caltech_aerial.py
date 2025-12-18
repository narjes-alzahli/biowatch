#!/usr/bin/env python3
"""
Build Caltech Aerial dataset - complete pipeline from zip files.

This script processes Caltech Aerial datasets (RGBT Pairs and Thermal Singles) directly from zip files:
1. Extracts both zip files
2. Processes segmentation masks (PNG format)
3. Converts masks to bounding boxes via connected components
4. Filters by minimum bbox size (20px for aerial/trap cameras)
5. Maps pixel values: 10 = vehicle, 11 = person/human
6. Processes both RGB and Thermal images
7. Saves annotations in COCO format

Usage:
    python3 scripts/build_caltech_aerial.py
"""

import sys
import shutil
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

from utils import extract_zip, AnnotationCollector


def get_connected_components(mask, class_value):
    """Extract bounding boxes from mask for a specific class value using flood fill."""
    # Create binary mask for this class
    binary_mask = (mask == class_value).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return []
    
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    bboxes = []
    
    def flood_fill(start_y, start_x):
        """Simple flood fill to find connected component."""
        if visited[start_y, start_x] or binary_mask[start_y, start_x] == 0:
            return None
        
        stack = [(start_y, start_x)]
        x_coords = []
        y_coords = []
        
        while stack:
            y, x = stack.pop()
            if visited[y, x] or binary_mask[y, x] == 0:
                continue
            
            visited[y, x] = True
            x_coords.append(x)
            y_coords.append(y)
            
            # Check neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if not visited[ny, nx] and binary_mask[ny, nx] == 1:
                        stack.append((ny, nx))
        
        if len(x_coords) > 0:
            return {
                'x': min(x_coords),
                'y': min(y_coords),
                'w': max(x_coords) - min(x_coords) + 1,
                'h': max(y_coords) - min(y_coords) + 1,
                'min_dim': min(max(x_coords) - min(x_coords) + 1, max(y_coords) - min(y_coords) + 1)
            }
        return None
    
    # Find all connected components
    for y in range(height):
        for x in range(width):
            if binary_mask[y, x] == 1 and not visited[y, x]:
                bbox = flood_fill(y, x)
                if bbox:
                    bboxes.append(bbox)
    
    return bboxes


def process_mask_file(mask_path, min_bbox_size=20):
    """
    Process a segmentation mask file and extract valid bounding boxes.
    
    Returns:
        dict with 'human' and 'vehicle' keys, each containing list of valid bboxes in COCO format
    """
    try:
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img)
        
        # Map: pixel value 10 = vehicle, 11 = person/human
        vehicle_boxes = get_connected_components(mask, 10)
        human_boxes = get_connected_components(mask, 11)
        
        # Filter by minimum size
        valid_vehicle = [b for b in vehicle_boxes if b['min_dim'] >= min_bbox_size]
        valid_human = [b for b in human_boxes if b['min_dim'] >= min_bbox_size]
        
        # Convert to COCO format [x, y, w, h]
        return {
            'human': [{'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h']} for b in valid_human],
            'vehicle': [{'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h']} for b in valid_vehicle]
        }
    except Exception as e:
        print(f"  Warning: Could not process {mask_path.name}: {e}")
        return {'human': [], 'vehicle': []}


def process_rgbt_pairs(extract_dir, output_dir, min_bbox_size=20, ann_collector=None):
    """Process RGBT Pairs dataset - extracts both RGB and Thermal images."""
    extract_dir = Path(extract_dir)
    output_dir = Path(output_dir)
    
    # Find the actual dataset directory (could be nested)
    annotations_dir = None
    thermal_dir = None
    rgb_dir = None
    
    # Try common paths
    possible_annotations = list(extract_dir.rglob('annotations'))
    possible_thermal = list(extract_dir.rglob('thermal8'))
    possible_rgb = list(extract_dir.rglob('eo'))
    if not possible_rgb:
        possible_rgb = list(extract_dir.rglob('rgb'))
    if not possible_rgb:
        possible_rgb = list(extract_dir.rglob('color'))
    
    if possible_annotations:
        annotations_dir = possible_annotations[0]
    if possible_thermal:
        thermal_dir = possible_thermal[0]
    if possible_rgb:
        rgb_dir = possible_rgb[0]
    
    if not annotations_dir or not thermal_dir:
        print(f"Error: Could not find annotations or thermal8 directories in {extract_dir}")
        return None
    
    # Create output directories for both RGB and Thermal
    output_thermal_human = output_dir / 'thermal' / 'human'
    output_thermal_vehicle = output_dir / 'thermal' / 'vehicle'
    output_rgb_human = output_dir / 'rgb' / 'human'
    output_rgb_vehicle = output_dir / 'rgb' / 'vehicle'
    
    for d in [output_thermal_human, output_thermal_vehicle, output_rgb_human, output_rgb_vehicle]:
        d.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'thermal': {
            'human': {'total': 0, 'copied': 0, 'filtered': 0},
            'vehicle': {'total': 0, 'copied': 0, 'filtered': 0}
        },
        'rgb': {
            'human': {'total': 0, 'copied': 0, 'filtered': 0},
            'vehicle': {'total': 0, 'copied': 0, 'filtered': 0}
        }
    }
    
    print(f"\nProcessing RGBT Pairs...")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Thermal images: {thermal_dir}")
    if rgb_dir:
        print(f"  RGB images: {rgb_dir}")
    else:
        print(f"  RGB images: Not found")
    
    # Get all mask files
    mask_files = list(annotations_dir.glob('*.png'))
    print(f"  Found {len(mask_files)} mask files")
    
    for mask_file in mask_files:
        # Find corresponding thermal and RGB images
        mask_stem = mask_file.stem
        thermal_path = None
        rgb_path = None
        
        if '_mask-' in mask_stem:
            base_name = mask_stem.split('_mask-')[0]
            frame = mask_stem.split('_mask-')[1]
            # Try .jpg first, then .png
            thermal_name = f"{base_name}_thermal-{frame}.jpg"
            thermal_path = thermal_dir / thermal_name
            if not thermal_path.exists():
                thermal_name = f"{base_name}_thermal-{frame}.png"
                thermal_path = thermal_dir / thermal_name
            
            # Find RGB image
            if rgb_dir:
                rgb_name = f"{base_name}_eo-{frame}.jpg"
                rgb_path = rgb_dir / rgb_name
                if not rgb_path.exists():
                    rgb_name = f"{base_name}_eo-{frame}.png"
                    rgb_path = rgb_dir / rgb_name
        else:
            # Try alternative naming patterns
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                thermal_path = thermal_dir / (mask_file.stem + ext)
                if thermal_path.exists():
                    break
                # Try with _thermal suffix
                thermal_path = thermal_dir / (mask_file.stem + '_thermal' + ext)
                if thermal_path.exists():
                    break
            
            if rgb_dir:
                for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                    rgb_path = rgb_dir / (mask_file.stem + ext)
                    if rgb_path.exists():
                        break
                    # Try with _eo suffix
                    rgb_path = rgb_dir / (mask_file.stem + '_eo' + ext)
                    if rgb_path.exists():
                        break
        
        if not thermal_path or not thermal_path.exists():
            continue
        
        # Process mask to get bboxes
        bboxes = process_mask_file(mask_path=mask_file, min_bbox_size=min_bbox_size)
        
        # Copy images if they have valid annotations
        has_human = len(bboxes['human']) > 0
        has_vehicle = len(bboxes['vehicle']) > 0
        
        if not has_human and not has_vehicle:
            continue
        
        # Get image dimensions
        with Image.open(thermal_path) as img:
            thermal_width, thermal_height = img.size
        
        rgb_width, rgb_height = thermal_width, thermal_height
        if rgb_path and rgb_path.exists():
            with Image.open(rgb_path) as img:
                rgb_width, rgb_height = img.size
        
        # Copy thermal images
        if has_human:
            stats['thermal']['human']['total'] += 1
            dest = output_thermal_human / thermal_path.name
            if not dest.exists():
                shutil.copy2(thermal_path, dest)
                stats['thermal']['human']['copied'] += 1
                
                # Add to annotations (convert to COCO format)
                if ann_collector:
                    human_bboxes = [{'category': 'human', 'bbox': [b['x'], b['y'], b['w'], b['h']], 'area': b['w'] * b['h']} 
                                   for b in bboxes['human']]
                    rel_path = Path('thermal') / 'human' / dest.name
                    ann_collector.add_image(rel_path, 'thermal', 'human', thermal_width, thermal_height, human_bboxes)
            stats['thermal']['human']['filtered'] += len(bboxes['human'])
        
        if has_vehicle:
            stats['thermal']['vehicle']['total'] += 1
            dest = output_thermal_vehicle / thermal_path.name
            if not dest.exists():
                shutil.copy2(thermal_path, dest)
                stats['thermal']['vehicle']['copied'] += 1
                
                # Add to annotations (convert to COCO format)
                if ann_collector:
                    vehicle_bboxes = [{'category': 'vehicle', 'bbox': [b['x'], b['y'], b['w'], b['h']], 'area': b['w'] * b['h']} 
                                     for b in bboxes['vehicle']]
                    rel_path = Path('thermal') / 'vehicle' / dest.name
                    ann_collector.add_image(rel_path, 'thermal', 'vehicle', thermal_width, thermal_height, vehicle_bboxes)
            stats['thermal']['vehicle']['filtered'] += len(bboxes['vehicle'])
        
        # Copy RGB images (if available)
        if rgb_path and rgb_path.exists():
            if has_human:
                stats['rgb']['human']['total'] += 1
                dest = output_rgb_human / rgb_path.name
                if not dest.exists():
                    shutil.copy2(rgb_path, dest)
                    stats['rgb']['human']['copied'] += 1
                    
                    # Add to annotations (convert to COCO format)
                    if ann_collector:
                        human_bboxes = [{'category': 'human', 'bbox': [b['x'], b['y'], b['w'], b['h']], 'area': b['w'] * b['h']} 
                                       for b in bboxes['human']]
                        rel_path = Path('rgb') / 'human' / dest.name
                        ann_collector.add_image(rel_path, 'rgb', 'human', rgb_width, rgb_height, human_bboxes)
                stats['rgb']['human']['filtered'] += len(bboxes['human'])
            
            if has_vehicle:
                stats['rgb']['vehicle']['total'] += 1
                dest = output_rgb_vehicle / rgb_path.name
                if not dest.exists():
                    shutil.copy2(rgb_path, dest)
                    stats['rgb']['vehicle']['copied'] += 1
                    
                    # Add to annotations (convert to COCO format)
                    if ann_collector:
                        vehicle_bboxes = [{'category': 'vehicle', 'bbox': [b['x'], b['y'], b['w'], b['h']], 'area': b['w'] * b['h']} 
                                         for b in bboxes['vehicle']]
                        rel_path = Path('rgb') / 'vehicle' / dest.name
                        ann_collector.add_image(rel_path, 'rgb', 'vehicle', rgb_width, rgb_height, vehicle_bboxes)
                stats['rgb']['vehicle']['filtered'] += len(bboxes['vehicle'])
    
    return stats


def process_thermal_singles(extract_dir, output_dir, min_bbox_size=20, ann_collector=None):
    """Process Thermal Singles dataset."""
    extract_dir = Path(extract_dir)
    output_dir = Path(output_dir)
    
    # Find the base directory (could be nested)
    base_dir = None
    possible_bases = list(extract_dir.rglob('labeled_thermal_singles'))
    if possible_bases:
        base_dir = possible_bases[0]
    else:
        # Try to find any directory with masks subdirectory
        for d in extract_dir.rglob('*'):
            if d.is_dir() and (d / 'masks').exists():
                base_dir = d.parent
                break
    
    if not base_dir:
        print(f"Error: Could not find thermal singles base directory in {extract_dir}")
        return None
    
    output_human = output_dir / 'thermal' / 'human'
    output_vehicle = output_dir / 'thermal' / 'vehicle'
    output_human.mkdir(parents=True, exist_ok=True)
    output_vehicle.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'human': {'total': 0, 'copied': 0, 'filtered': 0},
        'vehicle': {'total': 0, 'copied': 0, 'filtered': 0}
    }
    
    print(f"\nProcessing Thermal Singles...")
    print(f"  Base directory: {base_dir}")
    
    # Find all mask directories
    mask_dirs = list(base_dir.rglob('masks'))
    print(f"  Found {len(mask_dirs)} mask directories")
    
    for mask_dir in mask_dirs:
        # Find corresponding thermal8 directory
        sequence_dir = mask_dir.parent
        thermal_dir = sequence_dir / 'thermal8'
        
        if not thermal_dir.exists():
            continue
        
        # Process all masks in this sequence
        mask_files = list(mask_dir.glob('*.png'))
        
        for mask_file in mask_files:
            # Find corresponding thermal image (same name)
            thermal_path = thermal_dir / mask_file.name
            
            if not thermal_path.exists():
                continue
            
            # Process mask to get bboxes
            bboxes = process_mask_file(mask_file, min_bbox_size)
            
            # Copy thermal image if it has valid annotations
            has_human = len(bboxes['human']) > 0
            has_vehicle = len(bboxes['vehicle']) > 0
            
            # Get image dimensions
            with Image.open(thermal_path) as img:
                width, height = img.size
            
            if has_human:
                stats['human']['total'] += 1
                # Use sequence name in filename to avoid conflicts
                seq_name = sequence_dir.name
                dest_name = f"{seq_name}_{thermal_path.name}"
                dest = output_human / dest_name
                if not dest.exists():
                    shutil.copy2(thermal_path, dest)
                    stats['human']['copied'] += 1
                    
                    # Add to annotations (convert to COCO format)
                    if ann_collector:
                        human_bboxes = [{'category': 'human', 'bbox': [b['x'], b['y'], b['w'], b['h']], 'area': b['w'] * b['h']} 
                                       for b in bboxes['human']]
                        rel_path = Path('thermal') / 'human' / dest.name
                        ann_collector.add_image(rel_path, 'thermal', 'human', width, height, human_bboxes)
                stats['human']['filtered'] += len(bboxes['human'])
            
            if has_vehicle:
                stats['vehicle']['total'] += 1
                seq_name = sequence_dir.name
                dest_name = f"{seq_name}_{thermal_path.name}"
                dest = output_vehicle / dest_name
                if not dest.exists():
                    shutil.copy2(thermal_path, dest)
                    stats['vehicle']['copied'] += 1
                    
                    # Add to annotations (convert to COCO format)
                    if ann_collector:
                        vehicle_bboxes = [{'category': 'vehicle', 'bbox': [b['x'], b['y'], b['w'], b['h']], 'area': b['w'] * b['h']} 
                                         for b in bboxes['vehicle']]
                        rel_path = Path('thermal') / 'vehicle' / dest.name
                        ann_collector.add_image(rel_path, 'thermal', 'vehicle', width, height, vehicle_bboxes)
                stats['vehicle']['filtered'] += len(bboxes['vehicle'])
    
    return stats


def main():
    """Process both Caltech Aerial datasets from zip files."""
    print("=" * 70)
    print("Caltech Aerial Dataset Builder (from zip files)")
    print("=" * 70)
    
    min_bbox_size = 20  # Supports aerial/trap cameras
    zip_rgbt = Path('zips/caltech_aerial_labeled_rgbt_pairs.zip')
    zip_thermal = Path('zips/caltech_aerial_labeled_thermal_singles.zip')
    output_dir = Path('dataset')
    
    # Temporary extraction directories
    extract_rgbt = output_dir.parent / 'temp_caltech_rgbt_extract'
    extract_thermal = output_dir.parent / 'temp_caltech_thermal_extract'
    
    print(f"\nConfiguration:")
    print(f"  Min bbox size: {min_bbox_size}x{min_bbox_size} pixels")
    print(f"  RGBT Pairs zip: {zip_rgbt}")
    print(f"  Thermal Singles zip: {zip_thermal}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)
    
    # Initialize annotation collector
    ann_collector = AnnotationCollector(output_dir / 'annotations.json')
    
    all_stats = {
        'thermal': {
            'human': {'total': 0, 'copied': 0, 'filtered': 0},
            'vehicle': {'total': 0, 'copied': 0, 'filtered': 0}
        },
        'rgb': {
            'human': {'total': 0, 'copied': 0, 'filtered': 0},
            'vehicle': {'total': 0, 'copied': 0, 'filtered': 0}
        }
    }
    
    # Process RGBT Pairs
    if zip_rgbt.exists():
        print(f"\n[Step 1/2] Extracting RGBT Pairs...")
        if extract_rgbt.exists():
            shutil.rmtree(extract_rgbt)
        extract_zip(zip_rgbt, extract_rgbt)
        stats_rgbt = process_rgbt_pairs(extract_rgbt, output_dir, min_bbox_size, ann_collector)
        if stats_rgbt:
            for modality in ['thermal', 'rgb']:
                for cat in ['human', 'vehicle']:
                    all_stats[modality][cat]['total'] += stats_rgbt[modality][cat]['total']
                    all_stats[modality][cat]['copied'] += stats_rgbt[modality][cat]['copied']
                    all_stats[modality][cat]['filtered'] += stats_rgbt[modality][cat]['filtered']
        # Cleanup
        shutil.rmtree(extract_rgbt, ignore_errors=True)
    else:
        print(f"\nRGBT Pairs zip not found: {zip_rgbt}")
    
    # Process Thermal Singles
    if zip_thermal.exists():
        print(f"\n[Step 2/2] Extracting Thermal Singles...")
        if extract_thermal.exists():
            shutil.rmtree(extract_thermal)
        extract_zip(zip_thermal, extract_thermal)
        stats_thermal = process_thermal_singles(extract_thermal, output_dir, min_bbox_size, ann_collector)
        if stats_thermal:
            for cat in ['human', 'vehicle']:
                all_stats['thermal'][cat]['total'] += stats_thermal[cat]['total']
                all_stats['thermal'][cat]['copied'] += stats_thermal[cat]['copied']
                all_stats['thermal'][cat]['filtered'] += stats_thermal[cat]['filtered']
        # Cleanup
        shutil.rmtree(extract_thermal, ignore_errors=True)
    else:
        print(f"\nThermal Singles zip not found: {zip_thermal}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    
    print(f"\nTHERMAL MODALITY:")
    print(f"  Human:")
    print(f"    Images with valid human boxes: {all_stats['thermal']['human']['copied']}")
    print(f"    Total human boxes (≥{min_bbox_size}px): {all_stats['thermal']['human']['filtered']}")
    print(f"  Vehicle:")
    print(f"    Images with valid vehicle boxes: {all_stats['thermal']['vehicle']['copied']}")
    print(f"    Total vehicle boxes (≥{min_bbox_size}px): {all_stats['thermal']['vehicle']['filtered']}")
    
    print(f"\nRGB MODALITY:")
    print(f"  Human:")
    print(f"    Images with valid human boxes: {all_stats['rgb']['human']['copied']}")
    print(f"    Total human boxes (≥{min_bbox_size}px): {all_stats['rgb']['human']['filtered']}")
    print(f"  Vehicle:")
    print(f"    Images with valid vehicle boxes: {all_stats['rgb']['vehicle']['copied']}")
    print(f"    Total vehicle boxes (≥{min_bbox_size}px): {all_stats['rgb']['vehicle']['filtered']}")
    
    print(f"\nOutput locations:")
    print(f"  Thermal: {output_dir}/thermal/human/ and {output_dir}/thermal/vehicle/")
    print(f"  RGB: {output_dir}/rgb/human/ and {output_dir}/rgb/vehicle/")
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
