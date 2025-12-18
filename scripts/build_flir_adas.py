#!/usr/bin/env python3
"""
Build FLIR ADAS Thermal v2 dataset - complete pipeline.

This script processes FLIR ADAS dataset with quality filters:
1. Extracts zip file
2. Parses COCO format annotations
3. Filters by category (person → human, vehicles → vehicle, animals → animal)
4. Includes bicycle category in vehicles
5. Applies quality filters (min bbox size 50px, exclude heavily occluded)
6. Processes both thermal and RGB images
7. Uses both train and val splits
8. Subsamples video test frames
9. Copies images to dataset/thermal/ and dataset/rgb/

Usage:
    python3 scripts/build_flir_adas.py [--max-images-per-category N]
"""

import sys
import json
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

from utils import extract_zip, AnnotationCollector


# Category mappings from FLIR to our categories
# Category IDs from COCO format: 1=person, 2=bike, 3=car, 4=motor, 6=bus, 7=train, 8=truck, 9=boat, 17=dog, 18=deer, 79=other vehicle
HUMAN_CATEGORY_IDS = {1}  # person
ANIMAL_CATEGORY_IDS = {17, 18}  # dog, deer
VEHICLE_CATEGORY_IDS = {2, 3, 4, 6, 7, 8, 79}  # bike, car, motor, bus, train, truck, other vehicle
# Excluding: 9=boat, 5=airplane

# Occlusion filter: exclude heavily occluded (70%-90%)
HEAVILY_OCCLUDED = '70%_-_90%_occluded_(difficult_to_see)'


def parse_coco_annotations(coco_json_path, min_bbox_size=50):
    """
    Parse COCO format annotations and filter by quality.
    
    Args:
        coco_json_path: Path to coco.json file
        min_bbox_size: Minimum bounding box size (width or height)
    
    Returns:
        dict: {image_id: {category: [bboxes]}} where category is 'human', 'animal', or 'vehicle'
    """
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    # Create category ID to name mapping
    category_map = {c['id']: c['name'] for c in data['categories']}
    
    # Create image ID to filename mapping
    image_map = {img['id']: img['file_name'] for img in data['images']}
    
    # Process annotations
    image_annotations = defaultdict(lambda: defaultdict(list))
    
    for ann in data['annotations']:
        category_id = ann['category_id']
        
        # Map to our categories (bicycles are included in vehicle category)
        if category_id in HUMAN_CATEGORY_IDS:
            category = 'human'
        elif category_id in ANIMAL_CATEGORY_IDS:
            category = 'animal'
        elif category_id in VEHICLE_CATEGORY_IDS:
            category = 'vehicle'
        else:
            continue  # Skip other categories
        
        # Check occlusion filter (exclude heavily occluded)
        if 'extra_info' in ann:
            occluded = ann['extra_info'].get('occluded', '')
            if occluded == HEAVILY_OCCLUDED:
                continue
        
        # Check bounding box size
        bbox = ann['bbox']  # [x, y, width, height]
        width, height = bbox[2], bbox[3]
        if min(width, height) < min_bbox_size:
            continue
        
        # Add annotation
        image_id = ann['image_id']
        image_annotations[image_id][category].append(bbox)
    
    # Convert to {filename: {category: [bboxes]}}
    # Note: filenames in COCO JSON may include "data/" prefix, strip it
    result = {}
    for image_id, categories in image_annotations.items():
        if image_id in image_map:
            filename = image_map[image_id]
            # Strip "data/" prefix if present
            if filename.startswith('data/'):
                filename = filename[5:]  # Remove "data/" prefix
            result[filename] = categories
    
    return result


def get_video_info_from_filename(filename):
    """Extract video ID and frame number from FLIR video filename."""
    # Format: video-{id}-frame-{frame_number}-{hash}.jpg
    try:
        parts = Path(filename).stem.split('-')
        if len(parts) >= 4 and parts[0] == 'video' and parts[2] == 'frame':
            video_id = parts[1]
            frame_num = int(parts[3])
            return video_id, frame_num
    except (ValueError, IndexError):
        pass
    return None, None


def subsample_video_frames(image_files, max_frames_per_video=None, stride=None):
    """
    Subsample video frames, grouped by video ID.
    
    Args:
        image_files: List of image file paths
        max_frames_per_video: Maximum number of frames to keep per video
        stride: Take every Nth frame
    
    Returns:
        List of file paths to keep
    """
    if not image_files:
        return []
    
    # Group by video ID
    videos = defaultdict(list)
    for img_file in image_files:
        video_id, frame_num = get_video_info_from_filename(img_file.name)
        if video_id is not None:
            videos[video_id].append((frame_num, img_file))
        else:
            # If we can't parse, treat as single video
            videos['unknown'].append((999999, img_file))
    
    # Subsample each video
    selected_files = []
    for video_id, frames in videos.items():
        # Sort by frame number
        frames.sort(key=lambda x: x[0])
        
        # Apply stride
        if stride and stride > 1:
            frames = frames[::stride]
        
        # Apply max limit per video
        if max_frames_per_video and len(frames) > max_frames_per_video:
            # Distribute evenly
            indices = [int(i * (len(frames) - 1) / (max_frames_per_video - 1)) 
                      for i in range(max_frames_per_video)]
            frames = [frames[i] for i in indices]
        
        selected_files.extend([f[1] for f in frames])
    
    return selected_files


def process_split(extracted_dir, split_name, modality, output_base, min_bbox_size=50, max_images_per_category=None, ann_collector=None):
    """
    Process a single split (train/val) for a modality (thermal/rgb).
    
    Args:
        extracted_dir: Base extracted directory
        split_name: 'train' or 'val'
        modality: 'thermal' or 'rgb'
        output_base: Base output directory (dataset/)
        min_bbox_size: Minimum bounding box size
        max_images_per_category: Maximum images per category (None = no limit)
    
    Returns:
        dict: Counts of images processed per category
    """
    # Paths
    images_dir = extracted_dir / f'FLIR_ADAS_v2/images_{modality}_{split_name}/data'
    coco_json = extracted_dir / f'FLIR_ADAS_v2/images_{modality}_{split_name}/coco.json'
    
    if not coco_json.exists():
        print(f"Warning: {coco_json} not found, skipping")
        return {'human': 0, 'animal': 0, 'vehicle': 0}
    
    print(f"\nProcessing {modality} {split_name}...")
    
    # Parse annotations
    print(f"  Parsing annotations from {coco_json}...")
    image_annotations = parse_coco_annotations(coco_json, min_bbox_size)
    print(f"  Found {len(image_annotations)} images with valid annotations")
    
    # Count annotations per category
    category_counts = defaultdict(int)
    for categories in image_annotations.values():
        for category in categories.keys():
            category_counts[category] += 1
    
    print(f"  Category distribution: {dict(category_counts)}")
    
    # Process images
    category_image_counts = defaultdict(int)
    category_limits = defaultdict(int)
    
    for filename, categories in image_annotations.items():
        # Check if we've hit limits
        if max_images_per_category:
            skip = False
            for category in categories.keys():
                if category_limits[category] >= max_images_per_category:
                    skip = True
                    break
            if skip:
                continue
        
        # Copy image to appropriate category folders
        source_path = images_dir / filename
        if not source_path.exists():
            continue
        
        for category in categories.keys():
            if max_images_per_category and category_limits[category] >= max_images_per_category:
                continue
            
            # Determine output directory based on modality
            if modality == 'thermal':
                output_dir = output_base / 'thermal' / category
            else:  # rgb
                output_dir = output_base / 'rgb' / category
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            dest_path = output_dir / filename
            if not dest_path.exists():  # Avoid duplicates
                shutil.copy2(source_path, dest_path)
                category_image_counts[category] += 1
                category_limits[category] += 1
                
                # Add to annotations
                if ann_collector:
                    bboxes = []
                    for bbox in categories[category]:  # categories[category] is list of bboxes
                        bboxes.append({
                            'category': category,
                            'bbox': bbox,  # Already COCO format [x, y, w, h]
                            'area': bbox[2] * bbox[3]
                        })
                    
                    # Get image dimensions
                    from PIL import Image
                    with Image.open(source_path) as img:
                        width, height = img.size
                    
                    rel_path = Path(modality) / category / filename
                    ann_collector.add_image(rel_path, modality, category, width, height, bboxes)
    
    print(f"  Copied images: {dict(category_image_counts)}")
    return category_image_counts


def process_video_split(extracted_dir, split_name, modality, output_base, min_bbox_size=50, 
                        max_frames_per_video=100, stride=10, max_images_per_category=None, ann_collector=None):
    """
    Process video test split with subsampling.
    
    Args:
        extracted_dir: Base extracted directory
        split_name: 'test' (for video)
        modality: 'thermal' or 'rgb'
        output_base: Base output directory (dataset/)
        min_bbox_size: Minimum bounding box size
        max_frames_per_video: Maximum frames per video sequence
        stride: Frame stride for subsampling
        max_images_per_category: Maximum images per category (None = no limit)
    
    Returns:
        dict: Counts of images processed per category
    """
    # Paths
    images_dir = extracted_dir / f'FLIR_ADAS_v2/video_{modality}_{split_name}/data'
    coco_json = extracted_dir / f'FLIR_ADAS_v2/video_{modality}_{split_name}/coco.json'
    
    if not coco_json.exists():
        print(f"Warning: {coco_json} not found, skipping")
        return {'human': 0, 'animal': 0, 'vehicle': 0}
    
    print(f"\nProcessing {modality} video {split_name} (with subsampling)...")
    
    # Parse annotations
    print(f"  Parsing annotations from {coco_json}...")
    image_annotations = parse_coco_annotations(coco_json, min_bbox_size)
    print(f"  Found {len(image_annotations)} images with valid annotations")
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.JPG'))
    print(f"  Total video frames: {len(image_files)}")
    
    # Subsample frames (grouped by video ID)
    subsampled_files = subsample_video_frames(image_files, max_frames_per_video=max_frames_per_video, stride=stride)
    print(f"  After subsampling: {len(subsampled_files)} frames")
    
    # Filter to only include subsampled files that have valid annotations
    valid_files = [f for f in subsampled_files if f.name in image_annotations]
    print(f"  Valid annotated frames: {len(valid_files)}")
    
    # Process images
    category_image_counts = defaultdict(int)
    category_limits = defaultdict(int)
    
    for img_file in valid_files:
        filename = img_file.name
        categories = image_annotations[filename]
        
        # Check if we've hit limits
        if max_images_per_category:
            skip = False
            for category in categories.keys():
                if category_limits[category] >= max_images_per_category:
                    skip = True
                    break
            if skip:
                continue
        
        # Copy image to appropriate category folders
        for category in categories.keys():
            if max_images_per_category and category_limits[category] >= max_images_per_category:
                continue
            
            # Determine output directory based on modality
            if modality == 'thermal':
                output_dir = output_base / 'thermal' / category
            else:  # rgb
                output_dir = output_base / 'rgb' / category
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            dest_path = output_dir / filename
            if not dest_path.exists():  # Avoid duplicates
                shutil.copy2(img_file, dest_path)
                category_image_counts[category] += 1
                category_limits[category] += 1
                
                # Add to annotations
                if ann_collector:
                    bboxes = []
                    for bbox in categories[category]:  # categories[category] is list of bboxes
                        bboxes.append({
                            'category': category,
                            'bbox': bbox,  # Already COCO format [x, y, w, h]
                            'area': bbox[2] * bbox[3]
                        })
                    
                    # Get image dimensions
                    from PIL import Image
                    with Image.open(img_file) as img:
                        width, height = img.size
                    
                    rel_path = Path(modality) / category / filename
                    ann_collector.add_image(rel_path, modality, category, width, height, bboxes)
    
    print(f"  Copied images: {dict(category_image_counts)}")
    return category_image_counts


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build FLIR ADAS Thermal v2 dataset')
    parser.add_argument('--max-images-per-category', type=int, default=None,
                       help='Maximum images per category (default: no limit)')
    parser.add_argument('--video-max-frames', type=int, default=100,
                       help='Maximum frames per video sequence (default: 100)')
    parser.add_argument('--video-stride', type=int, default=10,
                       help='Frame stride for video subsampling (default: 10)')
    parser.add_argument('--min-bbox-size', type=int, default=20,
                       help='Minimum bounding box size in pixels (default: 20, supports aerial/trap cameras)')
    
    args = parser.parse_args()
    
    # Paths
    workspace = Path(__file__).parent.parent
    zip_path = workspace / 'zips' / 'FLIR_ADAS_Thermal_v2.zip'
    extract_dir = workspace / 'temp_flir_extract'
    output_dir = workspace / 'dataset'
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found")
        sys.exit(1)
    
    # Extract zip
    print("=" * 60)
    print("FLIR ADAS Thermal v2 Dataset Builder")
    print("=" * 60)
    extract_zip(zip_path, extract_dir)
    
    # Initialize annotation collector
    ann_collector = AnnotationCollector(output_dir / 'annotations.json')
    
    # Process splits
    total_counts = defaultdict(lambda: defaultdict(int))
    
    # Process train and val for both thermal and RGB
    for modality in ['thermal', 'rgb']:
        for split in ['train', 'val']:
            counts = process_split(
                extract_dir, split, modality, output_dir,
                min_bbox_size=args.min_bbox_size,
                max_images_per_category=args.max_images_per_category,
                ann_collector=ann_collector
            )
            for category, count in counts.items():
                total_counts[modality][category] += count
    
    # Process video test for both thermal and RGB (with subsampling)
    for modality in ['thermal', 'rgb']:
        counts = process_video_split(
            extract_dir, 'test', modality, output_dir,
            min_bbox_size=args.min_bbox_size,
            max_frames_per_video=args.video_max_frames,
            stride=args.video_stride,
            max_images_per_category=args.max_images_per_category,
            ann_collector=ann_collector
        )
        for category, count in counts.items():
            total_counts[modality][category] += count
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for modality in ['thermal', 'rgb']:
        print(f"\n{modality.upper()} Modality:")
        for category in ['human', 'animal', 'vehicle']:
            count = total_counts[modality][category]
            print(f"  {category.capitalize()}: {count} images")
        print(f"  Total: {sum(total_counts[modality].values())} images")
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    # Cleanup
    print(f"\nCleaning up temporary extraction directory...")
    shutil.rmtree(extract_dir, ignore_errors=True)
    print("✓ Done!")


if __name__ == '__main__':
    main()

