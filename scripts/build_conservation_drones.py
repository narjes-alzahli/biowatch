#!/usr/bin/env python3
"""
Build Conservation Drones dataset - complete pipeline.

This script processes Conservation Drones dataset with quality filters and subsampling:
1. Extracts and processes train + test zip files
2. Applies quality filters (noise, unknown species, min bbox size 20px for aerial/trap cameras)
3. Subsamples to max 100 frames per sequence
4. Outputs to dataset/thermal/

Usage:
    python3 scripts/build_conservation_drones.py
"""

import sys
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# Import shared utilities
from utils import extract_zip, AnnotationCollector


def parse_csv_annotations(csv_path):
    """Parse MOT-format CSV annotations."""
    annotations = defaultdict(list)
    
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return annotations
    
    print(f"Parsing annotations from {csv_path}...")
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            
            try:
                frame_number = int(row[0])
                object_id = int(row[1])
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                class_id = int(row[6])  # 0=animal, 1=human
                species = int(row[7])
                occlusion = int(row[8])
                noise = int(row[9])
                
                annotations[frame_number].append({
                    'object_id': object_id,
                    'bbox': (float(x), float(y), float(w), float(h)),
                    'class': class_id,
                    'species': species,
                    'occlusion': occlusion,
                    'noise': noise
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed row: {row[:5]}... ({e})")
                continue
    
    print(f"✓ Parsed {len(annotations)} frames with annotations")
    return annotations


def find_image_for_frame(frame_number, image_dir, video_id=None, sequence_id=None):
    """Find image file corresponding to frame number."""
    if video_id is not None and sequence_id is not None:
        patterns = [
            f"{video_id}_{sequence_id}_{frame_number:010d}.jpg",
            f"{video_id}_{sequence_id}_{frame_number:010d}.png",
            f"{video_id}_{sequence_id}_{frame_number:09d}.jpg",
            f"{video_id}_{sequence_id}_{frame_number:09d}.png",
            f"{video_id}_{sequence_id}_{frame_number:08d}.jpg",
            f"{video_id}_{sequence_id}_{frame_number:08d}.png",
        ]
        for pattern in patterns:
            img_path = image_dir / pattern
            if img_path.exists():
                return img_path
    
    patterns = [
        f"{frame_number:06d}.jpg", f"{frame_number:06d}.png",
        f"{frame_number:05d}.jpg", f"{frame_number:05d}.png",
        f"frame_{frame_number:06d}.jpg", f"frame_{frame_number:06d}.png",
        f"img_{frame_number:06d}.jpg", f"img_{frame_number:06d}.png",
        f"{frame_number}.jpg", f"{frame_number}.png",
    ]
    
    for pattern in patterns:
        img_path = image_dir / pattern
        if img_path.exists():
            return img_path
    
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        matches = list(image_dir.glob(f"*{frame_number}*{ext}"))
        matches.extend(list(image_dir.glob(f"*{frame_number}*{ext.upper()}")))
        if matches:
            return matches[0]
    
    return None


def process_conservation_drones(zip_path, output_dir, ann_collector, filter_noise=True, filter_occlusion=False, 
                                filter_unknown_species=True, min_bbox_size=20, max_per_sequence_human=100,
                                max_per_sequence_animal=300):
    """Process Conservation Drones dataset directly to output directory."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    
    # Track sequences for subsampling
    sequences = defaultdict(lambda: {'human': [], 'animal': []})
    
    extract_dir = Path("temp_extract") / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    extract_zip(zip_path, extract_dir)
    
    csv_files = list(extract_dir.rglob("*.csv"))
    if not csv_files:
        annotation_dirs = list(extract_dir.rglob("*annotation*"))
        annotation_dirs.extend(list(extract_dir.rglob("*label*")))
        for ann_dir in annotation_dirs:
            csv_files.extend(list(ann_dir.glob("*.csv")))
    
    if not csv_files:
        print(f"Error: No CSV annotation files found!")
        return {'human': 0, 'animal': 0, 'skipped_noise': 0, 'no_image': 0}
    
    print(f"\nFound {len(csv_files)} CSV file(s)")
    
    image_dirs = []
    for img_dir_name in ['images', 'Images', 'frames', 'Frames', 'data', 'Data']:
        img_dirs = list(extract_dir.rglob(img_dir_name))
        image_dirs.extend([d for d in img_dirs if d.is_dir()])
    
    if not image_dirs:
        for d in extract_dir.rglob("*"):
            if d.is_dir() and (any(d.glob("*.jpg")) or any(d.glob("*.png"))):
                image_dirs.append(d)
    
    if not image_dirs:
        print(f"Error: No image directories found!")
        return {'human': 0, 'animal': 0, 'skipped_noise': 0, 'no_image': 0}
    
    stats = {'human': 0, 'animal': 0, 'skipped_noise': 0, 'no_image': 0}
    
    for csv_file in csv_files:
        annotations = parse_csv_annotations(csv_file)
        
        csv_stem = csv_file.stem
        video_id = None
        sequence_id = None
        if '_' in csv_stem:
            parts = csv_stem.split('_')
            if len(parts) >= 2:
                video_id = parts[0]
                sequence_id = parts[1]
        
        image_dir = None
        if video_id and sequence_id:
            target_dir_name = f"{video_id}_{sequence_id}"
            for img_dir in image_dirs:
                target_path = img_dir / target_dir_name
                if target_path.exists() and target_path.is_dir():
                    image_dir = target_path
                    break
                if img_dir.name == target_dir_name:
                    image_dir = img_dir
                    break
        
        if not image_dir:
            csv_name = csv_file.stem.lower()
            for img_dir in image_dirs:
                if csv_name in img_dir.name.lower() or img_dir.name.lower() in csv_name:
                    image_dir = img_dir
                    break
        
        if not image_dir and image_dirs:
            image_dir = image_dirs[0]
        
        if not image_dir:
            continue
        
        seq_key = (video_id, sequence_id) if video_id and sequence_id else ('unknown', csv_stem)
        
        for frame_number, frame_annotations in annotations.items():
            valid_annotations = []
            for ann in frame_annotations:
                if filter_noise and ann['noise'] == 1:
                    continue
                if filter_occlusion and ann['occlusion'] == 1:
                    continue
                if filter_unknown_species and ann['species'] == -1:
                    continue
                
                bbox_w = float(ann['bbox'][2])
                bbox_h = float(ann['bbox'][3])
                if bbox_w < min_bbox_size or bbox_h < min_bbox_size:
                    continue
                
                valid_annotations.append(ann)
            
            if not valid_annotations:
                stats['skipped_noise'] += 1
                continue
            
            img_path = find_image_for_frame(frame_number, image_dir, video_id, sequence_id)
            if not img_path:
                stats['no_image'] += 1
                continue
            
            # Process both human and animal
            has_human = any(ann['class'] == 1 for ann in valid_annotations)
            has_animal = any(ann['class'] == 0 for ann in valid_annotations)
            
            # Get image dimensions once
            from PIL import Image
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except:
                continue
            
            # Process human category
            if has_human:
                human_bboxes = []
                for ann in valid_annotations:
                    if ann['class'] == 1:  # 1=human
                        x, y, w, h = ann['bbox']
                        human_bboxes.append({
                            'category': 'human',
                            'bbox': [x, y, w, h],  # COCO format: [x, y, width, height]
                            'area': w * h
                        })
                
                if human_bboxes:
                    sequences[seq_key]['human'].append({
                        'frame_num': frame_number,
                        'img_path': img_path,
                        'width': width,
                        'height': height,
                        'bboxes': human_bboxes
                    })
            
            # Process animal category
            if has_animal:
                animal_bboxes = []
                for ann in valid_annotations:
                    if ann['class'] == 0:  # 0=animal
                        x, y, w, h = ann['bbox']
                        animal_bboxes.append({
                            'category': 'animal',
                            'bbox': [x, y, w, h],  # COCO format: [x, y, width, height]
                            'area': w * h
                        })
                
                if animal_bboxes:
                    sequences[seq_key]['animal'].append({
                        'frame_num': frame_number,
                        'img_path': img_path,
                        'width': width,
                        'height': height,
                        'bboxes': animal_bboxes
                    })
    
    # Subsample and copy to output directory
    print(f"\nSubsampling sequences and copying to output directory...")
    for seq_key, seq_data in sequences.items():
        # Subsample human frames
        if seq_data['human']:
            human_frames = sorted(seq_data['human'], key=lambda x: x['frame_num'])
            max_human = max_per_sequence_human
            if len(human_frames) > max_human:
                indices = [int(i * (len(human_frames) - 1) / (max_human - 1)) for i in range(max_human)]
                human_frames = [human_frames[i] for i in indices]
            
            for frame_info in human_frames:
                dest = output_dir / 'human' / frame_info['img_path'].name
                # Handle duplicates
                if dest.exists():
                    stem = dest.stem
                    suffix = dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = output_dir / 'human' / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frame_info['img_path'], dest)
                stats['human'] += 1
                
                # Add annotation
                rel_path = Path('thermal') / 'human' / dest.name
                ann_collector.add_image(rel_path, 'thermal', 'human', 
                                      frame_info['width'], frame_info['height'], frame_info['bboxes'])
        
        # Subsample animal frames
        if seq_data['animal']:
            animal_frames = sorted(seq_data['animal'], key=lambda x: x['frame_num'])
            max_animal = max_per_sequence_animal
            if len(animal_frames) > max_animal:
                indices = [int(i * (len(animal_frames) - 1) / (max_animal - 1)) for i in range(max_animal)]
                animal_frames = [animal_frames[i] for i in indices]
            
            for frame_info in animal_frames:
                dest = output_dir / 'animal' / frame_info['img_path'].name
                # Handle duplicates
                if dest.exists():
                    stem = dest.stem
                    suffix = dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = output_dir / 'animal' / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frame_info['img_path'], dest)
                stats['animal'] += 1
                
                # Add annotation
                rel_path = Path('thermal') / 'animal' / dest.name
                ann_collector.add_image(rel_path, 'thermal', 'animal',
                                      frame_info['width'], frame_info['height'], frame_info['bboxes'])
    
    print(f"\n✓ Human: {stats['human']}, Animal: {stats['animal']}, Filtered: {stats['skipped_noise']}")
    
    shutil.rmtree(extract_dir.parent, ignore_errors=True)
    return stats


def main():
    """Build Conservation Drones dataset with complete pipeline."""
    print("=" * 70)
    print("Conservation Drones Dataset Builder")
    print("=" * 70)
    
    zip_files = [
        'zips/conservation_drones_train_real.zip',
        'zips/conservation_drones_test_real.zip'
    ]
    output_dir = Path('dataset/thermal')
    min_bbox_size = 20  # Supports aerial/trap cameras
    max_per_sequence_human = 100  # Max frames per sequence for humans
    max_per_sequence_animal = 300  # Max frames per sequence for animals
    
    filter_noise = True  # Exclude: noisy annotations are low quality
    filter_occlusion = False  # Include: occluded objects are common in real scenarios
    filter_unknown_species = True  # Exclude: unknown species aren't useful
    
    missing_zips = [z for z in zip_files if not Path(z).exists()]
    if missing_zips:
        print(f"ERROR: Missing zip files:")
        for z in missing_zips:
            print(f"  - {z}")
        return 1
    
    print(f"\nConfiguration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Processing: Both human and animal categories")
    print(f"  Min bbox size: {min_bbox_size}x{min_bbox_size} pixels (supports aerial/trap cameras)")
    print(f"  Filter noise: {filter_noise} (exclude low quality)")
    print(f"  Filter occlusion: {filter_occlusion} (include occluded objects for real-world scenarios)")
    print(f"  Filter unknown species: {filter_unknown_species} (exclude unknown)")
    print(f"  Max per sequence - Human: {max_per_sequence_human} frames")
    print(f"  Max per sequence - Animal: {max_per_sequence_animal} frames")
    print("=" * 70)
    
    # Initialize annotation collector
    ann_collector = AnnotationCollector(Path('dataset') / 'annotations.json')
    
    # Create output directories
    (output_dir / 'human').mkdir(parents=True, exist_ok=True)
    (output_dir / 'animal').mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing zip files...")
    total_stats = {'human': 0, 'animal': 0, 'skipped_noise': 0, 'no_image': 0}
    
    for zip_file in zip_files:
        zip_path = Path(zip_file)
        print(f"\nProcessing: {zip_path.name}")
        stats = process_conservation_drones(
            zip_path, output_dir, ann_collector, filter_noise, filter_occlusion,
            filter_unknown_species, min_bbox_size, max_per_sequence_human, max_per_sequence_animal
        )
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
    
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    
    human_count = len(list((output_dir / 'human').glob('*.jpg'))) if (output_dir / 'human').exists() else 0
    human_count += len(list((output_dir / 'human').glob('*.png'))) if (output_dir / 'human').exists() else 0
    
    animal_count = len(list((output_dir / 'animal').glob('*.jpg'))) if (output_dir / 'animal').exists() else 0
    animal_count += len(list((output_dir / 'animal').glob('*.png'))) if (output_dir / 'animal').exists() else 0
    
    print(f"\nFinal counts:")
    print(f"  Human: {human_count} images")
    print(f"  Animal: {animal_count} images")
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
