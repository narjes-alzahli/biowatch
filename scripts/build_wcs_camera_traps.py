#!/usr/bin/env python3
"""
Build WCS Camera Traps dataset - RGB animal images.

This script:
1. Downloads bounding box annotations from LILA (or uses local zip in zips/)
2. Filters for animal, vehicle, person categories (excludes empty images)
3. Applies quality filters (min bbox size 20px)
4. Uses local downloaded images (--local-images) OR downloads from cloud storage
5. Saves to temp_wcs_camera_traps/ (temp folder) and adds to dataset/

Usage:
    python3 scripts/build_wcs_camera_traps.py [--max-images N] [--target-count N] [--local-images DIR]
    
    --local-images: Use local downloaded images (e.g., wcs_images/) instead of downloading from URLs
    
Note: If download fails, manually download annotations zip to zips/wcs_20220205_bboxes_with_classes.zip
      from: https://lila.science/datasets/wcs-camera-traps
      
      Or download images using: gsutil -m rsync -r gs://public-datasets-lila/wcs-unzipped/animals ./wcs_images/animals
"""

import sys
import json
import requests
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import argparse

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress indicator
    def tqdm(iterable, desc="", total=None, unit=""):
        if total:
            print(f"{desc}: {total:,} {unit}")
        return iterable

from utils import AnnotationCollector

# LILA WCS Camera Traps URLs
# Prefer no_classes version (animal/person/vehicle categories)
BBOX_ANNOTATIONS_URL = "https://lilablobssc.blob.core.windows.net/wcs/wcs_20220205_bboxes_no_classes.zip"
BBOX_ANNOTATIONS_URL_WITH_CLASSES = "https://lilablobssc.blob.core.windows.net/wcs/wcs_20220205_bboxes_with_classes.zip"

# Category mapping - Using no_classes version which has: animal, person, vehicle
# We'll process all three categories (animal, human, vehicle)
# Exclude only "empty" category
EXCLUDED_CATEGORIES = {'empty'}


def download_annotations(output_file):
    """Download WCS bounding box annotations."""
    output_file = Path(output_file)
    
    # Check for local zip file first (prefer no_classes version)
    local_zip = Path('zips/wcs_20220205_bboxes_no_classes.zip')
    if not local_zip.exists():
        # Fallback to with_classes version
        local_zip = Path('zips/wcs_20220205_bboxes_with_classes.zip')
    
    if local_zip.exists():
        print(f"Found local annotations zip: {local_zip}")
        print(f"  Extracting annotations...")
        try:
            import zipfile
            with zipfile.ZipFile(local_zip, 'r') as z:
                # Find the JSON file in the zip
                for name in z.namelist():
                    if name.endswith('.json'):
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(name) as src, open(output_file, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        print(f"✓ Extracted annotations to {output_file}")
                        return output_file
        except Exception as e:
            print(f"  Warning: Could not extract from zip: {e}")
    
    # Download from URL
    print(f"Downloading annotations from LILA...")
    print(f"  URL: {BBOX_ANNOTATIONS_URL}")
    try:
        # Try with headers to avoid 403
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(BBOX_ANNOTATIONS_URL, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        # Save to temp zip first
        temp_zip = output_file.parent / 'wcs_bboxes_temp.zip'
        with open(temp_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract JSON from zip
        import zipfile
        with zipfile.ZipFile(temp_zip, 'r') as z:
            for name in z.namelist():
                if name.endswith('.json'):
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(name) as src, open(output_file, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    break
        
        temp_zip.unlink()
        print(f"✓ Downloaded and extracted annotations to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error downloading annotations: {e}")
        return None


def build_local_file_map(local_source_dir):
    """Build a fast lookup map of all available local image files.
    
    Returns a tuple: (file_map, filename_map)
    - file_map: maps annotation paths to actual file paths
    - filename_map: maps filenames to actual file paths (fallback for sequence ID mismatches)
    This avoids thousands of filesystem stat() calls.
    """
    if not local_source_dir or not Path(local_source_dir).exists():
        return {}, {}
    
    local_source_dir = Path(local_source_dir)
    file_map = {}
    filename_map = {}
    
    print(f"  Building file lookup map from {local_source_dir}...")
    # Find all image files recursively
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_files = []
    for ext in image_extensions:
        all_files.extend(local_source_dir.rglob(f'*{ext}'))
        all_files.extend(local_source_dir.rglob(f'*{ext.upper()}'))
    
    print(f"  Found {len(all_files):,} image files")
    
    # Build map: for each file, create all possible lookup keys
    for file_path in tqdm(all_files, desc="  Indexing files", unit="file", total=len(all_files)):
        # Get relative path from local_source_dir
        try:
            rel_path = file_path.relative_to(local_source_dir)
        except ValueError:
            continue
        
        # Build filename map (fallback for sequence ID mismatches)
        filename = file_path.name
        if filename not in filename_map:  # Use first match found
            filename_map[filename] = file_path
        
        # Create lookup keys for all possible annotation paths
        # Key format: "animals/0019/0918.jpg" or "vehicles/0019/0918.jpg" etc.
        parts = rel_path.parts
        
        # If file is in animals/animals/0019/0918.jpg, create keys:
        # - animals/animals/0019/0918.jpg (exact)
        # - animals/0019/0918.jpg (without nested animals)
        # - vehicles/0019/0918.jpg (alternative category)
        if len(parts) >= 3:
            if parts[0] == 'animals' and parts[1] == 'animals':
                # Nested structure: animals/animals/0019/0918.jpg
                numeric_part = '/'.join(parts[2:])  # "0019/0918.jpg"
                file_map[f'animals/{numeric_part}'] = file_path
                file_map[f'animals/animals/{numeric_part}'] = file_path
                file_map[f'vehicles/{numeric_part}'] = file_path
                file_map[f'humans/{numeric_part}'] = file_path
            elif parts[0] == 'animals':
                # Direct structure: animals/0019/0918.jpg
                numeric_part = '/'.join(parts[1:])  # "0019/0918.jpg"
                file_map[f'animals/{numeric_part}'] = file_path
                file_map[f'animals/animals/{numeric_part}'] = file_path
                file_map[f'vehicles/{numeric_part}'] = file_path
                file_map[f'humans/{numeric_part}'] = file_path
            elif parts[0] in ['vehicles', 'humans']:
                # vehicles/0019/0918.jpg or humans/0019/0918.jpg
                numeric_part = '/'.join(parts[1:])
                file_map[f'{parts[0]}/{numeric_part}'] = file_path
                file_map[f'animals/{numeric_part}'] = file_path
                file_map[f'animals/animals/{numeric_part}'] = file_path
        
        # Also add exact relative path as key
        file_map[str(rel_path)] = file_path
    
    print(f"  ✓ Built lookup map with {len(file_map):,} path entries")
    print(f"  ✓ Built filename map with {len(filename_map):,} filename entries")
    return file_map, filename_map


def copy_local_image(image_path, output_path, local_file_map, filename_map=None):
    """Copy image from local directory using pre-built file map.
    
    Uses pre-built file_map for fast lookups (avoids filesystem stat() calls).
    Falls back to filename-only matching if path doesn't match (handles sequence ID mismatches).
    Returns True if image was found and copied, False otherwise.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not local_file_map:
        return False
    
    import shutil
    
    # Try exact match first
    if image_path in local_file_map:
        source_path = local_file_map[image_path]
        try:
            shutil.copy2(source_path, output_path)
            return True
        except (FileNotFoundError, OSError):
            # File was deleted between map building and now, skip
            pass
    
    # Try alternative paths (handle category mismatches and nested structure)
    if '/' in image_path:
        parts = image_path.split('/')
        if len(parts) >= 2:
            numeric_part = '/'.join(parts[1:])  # e.g., "0019/0918.jpg"
            # Try with different category prefixes
            for prefix in ['animals', 'animals/animals', 'vehicles', 'humans']:
                alt_path = f'{prefix}/{numeric_part}'
                if alt_path in local_file_map:
                    source_path = local_file_map[alt_path]
                    try:
                        shutil.copy2(source_path, output_path)
                        return True
                    except (FileNotFoundError, OSError):
                        # File was deleted, try next
                        continue
    
    # FALLBACK: Match by filename only (handles sequence ID mismatches)
    # Annotation paths may have wrong sequence IDs, but filenames should match
    if filename_map:
        filename = Path(image_path).name
        if filename in filename_map:
            source_path = filename_map[filename]
            try:
                shutil.copy2(source_path, output_path)
                return True
            except (FileNotFoundError, OSError):
                pass
    
    return False


def filter_annotations(annotations_data, min_bbox_size=20, max_images=None, target_count=None):
    """Filter annotations for all categories (animal, person/human, vehicle) with valid bounding boxes."""
    images = annotations_data.get('images', [])
    annotations = annotations_data.get('annotations', [])
    categories = {cat['id']: cat['name'].lower() for cat in annotations_data.get('categories', [])}
    
    # Map categories to our categories
    # no_classes version has: animal, person, vehicle
    # We map: person -> human, animal -> animal, vehicle -> vehicle
    category_mapping = {
        'animal': 'animal',
        'person': 'human',
        'human': 'human',
        'vehicle': 'vehicle',
        'car': 'vehicle',
        'truck': 'vehicle',
        'bicycle': 'vehicle'
    }
    
    print(f"  Filtering {len(annotations):,} annotations for all categories (animal, human, vehicle)...")
    if target_count:
        print(f"  Target: {target_count:,} images per category (will stop early when reached)")
    
    # Step 1: Filter annotations first (much faster than checking all images)
    # Group by image_id and category: valid_annotations_by_image[img_id][category] = [bboxes]
    valid_annotations_by_image = defaultdict(lambda: defaultdict(list))
    total_annotations = len(annotations)
    
    # Use progress bar
    pbar = tqdm(annotations, desc="Filtering annotations", unit="ann", total=total_annotations)
    for ann in pbar:
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, '').lower()
        
        # Skip excluded categories
        if cat_name in EXCLUDED_CATEGORIES:
            continue
        
        # Map to our category
        our_category = category_mapping.get(cat_name, 'animal')  # Default to animal if unknown
        
        # Check bbox size
        bbox = ann['bbox']  # [x, y, width, height]
        if len(bbox) >= 4:
            w, h = bbox[2], bbox[3]
            if w >= min_bbox_size and h >= min_bbox_size:
                # valid_annotations_by_image[img_id][category] is a list of bboxes
                valid_annotations_by_image[ann['image_id']][our_category].append({
                    'bbox': bbox,
                    'area': ann.get('area', w * h)
                })
                
                # Update progress bar with current count
                pbar.set_postfix({'valid_images': len(valid_annotations_by_image)})
                
                # Early exit: if we have enough unique images and have a target, we can stop
                if target_count and len(valid_annotations_by_image) >= target_count * 1.1:
                    # We have 10% more than target (to account for some images not being in image_dict)
                    pbar.close()
                    print(f"    ✓ Found enough images ({len(valid_annotations_by_image):,}), stopping annotation filtering early")
                    break
    
    if not pbar.disable:
        pbar.close()
    
    # Count by category
    category_counts = defaultdict(int)
    for img_anns in valid_annotations_by_image.values():
        for cat in img_anns.keys():
            category_counts[cat] += 1
    
    print(f"  ✓ Found {len(valid_annotations_by_image):,} images with valid annotations:")
    for cat, count in sorted(category_counts.items()):
        print(f"    - {cat}: {count:,} images")
    
    # Step 2: Create image lookup and build valid images list
    # Group by category so we can process each category separately
    image_dict = {img['id']: img for img in images}
    valid_images_by_category = defaultdict(list)
    
    print(f"  Building image list from {len(valid_annotations_by_image):,} valid images...")
    pbar = tqdm(valid_annotations_by_image.items(), desc="Building image list", unit="img", total=len(valid_annotations_by_image))
    for img_id, category_bboxes_dict in pbar:
        if img_id in image_dict:
            # category_bboxes_dict is a dict: {category: [bboxes]}
            # Each image can have annotations for multiple categories
            # Create separate entries for each category
            for category, bboxes in category_bboxes_dict.items():
                valid_images_by_category[category].append({
                    'image': image_dict[img_id],
                    'bboxes': bboxes,
                    'category': category
                })
    
    if not pbar.disable:
        pbar.close()
    
    # Apply limits per category if specified
    if target_count:
        for category in valid_images_by_category:
            valid_images_by_category[category] = valid_images_by_category[category][:target_count]
    elif max_images:
        for category in valid_images_by_category:
            valid_images_by_category[category] = valid_images_by_category[category][:max_images]
    
    return valid_images_by_category


def main():
    """Build WCS Camera Traps dataset."""
    parser = argparse.ArgumentParser(description='Build WCS Camera Traps dataset')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum images to process (default: 10,000)')
    parser.add_argument('--target-count', type=int, default=None,
                       help='Target number of images (default: 10,000)')
    parser.add_argument('--min-bbox-size', type=int, default=20,
                       help='Minimum bounding box size in pixels (default: 20)')
    parser.add_argument('--no-limit', action='store_true',
                       help='Download all available images (no limit)')
    parser.add_argument('--local-images', type=str, default='wcs_images',
                       help='Local directory containing downloaded images (default: wcs_images). If provided, uses local files instead of downloading.')
    
    args = parser.parse_args()
    
    # Set default limit if not specified and --no-limit not used
    if not args.no_limit and args.max_images is None and args.target_count is None:
        args.target_count = 7000  # Default: 7,000 images
    
    print("=" * 70)
    print("WCS Camera Traps Dataset Builder")
    print("=" * 70)
    
    output_dir = Path('dataset')
    min_bbox_size = args.min_bbox_size
    
    # Show current stats and target
    try:
        import json
        with open('dataset/annotations.json', 'r') as f:
            data = json.load(f)
        rgb_animal = len([i for i in data['images'] if 'rgb/animal' in i['file_name']])
        thermal_animal = len([i for i in data['images'] if 'thermal/animal' in i['file_name']])
        print(f"\nCurrent dataset stats:")
        print(f"  RGB Animal: {rgb_animal} annotated images")
        print(f"  Thermal Animal: {thermal_animal} annotated images")
        if args.target_count:
            print(f"  Target: {args.target_count} images (to reach ~{rgb_animal + args.target_count} total)")
    except:
        pass
    
    # Download annotations
    annotations_file = Path('temp_wcs_annotations.json')
    if not annotations_file.exists():
        annotations_file = download_annotations(annotations_file)
        if not annotations_file or not annotations_file.exists():
            print("ERROR: Could not download annotations")
            return 1
    else:
        print(f"Using existing annotations file: {annotations_file}")
    
    # Load annotations
    print(f"\nLoading annotations...")
    print(f"  File size: {annotations_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"  This may take a moment for large files...")
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    print(f"  ✓ Loaded annotations")
    
    print(f"  Total images in dataset: {len(annotations_data.get('images', []))}")
    print(f"  Total annotations: {len(annotations_data.get('annotations', []))}")
    
    # Show categories
    categories = annotations_data.get('categories', [])
    print(f"\n  Categories found: {len(categories)}")
    for cat in categories:
        print(f"    - {cat['name']}")
    
    # Filter for all categories (animal, human, vehicle)
    limit_str = f" (limit: {args.target_count or args.max_images} per category)" if (args.target_count or args.max_images) else ""
    print(f"\nFiltering for all categories (animal, human, vehicle) (min bbox size: {min_bbox_size}px){limit_str}...")
    valid_images_by_category = filter_annotations(
        annotations_data, 
        min_bbox_size=min_bbox_size,
        max_images=args.max_images,
        target_count=args.target_count
    )
    
    total_images = sum(len(imgs) for imgs in valid_images_by_category.values())
    print(f"  Found {total_images} valid images across all categories")
    
    if not valid_images_by_category:
        print("No valid images found!")
        return 1
    
    # Use temp folder for processing
    temp_output_dir = Path('temp_wcs_camera_traps')
    temp_output_dir.mkdir(exist_ok=True)
    
    # Initialize annotation collector for temp folder
    temp_annotations_file = temp_output_dir / 'annotations.json'
    temp_ann_collector = AnnotationCollector(temp_annotations_file)
    
    # Also keep reference to main dataset collector for checking existing files
    ann_collector = AnnotationCollector(output_dir / 'annotations.json')
    
    # Build local file map once (if using local images)
    local_file_map = {}
    local_filename_map = {}
    local_images_dir = Path(args.local_images) if hasattr(args, 'local_images') and Path(args.local_images).exists() else None
    if local_images_dir:
        print(f"\n{'='*70}")
        print(f"Building local file lookup map...")
        print(f"{'='*70}")
        local_file_map, local_filename_map = build_local_file_map(local_images_dir)
        if not local_file_map:
            print(f"  ⚠ Warning: No files found in {local_images_dir}")
    else:
        print(f"\nNo local images directory found")
    
    # Process each category separately
    for category in ['animal', 'human', 'vehicle']:
        if category not in valid_images_by_category:
            continue
        
        valid_images = valid_images_by_category[category]
        print(f"\n{'='*70}")
        print(f"Processing {category.upper()} category: {len(valid_images)} images")
        print(f"{'='*70}")
        
        temp_category_dir = temp_output_dir / category
        temp_category_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing files and annotations for resume capability
        final_category_dir = output_dir / 'rgb' / category
        existing_files = {f.name for f in final_category_dir.glob('*.jpg')} if final_category_dir.exists() else set()
        existing_annotated = {Path(img['file_name']).name for img in ann_collector.images if f'rgb/{category}' in img['file_name']}
        
        print(f"  Found {len(existing_files)} existing files in dataset")
        print(f"  Found {len(existing_annotated)} existing annotated images")
        
        # Process images to temp folder for this category
        if local_file_map:
            print(f"\nCopying {category} images from local directory to temp folder...")
            pbar_desc = f"Copying {category}"
        else:
            print(f"\n⚠ No local images found - cannot process images without local directory")
            print(f"  Please provide --local-images directory with downloaded images")
            continue
        
        downloaded = 0
        skipped = 0
        failed = 0
        temp_annotation_queue = []
        
        pbar = tqdm(valid_images, desc=pbar_desc, unit="img", total=len(valid_images))
        for i, item in enumerate(pbar):
            img = item['image']
            bboxes = item['bboxes']
            category_name = item['category']
            
            # Get image path from annotations
            file_name = img.get('file_name', '')
            if not file_name:
                failed += 1
                pbar.set_postfix({'copied': downloaded, 'failed': failed, 'skipped': skipped})
                continue
            
            output_filename = Path(file_name).name
            
            # Skip if already annotated
            if output_filename in existing_annotated:
                skipped += 1
                pbar.set_postfix({'copied': downloaded, 'failed': failed, 'skipped': skipped})
                continue
            
            # Copy image to temp folder
            temp_path = temp_category_dir / output_filename
            
            # Handle duplicates in temp folder
            if temp_path.exists():
                stem = temp_path.stem
                suffix = temp_path.suffix
                counter = 1
                while temp_path.exists():
                    temp_path = temp_category_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                output_filename = temp_path.name
            
            # Skip if already in final dataset
            if output_filename in existing_files:
                skipped += 1
                pbar.set_postfix({'copied': downloaded, 'failed': failed, 'skipped': skipped})
                continue
            
            # Get additional image metadata for download attempts
            wcs_id = img.get('wcs_id')
            seq_id = img.get('seq_id')
            
            # Copy from local file map
            if not copy_local_image(file_name, temp_path, local_file_map, local_filename_map):
                failed += 1
                # Show periodic status updates
                total_attempts = downloaded + failed
                if total_attempts % 50 == 0 and total_attempts > 0:
                    success_rate = downloaded * 100 // total_attempts if total_attempts > 0 else 0
                    print(f"\n  Status: {downloaded} ✓ | {failed} ✗ | Success rate: {success_rate}%")
                pbar.set_postfix({'copied': downloaded, 'failed': failed, 'skipped': skipped})
                continue
            
            # Get image dimensions
            try:
                with Image.open(temp_path) as im:
                    width, height = im.size
            except Exception as e:
                print(f"  Warning: Could not read image {temp_path.name}: {e}")
                temp_path.unlink()
                failed += 1
                pbar.set_postfix({'copied': downloaded, 'failed': failed, 'skipped': skipped})
                continue
            
            downloaded += 1
            pbar.set_postfix({'copied': downloaded, 'failed': failed, 'skipped': skipped})
            
            # Add to temp annotation collector
            # Convert bboxes format: from list of dicts with 'bbox' to format expected by add_image
            bbox_list = []
            for bbox_info in bboxes:
                bbox_list.append({
                    'bbox': bbox_info['bbox'],  # COCO format: [x, y, width, height]
                    'category': category_name,
                    'area': bbox_info.get('area', bbox_info['bbox'][2] * bbox_info['bbox'][3])
                })
            
            # File path relative to temp_output_dir for annotations
            rel_path = temp_path.relative_to(temp_output_dir)
            temp_ann_collector.add_image(
                file_path=f'rgb/{category_name}/{rel_path.name}',
                modality='rgb',
                category=category_name,
                width=width,
                height=height,
                bboxes=bbox_list
            )
            
            # Also keep annotation queue for backward compatibility
            temp_annotation_queue.append({
                'file_name': str(rel_path),
                'bboxes': bboxes,
                'width': width,
                'height': height,
                'category': category_name
            })
            
            if args.target_count and downloaded >= args.target_count:
                break
        
        pbar.close()
        print(f"\n✓ Copied {downloaded} {category} images to temp folder")
        if skipped > 0:
            print(f"  Skipped: {skipped} images (already in dataset)")
        if failed > 0:
            print(f"  ⚠ Failed: {failed} images")
            print(f"\n⚠ WARNING: {failed} {category} images could not be found in local directory.")
            print(f"  The file_name in annotations may not match actual local file paths.")
            print(f"\n  Solutions:")
            print(f"  1. Check that all images are downloaded to {args.local_images}/")
            print(f"  2. Verify the directory structure matches annotation paths")
            print(f"  3. Some images may be in nested directories (e.g., animals/animals/...)")
        
        # Save annotation queue for this category
        annotation_queue_file = temp_output_dir / f'annotation_queue_{category}.json'
        with open(annotation_queue_file, 'w') as f:
            json.dump(temp_annotation_queue, f, indent=2)
        
        print(f"  Saved annotation data for {len(temp_annotation_queue)} {category} images")
        print(f"  Annotation queue: {annotation_queue_file}")
    
    # Save final annotations.json file
    print(f"\n{'='*70}")
    print(f"Saving annotations.json...")
    print(f"{'='*70}")
    temp_ann_collector.save()
    
    print(f"\n✓ Files are in: {temp_output_dir}/")
    print(f"  - animal/")
    print(f"  - human/")
    print(f"  - vehicle/")
    print(f"  - annotations.json")
    print(f"\nNote: Files are in temp folder. Review before adding to main dataset.")
    
    print("\n" + "=" * 70)
    print("COPY COMPLETE")
    print("=" * 70)
    print(f"\nFiles are in temp folder: {temp_output_dir}/")
    print(f"  - animal/")
    print(f"  - human/")
    print(f"  - vehicle/")
    print(f"  - annotations.json ({len(temp_ann_collector.images)} images, {len(temp_ann_collector.annotations)} annotations)")
    print(f"\nAnnotation queues (for reference):")
    print(f"  - {temp_output_dir}/annotation_queue_animal.json")
    print(f"  - {temp_output_dir}/annotation_queue_human.json")
    print(f"  - {temp_output_dir}/annotation_queue_vehicle.json")
    print(f"\nTo add to main dataset later, use a separate script or integrate manually.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

