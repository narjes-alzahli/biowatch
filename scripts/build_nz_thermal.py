#!/usr/bin/env python3
"""
Build New Zealand Wildlife Thermal Imaging dataset - selective video frame extraction.

This script:
1. Downloads metadata (5MB JSON file)
2. Analyzes which videos contain animal categories
3. Selectively downloads videos (not the full 33GB dataset)
4. Extracts frames from videos at track locations
5. Converts track trajectories to bounding boxes
6. Filters by minimum bbox size (20px)
7. Saves images and annotations

Usage:
    python3 scripts/build_nz_thermal.py [--max-videos N] [--target-count N]
"""

import sys
import json
import requests
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from utils import AnnotationCollector


# Animal categories to extract (excluding false positives, non-animal, and birds)
# Prioritizing big mammals, but including other mammals too
ANIMAL_CATEGORIES = {
    # Big mammals (priority)
    'deer', 'pig', 'goat', 'sheep', 'cow', 'dog', 'horse',
    # Medium mammals
    'possum', 'cat', 'hedgehog', 'rabbit', 'hare',
    # Small mammals
    'rodent', 'mustelid', 'stoat', 'ferret', 'weasel', 'rat', 'mouse'
    # Excluded: 'bird' - user requested no birds
}

# Base URLs for downloading
# Try multiple possible locations
METADATA_URLS = [
    "https://lilablobssc.blob.core.windows.net/nz-thermal/metadata.json",
    "https://storage.googleapis.com/public-datasets-lila/nz-thermal/metadata.json",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/nz-thermal/metadata.json"
]
VIDEO_BASE_URLS = [
    "https://lilablobssc.blob.core.windows.net/nz-thermal/videos",
    "https://storage.googleapis.com/public-datasets-lila/nz-thermal/videos",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/nz-thermal/videos"
]


def download_metadata(output_file):
    """Download or extract the main metadata file."""
    output_file = Path(output_file)
    
    # Check for local metadata file first
    local_metadata_zip = Path('zips/new-zealand-wildlife-thermal-imaging-metadata.json.zip')
    local_metadata_json = Path('zips/metadata.json')
    
    # Try local zip file
    if local_metadata_zip.exists():
        print(f"Found local metadata zip: {local_metadata_zip}")
        print(f"  Extracting metadata.json...")
        try:
            import zipfile
            with zipfile.ZipFile(local_metadata_zip, 'r') as z:
                # Look for metadata.json in the zip (filename is "new-zealand-wildlife-thermal-imaging.json")
                for name in z.namelist():
                    if 'metadata' in name.lower() or name.endswith('.json'):
                        # Extract to temp location first
                        temp_extract = output_file.parent / 'temp_metadata_extract'
                        temp_extract.mkdir(parents=True, exist_ok=True)
                        z.extract(name, temp_extract)
                        extracted_path = temp_extract / name
                        if extracted_path.exists():
                            # Ensure output directory exists
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(extracted_path, output_file)
                            shutil.rmtree(temp_extract, ignore_errors=True)
                            print(f"✓ Extracted metadata to {output_file}")
                            return output_file
        except Exception as e:
            print(f"  Warning: Could not extract from zip: {e}")
            import traceback
            traceback.print_exc()
    
    # Try local JSON file
    if local_metadata_json.exists():
        print(f"Found local metadata file: {local_metadata_json}")
        shutil.copy2(local_metadata_json, output_file)
        print(f"✓ Using local metadata: {output_file}")
        return output_file
    
    # Try to download from URLs
    for url in METADATA_URLS:
        try:
            print(f"Trying metadata URL: {url}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Metadata downloaded to {output_file}")
            return output_file
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    raise Exception(f"Could not find or download metadata. Please ensure metadata file is available.")


def analyze_metadata(metadata_file, target_count=None):
    """
    Analyze metadata to find videos with animal categories.
    Returns list of video info sorted by animal track count.
    """
    print(f"\nAnalyzing metadata...")
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    clips = data.get('clips', [])
    print(f"  Total videos: {len(clips)}")
    
    # Filter videos with animal categories
    animal_videos = []
    category_counts = defaultdict(int)
    
    for clip in clips:
        if clip.get('error'):
            continue
        
        labels = clip.get('labels', [])
        tracks = clip.get('tracks', [])
        
        # Check if video has animal categories
        animal_labels = [l for l in labels if l.lower() in ANIMAL_CATEGORIES]
        if not animal_labels:
            continue
        
        # Count animal tracks
        animal_tracks = []
        for track in tracks:
            track_tags = track.get('tags', [])
            for tag in track_tags:
                label = tag.get('label', '').lower()
                if label in ANIMAL_CATEGORIES:
                    animal_tracks.append(track)
                    category_counts[label] += 1
        
        if animal_tracks:
            animal_videos.append({
                'id': clip['id'],
                'labels': animal_labels,
                'tracks': animal_tracks,
                'track_count': len(animal_tracks),
                'width': clip.get('width'),
                'height': clip.get('height'),
                'filtered_video_filename': clip.get('filtered_video_filename'),
                'metadata_filename': clip.get('metadata_filename')
            })
    
    # Sort by track count (descending)
    animal_videos.sort(key=lambda x: x['track_count'], reverse=True)
    
    print(f"  Videos with animals: {len(animal_videos)}")
    print(f"  Total animal tracks: {sum(v['track_count'] for v in animal_videos)}")
    print(f"\n  Top animal categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {cat}: {count} tracks")
    
    # Create diverse video selection: prioritize big mammals, then mix other categories
    # Group videos by primary animal category
    videos_by_category = defaultdict(list)
    big_mammals = {'deer', 'pig', 'goat', 'sheep', 'cow', 'dog', 'horse'}
    
    for video in animal_videos:
        # Get primary category (most common label in video)
        if video['labels']:
            primary_cat = video['labels'][0].lower()
            videos_by_category[primary_cat].append(video)
    
    # Prioritize big mammals first, then interleave other categories
    diverse_videos = []
    
    # First, add big mammals (prioritized)
    for cat in sorted(big_mammals):
        if cat in videos_by_category:
            diverse_videos.extend(videos_by_category[cat])
    
    # Then interleave other mammal categories
    other_cats = [cat for cat in sorted(videos_by_category.keys()) if cat not in big_mammals]
    max_per_category = max(len(videos_by_category[cat]) for cat in other_cats) if other_cats else 0
    
    for i in range(max_per_category):
        for cat in other_cats:
            if i < len(videos_by_category[cat]):
                diverse_videos.append(videos_by_category[cat][i])
    
    # Add any remaining videos
    seen_ids = {v['id'] for v in diverse_videos}
    for video in animal_videos:
        if video['id'] not in seen_ids:
            diverse_videos.append(video)
    
    # Estimate how many videos we need
    if target_count:
        avg_tracks_per_video = sum(v['track_count'] for v in diverse_videos) / len(diverse_videos) if diverse_videos else 0
        estimated_videos = int(target_count / avg_tracks_per_video * 1.2) if avg_tracks_per_video > 0 else 0  # 20% buffer
        print(f"\n  Target: {target_count} images")
        print(f"  Estimated videos needed: ~{estimated_videos}")
        print(f"  (Will stop when target is reached)")
    
    print(f"\n  Diverse selection: {len(diverse_videos)} videos (interleaved by category)")
    return diverse_videos
    


def download_video(video_id, output_dir, use_filtered=True):
    """Download a single video file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_filtered:
        filename = f"{video_id}_filtered.mp4"
    else:
        filename = f"{video_id}.mp4"
    
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"    Video already exists: {filename}")
        return output_path
    
    # Try multiple base URLs
    for base_url in VIDEO_BASE_URLS:
        url = f"{base_url}/{filename}"
        try:
            print(f"    Downloading {filename} from {base_url}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r      Progress: {percent:.1f}%", end='', flush=True)
            
            print(f"\r      ✓ Downloaded {filename} ({downloaded / 1024 / 1024:.1f} MB)")
            return output_path
        except Exception as e:
            print(f"\n      Failed from {base_url}: {e}")
            if output_path.exists():
                output_path.unlink()
            continue
    
    print(f"      Error: Could not download {filename} from any URL")
    return None


def download_clip_metadata(video_id, output_dir):
    """Download individual clip metadata (contains track coordinates)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to get metadata filename from main metadata
    # For now, we'll construct it
    filename = f"{video_id}.json"
    
    output_path = output_dir / filename
    
    if output_path.exists():
        return output_path
    
    # Try multiple base URLs
    for base_url in VIDEO_BASE_URLS:
        # Remove /videos and add filename
        metadata_base = base_url.replace('/videos', '')
        url = f"{metadata_base}/{filename}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return output_path
        except:
            continue
    
    return None


def extract_track_bboxes(track_points, width, height):
    """
    Convert track points to bounding boxes.
    Track points are (x, y, frame) triplets.
    Returns list of bboxes per frame in COCO format [x, y, w, h].
    """
    if not track_points:
        return {}
    
    # Group points by frame
    frames_dict = defaultdict(list)
    for point in track_points:
        if len(point) >= 3:
            x, y, frame = point[0], point[1], int(point[2])
            if 0 <= x < width and 0 <= y < height:
                frames_dict[frame].append((x, y))
    
    # Create bboxes for each frame
    bboxes = {}
    for frame, points in frames_dict.items():
        if len(points) < 2:
            continue
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        x_min = max(0, int(min(xs)))
        y_min = max(0, int(min(ys)))
        x_max = min(width - 1, int(max(xs)))
        y_max = min(height - 1, int(max(ys)))
        
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        
        if w >= 20 and h >= 20:  # Min bbox size filter
            bboxes[frame] = [x_min, y_min, w, h]
    
    return bboxes


def extract_frames_from_video(video_path, frame_numbers, output_dir, video_id):
    """
    Extract specific frames from video using ffmpeg or opencv.
    Returns dict mapping frame_number to output image path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not frame_numbers:
        return {}
    
    frame_list = sorted(set(frame_numbers))
    extracted = {}
    
    # Try opencv first (if available)
    if HAS_OPENCV:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Could not open video")
            
            for frame_num in frame_list:
                output_path = output_dir / f"{video_id}_frame_{frame_num:06d}.jpg"
                
                if output_path.exists():
                    extracted[frame_num] = output_path
                    continue
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    cv2.imwrite(str(output_path), frame)
                    extracted[frame_num] = output_path
                else:
                    print(f"      Warning: Could not read frame {frame_num}")
            
            cap.release()
            return extracted
        except Exception as e:
            print(f"      OpenCV extraction failed: {e}, trying ffmpeg...")
    
    # Fallback to ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode != 0:
            raise Exception("ffmpeg not found")
    except:
        print("      ERROR: Neither OpenCV nor ffmpeg available!")
        print("      Please install one of:")
        print("        - pip install opencv-python")
        print("        - brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
        return {}
    
    # Extract frames using ffmpeg
    for frame_num in frame_list:
        output_path = output_dir / f"{video_id}_frame_{frame_num:06d}.jpg"
        
        if output_path.exists():
            extracted[frame_num] = output_path
            continue
        
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', f'select=eq(n\\,{frame_num})',
                '-vsync', 'vfr',
                '-frames:v', '1',
                '-q:v', '2',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and output_path.exists():
                extracted[frame_num] = output_path
            else:
                print(f"      Warning: Failed to extract frame {frame_num}")
        except Exception as e:
            print(f"      Error extracting frame {frame_num}: {e}")
    
    return extracted


def process_video(video_info, temp_dir, output_dir, ann_collector, min_bbox_size=20):
    """
    Process a single video: download, extract frames, create annotations.
    Returns number of images added.
    """
    video_id = video_info['id']
    tracks = video_info['tracks']
    width = video_info.get('width', 640)
    height = video_info.get('height', 480)
    
    print(f"\n  Processing video {video_id}...")
    print(f"    {len(tracks)} animal tracks")
    
    # Download video - try non-filtered first (better quality), fallback to filtered
    video_path = download_video(video_id, temp_dir / 'videos', use_filtered=False)
    if not video_path:
        # Fallback to filtered video
        video_path = download_video(video_id, temp_dir / 'videos', use_filtered=True)
    if not video_path:
        return 0
    
    # Download clip metadata for track coordinates
    clip_metadata_path = download_clip_metadata(video_id, temp_dir / 'metadata')
    
    # Load track coordinates if available
    track_coords = {}
    if clip_metadata_path and clip_metadata_path.exists():
        try:
            with open(clip_metadata_path, 'r') as f:
                clip_data = json.load(f)
                for track in clip_data.get('tracks', []):
                    track_id = track.get('start_frame', 0)  # Use start_frame as ID
                    points = track.get('points', [])
                    if points:
                        track_coords[track_id] = points
        except:
            pass
    
    # Collect frames to extract
    frames_to_extract = set()
    frame_bboxes = {}  # frame_num -> list of bboxes
    
    # Better frame sampling: extract 1 frame per track, spaced out
    # This avoids similar frames from overlapping tracks
    for track in tracks:
        start_frame = track.get('start_frame', 0)
        end_frame = track.get('end_frame', 0)
        
        # Extract 1 frame per track (middle of track for best visibility)
        # This gives us diversity across different tracks
        if end_frame > start_frame:
            # Use middle frame of track (best chance of clear object)
            middle_frame = start_frame + (end_frame - start_frame) // 2
            sample_frames = [middle_frame]
        else:
            sample_frames = [start_frame]
        
        # Get track points if available
        track_id = start_frame
        points = track_coords.get(track_id, [])
        
        # Convert points to bboxes
        if points:
            bboxes = extract_track_bboxes(points, width, height)
            for frame_num in sample_frames:
                if frame_num in bboxes:
                    frames_to_extract.add(frame_num)
                    if frame_num not in frame_bboxes:
                        frame_bboxes[frame_num] = []
                    frame_bboxes[frame_num].append(bboxes[frame_num])
        else:
            # If no track coordinates, use full-frame bbox (common for camera trap videos)
            # Animals are usually the main subject, so full frame is reasonable
            for frame_num in sample_frames:
                frames_to_extract.add(frame_num)
                if frame_num not in frame_bboxes:
                    frame_bboxes[frame_num] = []
                # Use full frame as bbox for camera trap videos
                # Animals are usually the main subject in thermal camera traps
                # Full frame bbox is reasonable for this use case
                bbox = [0, 0, width, height]
                if bbox[2] >= 20 and bbox[3] >= 20:  # Min size check
                    frame_bboxes[frame_num].append(bbox)
    
    if not frames_to_extract:
        print(f"    No valid frames to extract")
        return 0
    
    # Extract frames
    print(f"    Extracting {len(frames_to_extract)} frames...")
    extracted_frames = extract_frames_from_video(
        video_path, 
        list(frames_to_extract), 
        temp_dir / 'frames',
        video_id
    )
    
    # Delete video immediately after extraction to save space
    if video_path.exists():
        try:
            video_path.unlink()
            print(f"    ✓ Deleted video file to save space")
        except Exception as e:
            print(f"    Warning: Could not delete video: {e}")
    
    # Copy to output and add annotations
    images_added = 0
    output_animal_dir = output_dir / 'thermal' / 'animal'
    output_animal_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_num, frame_path in extracted_frames.items():
        if not frame_path.exists():
            continue
        
        # Get image dimensions
        try:
            with Image.open(frame_path) as img:
                img_width, img_height = img.size
        except:
            continue
        
        # Get bboxes for this frame
        bboxes = frame_bboxes.get(frame_num, [])
        
        # If no bboxes, skip (we need annotations)
        if not bboxes:
            continue
        
        # Copy image
        output_filename = f"nz_thermal_{video_id}_frame_{frame_num:06d}.jpg"
        output_path = output_animal_dir / output_filename
        
        # Skip if already exists (resume capability)
        if output_path.exists():
            continue
        
        shutil.copy2(frame_path, output_path)
        
        # Add annotations
        if ann_collector:
            animal_bboxes = [
                {'category': 'animal', 'bbox': bbox, 'area': bbox[2] * bbox[3]}
                for bbox in bboxes
            ]
            rel_path = Path('thermal') / 'animal' / output_filename
            ann_collector.add_image(rel_path, 'thermal', 'animal', img_width, img_height, animal_bboxes)
        
        images_added += 1
    
    print(f"    ✓ Added {images_added} images")
    return images_added


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build NZ Thermal dataset')
    parser.add_argument('--max-videos', type=int, default=5000, 
                       help='Maximum number of videos to process (default: 5000, or unlimited if target-count is set)')
    parser.add_argument('--target-count', type=int, default=10000,
                       help='Target number of images to extract (default: 10000, will stop when reached)')
    parser.add_argument('--min-bbox-size', type=int, default=20,
                       help='Minimum bounding box size in pixels (default: 20)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("New Zealand Wildlife Thermal Imaging Dataset Builder")
    print("=" * 70)
    
    # Check dependencies
    if not HAS_OPENCV:
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
            print("✓ ffmpeg found")
        except:
            print("⚠ WARNING: Neither OpenCV nor ffmpeg found!")
            print("  Please install one:")
            print("    pip install opencv-python")
            print("    OR: brew install ffmpeg (macOS)")
            return 1
    else:
        print("✓ OpenCV found")
    
    min_bbox_size = args.min_bbox_size
    max_videos = args.max_videos
    target_count = args.target_count
    
    print(f"\nConfiguration:")
    print(f"  Min bbox size: {min_bbox_size}x{min_bbox_size} pixels")
    print(f"  Max videos to process: {max_videos}")
    if target_count:
        print(f"  Target image count: {target_count}")
    print("=" * 70)
    
    # Setup directories
    temp_dir = Path('temp_nz_thermal')
    output_dir = Path('temp_nz_thermal_output')  # Temporary output folder
    metadata_file = temp_dir / 'metadata.json'
    
    # Download metadata
    if not metadata_file.exists():
        download_metadata(metadata_file)
    else:
        print(f"\nUsing existing metadata: {metadata_file}")
    
    # Analyze metadata
    animal_videos = analyze_metadata(metadata_file, target_count)
    
    if not animal_videos:
        print("\nNo animal videos found!")
        return 1
    
    # Limit number of videos (unless target_count is set, then use all available)
    if target_count:
        # When target_count is set, process all available videos until target is reached
        videos_to_process = animal_videos
        print(f"\nProcessing up to {len(videos_to_process)} videos (until {target_count} images reached)...")
    else:
        videos_to_process = animal_videos[:max_videos]
        print(f"\nProcessing {len(videos_to_process)} videos...")
    
    # Initialize annotation collector (use temporary output directory)
    ann_collector = AnnotationCollector(output_dir / 'annotations.json')
    
    # Check which videos have already been processed (by checking for output images)
    output_animal_dir = output_dir / 'thermal' / 'animal'
    processed_video_ids = set()
    if output_animal_dir.exists():
        # Extract video IDs from existing filenames (format: nz_thermal_{video_id}_frame_*.jpg)
        for img_file in output_animal_dir.glob('nz_thermal_*_frame_*.jpg'):
            # Extract video_id from filename: nz_thermal_{video_id}_frame_*.jpg
            parts = img_file.stem.split('_')
            if len(parts) >= 3:
                try:
                    video_id = int(parts[2])  # nz_thermal_{video_id}_frame_...
                    processed_video_ids.add(video_id)
                except ValueError:
                    pass
        if processed_video_ids:
            print(f"\nFound {len(processed_video_ids)} already-processed videos (will skip)")
    
    # Process videos
    total_images = len(ann_collector.images)  # Start with existing count
    processed = 0
    skipped = 0
    
    for video_info in videos_to_process:
        if target_count and total_images >= target_count:
            print(f"\n✓ Target count reached: {total_images} images")
            break
        
        video_id = video_info['id']
        
        # Skip if already processed
        if video_id in processed_video_ids:
            skipped += 1
            continue
        
        images_added = process_video(
            video_info, 
            temp_dir, 
            output_dir, 
            ann_collector,
            min_bbox_size
        )
        
        total_images += images_added
        processed += 1
        
        print(f"  Progress: {processed}/{len(videos_to_process)} videos, {total_images} images (skipped: {skipped})")
    
    # Save annotations
    print(f"\nSaving annotations...")
    ann_collector.save()
    
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"  Videos processed: {processed}")
    print(f"  Images added: {total_images}")
    print(f"  Output location: {output_dir}/thermal/animal/")
    print(f"  Annotations: {output_dir}/annotations.json")
    print("=" * 70)
    
    print(f"\nOutput files:")
    print(f"  Frames: {output_dir}/thermal/animal/")
    print(f"  Annotations: {output_dir}/annotations.json")
    print(f"  Temporary extraction files: {temp_dir}/ (can be deleted)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

