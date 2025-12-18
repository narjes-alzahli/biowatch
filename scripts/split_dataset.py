#!/usr/bin/env python3
"""
Split images from temporary folders into train/val/test splits.

This script takes images from dataset_temp/ and splits them into
train (70%), val (15%), and test (15%) sets, maintaining proper
distribution across categories.

Usage:
    python3 split_dataset.py [--train-ratio 0.7] [--val-ratio 0.15] [--test-ratio 0.15]
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter


def get_image_files(directory):
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    files = []
    for ext in image_extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(files)


def get_sequence_id(filepath):
    """
    Extract video sequence ID from filename.
    Format: {video_id}_{sequence_id}_{frame_number}.jpg
    Returns: (video_id, sequence_id) tuple or None
    """
    filename = Path(filepath).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        return (parts[0], parts[1])
    return None


def group_by_sequence(images):
    """Group images by video sequence to avoid data leakage."""
    sequences = {}
    for img in images:
        seq_id = get_sequence_id(img)
        if seq_id:
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append(img)
        else:
            # For images without sequence info, use filename as sequence
            seq_id = ('unknown', Path(img).stem)
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append(img)
    
    # Sort images within each sequence by filename
    for seq_id in sequences:
        sequences[seq_id].sort()
    
    return sequences


def split_images(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, split_by_sequence=True):
    """
    Split images from source directory into train/val/test directories.
    
    Args:
        source_dir: Source directory containing images
        train_dir: Destination train directory
        val_dir: Destination val directory
        test_dir: Destination test directory
        train_ratio: Ratio for training set (default 0.7)
        val_ratio: Ratio for validation set (default 0.15)
        test_ratio: Ratio for test set (default 0.15)
        split_by_sequence: Split by video sequence to avoid data leakage (default True)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Get all images
    images = get_image_files(source_dir)
    total = len(images)
    
    if total == 0:
        print(f"  No images found in {source_dir}")
        return 0, 0, 0
    
    # Create destination directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    if split_by_sequence:
        # Group by sequence to avoid data leakage
        sequences = group_by_sequence(images)
        sequence_list = list(sequences.items())
        random.shuffle(sequence_list)  # Shuffle sequences, not individual images
        
        # Calculate split based on sequences
        total_sequences = len(sequence_list)
        train_seq_count = int(total_sequences * train_ratio)
        val_seq_count = int(total_sequences * val_ratio)
        
        # Split sequences
        train_sequences = sequence_list[:train_seq_count]
        val_sequences = sequence_list[train_seq_count:train_seq_count + val_seq_count]
        test_sequences = sequence_list[train_seq_count + val_seq_count:]
        
        # Flatten to get images
        train_images = [img for seq_id, imgs in train_sequences for img in imgs]
        val_images = [img for seq_id, imgs in val_sequences for img in imgs]
        test_images = [img for seq_id, imgs in test_sequences for img in imgs]
    else:
        # Random split (original behavior)
        random.shuffle(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
    
    # Copy files
    for img in train_images:
        shutil.copy2(img, train_dir)
    
    for img in val_images:
        shutil.copy2(img, val_dir)
    
    for img in test_images:
        shutil.copy2(img, test_dir)
    
    return len(train_images), len(val_images), len(test_images)


def main():
    parser = argparse.ArgumentParser(description='Split dataset from temp folders into train/val/test')
    parser.add_argument('--source', type=str, default='dataset_temp',
                       help='Source directory (default: dataset_temp)')
    parser.add_argument('--dest', type=str, default='dataset',
                       help='Destination directory (default: dataset)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually copying files')
    parser.add_argument('--no-sequence-split', action='store_true',
                       help='Disable sequence-based splitting (use random split instead). Use this only if images are NOT from videos.')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    split_by_sequence = not args.no_sequence_split
    
    source_path = Path(args.source)
    dest_path = Path(args.dest)
    
    if not source_path.exists():
        print(f"Error: Source directory '{args.source}' does not exist!")
        return
    
    print(f"Splitting dataset from '{args.source}' to '{args.dest}'")
    print(f"Ratios: Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}, Test={args.test_ratio:.1%}")
    print(f"Split method: {'By video sequence' if split_by_sequence else 'Random'} (prevents data leakage from similar video frames)")
    print(f"Random seed: {args.seed}")
    print("-" * 60)
    
    # Statistics
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
    
    # Process each modality and category
    for modality in ['rgb', 'thermal']:
        modality_source = source_path / modality
        if not modality_source.exists():
            print(f"\nSkipping {modality} (directory not found)")
            continue
        
        print(f"\nProcessing {modality.upper()} modality:")
        
        for category in ['human', 'animal', 'vehicle']:
            category_source = modality_source / category
            if not category_source.exists():
                print(f"  Skipping {category} (directory not found)")
                continue
            
            # Destination paths
            train_dest = dest_path / modality / 'train' / category
            val_dest = dest_path / modality / 'val' / category
            test_dest = dest_path / modality / 'test' / category
            
            print(f"  {category.capitalize()}:", end=' ')
            
            if args.dry_run:
                images = get_image_files(category_source)
                total = len(images)
                if split_by_sequence:
                    sequences = group_by_sequence(images)
                    total_sequences = len(sequences)
                    train_seq = int(total_sequences * args.train_ratio)
                    val_seq = int(total_sequences * args.val_ratio)
                    test_seq = total_sequences - train_seq - val_seq
                    train_count = int(total * args.train_ratio)
                    val_count = int(total * args.val_ratio)
                    test_count = total - train_count - val_count
                    print(f"{total} images ({total_sequences} sequences) -> Train: ~{train_count}, Val: ~{val_count}, Test: ~{test_count}")
                else:
                    train_count = int(total * args.train_ratio)
                    val_count = int(total * args.val_ratio)
                    test_count = total - train_count - val_count
                    print(f"{total} images -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
                stats[f"{modality}/{category}"] = {
                    'train': train_count,
                    'val': val_count,
                    'test': test_count,
                    'total': total
                }
            else:
                train_count, val_count, test_count = split_images(
                    category_source,
                    train_dest,
                    val_dest,
                    test_dest,
                    args.train_ratio,
                    args.val_ratio,
                    args.test_ratio,
                    split_by_sequence
                )
                total = train_count + val_count + test_count
                print(f"{total} images -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
                stats[f"{modality}/{category}"] = {
                    'train': train_count,
                    'val': val_count,
                    'test': test_count,
                    'total': total
                }
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    total_all = sum(s['total'] for s in stats.values())
    
    print(f"\nTotal images: {total_all}")
    print(f"  Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"  Val:   {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"  Test:  {total_test} ({total_test/total_all*100:.1f}%)")
    
    print("\nBy category:")
    for key, s in sorted(stats.items()):
        print(f"  {key:20s}: {s['total']:5d} total -> "
              f"Train: {s['train']:5d}, Val: {s['val']:5d}, Test: {s['test']:5d}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were actually moved. Run without --dry-run to perform the split.")
    else:
        print("\n✓ Split complete!")


if __name__ == '__main__':
    main()
