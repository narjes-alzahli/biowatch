#!/usr/bin/env python3
"""
Re-split YOLO dataset to have better train/val ratio (80/20 instead of 92.8/7.2).
Moves images and labels from train to val to achieve desired split.
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def resplit_yolo_dataset(dataset_root, target_val_ratio=0.20, seed=42):
    """
    Re-split YOLO dataset to achieve target validation ratio.
    
    Args:
        dataset_root: Root directory of YOLO dataset (e.g., 'dataset_yolo_6ch')
        target_val_ratio: Target validation ratio (default: 0.20 = 20%)
        seed: Random seed for reproducibility
    """
    dataset_root = Path(dataset_root)
    train_img_dir = dataset_root / 'images' / 'train'
    val_img_dir = dataset_root / 'images' / 'val'
    train_label_dir = dataset_root / 'labels' / 'train'
    val_label_dir = dataset_root / 'labels' / 'val'
    
    # Check directories exist
    if not train_img_dir.exists():
        print(f"❌ Error: {train_img_dir} does not exist")
        return False
    
    # Create val directories if they don't exist
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all training images
    train_images = list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.npz'))
    total_images = len(train_images)
    
    if total_images == 0:
        print(f"❌ Error: No images found in {train_img_dir}")
        return False
    
    # Count current split
    val_images_existing = list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.npz'))
    current_val_count = len(val_images_existing)
    current_total = total_images + current_val_count
    current_val_ratio = current_val_count / current_total if current_total > 0 else 0
    
    print(f"\n📊 Current Split:")
    print(f"   Train: {total_images:,} images ({100*(1-current_val_ratio):.1f}%)")
    print(f"   Val: {current_val_count:,} images ({100*current_val_ratio:.1f}%)")
    print(f"   Total: {current_total:,} images")
    
    # Calculate target counts
    target_val_count = int(current_total * target_val_ratio)
    target_train_count = current_total - target_val_count
    
    # Calculate how many to move
    need_to_move = target_val_count - current_val_count
    
    if need_to_move <= 0:
        print(f"\n✅ Dataset already has {current_val_ratio*100:.1f}% validation (target: {target_val_ratio*100:.1f}%)")
        print(f"   No changes needed!")
        return True
    
    print(f"\n🎯 Target Split:")
    print(f"   Train: {target_train_count:,} images ({100*(1-target_val_ratio):.1f}%)")
    print(f"   Val: {target_val_count:,} images ({100*target_val_ratio:.1f}%)")
    print(f"\n📦 Need to move {need_to_move:,} images from train to val")
    
    # Confirm
    response = input(f"\nProceed with re-splitting? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return False
    
    # Set random seed
    random.seed(seed)
    
    # Shuffle and select images to move
    train_images_shuffled = train_images.copy()
    random.shuffle(train_images_shuffled)
    images_to_move = train_images_shuffled[:need_to_move]
    
    print(f"\n🔄 Moving {len(images_to_move):,} images and labels...")
    
    moved_count = 0
    for img_path in tqdm(images_to_move, desc="Moving files"):
        # Get corresponding label file
        label_name = img_path.stem + '.txt'
        label_path = train_label_dir / label_name
        
        # Move image
        try:
            shutil.move(str(img_path), str(val_img_dir / img_path.name))
            
            # Move label if it exists
            if label_path.exists():
                shutil.move(str(label_path), str(val_label_dir / label_name))
            else:
                # Create empty label file if it doesn't exist
                (val_label_dir / label_name).touch()
            
            moved_count += 1
        except Exception as e:
            print(f"\n⚠️  Warning: Could not move {img_path.name}: {e}")
    
    # Verify final split
    final_train_count = len(list(train_img_dir.glob('*.jpg'))) + len(list(train_img_dir.glob('*.npz')))
    final_val_count = len(list(val_img_dir.glob('*.jpg'))) + len(list(val_img_dir.glob('*.npz')))
    final_total = final_train_count + final_val_count
    final_val_ratio = final_val_count / final_total if final_total > 0 else 0
    
    print(f"\n✅ Re-splitting complete!")
    print(f"\n📊 Final Split:")
    print(f"   Train: {final_train_count:,} images ({100*(1-final_val_ratio):.1f}%)")
    print(f"   Val: {final_val_count:,} images ({100*final_val_ratio:.1f}%)")
    print(f"   Total: {final_total:,} images")
    print(f"   Moved: {moved_count:,} images")
    
    # Clear cache files (YOLO will regenerate them)
    cache_files = list(dataset_root.rglob('*.cache'))
    if cache_files:
        print(f"\n🗑️  Clearing {len(cache_files)} cache files (YOLO will regenerate)...")
        for cache_file in cache_files:
            cache_file.unlink()
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Re-split YOLO dataset for better train/val ratio')
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset_yolo_6ch'),
                       help='Root directory of YOLO dataset')
    parser.add_argument('--val-ratio', type=float, default=0.20,
                       help='Target validation ratio (default: 0.20 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("YOLO Dataset Re-splitting")
    print("=" * 70)
    
    success = resplit_yolo_dataset(args.dataset_root, args.val_ratio, args.seed)
    
    if success:
        print("\n" + "=" * 70)
        print("✅ Dataset re-split complete! Ready for training.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Re-splitting failed or cancelled.")
        print("=" * 70)

