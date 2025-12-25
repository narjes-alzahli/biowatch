"""
COCO-format dataset loader with RGB/thermal support.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from torchvision import transforms


class BioWatchDataset(Dataset):
    """
    Dataset loader for BioWatch multi-modal object detection.
    
    Supports:
    - RGB-only images
    - Thermal-only images
    - RGB+thermal paired images (when available)
    """
    
    def __init__(
        self,
        annotations_file: Path,
        dataset_root: Path,
        input_size: Tuple[int, int] = (640, 640),
        use_rgb: bool = True,
        use_thermal: bool = True,
        require_both_modalities: bool = False,
        use_augmentation: bool = True,
        mode: str = "train"
    ):
        """
        Args:
            annotations_file: Path to COCO format annotations JSON
            dataset_root: Root directory of dataset
            input_size: Target image size (height, width)
            use_rgb: Whether to use RGB images
            use_thermal: Whether to use thermal images
            require_both_modalities: If True, only use images with both RGB and thermal
            use_augmentation: Whether to apply data augmentation
            mode: 'train', 'val', or 'test'
        """
        self.dataset_root = Path(dataset_root)
        self.input_size = input_size
        self.use_rgb = use_rgb
        self.use_thermal = use_thermal
        self.require_both_modalities = require_both_modalities
        self.mode = mode
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image and annotation mappings
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Map image_id to annotations
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Filter images based on modality requirements
        self.valid_images = self._filter_images()
        
        # Filter by split if split information exists
        if self.mode and any('split' in img for img in self.valid_images):
            self.valid_images = [
                img for img in self.valid_images
                if img.get('split', 'train') == self.mode
            ]
        
        # Build paired image mapping (for RGB+thermal pairs)
        self.paired_images = self._build_paired_mapping()
        
        # Augmentation
        if use_augmentation and mode == "train":
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = None
        
        # Resize transform (always applied)
        self.resize = transforms.Resize(input_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    
    def _filter_images(self) -> List[Dict]:
        """Filter images based on modality requirements."""
        valid = []
        
        for img in self.images:
            file_name = img['file_name']
            has_rgb = 'rgb/' in file_name
            has_thermal = 'thermal/' in file_name
            
            # Check if image matches requirements
            if self.require_both_modalities:
                # Need both RGB and thermal
                if not (has_rgb and has_thermal):
                    continue
            else:
                # Can use single modality
                if has_rgb and not self.use_rgb:
                    continue
                if has_thermal and not self.use_thermal:
                    continue
                if not (has_rgb or has_thermal):
                    continue
            
            # Must have annotations
            if img['id'] not in self.img_to_anns:
                continue
            
            # Check if file actually exists (basic check)
            file_path = self.dataset_root / file_name
            if not file_path.exists():
                # Skip images where file doesn't exist
                continue
            
            valid.append(img)
        
        return valid
    
    def _build_paired_mapping(self) -> Dict[int, Optional[int]]:
        """
        Build mapping from RGB image_id to thermal image_id (and vice versa).
        
        Returns:
            Dict mapping image_id -> paired_image_id (or None if no pair)
        """
        paired = {}
        
        # Group images by base name (without modality prefix)
        rgb_images = {}
        thermal_images = {}
        
        for img in self.valid_images:
            file_name = img['file_name']
            img_id = img['id']
            
            if 'rgb/' in file_name:
                # Extract base name (e.g., "llvip_00001.jpg" from "rgb/human/llvip_00001.jpg")
                base_name = Path(file_name).name
                rgb_images[base_name] = img_id
            elif 'thermal/' in file_name:
                base_name = Path(file_name).name
                thermal_images[base_name] = img_id
        
        # Find pairs
        for base_name, rgb_id in rgb_images.items():
            if base_name in thermal_images:
                thermal_id = thermal_images[base_name]
                paired[rgb_id] = thermal_id
                paired[thermal_id] = rgb_id
        
        return paired
    
    def _load_image(self, file_path: Path) -> Optional[torch.Tensor]:
        """Load and preprocess image. Returns None if file doesn't exist."""
        if not file_path.exists():
            return None
        
        try:
            img = Image.open(file_path).convert('RGB')
            
            # Apply augmentation if training
            if self.transform:
                img = self.transform(img)
            
            # Resize
            img = self.resize(img)
            
            # Convert to tensor and normalize
            img = self.to_tensor(img)
            img = self.normalize(img)  # Normalize here
            
            return img
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return None
    
    def _get_annotations(self, image_id: int) -> List[Dict]:
        """Get annotations for an image."""
        anns = self.img_to_anns.get(image_id, [])
        
        # Convert to format: [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        boxes = []
        labels = []
        
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, w, h]
            x, y, w, h = bbox
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'] - 1)  # Convert to 0-indexed (human=0, animal=1, vehicle=2)
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        }
    
    def __len__(self) -> int:
        return len(self.valid_images)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with:
            - 'rgb': RGB image tensor (C, H, W) or None
            - 'thermal': Thermal image tensor (C, H, W) or None
            - 'target': Target dict with 'boxes' and 'labels'
            - 'image_id': Original image ID
            - 'modality': 'rgb', 'thermal', or 'both'
        """
        img_info = self.valid_images[idx]
        image_id = img_info['id']
        file_name = img_info['file_name']
        
        # Load images
        rgb_img = None
        thermal_img = None
        
        if 'rgb/' in file_name:
            rgb_path = self.dataset_root / file_name
            rgb_img = self._load_image(rgb_path)
            
            # Check for paired thermal image
            if image_id in self.paired_images:
                thermal_id = self.paired_images[image_id]
                # Find thermal image info
                thermal_info = next((img for img in self.valid_images if img['id'] == thermal_id), None)
                if thermal_info:
                    thermal_path = self.dataset_root / thermal_info['file_name']
                    thermal_img = self._load_image(thermal_path)
                    if thermal_img is None:
                        thermal_img = None  # File doesn't exist, skip thermal
        
        elif 'thermal/' in file_name:
            thermal_path = self.dataset_root / file_name
            thermal_img = self._load_image(thermal_path)
            
            # Check for paired RGB image
            if image_id in self.paired_images:
                rgb_id = self.paired_images[image_id]
                rgb_info = next((img for img in self.valid_images if img['id'] == rgb_id), None)
                if rgb_info:
                    rgb_path = self.dataset_root / rgb_info['file_name']
                    rgb_img = self._load_image(rgb_path)
                    if rgb_img is None:
                        rgb_img = None  # File doesn't exist, skip RGB
        
        # Determine modality
        if rgb_img is not None and thermal_img is not None:
            modality = 'both'
        elif rgb_img is not None:
            modality = 'rgb'
        else:
            modality = 'thermal'
        
        # Get annotations (use original image_id)
        target = self._get_annotations(image_id)
        
        # Scale boxes to input size
        orig_w, orig_h = img_info['width'], img_info['height']
        target_h, target_w = self.input_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        if len(target['boxes']) > 0:
            target['boxes'][:, [0, 2]] *= scale_x
            target['boxes'][:, [1, 3]] *= scale_y
        
        return {
            'rgb': rgb_img,
            'thermal': thermal_img,
            'target': target,
            'image_id': image_id,
            'modality': modality
        }
