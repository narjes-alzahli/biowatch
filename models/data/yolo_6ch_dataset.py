"""
Custom YOLO dataset that loads pre-created 6-channel numpy arrays.
Handles RGB-only, thermal-only, and both modalities.
Uses properly normalized and preprocessed images.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics.data import YOLODataset
from typing import Optional
import sys

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.image_preprocessing import create_6channel_from_single_image


class YOLO6ChannelDataset(YOLODataset):
    """
    Custom YOLO dataset that loads 6-channel images from .npz files.
    Images are properly normalized to [0, 1] range and preprocessed.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize parent YOLODataset
        super().__init__(*args, **kwargs)
    
    def load_image(self, i):
        """
        Load 6-channel image from compressed .npz file.
        Falls back to creating 6-channel image on-the-fly if .npz doesn't exist.
        """
        # Get image path from parent
        im_file = self.im_files[i]
        img_path = Path(im_file)
        
        # Try to load 6-channel compressed numpy array (.npz)
        npz_path = img_path.with_suffix('.npz')
        if npz_path.exists():
            # Load 6-channel image (H, W, 6) from compressed file
            # Expected: float32 array normalized to [0, 1]
            # But handle old uint8 format [0, 255] for backward compatibility
            data = np.load(npz_path)
            img_6ch = data['img']
            
            # Convert to float32
            img_6ch = img_6ch.astype(np.float32)
            
            # Normalize if values are in [0, 255] range (old format)
            if img_6ch.max() > 1.0:
                img_6ch = img_6ch / 255.0
            
            # Ensure values are in [0, 1] range (clamp if needed)
            img_6ch = np.clip(img_6ch, 0.0, 1.0)
            
            return img_6ch
        else:
            # Fallback: try .npy (old format) for compatibility
            npy_path = img_path.with_suffix('.npy')
            if npy_path.exists():
                img_6ch = np.load(npy_path).astype(np.float32)
                # Normalize if values are in [0, 255] range
                if img_6ch.max() > 1.0:
                    img_6ch = img_6ch / 255.0
                img_6ch = np.clip(img_6ch, 0.0, 1.0)
                return img_6ch
            
            # Final fallback: create 6-channel image on-the-fly using preprocessing
            # Try to detect if it's thermal based on path
            is_thermal = 'thermal' in str(img_path).lower()
            try:
                img_6ch = create_6channel_from_single_image(img_path, is_thermal=is_thermal)
                return img_6ch
            except Exception as e:
                # Last resort: load RGB and pad with zeros (old behavior)
                img = np.array(Image.open(im_file).convert('RGB'), dtype=np.float32)
                img = img / 255.0  # Normalize to [0, 1]
                zeros = np.zeros_like(img)
                img_6ch = np.concatenate([img, zeros], axis=2)
                return img_6ch

