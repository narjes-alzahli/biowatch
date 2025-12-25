"""
Unified image preprocessing for RGB and thermal images.
Ensures consistent normalization and preprocessing for both modalities.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
from typing import Tuple, Literal, Optional

try:
    import cv2
    HAS_CV2 = True
    # Create CLAHE object once and reuse (performance optimization)
    _CLAHE_OBJECT = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
except ImportError:
    HAS_CV2 = False
    _CLAHE_OBJECT = None


def normalize_image(img: np.ndarray, method: Literal['standard', 'minmax'] = 'minmax') -> np.ndarray:
    """
    Normalize image to [0, 1] range (minmax) or standardized (mean=0, std=1).
    
    Args:
        img: Image array (H, W, C) in any range
        method: 'minmax' for [0,1] or 'standard' for z-score normalization
    
    Returns:
        Normalized image as float32
    """
    img_float = img.astype(np.float32)
    
    if method == 'minmax':
        # Normalize to [0, 1]
        img_min = img_float.min()
        img_max = img_float.max()
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
        else:
            img_normalized = img_float * 0.0  # All same value, set to zeros
        return img_normalized
    else:  # standard
        # Z-score normalization
        mean = img_float.mean()
        std = img_float.std()
        if std > 0:
            return (img_float - mean) / std
        else:
            return img_float - mean


def preprocess_thermal_image(thermal_img: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
    """
    Preprocess thermal image for better detection performance.
    
    Args:
        thermal_img: Thermal image as numpy array (H, W) - grayscale
        enhance_contrast: Apply histogram equalization for better contrast
    
    Returns:
        Processed thermal image as RGB (H, W, 3) normalized to [0, 1]
    """
    # Ensure grayscale
    if len(thermal_img.shape) == 3:
        # Convert to grayscale if RGB
        thermal_img = np.mean(thermal_img, axis=2).astype(np.uint8)
    else:
        thermal_img = thermal_img.astype(np.uint8)
    
    # Apply histogram equalization for better contrast
    if enhance_contrast:
        if HAS_CV2:
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for better results
            # Reuse global CLAHE object for performance (creating it once per process)
            global _CLAHE_OBJECT
            if _CLAHE_OBJECT is None:
                _CLAHE_OBJECT = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            thermal_enhanced = _CLAHE_OBJECT.apply(thermal_img)
        else:
            # Fallback to PIL histogram equalization
            thermal_pil = Image.fromarray(thermal_img)
            thermal_enhanced = np.array(ImageOps.equalize(thermal_pil))
    else:
        thermal_enhanced = thermal_img
    
    # Convert to RGB by duplicating channels (but with enhanced values)
    thermal_rgb = np.stack([thermal_enhanced, thermal_enhanced, thermal_enhanced], axis=-1)
    
    # Normalize to [0, 1] range
    thermal_normalized = normalize_image(thermal_rgb, method='minmax')
    
    return thermal_normalized


def preprocess_rgb_image(rgb_img: np.ndarray) -> np.ndarray:
    """
    Preprocess RGB image with normalization.
    
    Args:
        rgb_img: RGB image as numpy array (H, W, 3) in 0-255 range
    
    Returns:
        Normalized RGB image (H, W, 3) in [0, 1] range
    """
    # Ensure RGB format
    if len(rgb_img.shape) == 2:
        # Grayscale to RGB
        rgb_img = np.stack([rgb_img, rgb_img, rgb_img], axis=-1)
    elif rgb_img.shape[2] == 4:
        # RGBA to RGB
        rgb_img = rgb_img[:, :, :3]
    
    # Normalize to [0, 1] range
    rgb_normalized = normalize_image(rgb_img, method='minmax')
    
    return rgb_normalized


def create_6channel_image(
    rgb_path: Optional[Path],
    thermal_path: Optional[Path],
    modality: Literal['rgb', 'thermal', 'both'] = 'both'
) -> Tuple[np.ndarray, Literal['rgb', 'thermal', 'both']]:
    """
    Create 6-channel image from RGB and/or thermal inputs.
    Properly normalizes and preprocesses both modalities.
    
    Args:
        rgb_path: Path to RGB image (or None)
        thermal_path: Path to thermal image (or None)
        modality: Expected modality type ('rgb', 'thermal', 'both')
    
    Returns:
        Tuple of (6-channel image as float32 [0,1], detected modality)
        Shape: (H, W, 6)
        Format: [RGB_channels, Thermal_channels]
    """
    rgb_data = None
    thermal_data = None
    detected_modality = 'both'
    
    # Load and preprocess RGB
    if rgb_path and rgb_path.exists():
        try:
            rgb_img = Image.open(rgb_path).convert('RGB')
            rgb_array = np.array(rgb_img, dtype=np.uint8)
            rgb_data = preprocess_rgb_image(rgb_array)
            if thermal_path is None or not thermal_path.exists():
                detected_modality = 'rgb'
        except Exception as e:
            print(f"Warning: Failed to load RGB image {rgb_path}: {e}")
    
    # Load and preprocess thermal
    if thermal_path and thermal_path.exists():
        try:
            thermal_img = Image.open(thermal_path)
            # Convert to grayscale array
            if thermal_img.mode == 'RGB':
                thermal_array = np.array(thermal_img.convert('L'))
            else:
                thermal_array = np.array(thermal_img.convert('L'))
            thermal_data = preprocess_thermal_image(thermal_array, enhance_contrast=True)
            if rgb_path is None or not rgb_path.exists():
                detected_modality = 'thermal'
        except Exception as e:
            print(f"Warning: Failed to load thermal image {thermal_path}: {e}")
    
    # Determine output dimensions
    if rgb_data is not None:
        h, w = rgb_data.shape[:2]
    elif thermal_data is not None:
        h, w = thermal_data.shape[:2]
    else:
        raise ValueError("At least one of rgb_path or thermal_path must be valid")
    
    # Create 6-channel image
    img_6ch = np.zeros((h, w, 6), dtype=np.float32)
    
    if modality == 'rgb' or (rgb_data is not None and thermal_data is None):
        # RGB camera: Use RGB in both halves
        if rgb_data is not None:
            img_6ch[:, :, :3] = rgb_data  # RGB channels
            img_6ch[:, :, 3:6] = rgb_data  # Duplicate RGB in thermal channels
        detected_modality = 'rgb'
    elif modality == 'thermal' or (thermal_data is not None and rgb_data is None):
        # Thermal camera: Use processed thermal in both halves
        if thermal_data is not None:
            img_6ch[:, :, :3] = thermal_data  # Thermal processed as RGB-like
            img_6ch[:, :, 3:6] = thermal_data  # Thermal in thermal channels
        detected_modality = 'thermal'
    else:
        # Both available: RGB in first 3, thermal in last 3
        if rgb_data is not None:
            img_6ch[:, :, :3] = rgb_data
        if thermal_data is not None:
            img_6ch[:, :, 3:6] = thermal_data
        
        # Ensure same size (resize if needed)
        if rgb_data is not None and thermal_data is not None:
            if rgb_data.shape[:2] != thermal_data.shape[:2]:
                # Resize thermal to match RGB using faster bilinear interpolation
                from PIL import Image as PILImage
                thermal_resized = PILImage.fromarray((thermal_data * 255).astype(np.uint8))
                thermal_resized = thermal_resized.resize((w, h), Image.BILINEAR)  # Faster than LANCZOS
                thermal_data = np.array(thermal_resized, dtype=np.float32) / 255.0
                img_6ch[:, :, 3:6] = thermal_data
        detected_modality = 'both'
    
    return img_6ch, detected_modality


def create_6channel_from_single_image(
    image_path: Path,
    is_thermal: bool = False
) -> np.ndarray:
    """
    Create 6-channel image from a single RGB or thermal image.
    For production use where camera type is known.
    
    Args:
        image_path: Path to image file
        is_thermal: True if this is a thermal camera image
    
    Returns:
        6-channel image (H, W, 6) normalized to [0, 1]
    """
    if is_thermal:
        return create_6channel_image(None, image_path, modality='thermal')[0]
    else:
        return create_6channel_image(image_path, None, modality='rgb')[0]

