"""
Configuration for BioWatch multi-modal object detection model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

@dataclass
class ModelConfig:
    """Model configuration."""
    # Model architecture
    backbone: str = "resnet50"  # resnet50, resnet101, efficientnet-b0, etc.
    fusion_method: Literal["early", "late", "feature"] = "feature"  # early: concat at input, late: concat at detection head, feature: fuse at feature level
    
    # Input settings
    input_size: tuple = (640, 640)  # (height, width)
    num_classes: int = 3  # human, animal, vehicle
    
    # Training settings
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # GPU settings
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cuda:1", "cpu"
    num_workers: int = 4  # Number of data loading workers
    
    # Data settings
    dataset_path: Path = Path("dataset")
    annotations_file: Path = Path("dataset/annotations.json")
    
    # Modality settings
    use_rgb: bool = True
    use_thermal: bool = True
    require_both_modalities: bool = False  # If True, only use paired RGB+thermal images
    
    # Augmentation
    use_augmentation: bool = True
    
    # Class weights (for handling imbalanced datasets)
    use_class_weights: bool = True  # If True, automatically compute class weights from dataset
    class_weights: Optional[list] = None  # Manual class weights [animal, human, vehicle] or None for auto
    
    # Performance optimizations
    use_mixed_precision: bool = True  # Use FP16 mixed precision training (saves ~50% memory, 1.5-2x speedup)
    use_gradient_checkpointing: bool = False  # Trade compute for memory (saves ~50% memory, ~20% slower)
    
    # Output settings
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    
    def __post_init__(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
