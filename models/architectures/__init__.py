"""Model architectures."""
from .multimodal_detector import MultiModalFasterRCNN, MultiModalBackbone, FeatureFusionModule

__all__ = ['MultiModalFasterRCNN', 'MultiModalBackbone', 'FeatureFusionModule']
