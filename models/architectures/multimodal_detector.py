"""
Multi-modal object detection model with RGB and thermal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from typing import Optional, Dict, Literal, List, Tuple
from torchvision.transforms import functional as F


class FeatureFusionModule(nn.Module):
    """Feature-level fusion module for RGB and thermal features."""
    
    def __init__(self, in_channels: int, fusion_method: str = "concat"):
        """
        Args:
            in_channels: Number of input channels per modality
            fusion_method: 'concat', 'add', or 'attention'
        """
        super().__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == "concat":
            out_channels = in_channels * 2
        elif fusion_method == "add":
            out_channels = in_channels
        elif fusion_method == "attention":
            out_channels = in_channels
            # Attention-based fusion
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels, 2, 1),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.out_channels = out_channels
    
    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and thermal features.
        
        Args:
            rgb_feat: RGB features (B, C, H, W)
            thermal_feat: Thermal features (B, C, H, W)
        
        Returns:
            Fused features (B, out_channels, H, W)
        """
        if self.fusion_method == "concat":
            return torch.cat([rgb_feat, thermal_feat], dim=1)
        elif self.fusion_method == "add":
            return rgb_feat + thermal_feat
        elif self.fusion_method == "attention":
            # Concatenate for attention computation
            concat = torch.cat([rgb_feat, thermal_feat], dim=1)
            attn_weights = self.attention(concat)  # (B, 2, H, W)
            # Apply attention
            rgb_weight = attn_weights[:, 0:1, :, :]
            thermal_weight = attn_weights[:, 1:2, :, :]
            return rgb_feat * rgb_weight + thermal_feat * thermal_weight


class MultiModalBackbone(nn.Module):
    """Backbone that processes RGB and/or thermal images."""
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        fusion_method: Literal["early", "late", "feature"] = "feature",
        pretrained: bool = True
    ):
        """
        Args:
            backbone_name: Name of backbone architecture
            fusion_method: 'early' (concat at input), 'late' (no fusion in backbone), 
                          'feature' (fuse at feature level)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.fusion_method = fusion_method
        
        # Load backbone
        if backbone_name.startswith("resnet"):
            if backbone_name == "resnet50":
                backbone = models.resnet50(pretrained=pretrained)
            elif backbone_name == "resnet101":
                backbone = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unknown ResNet: {backbone_name}")
            
            # Extract layers for feature extraction
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            
            out_channels = 2048 if "50" in backbone_name or "101" in backbone_name else 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.out_channels = out_channels
        
        # Modify first layer for early fusion (6 channels: RGB 3 + thermal 3)
        if fusion_method == "early":
            # Replace first conv layer to accept 6 channels (RGB 3 + thermal 3)
            self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize with pretrained weights (use RGB weights for RGB channels, average for thermal)
            if pretrained:
                pretrained_conv1 = models.resnet50(pretrained=True).conv1.weight
                with torch.no_grad():
                    # RGB channels: use pretrained weights
                    self.conv1.weight[:, :3] = pretrained_conv1
                    # Thermal channels: use average of pretrained weights
                    thermal_weight = pretrained_conv1.mean(dim=1, keepdim=True)
                    self.conv1.weight[:, 3:6] = thermal_weight.expand(-1, 3, -1, -1)
        
        # Feature-level fusion
        if fusion_method == "feature":
            # Create separate backbones for RGB and thermal
            self.rgb_backbone = self._create_backbone(backbone_name, pretrained)
            self.thermal_backbone = self._create_backbone(backbone_name, pretrained)
            
            # Fusion module
            self.fusion = FeatureFusionModule(out_channels, fusion_method="attention")
            self.out_channels = self.fusion.out_channels
    
    def _create_backbone(self, backbone_name: str, pretrained: bool):
        """Create a backbone network."""
        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        elif backbone_name == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
    
    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb: RGB image tensor (B, 3, H, W) or None
            thermal: Thermal image tensor (B, 3, H, W) or None
        
        Returns:
            Features (B, C, H', W')
        """
        if self.fusion_method == "early":
            # Early fusion: concatenate at input
            if rgb is not None and thermal is not None:
                x = torch.cat([rgb, thermal], dim=1)  # (B, 6, H, W) - RGB 3 + thermal 3
            elif rgb is not None:
                # Pad thermal channels with zeros
                x = torch.cat([rgb, torch.zeros_like(rgb)], dim=1)  # (B, 6, H, W)
            elif thermal is not None:
                # Pad RGB channels with zeros
                x = torch.cat([torch.zeros_like(thermal), thermal], dim=1)  # (B, 6, H, W)
            else:
                raise ValueError("At least one modality must be provided")
            
            # Standard forward pass
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            return x
        
        elif self.fusion_method == "feature":
            # Feature-level fusion
            rgb_feat = None
            thermal_feat = None
            
            if rgb is not None:
                rgb_feat = self.rgb_backbone(rgb)
            
            if thermal is not None:
                thermal_feat = self.thermal_backbone(thermal)
            
            # Fuse features
            if rgb_feat is not None and thermal_feat is not None:
                return self.fusion(rgb_feat, thermal_feat)
            elif rgb_feat is not None:
                return rgb_feat
            elif thermal_feat is not None:
                return thermal_feat
            else:
                raise ValueError("At least one modality must be provided")
        
        else:  # late fusion - handled by detection head
            # Process each modality separately
            if rgb is not None:
                return self._forward_single(rgb)
            elif thermal is not None:
                return self._forward_single(thermal)
            else:
                raise ValueError("At least one modality must be provided")
    
    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single modality."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class CustomRCNNTransform(GeneralizedRCNNTransform):
    """Custom transform that handles 6-channel images (RGB + Thermal)."""
    
    def __init__(self, min_size, max_size, image_mean, image_std):
        # Initialize parent with 3-channel dummy values (parent expects 3 channels)
        super().__init__(min_size, max_size, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        # Override image_mean and image_std to handle 6 channels
        # Parent stores them as tensors with shape (C, 1, 1)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(-1, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(-1, 1, 1)
    
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Override normalize to handle 6-channel images.
        Images are already normalized in dataset, so just return as-is for 6-channel.
        """
        if not image.is_floating_point():
            raise TypeError(f"Expected floating point tensor, got {image.dtype}")
        
        # Check if 6-channel image - if so, skip normalization (already done in dataset)
        if image.shape[0] == 6:
            return image
        
        # For 3-channel, use parent's normalize logic manually
        dtype, device = image.dtype, image.device
        parent_mean = torch.as_tensor([0.0, 0.0, 0.0], dtype=dtype, device=device).view(-1, 1, 1)
        parent_std = torch.as_tensor([1.0, 1.0, 1.0], dtype=dtype, device=device).view(-1, 1, 1)
        return (image - parent_mean) / parent_std
    
    def forward(self, images, targets=None):
        """
        Override forward to handle 6-channel images in normalize step.
        """
        # Process each image
        processed_images = []
        processed_targets = []
        
        for image, target in zip(images, targets) if targets else zip(images, [None] * len(images)):
            # Resize image
            image, scale = self._resize_image(image)
            
            # Normalize - use our override
            image = self.normalize(image)
            
            processed_images.append(image)
            if target is not None:
                # Scale target boxes
                if 'boxes' in target and len(target['boxes']) > 0:
                    target = target.copy()
                    target['boxes'] = target['boxes'] * scale
                processed_targets.append(target)
        
        if targets:
            return processed_images, processed_targets
        return processed_images


class BackboneWrapper(nn.Module):
    """Wrapper to make MultiModalBackbone compatible with torchvision's Faster R-CNN."""
    
    def __init__(self, backbone: MultiModalBackbone):
        super().__init__()
        self.backbone = backbone
        # Required by torchvision's Faster R-CNN
        self.out_channels = backbone.out_channels
        
    def forward(self, x):
        """
        Forward pass compatible with torchvision's Faster R-CNN.
        
        Args:
            x: Dict with '0' key containing concatenated RGB+thermal images, or tensor directly
        
        Returns:
            Dict with '0' key containing features
        """
        # Handle both dict and tensor inputs
        if isinstance(x, dict):
            # Extract the concatenated image (RGB + thermal = 6 channels)
            img = x['0']  # (B, 6, H, W)
        elif isinstance(x, torch.Tensor):
            # Direct tensor input
            img = x  # (B, 6, H, W)
        else:
            raise TypeError(f"Unexpected input type: {type(x)}")
        
        # Split back into RGB and thermal
        rgb = img[:, :3, :, :]  # First 3 channels
        thermal = img[:, 3:6, :, :]  # Last 3 channels
        
        # Get features from backbone
        features = self.backbone(rgb=rgb, thermal=thermal)
        
        # Return in format expected by Faster R-CNN
        return {'0': features}


class MultiModalFasterRCNN(nn.Module):
    """Multi-modal Faster R-CNN detector."""
    
    def __init__(
        self,
        num_classes: int = 3,
        backbone_name: str = "resnet50",
        fusion_method: Literal["early", "late", "feature"] = "feature",
        pretrained: bool = True,
        min_size: int = 640,
        max_size: int = 640
    ):
        """
        Args:
            num_classes: Number of object classes (excluding background)
            backbone_name: Backbone architecture name
            fusion_method: Fusion strategy
            pretrained: Whether to use pretrained weights
            min_size: Minimum image size for transform
            max_size: Maximum image size for transform
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # Create backbone
        multimodal_backbone = MultiModalBackbone(backbone_name, fusion_method, pretrained)
        backbone_out_channels = multimodal_backbone.out_channels
        
        # Wrap backbone for torchvision compatibility
        wrapped_backbone = BackboneWrapper(multimodal_backbone)
        
        # Create anchor generator (for RPN)
        # Since we only have one feature map ('0'), we need one anchor size tuple
        anchor_sizes = ((32, 64, 128, 256, 512),)  # All sizes for single feature map
        aspect_ratios = ((0.5, 1.0, 2.0),)  # One tuple for one feature map
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # Create RPN head
        rpn_head = RPNHead(backbone_out_channels, anchor_generator.num_anchors_per_location()[0])
        
        # Create RoI pooling
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create RoI heads (classification + bbox regression)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
        
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            backbone_out_channels * resolution ** 2,
            representation_size
        )
        
        box_predictor = FastRCNNPredictor(representation_size, num_classes + 1)  # +1 for background
        
        # Create RoI heads
        roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )
        
        # Create custom transform for 6-channel images
        # Images are already normalized in dataset, so we skip normalization
        transform = CustomRCNNTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.0] * 6,  # Not used (normalization skipped)
            image_std=[1.0] * 6    # Not used (normalization skipped)
        )
        # Force override normalize method (monkey-patch to ensure it's used)
        def normalize_6ch(self, image):
            if not image.is_floating_point():
                raise TypeError(f"Expected floating point tensor, got {image.dtype}")
            if image.shape[0] == 6:
                return image  # Skip normalization for 6-channel (already normalized)
            # For 3-channel, use identity (mean=0, std=1)
            return image
        import types
        transform.normalize = types.MethodType(normalize_6ch, transform)
        
        # Create Faster R-CNN model
        self.model = FasterRCNN(
            backbone=wrapped_backbone,
            num_classes=num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            roi_heads=roi_heads,
            transform=transform
        )
        
        self.backbone_out_channels = backbone_out_channels
    
    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            rgb: RGB images (B, 3, H, W) or None
            thermal: Thermal images (B, 3, H, W) or None
            targets: List of target dicts with 'boxes' and 'labels'
        
        Returns:
            During training: Dict with loss components
            During inference: List of dicts with 'boxes', 'labels', 'scores'
        """
        # Concatenate RGB and thermal for early fusion
        if rgb is not None and thermal is not None:
            images = torch.cat([rgb, thermal], dim=1)  # (B, 6, H, W)
        elif rgb is not None:
            images = torch.cat([rgb, torch.zeros_like(rgb)], dim=1)  # (B, 6, H, W)
        elif thermal is not None:
            images = torch.cat([torch.zeros_like(thermal), thermal], dim=1)  # (B, 6, H, W)
        else:
            raise ValueError("At least one modality must be provided")
        
        # Temporarily replace transform's normalize to handle 6 channels
        # Store original normalize method
        original_normalize = self.model.transform.normalize
        
        # Create bound method for our normalize (needs self as first arg)
        def normalize_6ch(self_transform, image):
            if not image.is_floating_point():
                raise TypeError(f"Expected floating point tensor, got {image.dtype}")
            if image.shape[0] == 6:
                return image  # Skip normalization for 6-channel (already normalized)
            # For 3-channel, call original normalize
            return original_normalize(image)
        
        # Replace normalize method
        import types
        self.model.transform.normalize = types.MethodType(normalize_6ch, self.model.transform)
        
        try:
            # Convert to list format expected by Faster R-CNN
            image_list = [img for img in images]
            
            # Prepare targets (convert to list of dicts if needed)
            if targets is not None:
                # Targets should already be in correct format from dataset
                target_list = targets
            else:
                target_list = None
            
            # Forward through Faster R-CNN
            if self.training and target_list is not None:
                # Training mode: returns loss dict
                return self.model(image_list, target_list)
            else:
                # Inference mode: returns predictions
                return self.model(image_list)
        finally:
            # Restore original normalize
            self.model.transform.normalize = original_normalize
