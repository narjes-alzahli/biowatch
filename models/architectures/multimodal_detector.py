"""
Multi-modal object detection model with RGB and thermal fusion.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from typing import Optional, Dict, Literal


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
        
        # Modify first layer for early fusion (4 channels: RGB + thermal)
        if fusion_method == "early":
            # Replace first conv layer to accept 4 channels
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize with pretrained weights (average RGB weights for thermal channel)
            if pretrained:
                pretrained_conv1 = models.resnet50(pretrained=True).conv1.weight
                with torch.no_grad():
                    self.conv1.weight[:, :3] = pretrained_conv1
                    self.conv1.weight[:, 3:4] = pretrained_conv1.mean(dim=1, keepdim=True)
        
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
                x = torch.cat([rgb, thermal], dim=1)  # (B, 4, H, W)
            elif rgb is not None:
                # Pad thermal channel with zeros
                x = torch.cat([rgb, torch.zeros_like(rgb[:, :1, :, :])], dim=1)
            elif thermal is not None:
                # Pad RGB channels with zeros
                x = torch.cat([torch.zeros_like(thermal), thermal], dim=1)
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


class MultiModalFasterRCNN(nn.Module):
    """Multi-modal Faster R-CNN detector."""
    
    def __init__(
        self,
        num_classes: int = 3,
        backbone_name: str = "resnet50",
        fusion_method: Literal["early", "late", "feature"] = "feature",
        pretrained: bool = True
    ):
        """
        Args:
            num_classes: Number of object classes (excluding background)
            backbone_name: Backbone architecture name
            fusion_method: Fusion strategy
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # Create backbone
        self.backbone = MultiModalBackbone(backbone_name, fusion_method, pretrained)
        
        # For now, we'll use a simplified approach
        # In a full implementation, we'd integrate with torchvision's detection models
        # This is a placeholder structure
        self.backbone_out_channels = self.backbone.out_channels
    
    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        targets: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            rgb: RGB images (B, 3, H, W) or None
            thermal: Thermal images (B, 3, H, W) or None
            targets: List of target dicts with 'boxes' and 'labels'
        
        Returns:
            Dictionary with 'boxes', 'labels', 'scores' during inference,
            or loss dict during training
        """
        # Extract features
        features = self.backbone(rgb=rgb, thermal=thermal)
        
        # TODO: Integrate with Faster R-CNN detection head
        # For now, return features as placeholder
        return {
            'features': features
        }
