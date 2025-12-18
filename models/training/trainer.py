"""
Training script for multi-modal object detection.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, Optional

from ..config import ModelConfig
from ..data.dataset import BioWatchDataset
from ..architectures.multimodal_detector import MultiModalFasterRCNN


class Trainer:
    """Trainer for multi-modal object detection model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = MultiModalFasterRCNN(
            num_classes=config.num_classes,
            backbone_name=config.backbone,
            fusion_method=config.fusion_method,
            pretrained=True
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # Create datasets
        self.train_dataset = BioWatchDataset(
            annotations_file=config.annotations_file,
            dataset_root=config.dataset_path,
            input_size=config.input_size,
            use_rgb=config.use_rgb,
            use_thermal=config.use_thermal,
            require_both_modalities=config.require_both_modalities,
            use_augmentation=config.use_augmentation,
            mode="train"
        )
        
        self.val_dataset = BioWatchDataset(
            annotations_file=config.annotations_file,
            dataset_root=config.dataset_path,
            input_size=config.input_size,
            use_rgb=config.use_rgb,
            use_thermal=config.use_thermal,
            require_both_modalities=False,  # Allow single modality in val
            use_augmentation=False,
            mode="val"
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate_fn
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        rgb_images = []
        thermal_images = []
        targets = []
        image_ids = []
        modalities = []
        
        for item in batch:
            if item['rgb'] is not None:
                rgb_images.append(item['rgb'])
            else:
                rgb_images.append(None)
            
            if item['thermal'] is not None:
                thermal_images.append(item['thermal'])
            else:
                thermal_images.append(None)
            
            targets.append(item['target'])
            image_ids.append(item['image_id'])
            modalities.append(item['modality'])
        
        # Stack images - handle None values by creating zero tensors
        batch_size = len(batch)
        if rgb_images[0] is not None:
            rgb_batch = torch.stack([img if img is not None else torch.zeros_like(rgb_images[0]) 
                                    for img in rgb_images])
        else:
            rgb_batch = None
        
        if thermal_images[0] is not None:
            thermal_batch = torch.stack([img if img is not None else torch.zeros_like(thermal_images[0]) 
                                       for img in thermal_images])
        else:
            thermal_batch = None
        
        return {
            'rgb': rgb_batch,
            'thermal': thermal_batch,
            'targets': targets,
            'image_ids': image_ids,
            'modalities': modalities
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            rgb = batch['rgb']
            thermal = batch['thermal']
            targets = batch['targets']
            
            # Move to device and handle None values
            if rgb is not None:
                rgb = rgb.to(self.device)
                # Check if all images in batch are None (zero tensors)
                if rgb.sum() == 0:
                    rgb = None
            if thermal is not None:
                thermal = thermal.to(self.device)
                if thermal.sum() == 0:
                    thermal = None
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(rgb=rgb, thermal=thermal, targets=targets)
            
            # Compute loss (placeholder - will be replaced with actual detection loss)
            loss = self._compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _compute_loss(self, outputs: Dict, targets: list) -> torch.Tensor:
        """Compute detection loss."""
        # TODO: Implement actual detection loss (classification + bbox regression)
        # For now, return a placeholder
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                rgb = batch['rgb']
                thermal = batch['thermal']
                targets = batch['targets']
                
                if rgb is not None:
                    rgb = rgb.to(self.device)
                if thermal is not None:
                    thermal = thermal.to(self.device)
                
                outputs = self.model(rgb=rgb, thermal=thermal, targets=targets)
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float, path: Path):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop."""
        print(f"Starting training...")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Fusion method: {self.config.fusion_method}")
        print(f"  Backbone: {self.config.backbone}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.config.checkpoint_dir / f"best_model_epoch_{epoch+1}.pth"
                self.save_checkpoint(epoch, val_loss, checkpoint_path)
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(epoch, val_loss, checkpoint_path)
        
        # Save final model
        final_path = self.config.checkpoint_dir / "final_model.pth"
        self.save_checkpoint(self.config.num_epochs - 1, val_loss, final_path)
        
        # Save training history
        history_path = self.config.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
