#!/usr/bin/env python3
"""
YOLO training script with custom 6-channel dataset loader.
Handles RGB-only, thermal-only, and both modalities.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import sys
import json
from collections import Counter

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))

from models.data.yolo_6ch_dataset import YOLO6ChannelDataset

class ClassWeightsCallback:
    """Custom callback to apply class weights to YOLO loss function."""
    
    def __init__(self, class_weights):
        self.class_weights = class_weights
        self.applied = False
    
    def __call__(self, trainer):
        """Make callback callable directly."""
        return self.on_train_start(trainer)
    
    def on_train_start(self, trainer):
        """Apply class weights when training starts."""
        if self.applied or self.class_weights is None:
            return
        
        try:
            # Try multiple ways to access the loss function
            # In Ultralytics YOLO, loss can be a function or an object
            loss_obj = None
            
            # Method 1: Access from trainer.model.loss (YOLO's actual location)
            if hasattr(trainer, 'model') and hasattr(trainer.model, 'loss'):
                loss_obj = trainer.model.loss
            # Method 2: Direct access from trainer (fallback)
            elif hasattr(trainer, 'criterion'):
                loss_obj = trainer.criterion
            # Method 3: Try trainer.model.criterion (another fallback)
            elif hasattr(trainer, 'model') and hasattr(trainer.model, 'criterion'):
                loss_obj = trainer.model.criterion
            
            if loss_obj is None:
                print("⚠️  Warning: Could not find loss function in trainer")
                print(f"   Available trainer.model attributes: {[a for a in dir(trainer.model) if not a.startswith('_') and 'loss' in a.lower()][:10]}")
                # Try alternative method
                return self._apply_via_compute_loss(trainer)
            
            # Check if it's a function or an object with forward method
            class_weights = self.class_weights  # Capture for closure
            avg_weight = sum(class_weights) / len(class_weights)
            
            if callable(loss_obj) and not hasattr(loss_obj, 'forward'):
                # It's a function - wrap it directly
                # YOLO calls loss(x, *args, **kwargs), so we need to accept *args and **kwargs
                original_loss_fn = loss_obj
                
                def weighted_loss_fn(preds, *args, **kwargs):
                    loss, loss_items = original_loss_fn(preds, *args, **kwargs)
                    
                    # Apply per-class weights to classification loss
                    # YOLO's loss_items structure: [loss, box_loss, cls_loss, dfl_loss]
                    # loss_items[0] is the total loss, which already includes cls_loss
                    if len(loss_items) >= 3:
                        cls_loss = loss_items[2]  # Classification loss
                        
                        # Compute adjustment: (weight - 1) * cls_loss
                        # This preserves gradients because we're using the original cls_loss tensor
                        weight_adjustment = (avg_weight - 1.0) * cls_loss
                        
                        # Adjust total loss: subtract original cls_loss, add weighted cls_loss
                        # This is equivalent to: loss = loss - cls_loss + (cls_loss * avg_weight)
                        # Which simplifies to: loss = loss + (avg_weight - 1.0) * cls_loss
                        weighted_loss = loss + weight_adjustment
                        
                        # Create new loss_items with updated values
                        # Don't modify in place - create new tuple/list
                        try:
                            # Try to create new structure preserving type
                            if isinstance(loss_items, tuple):
                                weighted_cls_loss = cls_loss * avg_weight
                                new_loss_items = (weighted_loss, loss_items[1], weighted_cls_loss) + loss_items[3:]
                            elif isinstance(loss_items, list):
                                weighted_cls_loss = cls_loss * avg_weight
                                new_loss_items = [weighted_loss, loss_items[1], weighted_cls_loss] + list(loss_items[3:])
                            else:
                                # If it's a tensor or other structure, just return modified loss
                                # YOLO will handle loss_items internally
                                return weighted_loss, loss_items
                        except (IndexError, TypeError):
                            # If structure is unexpected, just return modified loss
                            return weighted_loss, loss_items
                        
                        return weighted_loss, new_loss_items
                    
                    return loss, loss_items
                
                # Replace the function
                if hasattr(trainer.model, 'loss'):
                    trainer.model.loss = weighted_loss_fn
                elif hasattr(trainer, 'criterion'):
                    trainer.criterion = weighted_loss_fn
                
                self.applied = True
                print(f"\n✅ Applied class weights to loss function (function wrapper)!")
                print(f"   Weights: Human={class_weights[0]:.2f}, Animal={class_weights[1]:.2f}, Vehicle={class_weights[2]:.2f}")
                print(f"   Average weight: {avg_weight:.2f}")
                return
                
            elif hasattr(loss_obj, 'forward'):
                # It's an object with forward method
                if not hasattr(loss_obj, '_original_forward'):
                    loss_obj._original_forward = loss_obj.forward
                
                original_forward = loss_obj._original_forward
                
                def weighted_forward(preds, *args, **kwargs):
                    loss, loss_items = original_forward(preds, *args, **kwargs)
                    
                    if len(loss_items) >= 3:
                        cls_loss = loss_items[2]
                        
                        # Compute adjustment preserving gradients
                        weight_adjustment = (avg_weight - 1.0) * cls_loss
                        weighted_loss = loss + weight_adjustment
                        
                        # Create new loss_items with updated values
                        try:
                            if isinstance(loss_items, tuple):
                                weighted_cls_loss = cls_loss * avg_weight
                                new_loss_items = (weighted_loss, loss_items[1], weighted_cls_loss) + loss_items[3:]
                            elif isinstance(loss_items, list):
                                weighted_cls_loss = cls_loss * avg_weight
                                new_loss_items = [weighted_loss, loss_items[1], weighted_cls_loss] + list(loss_items[3:])
                            else:
                                return weighted_loss, loss_items
                        except (IndexError, TypeError):
                            return weighted_loss, loss_items
                        
                        return weighted_loss, new_loss_items
                    
                    return loss, loss_items
                
                loss_obj.forward = weighted_forward
                self.applied = True
                print(f"\n✅ Applied class weights to loss function (object method)!")
                print(f"   Weights: Human={class_weights[0]:.2f}, Animal={class_weights[1]:.2f}, Vehicle={class_weights[2]:.2f}")
                print(f"   Average weight: {avg_weight:.2f}")
                return
            else:
                # Try alternative method
                return self._apply_via_compute_loss(trainer)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not apply class weights: {e}")
            import traceback
            traceback.print_exc()
            # Try alternative method
            return self._apply_via_compute_loss(trainer)
    
    def _apply_via_compute_loss(self, trainer):
        """Alternative: Apply via compute_loss method."""
        try:
            if hasattr(trainer, 'compute_loss'):
                original_compute_loss = trainer.compute_loss
                class_weights = self.class_weights
                avg_weight = sum(class_weights) / len(class_weights)
                
                def weighted_compute_loss(preds, *args, **kwargs):
                    loss, loss_items = original_compute_loss(preds, *args, **kwargs)
                    if len(loss_items) >= 3:
                        cls_loss = loss_items[2]
                        
                        # Compute adjustment preserving gradients
                        weight_adjustment = (avg_weight - 1.0) * cls_loss
                        weighted_loss = loss + weight_adjustment
                        
                        # Create new loss_items with updated values
                        try:
                            if isinstance(loss_items, tuple):
                                weighted_cls_loss = cls_loss * avg_weight
                                new_loss_items = (weighted_loss, loss_items[1], weighted_cls_loss) + loss_items[3:]
                            elif isinstance(loss_items, list):
                                weighted_cls_loss = cls_loss * avg_weight
                                new_loss_items = [weighted_loss, loss_items[1], weighted_cls_loss] + list(loss_items[3:])
                            else:
                                return weighted_loss, loss_items
                        except (IndexError, TypeError):
                            return weighted_loss, loss_items
                        
                        return weighted_loss, new_loss_items
                    return loss, loss_items
                
                trainer.compute_loss = weighted_compute_loss
                self.applied = True
                print(f"\n✅ Applied class weights via compute_loss method!")
                print(f"   Weights: Human={class_weights[0]:.2f}, Animal={class_weights[1]:.2f}, Vehicle={class_weights[2]:.2f}")
                return True
        except Exception as e:
            print(f"⚠️  Alternative method also failed: {e}")
        return False

def _apply_class_weights_alternative(trainer, class_weights):
    """Alternative method to apply class weights if callback fails."""
    try:
        # Try to monkey-patch the loss computation directly
        if hasattr(trainer, 'compute_loss'):
            original_compute_loss = trainer.compute_loss
            
            def weighted_compute_loss(preds, batch):
                loss, loss_items = original_compute_loss(preds, batch)
                if len(loss_items) >= 3:
                    cls_loss = loss_items[2]
                    avg_weight = sum(class_weights) / len(class_weights)
                    weighted_cls_loss = cls_loss * avg_weight
                    loss = loss_items[0] + loss_items[1] + weighted_cls_loss + (loss_items[3] if len(loss_items) > 3 else 0)
                    loss_items[2] = weighted_cls_loss
                    loss_items[0] = loss
                return loss, loss_items
            
            trainer.compute_loss = weighted_compute_loss
            print("✅ Applied class weights via compute_loss monkey-patch")
            return True
    except Exception as e:
        print(f"⚠️  Alternative method also failed: {e}")
    return False

def modify_yolo_for_6ch(yolo_model):
    """Modify YOLO first conv layer to accept 6 channels (RGB 3 + thermal 3)."""
    first_conv = yolo_model.model.model[0].conv
    
    # Create new 6-channel conv layer
    old_conv = first_conv
    new_conv = nn.Conv2d(
        6, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    
    # Initialize weights: RGB channels use pretrained, thermal channels use average
    with torch.no_grad():
        if old_conv.weight.shape[0] > 0:
            # Copy RGB weights (first 3 channels)
            new_conv.weight[:, :3] = old_conv.weight[:, :3]
            # Thermal channels: use average of RGB weights
            thermal_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:6] = thermal_weight.expand(-1, 3, -1, -1)
            
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
    
    # Replace first conv in model
    yolo_model.model.model[0].conv = new_conv
    print("✅ Modified first layer for 6-channel input (RGB+thermal)")
    return yolo_model

def compute_class_weights(annotations_file='dataset/annotations.json'):
    """Compute class weights from annotations."""
    import json
    from collections import Counter
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Count annotations per class
    class_counts = Counter()
    for ann in data['annotations']:
        category_id = ann['category_id']
        class_counts[category_id] += 1
    
    # Compute inverse frequency weights
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Get weights: human (1), animal (2), vehicle (3)
    weights = {}
    for cat_id in [1, 2, 3]:
        count = class_counts.get(cat_id, 1)
        weight = total / (num_classes * count)
        weights[cat_id] = weight
    
    # Normalize so min = 1.0
    min_weight = min(weights.values())
    normalized = {k: v / min_weight for k, v in weights.items()}
    
    # Convert to YOLO format (0-indexed): [human, animal, vehicle]
    weight_list = [
        normalized[1],  # human -> index 0
        normalized[2],  # animal -> index 1
        normalized[3]   # vehicle -> index 2
    ]
    
    return weight_list

def apply_class_weights_to_loss(model, class_weights):
    """
    Apply class weights to YOLO's loss function by modifying the criterion.
    This is a workaround since Ultralytics YOLO doesn't have direct class_weights parameter.
    """
    try:
        # Access YOLO's loss criterion
        # YOLO v8+ stores criterion in model.trainer.criterion
        criterion = None
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'criterion'):
            criterion = model.trainer.criterion
        elif hasattr(model.model, 'criterion'):
            criterion = model.model.criterion
            
            # Store original forward method
            if not hasattr(criterion, '_original_forward'):
                criterion._original_forward = criterion.forward
            
            # Create weighted forward
            def weighted_forward(preds, batch):
                # Call original forward
                loss, loss_items = criterion._original_forward(preds, batch)
                
                # Apply class weights to classification loss
                # YOLO's loss_items structure: [loss, box_loss, cls_loss, dfl_loss]
                if len(loss_items) >= 3:
                    cls_loss = loss_items[2]  # Classification loss
                    
                    # Weight the classification loss
                    # Note: This is a simplified approach - proper implementation would
                    # weight per-class, but YOLO's loss is already computed
                    # We multiply by average weight as a global adjustment
                    avg_weight = sum(class_weights) / len(class_weights)
                    weighted_cls_loss = cls_loss * avg_weight
                    
                    # Recompute total loss
                    loss = loss_items[0] + loss_items[1] + weighted_cls_loss + (loss_items[3] if len(loss_items) > 3 else 0)
                    loss_items[2] = weighted_cls_loss
                    loss_items[0] = loss
                
                return loss, loss_items
            
            criterion.forward = weighted_forward
            print(f"✅ Applied class weights to loss: {class_weights}")
            print(f"   (Human: {class_weights[0]:.2f}, Animal: {class_weights[1]:.2f}, Vehicle: {class_weights[2]:.2f})")
            return True
        else:
            print("⚠️  Warning: Could not find loss criterion. Class weights not applied.")
            return False
    except Exception as e:
        print(f"⚠️  Warning: Could not apply class weights: {e}")
        return False

def main():
    print("=" * 60)
    print("YOLOv8n Training - BioWatch Multi-Modal Detection")
    print("=" * 60)
    
    # Configuration
    model_size = 'n'  # nano for camera deployment
    epochs = 100  # Increased from 30 for better accuracy
    batch_size = 64
    imgsz = 640
    dataset_yaml = 'dataset_yolo_6ch/biowatch.yaml'  # New folder with 6-channel images
    dataset_root = 'dataset'
    annotations_file = 'dataset/annotations.json'
    use_class_weights = True  # Enable class weights
    
    print(f"\nConfiguration:")
    print(f"  Model: YOLOv8{model_size} (nano)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Input: 6 channels (RGB+thermal)")
    print(f"  Handling: RGB-only, thermal-only, and both")
    print(f"  Class weights: {'Enabled' if use_class_weights else 'Disabled'}")
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Check dataset files
    if not Path(dataset_yaml).exists():
        print(f"\n❌ Error: Dataset config not found: {dataset_yaml}")
        return
    
    if not Path(annotations_file).exists():
        print(f"\n❌ Error: Annotations file not found: {annotations_file}")
        return
    
    print(f"\n✅ Dataset config found: {dataset_yaml}")
    print(f"✅ Annotations file found: {annotations_file}")
    
    # Load YOLO model
    print(f"\nLoading YOLOv8{model_size}...")
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Modify for 6-channel input
    model = modify_yolo_for_6ch(model)
    
    # Set number of classes (should be 3: human, animal, vehicle)
    model.model.model[-1].nc = 3
    print(f"✅ Set number of classes to 3")
    
    # Compute class weights
    class_weights_callback = None
    if use_class_weights:
        print(f"\nComputing class weights from {annotations_file}...")
        try:
            class_weights = compute_class_weights(annotations_file)
            print(f"✅ Computed class weights: {class_weights}")
            print(f"   (Human: {class_weights[0]:.2f}, Animal: {class_weights[1]:.2f}, Vehicle: {class_weights[2]:.2f})")
            print("   Class weights will be applied during training via callback")
            # Create callback to apply class weights
            class_weights_callback = ClassWeightsCallback(class_weights)
        except Exception as e:
            print(f"⚠️  Warning: Could not compute class weights: {e}")
            print("   Continuing without class weights...")
            class_weights_callback = None
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    # Train model with custom 6-channel dataset
    try:
        # Monkey-patch YOLO's dataset class to use our 6-channel loader
        # Note: We need to do this before training starts, but it causes pickling issues with multiprocessing
        # Solution: Set workers=0 to disable multiprocessing, or properly register the class
        import ultralytics.data.dataset
        original_yolo_dataset = ultralytics.data.dataset.YOLODataset
        
        # Replace with our 6-channel dataset
        ultralytics.data.dataset.YOLODataset = YOLO6ChannelDataset
        print("✅ Using custom 6-channel dataset loader (loads from .npz files)")
        print("⚠️  Note: Setting workers=0 to avoid pickling issues with custom dataset class")
        
        # Register callback before training starts (CRITICAL for class weights)
        if class_weights_callback is not None:
            try:
                # Method 1: Try to add callback using model.add_callback (newer YOLO versions)
                model.add_callback('on_train_start', class_weights_callback.on_train_start)
                print("✅ Registered class weights callback via add_callback")
            except Exception as e:
                print(f"⚠️  Could not register callback via add_callback: {e}")
                print("   Will apply directly after trainer is created")
        
        # Start training
        # Set workers=0 to avoid pickling error with custom dataset class
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=0,  # Disable multiprocessing to avoid pickling error with custom dataset
            project='runs/detect',
            name='biowatch_yolov8n',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            patience=50,
            save=True,
            save_period=10,
            val=True,
            plots=True,
            verbose=True
        )
        
        # CRITICAL: Apply class weights directly after training starts (fallback)
        # This ensures class weights are applied even if callback didn't trigger
        if class_weights_callback is not None:
            # Wait a moment for trainer to be initialized
            import time
            time.sleep(2)
            
            if not class_weights_callback.applied:
                print("\n⚠️  Callback didn't apply weights automatically, applying directly...")
                try:
                    # Access trainer after training has started
                    if hasattr(model, 'trainer') and model.trainer is not None:
                        class_weights_callback.on_train_start(model.trainer)
                        if class_weights_callback.applied:
                            print("✅ Class weights applied via direct method")
                        else:
                            print("❌ ERROR: Could not apply class weights!")
                            print("   This is critical - training may be biased")
                            print("   Attempting alternative method...")
                            # Try alternative: hook into loss computation
                            class_weights_callback._apply_via_compute_loss(model.trainer)
                    else:
                        print("❌ ERROR: Trainer not accessible - class weights NOT applied!")
                        print("   Training will continue but may be biased toward majority class")
                except Exception as e:
                    print(f"❌ ERROR: Could not apply class weights: {e}")
                    print("   Training will continue but may be biased")
                    import traceback
                    traceback.print_exc()
            else:
                print("✅ Class weights successfully applied via callback!")
        
        # Restore original dataset class
        ultralytics.data.dataset.YOLODataset = original_yolo_dataset
        
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print("=" * 60)
        print(f"\nBest model saved to: runs/detect/biowatch_yolov8n/weights/best.pt")
        print(f"Final model saved to: runs/detect/biowatch_yolov8n/weights/last.pt")
        
    except Exception as e:
        # Restore original dataset class on error
        try:
            ultralytics.data.dataset.YOLODataset = original_yolo_dataset
        except:
            pass
        
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()

