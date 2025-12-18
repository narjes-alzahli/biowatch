#!/usr/bin/env python3
"""
Training script for BioWatch multi-modal object detection model.
"""

import argparse
from pathlib import Path
from models.config import ModelConfig
from models.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train BioWatch multi-modal object detection model')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101'],
                       help='Backbone architecture')
    parser.add_argument('--fusion-method', type=str, default='feature',
                       choices=['early', 'late', 'feature'],
                       help='Fusion method: early (concat at input), late (at detection head), feature (at feature level)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    
    # Data arguments
    parser.add_argument('--dataset-path', type=Path, default=Path('dataset'),
                       help='Path to dataset directory')
    parser.add_argument('--annotations', type=Path, default=Path('dataset/annotations.json'),
                       help='Path to annotations JSON file')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                       help='Input image size (height width)')
    
    # Modality arguments
    parser.add_argument('--use-rgb', action='store_true', default=True,
                       help='Use RGB images')
    parser.add_argument('--use-thermal', action='store_true', default=True,
                       help='Use thermal images')
    parser.add_argument('--require-both', action='store_true',
                       help='Require both RGB and thermal (only use paired images)')
    
    # Other arguments
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                       help='Output directory')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'),
                       help='Checkpoint directory')
    parser.add_argument('--resume', type=Path, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig(
        backbone=args.backbone,
        fusion_method=args.fusion_method,
        input_size=tuple(args.input_size),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dataset_path=args.dataset_path,
        annotations_file=args.annotations,
        use_rgb=args.use_rgb,
        use_thermal=args.use_thermal,
        require_both_modalities=args.require_both,
        use_augmentation=not args.no_augmentation,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
