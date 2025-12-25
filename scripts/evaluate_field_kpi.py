#!/usr/bin/env python3
"""
Field Evaluation Script for KPI-based Detection Accuracy Measurement

This script evaluates the model according to the KPI requirements:
- Detection Accuracy ≥ 85%: TP / (TP + FP + FN)
- Calculated on 100 scripted test events (33 human, 33 animal, 34 vehicle)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import torch
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install: pip install ultralytics torch")
    sys.exit(1)

# Import preprocessing module for 6-channel image creation
from utils.image_preprocessing import create_6channel_from_single_image


class FieldKPIEvaluator:
    """Evaluate model according to KPI requirements for field testing."""
    
    def __init__(self, model_path: str, annotations_file: str, conf_threshold: float = 0.25):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            annotations_file: Path to COCO format annotations JSON
            conf_threshold: Confidence threshold for detections (default 0.25)
        """
        self.model_path = Path(model_path)
        self.annotations_file = Path(annotations_file)
        self.conf_threshold = conf_threshold
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = YOLO(str(model_path))
        
        # Load annotations
        print(f"Loading annotations from {annotations_file}...")
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_id_to_annotations = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.image_id_to_annotations[ann['image_id']].append(ann)
        
        # Class names
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.class_names = ['human', 'animal', 'vehicle']  # YOLO format (0-indexed)
        
    def iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] in absolute coordinates
            box2: [x1, y1, x2, y2] in absolute coordinates
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def coco_to_xyxy(self, annotation: Dict) -> List[float]:
        """Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]."""
        x, y, w, h = annotation['bbox']
        return [x, y, x + w, y + h]
    
    def yolo_to_xyxy(self, box: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert YOLO format [x_center, y_center, width, height] (normalized) to [x1, y1, x2, y2] (absolute)."""
        x_center, y_center, width, height = box
        x1 = (x_center - width / 2) * img_width
        y1 = (y_center - height / 2) * img_height
        x2 = (x_center + width / 2) * img_width
        y2 = (y_center + height / 2) * img_height
        return [x1, y1, x2, y2]
    
    def evaluate_image(self, image_id: int, iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate a single image.
        
        Returns:
            Dictionary with TP, FP, FN counts per class
        """
        img_info = self.image_id_to_info[image_id]
        img_path = Path(img_info['file_name'])
        
        # Skip if image doesn't exist
        if not img_path.exists():
            # Try relative to dataset root
            img_path = Path('dataset') / img_info['file_name']
            if not img_path.exists():
                print(f"Warning: Image not found: {img_info['file_name']}")
                return {'tp': 0, 'fp': 0, 'fn': 0, 'details': []}
        
        # Get ground truth annotations
        gt_annotations = self.image_id_to_annotations[image_id]
        gt_boxes = [self.coco_to_xyxy(ann) for ann in gt_annotations]
        gt_classes = [ann['category_id'] - 1 for ann in gt_annotations]  # Convert to 0-indexed
        
        # For 6-channel model, we need to create 6-channel image
        # Try to find corresponding .npz file first (from dataset preparation)
        npz_path = None
        train_npz = Path('dataset_yolo_6ch') / 'images' / 'train' / img_path.with_suffix('.npz').name
        val_npz = Path('dataset_yolo_6ch') / 'images' / 'val' / img_path.with_suffix('.npz').name
        
        if train_npz.exists():
            npz_path = train_npz
        elif val_npz.exists():
            npz_path = val_npz
        
        if npz_path and npz_path.exists():
            # Use pre-prepared 6-channel image from dataset
            data = np.load(npz_path)
            img_6ch = data['img'].astype(np.float32)
            
            # Normalize if needed (handle old uint8 format)
            if img_6ch.max() > 1.0:
                img_6ch = img_6ch / 255.0
            img_6ch = np.clip(img_6ch, 0.0, 1.0)
        else:
            # Create 6-channel image on-the-fly using preprocessing
            # Detect if it's thermal based on path or image name
            is_thermal = 'thermal' in str(img_path).lower() or '_thermal' in img_path.stem.lower()
            img_6ch = create_6channel_from_single_image(img_path, is_thermal=is_thermal)
        
        # YOLO predict expects numpy array in (H, W, C) format
        # Convert from [0,1] float32 to [0,255] uint8 for YOLO (it will normalize internally)
        img_6ch_uint8 = (img_6ch * 255.0).astype(np.uint8)
        
        # Run inference - YOLO will handle the 6-channel input
        results = self.model.predict(img_6ch_uint8, conf=self.conf_threshold, verbose=False)
        
        # Extract predictions
        pred_boxes = []
        pred_classes = []
        pred_confidences = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            img_height, img_width = results[0].orig_shape
            
            for i in range(len(boxes)):
                # YOLO returns boxes in xyxy format already
                box = boxes.xyxy[i].cpu().numpy().tolist()
                class_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                
                pred_boxes.append(box)
                pred_classes.append(class_id)
                pred_confidences.append(conf)
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        tp = 0
        fp = 0
        details = []
        
        # Match each prediction to best ground truth
        for pred_idx, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt or gt_class != pred_class:
                    continue
                
                iou_val = self.iou(pred_box, gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                details.append({
                    'type': 'TP',
                    'class': self.class_names[pred_class],
                    'confidence': pred_confidences[pred_idx],
                    'iou': best_iou
                })
            else:
                fp += 1
                details.append({
                    'type': 'FP',
                    'class': self.class_names[pred_class],
                    'confidence': pred_confidences[pred_idx],
                    'iou': best_iou if best_gt_idx >= 0 else 0.0
                })
        
        # Count false negatives (unmatched ground truth)
        fn = len(gt_boxes) - len(matched_gt)
        for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if gt_idx not in matched_gt:
                details.append({
                    'type': 'FN',
                    'class': self.class_names[gt_class],
                    'confidence': 0.0,
                    'iou': 0.0
                })
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'details': details,
            'image_id': image_id,
            'image_name': img_info['file_name']
        }
    
    def evaluate_test_set(self, test_image_ids: List[int] = None, iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate on test set.
        
        Args:
            test_image_ids: List of image IDs to evaluate. If None, uses all images.
            iou_threshold: IoU threshold for matching (default 0.5)
            
        Returns:
            Dictionary with overall metrics and per-class metrics
        """
        if test_image_ids is None:
            test_image_ids = list(self.image_id_to_info.keys())
        
        print(f"\nEvaluating {len(test_image_ids)} images...")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for img_id in test_image_ids:
            if img_id not in self.image_id_to_info:
                continue
                
            result = self.evaluate_image(img_id, iou_threshold)
            
            total_tp += result['tp']
            total_fp += result['fp']
            total_fn += result['fn']
            
            # Count per class
            for detail in result['details']:
                class_name = detail['class']
                if detail['type'] == 'TP':
                    per_class_stats[class_name]['tp'] += 1
                elif detail['type'] == 'FP':
                    per_class_stats[class_name]['fp'] += 1
                elif detail['type'] == 'FN':
                    per_class_stats[class_name]['fn'] += 1
        
        # Calculate metrics
        total_detections = total_tp + total_fp + total_fn
        if total_detections > 0:
            accuracy = total_tp / total_detections
        else:
            accuracy = 0.0
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        # Per-class metrics
        per_class_metrics = {}
        for class_name in self.class_names:
            stats = per_class_stats[class_name]
            class_tp = stats['tp']
            class_fp = stats['fp']
            class_fn = stats['fn']
            class_total = class_tp + class_fp + class_fn
            
            if class_total > 0:
                class_accuracy = class_tp / class_total
            else:
                class_accuracy = 0.0
            
            class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0.0
            class_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0.0
            
            per_class_metrics[class_name] = {
                'tp': class_tp,
                'fp': class_fp,
                'fn': class_fn,
                'accuracy': class_accuracy,
                'precision': class_precision,
                'recall': class_recall
            }
        
        return {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_detections': total_detections,
            'accuracy': accuracy,  # KPI metric: TP / (TP + FP + FN)
            'precision': precision,
            'recall': recall,
            'per_class': per_class_metrics,
            'kpi_target': 0.85,  # ≥ 85%
            'kpi_met': accuracy >= 0.85
        }
    
    def print_results(self, results: Dict):
        """Print evaluation results in a readable format."""
        print("\n" + "="*70)
        print("FIELD EVALUATION RESULTS (KPI-based)")
        print("="*70)
        
        print(f"\nOverall Metrics:")
        print(f"  True Positives (TP):  {results['total_tp']}")
        print(f"  False Positives (FP): {results['total_fp']}")
        print(f"  False Negatives (FN): {results['total_fn']}")
        print(f"  Total Detections:     {results['total_detections']}")
        print(f"\n  Detection Accuracy:   {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  KPI Target:          ≥ 85.00%")
        print(f"  KPI Status:          {'✓ PASSED' if results['kpi_met'] else '✗ FAILED'}")
        print(f"\n  Precision:            {results['precision']:.4f} ({results['precision']*100:.2f}%)")
        print(f"  Recall:               {results['recall']:.4f} ({results['recall']*100:.2f}%)")
        
        print(f"\nPer-Class Metrics:")
        print("-" * 70)
        for class_name, metrics in results['per_class'].items():
            print(f"\n{class_name.upper()}:")
            print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO model for field testing KPI requirements'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLO model (.pt file)'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='dataset/annotations.json',
        help='Path to COCO format annotations JSON'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IoU threshold for matching (default: 0.5)'
    )
    parser.add_argument(
        '--test-ids',
        type=str,
        default=None,
        help='Path to JSON file with list of test image IDs (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FieldKPIEvaluator(
        model_path=args.model,
        annotations_file=args.annotations,
        conf_threshold=args.conf
    )
    
    # Get test image IDs
    test_image_ids = None
    if args.test_ids:
        with open(args.test_ids, 'r') as f:
            test_image_ids = json.load(f)
    
    # Evaluate
    results = evaluator.evaluate_test_set(test_image_ids, iou_threshold=args.iou)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results to file
    output_file = Path('field_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

