#!/usr/bin/env python3
"""
Shared utility functions for dataset processing scripts.
"""

import json
import zipfile
from pathlib import Path
from collections import defaultdict


def extract_zip(zip_path, extract_to):
    """Extract zip file to directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


class AnnotationCollector:
    """Collect and save annotations in COCO format."""
    
    def __init__(self, output_file='dataset/annotations.json'):
        self.output_file = Path(output_file)
        self.categories = [
            {'id': 1, 'name': 'human', 'supercategory': 'person'},
            {'id': 2, 'name': 'animal', 'supercategory': 'animal'},
            {'id': 3, 'name': 'vehicle', 'supercategory': 'vehicle'}
        ]
        self.category_map = {'human': 1, 'animal': 2, 'vehicle': 3}
        
        self.images = []
        self.annotations = []
        self.image_id = 0
        self.ann_id = 0
        
        # Load existing if file exists
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    self.images = data.get('images', [])
                    self.annotations = data.get('annotations', [])
                    if self.images:
                        self.image_id = max(img['id'] for img in self.images) + 1
                    if self.annotations:
                        self.ann_id = max(ann['id'] for ann in self.annotations) + 1
            except:
                pass
    
    def add_image(self, file_path, modality, category, width, height, bboxes):
        """
        Add an image with its annotations.
        
        Args:
            file_path: Path to image file (relative to dataset/)
            modality: 'rgb' or 'thermal'
            category: 'human', 'animal', or 'vehicle'
            width: Image width
            height: Image height
            bboxes: List of dicts with 'bbox' (COCO format: [x, y, w, h]) and 'category'
        """
        # Filter bboxes to match category
        filtered_bboxes = [b for b in bboxes if b.get('category', category) == category]
        if not filtered_bboxes:
            return
        
        # Add image
        image_entry = {
            'id': self.image_id,
            'file_name': str(file_path),
            'width': width,
            'height': height
        }
        self.images.append(image_entry)
        
        # Add annotations
        for bbox_info in filtered_bboxes:
            bbox = bbox_info['bbox']
            cat = bbox_info.get('category', category)
            
            self.annotations.append({
                'id': self.ann_id,
                'image_id': self.image_id,
                'category_id': self.category_map[cat],
                'bbox': bbox,  # COCO format: [x, y, width, height]
                'area': bbox_info.get('area', bbox[2] * bbox[3]),
                'iscrowd': 0
            })
            self.ann_id += 1
        
        self.image_id += 1
    
    def save(self):
        """Save annotations to JSON file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        coco_data = {
            'info': {
                'description': 'BioWatch Dataset Annotations',
                'version': '1.0',
                'year': 2024
            },
            'licenses': [],
            'categories': self.categories,
            'images': self.images,
            'annotations': self.annotations
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  Saved {len(self.images)} images, {len(self.annotations)} annotations to {self.output_file}")

