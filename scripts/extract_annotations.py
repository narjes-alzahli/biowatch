#!/usr/bin/env python3
"""
Extract bounding box annotations from original datasets and save in COCO format.

This script:
1. Scans the dataset folder for all images
2. Matches them back to their source datasets
3. Extracts annotations from the original zip files
4. Saves all annotations in COCO JSON format

Usage:
    python3 scripts/extract_annotations.py [--output dataset/annotations.json]
"""

import sys
import json
import xml.etree.ElementTree as ET
import csv
import zipfile
import tempfile
from pathlib import Path
from collections import defaultdict
from PIL import Image


def parse_pascal_voc_xml(xml_path):
    """Parse PASCAL VOC XML annotation."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text) if size is not None else 0
        height = int(size.find('height').text) if size is not None else 0
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is None:
                continue
            name = name.text.lower()
            
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            
            # Map to our categories
            if name == 'person':
                category = 'human'
            elif name in ['cat', 'dog', 'bird', 'cow', 'horse', 'sheep']:
                category = 'animal'
            elif name in ['car', 'bus', 'bicycle', 'motorbike', 'train']:
                category = 'vehicle'
            else:
                continue
            
            objects.append({
                'category': category,
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  # COCO: [x, y, w, h]
                'area': (xmax - xmin) * (ymax - ymin)
            })
        
        return {
            'width': width,
            'height': height,
            'objects': objects
        }
    except Exception as e:
        return None


def parse_coco_json(coco_path, image_filename):
    """Extract annotations for specific image from COCO JSON."""
    try:
        with open(coco_path, 'r') as f:
            data = json.load(f)
        
        # Find image
        image_info = None
        for img in data['images']:
            # Handle different filename formats
            fn = img['file_name']
            if fn == image_filename or fn.endswith(image_filename) or \
               image_filename.endswith(fn.split('/')[-1]):
                image_info = img
                break
        
        if not image_info:
            return None
        
        # Category mappings
        category_map = {c['id']: c['name'] for c in data['categories']}
        HUMAN_IDS = {1}  # person
        ANIMAL_IDS = {17, 18}  # dog, deer
        VEHICLE_IDS = {3, 4, 6, 7, 8, 79}  # car, motor, bus, train, truck, other vehicle
        
        objects = []
        for ann in data['annotations']:
            if ann['image_id'] != image_info['id']:
                continue
            
            cat_id = ann['category_id']
            if cat_id in HUMAN_IDS:
                category = 'human'
            elif cat_id in ANIMAL_IDS:
                category = 'animal'
            elif cat_id in VEHICLE_IDS:
                category = 'vehicle'
            else:
                continue
            
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            objects.append({
                'category': category,
                'bbox': bbox,
                'area': ann.get('area', bbox[2] * bbox[3])
            })
        
        return {
            'width': image_info.get('width', 0),
            'height': image_info.get('height', 0),
            'objects': objects
        }
    except Exception as e:
        return None


def extract_from_pascal_voc(image_name, zips_dir):
    """Extract annotation from PASCAL VOC zip."""
    zip_path = zips_dir / 'PASCAL_VOC_2012.zip'
    if not zip_path.exists():
        return None
    
    # Extract image ID (e.g., "2007_000027" from "2007_000027.jpg")
    image_id = Path(image_name).stem
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find annotation file
            ann_path = f"VOC2012_train_val/VOC2012_train_val/Annotations/{image_id}.xml"
            if ann_path not in z.namelist():
                # Try alternative path
                ann_path = f"VOC2012_train_val/VOC2012_train_val/VOC2012/Annotations/{image_id}.xml"
            if ann_path not in z.namelist():
                return None
            
            z.extract(ann_path, tmpdir)
            xml_path = Path(tmpdir) / ann_path
            return parse_pascal_voc_xml(xml_path)
    
    return None


def extract_from_llvip(image_name, zips_dir):
    """Extract annotation from LLVIP zip."""
    zip_path = zips_dir / 'LLVIP.zip'
    if not zip_path.exists():
        return None
    
    # Extract image ID (e.g., "010001" from "llvip_010001.jpg")
    image_id = image_name.replace('llvip_', '').replace('.jpg', '')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            ann_path = f"LLVIP/Annotations/{image_id}.xml"
            if ann_path not in z.namelist():
                return None
            
            z.extract(ann_path, tmpdir)
            xml_path = Path(tmpdir) / ann_path
            return parse_pascal_voc_xml(xml_path)
    
    return None


def extract_from_flir_adas(image_name, modality, zips_dir):
    """Extract annotation from FLIR ADAS zip."""
    zip_path = zips_dir / 'FLIR_ADAS_Thermal_v2.zip'
    if not zip_path.exists():
        return None
    
    # FLIR uses COCO format, need to find the right split
    splits = ['train', 'val', 'test']
    if 'video' in image_name:
        split_type = 'video'
        split_name = 'test'
    else:
        split_type = 'images'
        split_name = 'train'  # Try train first
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Try different splits
            for split in splits:
                if split_type == 'video':
                    coco_path = f"FLIR_ADAS_v2/video_{modality}_{split}/coco.json"
                else:
                    coco_path = f"FLIR_ADAS_v2/images_{modality}_{split}/coco.json"
                
                if coco_path in z.namelist():
                    z.extract(coco_path, tmpdir)
                    result = parse_coco_json(Path(tmpdir) / coco_path, image_name)
                    if result:
                        return result
    
    return None


def extract_annotations(dataset_dir, zips_dir, output_file):
    """Extract all annotations and save in COCO format."""
    dataset_path = Path(dataset_dir)
    zips_path = Path(zips_dir)
    
    categories = [
        {'id': 1, 'name': 'human', 'supercategory': 'person'},
        {'id': 2, 'name': 'animal', 'supercategory': 'animal'},
        {'id': 3, 'name': 'vehicle', 'supercategory': 'vehicle'}
    ]
    cat_map = {'human': 1, 'animal': 2, 'vehicle': 3}
    
    coco_data = {
        'info': {'description': 'BioWatch Dataset', 'version': '1.0'},
        'licenses': [],
        'categories': categories,
        'images': [],
        'annotations': []
    }
    
    image_id = 0
    ann_id = 0
    
    print("=" * 60)
    print("Extracting Annotations")
    print("=" * 60)
    
    for modality in ['rgb', 'thermal']:
        for category in ['human', 'animal', 'vehicle']:
            cat_dir = dataset_path / modality / category
            if not cat_dir.exists():
                continue
            
            print(f"\n{modality}/{category}...")
            images = list(cat_dir.glob('*.jpg')) + list(cat_dir.glob('*.JPG')) + \
                    list(cat_dir.glob('*.png')) + list(cat_dir.glob('*.PNG'))
            
            extracted = 0
            for img_file in images:
                img_name = img_file.name
                annotation = None
                
                # Determine source and extract
                if any(img_name.startswith(f'{y}_') for y in ['2007', '2008', '2009', '2010', '2011', '2012']):
                    annotation = extract_from_pascal_voc(img_name, zips_path)
                elif img_name.startswith('llvip_'):
                    annotation = extract_from_llvip(img_name, zips_path)
                elif img_name.startswith('video-'):
                    annotation = extract_from_flir_adas(img_name, modality, zips_path)
                # TODO: Add Conservation Drones, Caltech, etc.
                
                if annotation and annotation['objects']:
                    # Filter to only objects matching current category
                    filtered_objs = [o for o in annotation['objects'] if o['category'] == category]
                    if not filtered_objs:
                        continue
                    
                    # Get image dimensions
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size
                    except:
                        width = annotation.get('width', 0)
                        height = annotation.get('height', 0)
                    
                    # Add image
                    coco_data['images'].append({
                        'id': image_id,
                        'file_name': f"{modality}/{category}/{img_name}",
                        'width': width,
                        'height': height
                    })
                    
                    # Add annotations
                    for obj in filtered_objs:
                        coco_data['annotations'].append({
                            'id': ann_id,
                            'image_id': image_id,
                            'category_id': cat_map[obj['category']],
                            'bbox': obj['bbox'],
                            'area': obj['area'],
                            'iscrowd': 0
                        })
                        ann_id += 1
                    
                    image_id += 1
                    extracted += 1
            
            print(f"  Extracted: {extracted}/{len(images)}")
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Saved: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    print(f"Output: {output_path}")
    print(f"{'=' * 60}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract bounding box annotations')
    parser.add_argument('--dataset-dir', default='dataset', help='Dataset directory')
    parser.add_argument('--zips-dir', default='zips', help='Zips directory')
    parser.add_argument('--output', default='dataset/annotations.json', help='Output file')
    
    args = parser.parse_args()
    extract_annotations(args.dataset_dir, args.zips_dir, args.output)


if __name__ == '__main__':
    main()
