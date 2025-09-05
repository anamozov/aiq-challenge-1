#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format for coin detection dataset.
"""

import json
import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, train_ratio=0.8):
    """
    Convert COCO format annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO annotation JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format dataset
        train_ratio: Ratio of images to use for training (default: 0.8)
    """
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_path = Path(output_dir)
    train_images_dir = output_path / 'images' / 'train2017'
    val_images_dir = output_path / 'images' / 'val2017'
    train_labels_dir = output_path / 'labels' / 'train2017'
    val_labels_dir = output_path / 'labels' / 'val2017'
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create category mapping (COCO category_id -> YOLO class_id)
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    # Filter out the supercategory and map coin class to 0
    coin_categories = {cat_id: 0 for cat_id, cat_name in categories.items() 
                      if cat_name != 'Coin-Dataset'}
    
    print(f"Found categories: {categories}")
    print(f"Coin categories mapped to class 0: {coin_categories}")
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Create image info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Get all image IDs and shuffle for train/val split
    all_image_ids = list(image_info.keys())
    random.shuffle(all_image_ids)
    
    # Split into train and val
    split_idx = int(len(all_image_ids) * train_ratio)
    train_image_ids = all_image_ids[:split_idx]
    val_image_ids = all_image_ids[split_idx:]
    
    print(f"Total images: {len(all_image_ids)}")
    print(f"Train images: {len(train_image_ids)}")
    print(f"Val images: {len(val_image_ids)}")
    
    # Process images and annotations
    def process_split(image_ids, source_images_dir, dest_images_dir, labels_dir, split_name):
        print(f"\nProcessing {split_name} split...")
        
        for i, image_id in enumerate(image_ids):
            if i % 50 == 0:
                print(f"Processing {split_name} image {i+1}/{len(image_ids)}")
            
            img_info = image_info[image_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            src_img_path = Path(source_images_dir) / img_filename
            dst_img_path = dest_images_dir / img_filename
            
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image not found: {src_img_path}")
                continue
            
            # Create YOLO annotation file
            label_filename = img_filename.replace('.jpg', '.txt')
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        category_id = ann['category_id']
                        
                        # Skip if not a coin category
                        if category_id not in coin_categories:
                            continue
                        
                        # Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized)
                        x, y, w, h = ann['bbox']
                        
                        # Convert to center coordinates and normalize
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        w_norm = w / img_width
                        h_norm = h / img_height
                        
                        # Write YOLO format: class_id x_center y_center width height
                        f.write(f"{coin_categories[category_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # Process train and validation splits
    process_split(train_image_ids, images_dir, train_images_dir, train_labels_dir, "train")
    process_split(val_image_ids, images_dir, val_images_dir, val_labels_dir, "val")
    
    # Create train2017.txt and val2017.txt files
    def create_txt_files(image_ids, images_dir, txt_filename):
        txt_path = output_path / txt_filename
        with open(txt_path, 'w') as f:
            for image_id in image_ids:
                img_info = image_info[image_id]
                img_filename = img_info['file_name']
                # Write relative path from dataset root
                f.write(f"images/{'train2017' if txt_filename == 'train2017.txt' else 'val2017'}/{img_filename}\n")
    
    create_txt_files(train_image_ids, train_images_dir, 'train2017.txt')
    create_txt_files(val_image_ids, val_images_dir, 'val2017.txt')
    
    print(f"\nConversion completed!")
    print(f"Dataset saved to: {output_path}")
    print(f"Train images: {len(train_image_ids)}")
    print(f"Val images: {len(val_image_ids)}")
    
    return output_path

if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Paths
    coco_json_path = "../coin-dataset/_annotations.coco.json"
    images_dir = "../coin-dataset"
    output_dir = "Dataset/COCO"
    
    # Convert dataset
    convert_coco_to_yolo(coco_json_path, images_dir, output_dir)
