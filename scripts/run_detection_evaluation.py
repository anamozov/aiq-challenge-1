#!/usr/bin/env python3
"""
Script to run coin detection on all images in the dataset and evaluate using the eval.py script.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# Add the app directory to the path so we can import the detection pipeline
sys.path.append('app')
from pipeline.detect import segment_circles_and_mask, Circle


def circle_to_coco_bbox(circle: Tuple[float, float, float]) -> List[float]:
    """
    Convert circle [cx, cy, r] to COCO bbox [x, y, width, height]
    """
    cx, cy, r = circle
    x = cx - r
    y = cy - r
    w = 2 * r
    h = 2 * r
    return [x, y, w, h]


def load_coco_annotations(coco_file: str) -> Dict[str, List[List[float]]]:
    """
    Load COCO annotations and keep in COCO bbox format.
    Returns dict mapping image filename to list of bboxes.
    """
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from image_id to filename
    id_to_filename = {}
    for img in coco_data['images']:
        id_to_filename[img['id']] = img['file_name']
    
    # Keep annotations in COCO bbox format
    image_bboxes = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        filename = id_to_filename[image_id]
        
        if filename not in image_bboxes:
            image_bboxes[filename] = []
        
        # Keep COCO bbox format: [x, y, width, height]
        bbox = ann['bbox']
        image_bboxes[filename].append(bbox)
    
    return image_bboxes


def run_detection_on_dataset(dataset_dir: str) -> Dict[str, List[List[float]]]:
    """
    Run detection on all images in the dataset.
    Returns dict mapping image filename to list of detected bboxes in COCO format.
    """
    dataset_path = Path(dataset_dir)
    predictions = {}
    
    # Get all image files
    image_files = list(dataset_path.glob("*.jpg"))
    print(f"Found {len(image_files)} images to process")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
        
        # Load image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Warning: Could not load {image_path}")
            continue
        
        # Run detection
        try:
            mask, circles = segment_circles_and_mask(image_bgr)
            
            # Convert Circle objects to COCO bbox format
            bbox_list = [circle_to_coco_bbox((c.cx, c.cy, c.r)) for c in circles]
            predictions[image_path.name] = bbox_list
            
            print(f"  Detected {len(circles)} circles")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            predictions[image_path.name] = []
    
    return predictions


def main():
    dataset_dir = "/home/anamozov/data/codes/mine/AIQ/coin-dataset"
    coco_file = os.path.join(dataset_dir, "_annotations.coco.json")
    
    print("Loading COCO annotations...")
    ground_truth = load_coco_annotations(coco_file)
    print(f"Loaded ground truth for {len(ground_truth)} images")
    
    print("\nRunning detection on all images...")
    predictions = run_detection_on_dataset(dataset_dir)
    print(f"Generated predictions for {len(predictions)} images")
    
    # Save predictions to JSON file
    predictions_file = "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {predictions_file}")
    
    # Convert ground truth to the format expected by eval.py
    gt_file = "ground_truth.json"
    with open(gt_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground truth to {gt_file}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    import subprocess
    result = subprocess.run([
        "python3", "scripts/eval.py", 
        "--pred", predictions_file,
        "--gt", gt_file
    ], capture_output=True, text=True)
    
    print("Evaluation Results:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    print(f"\nEvaluation completed. Check {predictions_file} and {gt_file} for detailed results.")


if __name__ == "__main__":
    main()
