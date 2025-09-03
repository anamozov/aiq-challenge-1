#!/usr/bin/env python3
"""
Unified evaluation script that:
1. Takes a list of images from a text file
2. Extracts ground truth data for those images from COCO annotations
3. Runs image processing-based detection on those images
4. Evaluates the results using IoU matching

Usage:
    python3 evaluate_on_image_list.py <image_list.txt> <coco_annotations.json> [--iou-threshold 0.5]
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os
import argparse

# Add the project root to the path so we can import the detection pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline.detect import segment_circles_and_mask, Circle


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


def load_image_list(image_list_file: str) -> List[str]:
    """
    Load list of image filenames from a text file.
    Each line should contain a path to an image file.
    """
    image_list = []
    with open(image_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Extract just the filename from the path
                filename = os.path.basename(line)
                image_list.append(filename)
    
    return image_list


def filter_ground_truth_for_image_list(ground_truth: Dict[str, List[List[float]]], image_list: List[str]) -> Dict[str, List[List[float]]]:
    """
    Filter ground truth to only include images that are in the image list.
    """
    image_set = set(image_list)
    filtered_gt = {}
    
    for filename, bboxes in ground_truth.items():
        if filename in image_set:
            filtered_gt[filename] = bboxes
    
    print(f"Filtered ground truth from {len(ground_truth)} to {len(filtered_gt)} images")
    return filtered_gt


def find_image_paths(image_list: List[str], search_dirs: List[str]) -> Dict[str, str]:
    """
    Find the full paths for images in the image list by searching in the given directories.
    Returns dict mapping filename to full path.
    """
    image_paths = {}
    
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            print(f"Warning: Search directory {search_dir} does not exist")
            continue
            
        for image_file in search_path.rglob("*.jpg"):
            if image_file.name in image_list:
                image_paths[image_file.name] = str(image_file)
    
    print(f"Found {len(image_paths)} out of {len(image_list)} images")
    return image_paths


def run_detection_on_image_list(image_paths: Dict[str, str], ground_truth: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
    """
    Run detection on images that have ground truth annotations.
    Returns dict mapping image filename to list of detected bboxes in COCO format.
    """
    predictions = {}
    
    # Filter to only images that have both paths and ground truth
    valid_images = set(image_paths.keys()) & set(ground_truth.keys())
    valid_images = list(valid_images)
    
    print(f"Running detection on {len(valid_images)} images with both paths and ground truth")
    
    for i, filename in enumerate(valid_images):
        image_path = image_paths[filename]
        print(f"Processing {i+1}/{len(valid_images)}: {filename}")
        
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: Could not load {image_path}")
            continue
        
        # Run detection using the image processing pipeline
        try:
            mask, circles = segment_circles_and_mask(image_bgr)
            
            # Convert Circle objects to COCO bbox format
            bbox_list = [circle_to_coco_bbox((c.cx, c.cy, c.r)) for c in circles]
            predictions[filename] = bbox_list
            
            print(f"  Detected {len(circles)} circles")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            predictions[filename] = []
    
    return predictions


def bbox_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two bounding boxes in COCO format (x, y, width, height)"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_bboxes(pred_bboxes: List[Tuple[float, float, float, float]], 
                gt_bboxes: List[Tuple[float, float, float, float]], 
                iou_thr=0.5):
    """Match predicted bboxes with ground truth bboxes using IoU threshold"""
    matched_pred = set()
    matched_gt = set()
    
    for gi, gt_bbox in enumerate(gt_bboxes):
        best_iou = -1.0
        best_pi = -1
        
        for pi, pred_bbox in enumerate(pred_bboxes):
            if pi in matched_pred:
                continue
            
            iou = bbox_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        
        if best_iou >= iou_thr and best_pi >= 0:
            matched_pred.add(best_pi)
            matched_gt.add(gi)
    
    tp = len(matched_pred)
    fp = len(pred_bboxes) - tp
    fn = len(gt_bboxes) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    
    return tp, fp, fn, precision, recall


def evaluate_predictions(predictions: Dict[str, List[List[float]]], 
                        ground_truth: Dict[str, List[List[float]]], 
                        iou_threshold: float = 0.5):
    """
    Evaluate predictions against ground truth using IoU matching.
    """
    # Get common images
    image_ids = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    print(f"Evaluating {len(image_ids)} images with IoU threshold {iou_threshold}")

    agg_tp = agg_fp = agg_fn = 0
    for iid in image_ids:
        # Convert to tuples for bbox matching (already in COCO format)
        pred_bboxes = [tuple(map(float, bbox)) for bbox in predictions[iid]]
        gt_bboxes = [tuple(map(float, bbox)) for bbox in ground_truth[iid]]
        
        tp, fp, fn, precision, recall = match_bboxes(pred_bboxes, gt_bboxes, iou_threshold)
        print(f"image {iid}: TP={tp} FP={fp} FN={fn} P={precision:.3f} R={recall:.3f}")
        agg_tp += tp
        agg_fp += fp
        agg_fn += fn

    p = agg_tp / (agg_tp + agg_fp) if agg_tp + agg_fp > 0 else 0.0
    r = agg_tp / (agg_tp + agg_fn) if agg_tp + agg_fn > 0 else 0.0
    if p + r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0.0
    print(f"Overall: P={p:.3f} R={r:.3f} F1={f1:.3f}")
    
    return p, r, f1


def main():
    parser = argparse.ArgumentParser(description='Evaluate image processing-based coin detection on a list of images')
    parser.add_argument('image_list', help='Text file containing list of image paths (one per line)')
    parser.add_argument('coco_annotations', help='COCO format annotations JSON file')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--search-dirs', nargs='+', default=['../../coin-dataset', '../../YOLOv11/Dataset/COCO/images'], 
                       help='Directories to search for images (default: coin-dataset and YOLOv11 val2017)')
    parser.add_argument('--output-prefix', default='evaluation', help='Prefix for output files (default: evaluation)')
    
    args = parser.parse_args()
    
    print(f"Loading image list from: {args.image_list}")
    image_list = load_image_list(args.image_list)
    print(f"Loaded {len(image_list)} images from list")
    
    print(f"Loading COCO annotations from: {args.coco_annotations}")
    ground_truth = load_coco_annotations(args.coco_annotations)
    print(f"Loaded ground truth for {len(ground_truth)} images")
    
    # Filter ground truth to only include images in the list
    filtered_ground_truth = filter_ground_truth_for_image_list(ground_truth, image_list)
    
    # Find image paths
    print(f"Searching for images in: {args.search_dirs}")
    image_paths = find_image_paths(image_list, args.search_dirs)
    
    # Run detection
    print(f"\nRunning image processing-based detection...")
    predictions = run_detection_on_image_list(image_paths, filtered_ground_truth)
    print(f"Generated predictions for {len(predictions)} images")
    
    # Save predictions and ground truth
    predictions_file = f"{args.output_prefix}_predictions.json"
    gt_file = f"{args.output_prefix}_ground_truth.json"
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {predictions_file}")
    
    with open(gt_file, 'w') as f:
        json.dump(filtered_ground_truth, f, indent=2)
    print(f"Saved filtered ground truth to {gt_file}")
    
    # Run evaluation
    print(f"\nRunning evaluation...")
    precision, recall, f1 = evaluate_predictions(predictions, filtered_ground_truth, args.iou_threshold)
    
    print(f"\nEvaluation completed.")
    print(f"Results: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"Predictions saved to: {predictions_file}")
    print(f"Filtered ground truth saved to: {gt_file}")


if __name__ == "__main__":
    main()
