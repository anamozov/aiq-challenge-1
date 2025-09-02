import json
from pathlib import Path
import argparse
import cv2
import numpy as np
from typing import List, Tuple


def circle_to_bbox(circle: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """Convert circle (cx, cy, r) to COCO bbox format (x, y, width, height)"""
    cx, cy, r = circle
    x = cx - r
    y = cy - r
    w = 2 * r
    h = 2 * r
    return x, y, w, h


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


def load_coco_annotations(coco_file: str) -> dict:
    """Load COCO annotations and return dict mapping filename to list of bboxes"""
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from image_id to filename
    id_to_filename = {}
    for img in coco_data['images']:
        id_to_filename[img['id']] = img['file_name']
    
    # Convert annotations to bboxes per image
    image_bboxes = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        filename = id_to_filename[image_id]
        
        if filename not in image_bboxes:
            image_bboxes[filename] = []
        
        # COCO bbox format: [x, y, width, height]
        bbox = ann['bbox']
        image_bboxes[filename].append(bbox)
    
    return image_bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions JSON (COCO bbox format)")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground-truth JSON (COCO bbox format)")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for matching (default: 0.5)")
    args = parser.parse_args()

    # Load predictions (COCO bbox format)
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    
    # Load ground truth (COCO bbox format)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    # Expected format per image: {image_id: [[x, y, w, h], ...]} for both pred and gt
    image_ids = sorted(set(pred.keys()) & set(gt.keys()))
    print(f"Evaluating {len(image_ids)} images with IoU threshold {args.iou_thr}")

    agg_tp = agg_fp = agg_fn = 0
    for iid in image_ids:
        # Convert to tuples for bbox matching (already in COCO format)
        pred_bboxes = [tuple(map(float, bbox)) for bbox in pred[iid]]
        gt_bboxes = [tuple(map(float, bbox)) for bbox in gt[iid]]
        
        tp, fp, fn, precision, recall = match_bboxes(pred_bboxes, gt_bboxes, args.iou_thr)
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


if __name__ == "__main__":
    main()



