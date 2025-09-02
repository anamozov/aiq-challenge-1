import json
from pathlib import Path
import argparse
import cv2
import numpy as np
from typing import List, Tuple


def iou_circles(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> float:
    # Approximate IoU via rasterization for simplicity
    rmax = int(max(c1[2], c2[2]) * 2 + 10)
    canvas = np.zeros((rmax, rmax), dtype=np.uint8)
    off = rmax // 2
    cv2.circle(canvas, (off, off), int(c1[2]), 255, -1)
    a1 = (canvas > 0).sum()
    canvas2 = np.zeros_like(canvas)
    cv2.circle(canvas2, (off, off), int(c2[2]), 255, -1)
    a2 = (canvas2 > 0).sum()
    inter = ((canvas & canvas2) > 0).sum()
    union = a1 + a2 - inter
    return float(inter) / float(union) if union > 0 else 0.0


def match_circles(pred: List[Tuple[float, float, float]], gt: List[Tuple[float, float, float]], iou_thr=0.3):
    matched_pred = set()
    matched_gt = set()
    for gi, g in enumerate(gt):
        best = -1.0
        best_pi = -1
        for pi, p in enumerate(pred):
            if pi in matched_pred:
                continue
            iou = iou_circles(p, g)
            if iou > best:
                best = iou
                best_pi = pi
        if best >= iou_thr and best_pi >= 0:
            matched_pred.add(best_pi)
            matched_gt.add(gi)
    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return tp, fp, fn, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground-truth JSON")
    args = parser.parse_args()

    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    # Expected format per image: {image_id: [[cx, cy, r], ...]}
    image_ids = sorted(set(pred.keys()) & set(gt.keys()))

    agg_tp = agg_fp = agg_fn = 0
    for iid in image_ids:
        p = [tuple(map(float, c)) for c in pred[iid]]
        g = [tuple(map(float, c)) for c in gt[iid]]
        tp, fp, fn, precision, recall = match_circles(p, g)
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



