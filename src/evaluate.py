#!/usr/bin/env python3
"""
Evaluates predictions (binary masks) against ground-truth masks.
Computes IoU, Dice, Normalized Corner Distance (NCD).
Writes CSV report.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from skimage import measure
from shapely.geometry import Polygon
from shapely.ops import unary_union
import itertools
import math

def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter==0 else 0.0
    return inter/union

def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2*inter/denom

def mask_to_poly(mask):
    contours = measure.find_contours(mask.astype('uint8'), 0.5)
    polys = []
    for c in contours:
        pts = [(float(p[1]), float(p[0])) for p in c]
        try:
            poly = Polygon(pts)
            if poly.is_valid and poly.area > 1:
                polys.append(poly)
        except Exception:
            continue
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.geom_type == "MultiPolygon":
        merged = max(list(merged.geoms), key=lambda a: a.area)
    return merged

def corners_from_poly(poly):
    if poly is None:
        return None
    rect = poly.minimum_rotated_rectangle
    pts = list(rect.exterior.coords)[:-1]
    return pts

def normalized_corner_distance(pred_c, gt_c):
    if pred_c is None or gt_c is None:
        return None
    pred = np.array(pred_c)
    gt = np.array(gt_c)
    diag = np.linalg.norm(gt.max(axis=0) - gt.min(axis=0))
    if diag == 0:
        return None
    best = float('inf')
    for perm in itertools.permutations(range(len(pred))):
        d = np.mean(np.linalg.norm(pred[list(perm)] - gt, axis=1))
        if d < best:
            best = d
    return best / diag

def main(pred_dir, gt_dir, out_csv):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    rows = []
    for p in sorted(gt_dir.iterdir()):
        if p.suffix.lower() not in ('.png','.jpg','.jpeg'):
            continue
        fn = p.name
        gt_mask = np.array(Image.open(p).convert('L')) > 127
        # try find exact same name in pred_dir, otherwise try name_mask.png
        pred_candidate = pred_dir / fn
        if not pred_candidate.exists():
            alt = pred_dir / (p.stem + '_mask.png')
            if alt.exists():
                pred_candidate = alt
            else:
                # try any file with same stem
                found = list(pred_dir.glob(p.stem + '*'))
                pred_candidate = found[0] if found else None
        if pred_candidate is None or not pred_candidate.exists():
            print("Missing prediction for", fn)
            continue
        pred_mask = np.array(Image.open(pred_candidate).convert('L')) > 127
        iou_v = iou(pred_mask, gt_mask)
        dice_v = dice(pred_mask, gt_mask)
        pred_poly = mask_to_poly(pred_mask)
        gt_poly = mask_to_poly(gt_mask)
        ncd = normalized_corner_distance(corners_from_poly(pred_poly), corners_from_poly(gt_poly))
        rows.append({'image': fn, 'iou': iou_v, 'dice': dice_v, 'ncd': ncd})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved report to", out_csv)
    print(df.describe())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--out', default='report.csv')
    args = parser.parse_args()
    main(args.pred_dir, args.gt_dir, args.out)
