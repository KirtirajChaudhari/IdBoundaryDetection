#!/usr/bin/env python3
"""
merge_datasets.py

Merges multiple dataset sources into a single data/ structure:
  data/images/{train,val,test}
  data/masks/{train,val,test}
Produces data/manifest.csv.

Supports:
 - flat image_dir + mask_dir pairs,
 - COCO -> masks (if pycocotools is available),
 - MIDV-500 layout (root_dir with many subfolders containing images/ and ground_truth/ JSONs).
 - optional frame CSV for custom splits (path provided in merge_config.json)

Usage:
  python scripts/merge_datasets.py --config scripts/merge_config.json --out data --seed 42
"""
from pathlib import Path
import argparse
import json
import random
from PIL import Image, ImageDraw
import numpy as np
import shutil
import csv
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
    HAS_PYCOCOTOOLS = True
except Exception:
    HAS_PYCOCOTOOLS = False

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_binary_mask(mask_arr: np.ndarray, out_path: Path):
    # mask_arr: boolean or 0/1 array with shape (H,W)
    img = Image.fromarray((mask_arr.astype(np.uint8) * 255))
    img.save(str(out_path))

def polygon_to_mask(polygon_pts, image_size):
    # polygon_pts: list of (x,y) absolute coords in image pixel space
    W, H = image_size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    # ensure integer coords
    flat = [tuple(map(float, p)) for p in polygon_pts]
    # PIL expects sequence of tuples (x,y)
    try:
        draw.polygon(flat, outline=1, fill=1)
    except Exception:
        # fallback: round to ints
        flat_int = [tuple(map(int, map(round, p))) for p in polygon_pts]
        draw.polygon(flat_int, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

def parse_midv_groundtruth_json(gt_json_path):
    """
    Tries to read ground-truth JSON from MIDV-500 ground_truth folder.
    Returns list of polygons (each a list of (x,y)).
    Heuristics: JSON sometimes contains 'quad' or 'quad_points' or 'points' or 'polygons'.
    """
    try:
        with open(gt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    polygons = []
    # Common keys
    if isinstance(data, dict):
        # If top-level keys like 'quad' or 'points'
        for key in ('quad', 'quad_points', 'points', 'polygon', 'polygons'):
            if key in data:
                obj = data[key]
                if isinstance(obj, list) and len(obj) >= 4:
                    # obj might be [x1,y1,x2,y2...]
                    if all(isinstance(v, (int, float)) for v in obj):
                        pts = [(obj[i], obj[i+1]) for i in range(0, len(obj), 2)]
                        polygons.append(pts)
                    else:
                        # maybe list of pairs
                        if all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in obj):
                            pts = [(float(pt[0]), float(pt[1])) for pt in obj]
                            polygons.append(pts)
        # If contains keys like 'annotations' or 'shapes'
        for main_key in ('annotations', 'shapes', 'polygons'):
            if main_key in data and isinstance(data[main_key], list):
                for item in data[main_key]:
                    # try find 'points' or 'polygon' inside
                    for sub in ('points', 'polygon', 'poly'):
                        if sub in item:
                            pts = item[sub]
                            if isinstance(pts, list) and len(pts) >= 4:
                                # handle [x,y,...] or [[x,y],...]
                                if all(isinstance(v, (int, float)) for v in pts):
                                    pts_list = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
                                else:
                                    pts_list = [(float(p[0]), float(p[1])) for p in pts]
                                polygons.append(pts_list)
        # some files have a top-level list of points under filename keys
    elif isinstance(data, list):
        # If JSON is a list of points or list of objects with 'points'
        for item in data:
            if isinstance(item, dict):
                if 'points' in item:
                    pts = item['points']
                    polygons.append([(float(p[0]), float(p[1])) for p in pts])
            elif isinstance(item, list) and len(item) >= 4:
                if all(isinstance(v, (int, float)) for v in item):
                    polygons.append([(item[i], item[i+1]) for i in range(0, len(item), 2)])
    return polygons

def dataset_files_from_dir(image_dir: Path):
    exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

    files = []
    for p in image_dir.rglob('*'):
        # Skip MacOS junk files: "._filename"
        if p.name.startswith("._"):
            continue

        if p.suffix.lower() in exts:
            files.append(p)

    return sorted(files)

def process_flat_dataset(entry, out_root: Path, ratios, seed, frame_csv_path=None):
    """
    entry: dict with keys name, image_dir, mask_dir (optional), coco_json (optional), split_file (optional)
    """
    rng = random.Random(seed)
    name = entry.get('name')
    image_dir = Path(entry['image_dir'])
    mask_dir = Path(entry['mask_dir']) if entry.get('mask_dir') else None
    coco_json = entry.get('coco_json')
    split_file = entry.get('split_file')  # optional explicit split map
    files = dataset_files_from_dir(image_dir)
    if not files:
        print(f"[WARN] No images found for dataset {name} at {image_dir}")
        return []

    # map filename -> split
    split_map = {}
    if split_file:
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                for line in f:
                    fn,sp = line.strip().split(',')
                    split_map[fn] = sp
        except Exception:
            print("[WARN] Could not read split_file:", split_file)

    # random split if not provided
    if not split_map:
        n = len(files)
        idxs = list(range(n))
        rng.shuffle(idxs)
        n_train = int(ratios[0]*n)
        n_val = int(ratios[1]*n)
        for i, idx in enumerate(idxs):
            fn = files[idx].name
            if i < n_train:
                split_map[fn] = 'train'
            elif i < n_train + n_val:
                split_map[fn] = 'val'
            else:
                split_map[fn] = 'test'

    entries_out = []
    for img_path in tqdm(files, desc=f"Processing {name}"):
        fn = img_path.name
        split = split_map.get(fn, 'train')
        dst_img_dir = out_root / 'images' / split
        dst_mask_dir = out_root / 'masks' / split
        ensure_dir(dst_img_dir); ensure_dir(dst_mask_dir)
        dst_name = f"{name}__{fn}"
        dst_img_path = dst_img_dir / dst_name
        shutil.copyfile(img_path, dst_img_path)

        # determine mask source
        mask_saved = False
        dst_mask_path = dst_mask_dir / dst_name
        # 1) mask_dir has same filename
        if mask_dir:
            cand = mask_dir / fn
            if not cand.exists():
                cand = mask_dir / (Path(fn).stem + '.png')
            if cand.exists():
                try:
                    im = Image.open(cand).convert('L')
                    arr = np.array(im)
                    bin_mask = (arr > 127).astype(np.uint8)
                    save_binary_mask(bin_mask, dst_mask_path)
                    mask_saved = True
                except Exception:
                    mask_saved = False

        # 2) coco_json handling
        if (not mask_saved) and coco_json and HAS_PYCOCOTOOLS:
            try:
                coco = COCO(coco_json)
                img_infos = coco.getImgIds()
                # find by file name
                img_id = None
                for iid in coco.getImgIds():
                    info = coco.loadImgs(iid)[0]
                    if info['file_name'] == fn:
                        img_id = info['id']; info_meta = info; break
                if img_id is not None:
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    anns = coco.loadAnns(ann_ids)
                    H = info_meta.get('height'); W = info_meta.get('width')
                    if H is None or W is None:
                        with Image.open(img_path) as tmp:
                            W, H = tmp.size
                    mask_combined = np.zeros((H, W), dtype=np.uint8)
                    for ann in anns:
                        rle = maskUtils.frPyObjects(ann['segmentation'], H, W)
                        rle = maskUtils.merge(rle)
                        m = maskUtils.decode(rle)
                        if m.ndim == 3:
                            m = np.any(m, axis=2).astype(np.uint8)
                        mask_combined = np.logical_or(mask_combined, m)
                    save_binary_mask(mask_combined.astype(np.uint8), dst_mask_path)
                    mask_saved = True
            except Exception:
                pass

        # 3) fallback: try to find masks in same folder tree using stem
        if (not mask_saved):
            stem = Path(fn).stem
            found = None
            if mask_dir:
                candidates = list(mask_dir.rglob(f"{stem}*"))
                if candidates:
                    found = candidates[0]
            if not found:
                # search near image directory
                candidates = list(img_path.parent.rglob(f"{stem}*"))
                for cand in candidates:
                    if cand.suffix.lower() in ('.png', '.jpg', '.jpeg') and cand != img_path:
                        # treat as mask if its content is grayscale or small file - best-effort
                        found = cand; break
            if found:
                try:
                    im = Image.open(found).convert('L')
                    arr = np.array(im)
                    bin_mask = (arr > 127).astype(np.uint8)
                    save_binary_mask(bin_mask, dst_mask_path)
                    mask_saved = True
                except Exception:
                    mask_saved = False

        # 4) if still not saved -> write empty mask of same size
        if not mask_saved:
            try:
                with Image.open(img_path) as tmp:
                    W, H = tmp.size
            except Exception:
                print(f"[WARN] Skipping unreadable image: {img_path}")
                # skip adding this sample to the outputs; continue to next image
                continue

            empty = np.zeros((H, W), dtype=np.uint8)
            save_binary_mask(empty, dst_mask_path)

        entries_out.append({
            'orig_filename': str(img_path),
            'dst_image': str(dst_img_path),
            'dst_mask': str(dst_mask_path),
            'dataset': name,
            'split': split
        })
    return entries_out


def process_midv500(entry, out_root: Path, ratios, seed):
    """
    entry: dict with 'root_dir' pointing to the midv500 dataset root.
    For each subfolder (01_*, 02_* ...), this function reads images from sub/images
    and ground-truth JSONs from sub/ground_truth (if present) and rasterizes polygons into masks.
    """
    rng = random.Random(seed)
    root = Path(entry['root_dir'])
    if not root.exists():
        print("[WARN] midv500 root does not exist:", root)
        return []
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    all_files = []
    for sub in subdirs:
        imgs_dir = sub / 'images'
        gt_dir = sub / 'ground_truth'
        if not imgs_dir.exists():
            continue
        files = list(imgs_dir.glob('*'))
        for f in files:
            if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'):
                all_files.append((sub.name, f, gt_dir))

    if not all_files:
        print("[WARN] No midv500 images found under", root)
        return []

    # random split
    n = len(all_files)
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    split_map = {}
    for i, idx in enumerate(idxs):
        if i < n_train:
            split_map[idx] = 'train'
        elif i < n_train + n_val:
            split_map[idx] = 'val'
        else:
            split_map[idx] = 'test'

    entries_out = []
    for i, (folder_name, img_path, gt_dir) in enumerate(tqdm(all_files, desc="Processing midv500")):
        split = split_map.get(i, 'train')
        dst_img_dir = out_root / 'images' / split
        dst_mask_dir = out_root / 'masks' / split
        ensure_dir(dst_img_dir); ensure_dir(dst_mask_dir)
        fn = img_path.name
        dst_name = f"midv500_{folder_name}__{fn}"
        dst_img_path = dst_img_dir / dst_name
        shutil.copyfile(img_path, dst_img_path)

        # Attempt to find a ground truth JSON with matching stem in gt_dir
        mask_saved = False
        dst_mask_path = dst_mask_dir / dst_name
        if gt_dir.exists():
            # typical file names in MIDV500: same stem with .json
            candidate_json = gt_dir / (Path(fn).stem + '.json')
            if not candidate_json.exists():
                # try any JSON that contains the image stem
                for p in gt_dir.glob('*.json'):
                    if Path(fn).stem in p.name:
                        candidate_json = p
                        break
            if candidate_json.exists():
                polygons = parse_midv_groundtruth_json(candidate_json)
                if polygons:
                    # choose first polygon (document quad) or union if multiple
                    try:
                        with Image.open(img_path) as im:
                            W, H = im.size
                        mask_combined = np.zeros((H, W), dtype=np.uint8)
                        for poly in polygons:
                            mask = polygon_to_mask(poly, (W, H))
                            mask_combined = np.logical_or(mask_combined, mask)
                        save_binary_mask(mask_combined.astype(np.uint8), dst_mask_path)
                        mask_saved = True
                    except Exception:
                        mask_saved = False

        if not mask_saved:
            # fallback: empty mask
            with Image.open(img_path) as im:
                W, H = im.size
            save_binary_mask(np.zeros((H, W), dtype=np.uint8), dst_mask_path)

        entries_out.append({
            'orig_filename': str(img_path),
            'dst_image': str(dst_img_path),
            'dst_mask': str(dst_mask_path),
            'dataset': 'midv500',
            'split': split
        })
    return entries_out

def main():
    parser = argparse.ArgumentParser(description="Merge many datasets into unified data/ layout")
    parser.add_argument('--config', required=True, help='path to merge_config.json')
    parser.add_argument('--out', required=True, help='output root (data/)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    out_root = Path(args.out)
    ensure_dir(out_root / 'images' / 'train')
    ensure_dir(out_root / 'masks' / 'train')
    master = []
    ratios = tuple(cfg.get('ratios', [0.7, 0.15, 0.15]))
    frame_csv = cfg.get('frame_csv')

    for ds in cfg.get('datasets', []):
        if ds.get('mode') == 'midv500' and 'root_dir' in ds:
            print("[INFO] Processing MIDV-500 layout for", ds['root_dir'])
            entries = process_midv500(ds, out_root, ratios, args.seed)
        else:
            entries = process_flat_dataset(ds, out_root, ratios, args.seed)
        master.extend(entries)

    # write manifest csv
    manifest_path = out_root / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['orig_filename', 'dst_image', 'dst_mask', 'dataset', 'split']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in master:
            writer.writerow(row)
    print("Saved manifest to:", manifest_path)
    # summary
    from collections import Counter
    splits = Counter([r['split'] for r in master])
    datasets = Counter([r['dataset'] for r in master])
    print("Split counts:", splits)
    print("Datasets merged:", datasets)

if __name__ == '__main__':
    main()
