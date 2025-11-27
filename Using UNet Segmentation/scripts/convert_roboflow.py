#!/usr/bin/env python3
"""
Simple helper to convert Roboflow PNG mask export (images/ and masks/) into data/ layout.
This script is intentionally minimal — primarily for manual/export conversions.
"""
from pathlib import Path
from PIL import Image
import argparse
import shutil

def main(src_dir, out_images, out_masks):
    src = Path(src_dir)
    imgs = src / 'images'
    masks = src / 'masks'
    out_images = Path(out_images)
    out_masks = Path(out_masks)
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    for p in imgs.iterdir():
        if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        dst_img = out_images / p.name
        shutil.copyfile(p, dst_img)
        mask_candidate = masks / p.name
        if mask_candidate.exists():
            im = Image.open(mask_candidate).convert('L')
            im = im.point(lambda v: 255 if v>127 else 0)
            im.save(out_masks / p.name)
        else:
            # make empty mask
            with Image.open(p) as im:
                W, H = im.size
            empty = Image.new('L', (W, H), 0)
            empty.save(out_masks / p.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir')
    parser.add_argument('out_images')
    parser.add_argument('out_masks')
    args = parser.parse_args()
    main(args.src_dir, args.out_images, args.out_masks)
