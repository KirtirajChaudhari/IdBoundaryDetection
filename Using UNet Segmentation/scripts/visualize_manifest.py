#!/usr/bin/env python3
"""
Creates a small QA grid for random samples from data/manifest.csv.
Saves a PNG with image + mask overlay thumbnails for manual inspection.
"""
from pathlib import Path
import argparse
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageFont
import math

def overlay_mask(image, mask, alpha=0.35):
    img = image.convert('RGBA')
    mask_col = Image.new('RGBA', img.size, (255,0,0,int(255*alpha)))
    mask_bin = mask.convert('L').point(lambda p: 255 if p>127 else 0)
    mask_col.putalpha(mask_bin)
    composite = Image.alpha_composite(img, mask_col)
    return composite.convert('RGB')

def make_grid(samples, cols=3, thumb_size=(320,240), out_path='qa_grid.png'):
    rows = math.ceil(len(samples)/cols)
    W = cols*thumb_size[0]
    H = rows*thumb_size[1]
    canvas = Image.new('RGB', (W, H), (255,255,255))
    for idx, (img_path, mask_path) in enumerate(samples):
        try:
            img = Image.open(img_path).convert('RGB').resize(thumb_size)
            mask = Image.open(mask_path).convert('L').resize(thumb_size)
            over = overlay_mask(img, mask)
            x = (idx%cols)*thumb_size[0]
            y = (idx//cols)*thumb_size[1]
            canvas.paste(over, (x,y))
        except Exception:
            continue
    canvas.save(out_path)
    print("Saved QA grid to", out_path)

def main(manifest, out, n=9):
    df = pd.read_csv(manifest)
    df = df.sample(n=min(n, len(df))).reset_index(drop=True)
    samples = []
    for _, row in df.iterrows():
        img = Path(row['dst_image'])
        mask = Path(row['dst_mask'])
        if img.exists() and mask.exists():
            samples.append((img, mask))
    make_grid(samples, cols=3, out_path=out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', default='qa_grid.png')
    parser.add_argument('--n', type=int, default=9)
    args = parser.parse_args()
    main(args.manifest, args.out, args.n)
