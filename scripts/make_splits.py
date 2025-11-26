#!/usr/bin/env python3
"""
Creates reproducible train/val/test splits from data/manifest.csv if you want to re-split.
Writes new files into data/images/{train,val,test} and data/masks/... (moves files).
"""
from pathlib import Path
import argparse
import pandas as pd
import random
import shutil

def main(manifest, out_root, seed=42, ratios=(0.7,0.15,0.15)):
    manifest = Path(manifest)
    df = pd.read_csv(manifest)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n*ratios[0])
    n_val = int(n*ratios[1])
    df['split'] = ['train']*n_train + ['val']*n_val + ['test']*(n - n_train - n_val)
    out_root = Path(out_root)
    for _, row in df.iterrows():
        src_img = Path(row['dst_image'])
        src_mask = Path(row['dst_mask'])
        split = row['split']
        dst_img = out_root / 'images' / split / src_img.name
        dst_mask = out_root / 'masks' / split / src_mask.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        # move (if on same filesystem, else copy)
        shutil.copyfile(src_img, dst_img)
        shutil.copyfile(src_mask, dst_mask)
    # write new manifest
    new_manifest = out_root / 'manifest_resplit.csv'
    df.to_csv(new_manifest, index=False)
    print("Wrote new manifest to:", new_manifest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.manifest, args.out, seed=args.seed)
