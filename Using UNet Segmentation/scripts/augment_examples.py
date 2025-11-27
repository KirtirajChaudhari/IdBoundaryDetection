#!/usr/bin/env python3
"""
Generates a small set of augmented images + masks for quick demonstration.
Uses albumentations pipeline to augment a few images from data/images/train.
"""
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import albumentations as A
from tqdm import tqdm

def augment_one(img, mask, aug):
    arr = np.array(img)
    m = np.array(mask)
    out = aug(image=arr, mask=m)
    img_out = Image.fromarray((out['image']).astype('uint8'))
    mask_out = Image.fromarray((out['mask']*255).astype('uint8'))
    return img_out, mask_out

def main(data_root, n=10):
    data_root = Path(data_root)
    src_dir = data_root / 'images' / 'train'
    mask_dir = data_root / 'masks' / 'train'
    out_dir = data_root / 'aug_examples'
    out_dir.mkdir(parents=True, exist_ok=True)
    aug = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.OneOf([A.RandomBrightnessContrast(), A.HueSaturationValue()]),
        A.Perspective(scale=(0.02,0.2)),
        A.GaussNoise(),
        A.CLAHE(),
        A.Resize(512, 512)
    ])
    files = list(src_dir.glob('*'))
    files = files[:n]
    for f in tqdm(files):
        mask_f = mask_dir / f.name
        if not mask_f.exists():
            continue
        img = Image.open(f).convert('RGB')
        mask = Image.open(mask_f).convert('L')
        img_out, mask_out = augment_one(img, mask, aug)
        img_out.save(out_dir / f.name)
        mask_out.save(out_dir / f.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--n', type=int, default=10)
    args = parser.parse_args()
    main(args.data, n=args.n)
