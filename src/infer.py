#!/usr/bin/env python3
"""
Inference script. Outputs:
 - overlay PNG with detected min-rotated-rect (4 corners)
 - corners JSON
 - optional warped frontalized crop
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from src.model import build_model
from src.utils import save_mask, mask_to_polygon, polygon_to_minrect_corners, draw_overlay, warp_to_rect
import json

def preprocess(img_pil, target=(512,512)):
    img_resized = img_pil.resize(target, Image.BILINEAR)
    arr = np.array(img_resized).astype('float32')/255.0
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    arr = (arr - mean) / std
    tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).float()
    return tensor, img_resized

def run(image_path, checkpoint, out_dir, device):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(preferred='smp')
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    inp, resized = preprocess(img, target=(512,512))
    inp = inp.to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()  # HxW in resized
    # resize mask back to original
    mask_resized = (probs >= 0.5).astype('uint8')
    mask = Image.fromarray((mask_resized*255).astype('uint8')).resize((orig_w, orig_h), Image.NEAREST)
    mask_arr = np.array(mask) // 255
    mask_path = out_dir / f"{image_path.stem}_mask.png"
    save_mask(mask_arr, mask_path)

    poly = mask_to_polygon(mask_arr)
    corners = polygon_to_minrect_corners(poly) if poly is not None else None
    overlay_path = out_dir / f"{image_path.stem}_overlay.png"
    draw_overlay(img, corners if corners else [], overlay_path)

    warp_path = None
    if corners:
        img_np = np.array(img)
        warped = warp_to_rect(img_np, corners, dst_size=(600,400))
        if warped is not None:
            warp_path = out_dir / f"{image_path.stem}_warp.png"
            Image.fromarray(warped).save(warp_path)

    corners_json = out_dir / f"{image_path.stem}_corners.json"
    corners_list = [{"x": float(x), "y": float(y)} for (x,y) in corners] if corners else []
    with corners_json.open('w', encoding='utf-8') as f:
        json.dump({"corners": corners_list}, f, indent=2)

    print("Saved:", mask_path, overlay_path, corners_json, warp_path if warp_path else "")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('checkpoint')
    parser.add_argument('out_dir')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    run(args.image, args.checkpoint, args.out_dir, args.device)
