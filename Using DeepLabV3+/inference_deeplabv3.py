# inference_deeplabv3.py
"""
Inference for single-class DeepLabV3+ model.
Produces: mask PNG, overlay PNG (alpha), outline PNG, polygon JSON for each input image.
Usage example:
    python inference_deeplabv3.py --ckpt "D:/Codes/CampusDrive/project_root/runs/deeplabv3_exp1/deeplabv3p_best.pth" \
        --input_dir "D:/Codes/CampusDrive/project_root/datasets/test" \
        --out_dir "D:/Codes/CampusDrive/project_root/runs/deeplabv3_exp1/infer" \
        --img_size 512 --device cuda
"""
import argparse
from pathlib import Path
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm

def load_model(ckpt_path, device):
    model = smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=3, classes=1, encoder_weights=None)
    model.to(device)
    ck = torch.load(ckpt_path, map_location=device)
    # attempt robust loading keys
    if "model_state_dict" in ck:
        state = ck["model_state_dict"]
    else:
        state = ck
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def preprocess(img_bgr, img_size):
    # BGR -> RGB, Resize, normalize to imagenet
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_size is not None:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img = (img - mean) / std
    # to CHW
    img = img.transpose(2,0,1)
    return img

def postprocess_mask(mask_pred, orig_shape):
    # mask_pred: numpy HxW values [0..1]
    # threshold to binary and resize back to orig shape
    mask = (mask_pred >= 0.5).astype(np.uint8) * 255
    if mask.shape[:2] != orig_shape[:2]:
        mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def mask_to_contour_coords(mask):
    # find largest contour(s) and return list of polygons (list of (x,y) pairs)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 20:  # skip tiny
            continue
        # simplify contour
        eps = 0.01 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True)
        coords = [(int(pt[0][0]), int(pt[0][1])) for pt in poly]
        if len(coords) >= 3:
            polys.append(coords)
    return polys

def draw_overlay_and_outline(orig_bgr, mask, polys, overlay_alpha=0.4, outline_color=(0,0,255), outline_thickness=3):
    # orig_bgr: BGR uint8
    overlay = orig_bgr.copy()
    # color mask region (green)
    color = (0,255,0)
    mask_bool = (mask > 127).astype(np.uint8)
    colored = np.zeros_like(orig_bgr, dtype=np.uint8)
    colored[:] = color
    overlay = np.where(mask_bool[:,:,None], (overlay * (1-overlay_alpha) + colored * overlay_alpha).astype(np.uint8), overlay)
    # draw outlines on a copy (red by default)
    outlined = orig_bgr.copy()
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(outlined, [pts], isClosed=True, color=outline_color, thickness=outline_thickness, lineType=cv2.LINE_AA)
    return overlay, outlined

def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    ckpt = Path(args.ckpt)
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    model = load_model(str(ckpt), device)

    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    masks_dir = out / "masks"
    overlays_dir = out / "overlays"
    outlines_dir = out / "outlines"
    json_dir = out / "polygons"
    for d in (masks_dir, overlays_dir, outlines_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    ext = tuple(x.lower() for x in args.img_ext.split(","))
    images = sorted([p for p in inp.iterdir() if p.suffix.lower() in ext])
    if len(images) == 0:
        print("No images found in", inp)
        return

    for p in tqdm(images, desc="inference"):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print("could not open", p); continue
        orig_h, orig_w = img_bgr.shape[:2]
        img_proc = preprocess(img_bgr, args.img_size)
        tensor = torch.from_numpy(img_proc).unsqueeze(0).to(device)

        with torch.no_grad():
            out_pred = model(tensor)  # shape [B,1,H,W]
            out_sig = torch.sigmoid(out_pred)
            out_np = out_sig.squeeze().cpu().numpy()

        mask = postprocess_mask(out_np, (orig_h, orig_w, 3))
        # save mask
        mask_path = masks_dir / (p.stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask)

        polys = mask_to_contour_coords(mask)

        overlay_img, outline_img = draw_overlay_and_outline(img_bgr, mask, polys, overlay_alpha=args.alpha,
                                                            outline_color=(0,0,255), outline_thickness=args.thickness)

        cv2.imwrite(str(overlays_dir / (p.stem + "_overlay.png")), overlay_img)
        cv2.imwrite(str(outlines_dir / (p.stem + "_outline.png")), outline_img)

        # save polygon json (list of polygons)
        poly_json = {"file": str(p.name), "polygons": polys}
        with open(json_dir / (p.stem + "_poly.json"), "w", encoding="utf8") as f:
            json.dump(poly_json, f, indent=2)

    print("Done. Outputs in:", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth (deeplabv3p_best.pth)")
    parser.add_argument("--input_dir", required=True, help="Folder with input images")
    parser.add_argument("--out_dir", required=True, help="Folder to write mask/overlay/outline/json")
    parser.add_argument("--img_size", type=int, default=512, help="Resize size used at inference")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha")
    parser.add_argument("--thickness", type=int, default=3, help="Outline thickness")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--img_ext", default=".jpg,.jpeg,.png", help="comma-separated accepted ext")
    args = parser.parse_args()
    run_inference(args)
