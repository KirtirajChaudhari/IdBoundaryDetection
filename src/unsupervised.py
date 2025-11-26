#!/usr/bin/env python3
"""
Unsupervised baseline: ViT patch embeddings + KMeans clustering to produce a foreground mask.
Useful for quick demos if you don't have a trained checkpoint.
"""
import argparse
from pathlib import Path
import torch
import timm
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import math

def extract_patch_embeddings(img_pil, model_name='vit_base_patch16_224', device='cpu'):
    model = timm.create_model(model_name, pretrained=True)
    model.eval().to(device)
    # remove head
    if hasattr(model, 'head'):
        model.head = torch.nn.Identity()
    # prepare image
    img = img_pil.resize((224,224)).convert('RGB')
    arr = np.array(img).astype('float32')/255.0
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    arr = (arr - mean) / std
    x = torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.forward_features(x)  # (B, num_patches+1, dim) or (B, dim, h, w)
    if feats.ndim == 3:
        # (1, N, D)
        feats = feats[0,1:,:].cpu().numpy()  # drop cls token
    else:
        # (1, D, h, w) -> reshape
        feats = feats[0].cpu().numpy()
        C,H,W = feats.shape
        feats = feats.reshape(C, H*W).T
    return feats

def create_mask_by_kmeans(feats, img_size=(224,224), k=2):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(feats)
    labels = kmeans.labels_
    h,w = int(math.sqrt(len(labels))), int(math.sqrt(len(labels)))
    mask = (labels.reshape(h,w) != labels.reshape(h,w)[0,0]).astype(np.uint8)  # heuristics
    # upsample to original
    return mask

def run(image_path, out_path):
    image_path = Path(image_path)
    img = Image.open(image_path).convert('RGB')
    feats = extract_patch_embeddings(img)
    mask_small = create_mask_by_kmeans(feats)
    mask_pil = Image.fromarray((mask_small*255).astype('uint8')).resize(img.size, Image.NEAREST)
    mask_pil.save(out_path)
    print("Saved unsupervised mask to", out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('out')
    args = parser.parse_args()
    run(args.image, args.out)
