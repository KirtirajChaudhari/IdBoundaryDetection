"""
Utility helpers: mask <-> polygon, overlay drawing, warp (projective) using scikit-image.
"""
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure, transform as sktf
from shapely.geometry import Polygon
from shapely.ops import unary_union

def save_mask(mask_arr, path):
    # mask_arr: 2D array (bool, 0/1 or 0..255)
    arr = (mask_arr.astype(np.uint8) * 255) if mask_arr.max() <= 1 else mask_arr.astype(np.uint8)
    Image.fromarray(arr).save(path)

from shapely.geometry import Polygon
from shapely.ops import unary_union
from skimage import measure
import numpy as np

def mask_to_polygon(mask):
    """
    Convert a binary mask (H,W) to a single dominant shapely Polygon.

    - Finds contours with skimage.measure.find_contours
    - Builds polygons, filters tiny/invalid ones
    - Merges them with unary_union
    - If result is MultiPolygon, picks the largest-by-area component
    """
    mask = mask.astype("uint8")
    contours = measure.find_contours(mask, 0.5)

    polys = []
    for c in contours:
        # c is (N, 2) array in (row, col) format => (y, x)
        coords = [(float(pt[1]), float(pt[0])) for pt in c]
        try:
            p = Polygon(coords)
            if p.is_valid and p.area > 10:
                polys.append(p)
        except Exception:
            continue

    if not polys:
        return None

    merged = unary_union(polys)

    # Shapely 2.x: MultiPolygon is not directly iterable; use .geoms
    if merged.geom_type == "MultiPolygon":
        merged = max(list(merged.geoms), key=lambda a: a.area)

    return merged

def polygon_to_minrect_corners(poly):
    if poly is None:
        return None
    rect = poly.minimum_rotated_rectangle
    pts = list(rect.exterior.coords)[:-1]
    return pts

def draw_overlay(image_pil, corners, out_path):
    draw = ImageDraw.Draw(image_pil)
    if corners:
        draw.line(corners + [corners[0]], width=6, fill=(255,0,0))
        for (x,y) in corners:
            r = 6
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(0,255,0))
    image_pil.save(out_path)

def warp_to_rect(img_np, src_quad, dst_size=(600,400)):
    # src_quad: list of 4 (x,y) in image coords
    src = np.array(src_quad[:4])
    dst = np.array([(0,0),(dst_size[0]-1,0),(dst_size[0]-1,dst_size[1]-1),(0,dst_size[1]-1)])
    tform = sktf.ProjectiveTransform()
    if not tform.estimate(src, dst):
        return None
    warped = sktf.warp(img_np, tform, output_shape=(dst_size[1], dst_size[0]))
    return (warped*255).astype('uint8')
