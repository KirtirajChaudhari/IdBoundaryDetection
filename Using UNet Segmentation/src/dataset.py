"""
PyTorch Dataset for images + binary masks.
Uses albumentations for transforms and is robust to unreadable/corrupt images.
"""

from pathlib import Path
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.RandomRotate90(),
            A.Flip(),
            A.OneOf([A.GridDistortion(), A.Perspective()], p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


class CardDataset(Dataset):
    def __init__(self, images_dir, masks_dir, filenames=None, transforms=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        if filenames is not None:
            self.filenames = list(filenames)
        else:
            exts = {".jpg", ".jpeg", ".png"}
            self.filenames = sorted(
                [p.name for p in self.images_dir.iterdir() if p.suffix.lower() in exts]
            )

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def _load_pair(self, fn):
        img_path = self.images_dir / fn
        mask_path = self.masks_dir / fn

        # load as numpy for albumentations
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype("uint8")  # binary 0/1

        return img, mask

    def __getitem__(self, idx):
        # Robust loader: if a file is unreadable, skip forward to the next valid one.
        original_idx = idx
        attempts = 0
        while attempts < len(self.filenames):
            fn = self.filenames[idx]
            try:
                img, mask = self._load_pair(fn)
                break
            except (UnidentifiedImageError, OSError) as e:
                # Log once per offending file
                print(f"[WARN] Skipping unreadable image or mask: {fn} ({e})")
                # move to next index (wrap around)
                idx = (idx + 1) % len(self.filenames)
                attempts += 1
        else:
            # If all attempts failed, raise a clear error
            raise RuntimeError("All images in this dataset appear to be unreadable/corrupt.")

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)
            img_t = augmented["image"]                  # torch.Tensor [C,H,W]
            mask_t = augmented["mask"].unsqueeze(0).float()  # [1,H,W]
        else:
            img_t = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            mask_t = torch.tensor(mask).unsqueeze(0).float()

        return img_t, mask_t
