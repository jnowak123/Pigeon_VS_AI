import io
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GEOMETRIC_OPS = {
    "original": lambda img: img,
    "rot90": lambda img: img.rotate(-90, expand=True),
    "rot180": lambda img: img.rotate(-180),
    "rot270": lambda img: img.rotate(-270, expand=True),
    "flip_h": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    "flip_v": lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
}


class BreakHisDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        magnification: int | None = None,
        input_size: int = 224,
        blur_and_compress: bool = False,
    ):
        df = manifest.copy()
        if magnification is not None:
            df = df[df["magnification"] == magnification]
        self.records = df.reset_index(drop=True)
        self.blur_and_compress = blur_and_compress

        self.resize_and_normalize = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")

        aug_fn = GEOMETRIC_OPS.get(row["augmentation"], GEOMETRIC_OPS["original"])
        img = aug_fn(img)

        if self.blur_and_compress and row["augmentation"] != "original":
            img = self._apply_blur_compress(img)

        img = self.resize_and_normalize(img)
        label = 1 if row["label"] == "malignant" else 0
        return img, label

    @staticmethod
    def _apply_blur_compress(img):
        if random.random() < 0.3:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        if random.random() < 0.3:
            quality = random.randint(30, 70)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img
