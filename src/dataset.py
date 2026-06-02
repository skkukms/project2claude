"""Worker-safe zip-backed image dataset.

The training data is distributed as a zip of equal-resolution PNG/JPG files.
"""
from __future__ import annotations

import io
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ZipImageDataset(Dataset):
    def __init__(
        self,
        zip_path: str | os.PathLike,
        flip: bool = True,
        normalize_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.zip_path = Path(zip_path)
        if not self.zip_path.is_file():
            raise FileNotFoundError(f"Zip not found: {self.zip_path}")
        self.flip = flip
        self.lo, self.hi = normalize_range
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            names = sorted(
                n for n in zf.namelist()
                if n.lower().endswith((".png", ".jpg", ".jpeg"))
            )
        if not names:
            raise RuntimeError(f"No PNG/JPG files found in {self.zip_path}")
        self._names: list[str] = names
        self._zf: zipfile.ZipFile | None = None

    def _ensure_open(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        zf = self._ensure_open()
        with zf.open(self._names[idx], "r") as f:
            buf = f.read()
        img = Image.open(io.BytesIO(buf)).convert("RGB")
        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img, dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = t * (self.hi - self.lo) + self.lo
        return t

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_zf"] = None
        return state


def infinite_loader(loader):
    """Yield batches forever by restarting iteration when exhausted."""
    while True:
        for batch in loader:
            yield batch
