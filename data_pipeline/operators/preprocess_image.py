# data_pipeline/operators/preprocess_image.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Airflow operator: resize, normalise, and deskew document images.

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_HEIGHT = 64
TARGET_WIDTH = 256


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew using the image moments method."""
    coords = np.column_stack(np.where(image < 128))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _resize_pad(image: np.ndarray) -> np.ndarray:
    """Resize to TARGET_HEIGHT keeping aspect ratio, pad width to TARGET_WIDTH."""
    h, w = image.shape
    scale = TARGET_HEIGHT / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, TARGET_HEIGHT))
    if new_w >= TARGET_WIDTH:
        return resized[:, :TARGET_WIDTH]
    padded = np.full((TARGET_HEIGHT, TARGET_WIDTH), 255, dtype=np.uint8)
    padded[:, :new_w] = resized
    return padded


def preprocess_images(input_dir: str, output_dir: str) -> int:
    """
    Preprocess all images in *input_dir* and save to *output_dir*.

    Steps: grayscale → deskew → resize-and-pad → save as PNG.

    Returns
    -------
    int
        Number of images processed.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    count = 0
    for img_file in in_path.rglob("*"):
        if img_file.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Cannot read %s — skipping", img_file)
            continue
        img = _deskew(img)
        img = _resize_pad(img)

        rel = img_file.relative_to(in_path)
        target = out_path / rel.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), img)
        count += 1

    logger.info("Preprocessed %d images → %s", count, out_path)
    return count
