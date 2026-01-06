# distributed_training/utils/image_utils.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4 fine-tuned on MC-OCR 2021

import cv2
import numpy as np

from utils import config


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(file_path: str) -> np.ndarray:
    """Load an image in BGR uint8 (OpenCV default)."""
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {file_path}")
    return img


# ---------------------------------------------------------------------------
# PP-OCRv4 recognition pre-processing
# (mirrors PaddleOCR RecResizeImg transform)
# ---------------------------------------------------------------------------

def preprocess_for_rec(
    image: np.ndarray,
    target_height: int = config.rec_image_height,
    target_width:  int = config.rec_image_width,
) -> np.ndarray:
    """
    Resize and normalise a BGR crop for the PP-OCRv4 recognition model.

    Steps
    -----
    1. Resize to (target_height, variable_width) preserving aspect ratio
    2. Pad or crop width to target_width
    3. BGR → RGB, divide by 255, subtract mean (0.5) / std (0.5)
    4. Return shape (1, 3, H, W) float32  (batch=1, CHW, normalised)
    """
    h, w = image.shape[:2]
    ratio = target_height / h
    new_w = max(1, int(w * ratio))

    resized = cv2.resize(image, (new_w, target_height))

    # Pad right with 127 (grey) if narrower than target_width
    if new_w < target_width:
        padded = np.full((target_height, target_width, 3), 127, dtype=np.uint8)
        padded[:, :new_w, :] = resized
        resized = padded
    else:
        resized = resized[:, :target_width, :]

    # Normalise: (pixel / 255 - 0.5) / 0.5  == pixel / 127.5 - 1
    img_f = resized[:, :, ::-1].astype(np.float32)   # BGR → RGB
    img_f = (img_f / 255.0 - 0.5) / 0.5

    # HWC → CHW → batch
    img_f = img_f.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
    return img_f


# ---------------------------------------------------------------------------
# PP-OCRv4 detection pre-processing
# (mirrors PaddleOCR DetResizeForTest transform)
# ---------------------------------------------------------------------------

def preprocess_for_det(
    image: np.ndarray,
    short_side: int = config.det_short_side,
) -> tuple:
    """
    Resize the input document image so the short side equals `short_side`
    (multiple of 32).  Returns the normalised CHW tensor and the scale factors.

    Returns
    -------
    tensor : np.ndarray  shape (1, 3, H\', W\') float32
    ratio_h, ratio_w : float  scale factors for mapping predictions back
    """
    h, w = image.shape[:2]
    if h < w:
        scale = short_side / h
    else:
        scale = short_side / w

    new_h = max(32, int(round(h * scale / 32) * 32))
    new_w = max(32, int(round(w * scale / 32) * 32))

    resized = cv2.resize(image, (new_w, new_h))

    img_f = resized[:, :, ::-1].astype(np.float32)  # BGR → RGB
    # ImageNet mean/std normalisation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_f = (img_f / 255.0 - mean) / std

    img_f = img_f.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
    return img_f, new_h / h, new_w / w


# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------

def random_brightness(image: np.ndarray, delta: float = 0.2) -> np.ndarray:
    factor = 1.0 + np.random.uniform(-delta, delta)
    return np.clip(image * factor, 0.0, 1.0).astype(np.float32)


def random_noise(image: np.ndarray, stddev: float = 0.02) -> np.ndarray:
    noise = np.random.normal(0, stddev, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)
