# data_pipeline/operators/load_image.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Airflow operator: load raw images from the MC-OCR 2021 dataset source
# into the working directory for downstream processing.

import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def load_images(source_dir: str, dest_dir: str) -> int:
    """
    Copy all supported image files from *source_dir* to *dest_dir*.

    Parameters
    ----------
    source_dir : str
        Root directory of the MC-OCR 2021 dataset.
    dest_dir : str
        Destination directory for raw images.

    Returns
    -------
    int
        Number of images copied.
    """
    src = Path(source_dir)
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    count = 0
    for img_path in src.rglob("*"):
        if img_path.suffix.lower() in SUPPORTED_EXTS:
            rel = img_path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, target)
            count += 1

    logger.info("Loaded %d images from %s → %s", count, src, dst)
    return count
