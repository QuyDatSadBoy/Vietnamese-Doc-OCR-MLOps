# distributed_training/utils/dataset.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4 fine-tuned on MC-OCR 2021
#
# Converts MC-OCR 2021 annotations to PaddleOCR SimpleDataSet format:
#   <image_path>\t<label_text>

import os
import json
from pathlib import Path

from utils import config
from constants import SUPPORTED_IMAGE_EXTS


# ---------------------------------------------------------------------------
# MC-OCR 2021 dataset converter
# ---------------------------------------------------------------------------

def build_paddleocr_label_file(
    annotation_dir: str,
    image_root: str,
    output_file: str,
    split: str = "train",
) -> int:
    """
    Parse MC-OCR 2021 COCO-style JSON annotations and write a PaddleOCR
    SimpleDataSet label file.

    MC-OCR 2021 annotation structure (COCO-like):
      annotations[i]:
        image_id  : int
        bbox      : [x, y, w, h]
        text      : str          <- ground-truth OCR text

    Output line format (PaddleOCR SimpleDataSet):
      <relative/path/to/crop.jpg>\t<label_text>

    Parameters
    ----------
    annotation_dir : str  Folder containing train_annotations.json / val_annotations.json
    image_root     : str  Root of the cropped word/line images
    output_file    : str  Destination .txt file
    split          : str  "train" or "val"

    Returns
    -------
    int  Number of samples written
    """
    ann_path = os.path.join(annotation_dir, f"{split}_annotations.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build id → filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}

    lines = []
    for ann in coco.get("annotations", []):
        img_id  = ann["image_id"]
        text    = ann.get("text", "").strip()
        if not text:
            continue
        img_name = id_to_file.get(img_id, "")
        if not img_name:
            continue
        img_path = os.path.join(image_root, img_name)
        if not any(img_path.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTS):
            continue
        lines.append(f"{img_path}\t{text}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return len(lines)


def verify_label_file(label_file: str, max_text_length: int = config.max_text_length) -> dict:
    """
    Validate a PaddleOCR label file.

    Returns a summary dict:
      total, empty_labels, too_long, valid
    """
    stats = {"total": 0, "empty_labels": 0, "too_long": 0, "valid": 0}
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            stats["total"] += 1
            parts = line.split("\t", 1)
            if len(parts) < 2 or not parts[1]:
                stats["empty_labels"] += 1
                continue
            if len(parts[1]) > max_text_length:
                stats["too_long"] += 1
                continue
            stats["valid"] += 1
    return stats
