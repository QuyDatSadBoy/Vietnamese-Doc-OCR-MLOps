# api/triton_client.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4 served via NVIDIA Triton + KServe ModelMesh
#
# Full OCR pipeline:
#   1. Preprocess document image
#   2. Call vn_doc_ocr_det  → text bounding boxes
#   3. Crop each box
#   4. Call vn_doc_ocr_rec  → recognised text per box
#   5. CTC greedy decode

import os, sys, json
import numpy as np
import cv2
import requests

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from constants import IDX_TO_CHAR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRITON_URL   = os.getenv("TRITON_URL",       "http://localhost:8000")
DET_MODEL    = os.getenv("DET_MODEL_NAME",   "vn_doc_ocr_det")
REC_MODEL    = os.getenv("REC_MODEL_NAME",   "vn_doc_ocr_rec")
MODEL_VER    = os.getenv("TRITON_MODEL_VER", "1")

REC_H, REC_W = 48, 320   # PP-OCRv4 recognition input size


# ---------------------------------------------------------------------------
# Helper: Triton HTTP v2 inference
# ---------------------------------------------------------------------------

def _triton_infer(model_name: str, input_name: str, data: np.ndarray, output_name: str) -> np.ndarray:
    payload = {
        "inputs": [{
            "name":     input_name,
            "shape":    list(data.shape),
            "datatype": "FP32",
            "data":     data.flatten().tolist(),
        }],
        "outputs": [{"name": output_name}],
    }
    url = f"{TRITON_URL}/v2/models/{model_name}/versions/{MODEL_VER}/infer"
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    out = resp.json()["outputs"][0]
    return np.array(out["data"], dtype=np.float32).reshape(out["shape"])


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_for_det(image: np.ndarray, short_side: int = 736) -> tuple:
    """Resize image to fixed short side (multiple of 32), normalise ImageNet."""
    h, w = image.shape[:2]
    scale = short_side / min(h, w)
    new_h = max(32, int(round(h * scale / 32) * 32))
    new_w = max(32, int(round(w * scale / 32) * 32))
    resized = cv2.resize(image, (new_w, new_h))
    img_f = resized[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_f = (img_f - mean) / std
    tensor = img_f.transpose(2, 0, 1)[np.newaxis, ...]   # (1, 3, H, W)
    return tensor, new_h / h, new_w / w


def preprocess_for_rec(crop: np.ndarray) -> np.ndarray:
    """Resize crop to (48, 320), normalise to [-1, 1], return (1, 3, 48, 320)."""
    h, w = crop.shape[:2]
    new_w = max(1, int(w * REC_H / h))
    resized = cv2.resize(crop, (min(new_w, REC_W), REC_H))
    pad = np.full((REC_H, REC_W, 3), 127, dtype=np.uint8)
    pad[:, :resized.shape[1]] = resized
    img_f = pad[:, :, ::-1].astype(np.float32)   # BGR → RGB
    img_f = (img_f / 255.0 - 0.5) / 0.5
    return img_f.transpose(2, 0, 1)[np.newaxis, ...]   # (1, 3, 48, 320)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def decode_prob_map(prob_map: np.ndarray, orig_h: int, orig_w: int,
                    thresh: float = 0.3, box_thresh: float = 0.6,
                    unclip_ratio: float = 1.5) -> list:
    """
    Convert DB probability map to bounding boxes.
    Returns list of (x1, y1, x2, y2) in original image coordinates.
    """
    bitmap = (prob_map[0, 0] > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    scale_h = orig_h / bitmap.shape[0]
    scale_w = orig_w / bitmap.shape[1]
    for cnt in contours:
        if cv2.contourArea(cnt) < 16:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Unclip
        dx = int(w * (unclip_ratio - 1) / 2)
        dy = int(h * (unclip_ratio - 1) / 2)
        x1 = max(0, int((x - dx) * scale_w))
        y1 = max(0, int((y - dy) * scale_h))
        x2 = min(orig_w, int((x + w + dx) * scale_w))
        y2 = min(orig_h, int((y + h + dy) * scale_h))
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def ctc_greedy_decode(logits: np.ndarray) -> str:
    """CTC greedy decode. logits shape: (time_steps, vocab_size)."""
    best = np.argmax(logits, axis=-1)
    merged = []
    prev = -1
    for tok in best:
        if tok != prev:
            merged.append(int(tok))
        prev = tok
    return "".join(IDX_TO_CHAR[t] for t in merged if t != 0)


# ---------------------------------------------------------------------------
# Full OCR pipeline
# ---------------------------------------------------------------------------

def ocr(image_path: str) -> list[dict]:
    """
    Run the full PaddleOCR PP-OCRv4 pipeline against Triton.

    Returns
    -------
    list of {"box": (x1,y1,x2,y2), "text": str}
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img.shape[:2]

    # --- Detection ---
    det_input, rh, rw = preprocess_for_det(img)
    prob_map = _triton_infer(DET_MODEL, "x", det_input, "sigmoid_0.tmp_0")
    boxes = decode_prob_map(prob_map, orig_h, orig_w)

    # --- Recognition ---
    results = []
    for (x1, y1, x2, y2) in boxes:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rec_input = preprocess_for_rec(crop)
        logits = _triton_infer(REC_MODEL, "x", rec_input, "softmax_0.tmp_0")
        text = ctc_greedy_decode(logits[0])   # logits shape (1, T, vocab)
        if text:
            results.append({"box": (x1, y1, x2, y2), "text": text})

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PP-OCRv4 inference via Triton")
    parser.add_argument("image", help="Path to document image")
    parser.add_argument("--url", default=TRITON_URL)
    args = parser.parse_args()
    TRITON_URL = args.url
    for item in ocr(args.image):
        print(f"{item['box']}  →  {item['text']}")
