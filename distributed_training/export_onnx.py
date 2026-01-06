# distributed_training/export_onnx.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4  →  ONNX  →  Triton / KServe ModelMesh
#
# Exports the fine-tuned PaddleOCR detection and recognition models to ONNX
# format so they can be served by NVIDIA Triton Inference Server.
#
# Usage:
#   python export_onnx.py --det   # export detection model
#   python export_onnx.py --rec   # export recognition model
#   python export_onnx.py --all   # export both

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "model_repo")

DET_INFERENCE_DIR  = os.path.join(BASE_DIR, "output", "det_inference")
DET_ONNX_OUT       = os.path.join(OUTPUT_DIR, "vn_doc_ocr_det", "1", "model.onnx")

REC_INFERENCE_DIR  = os.path.join(BASE_DIR, "output", "rec_inference")
REC_ONNX_OUT       = os.path.join(OUTPUT_DIR, "vn_doc_ocr_rec", "1", "model.onnx")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 – Export Paddle SavedModel → Inference Model
# ---------------------------------------------------------------------------

def export_paddle_inference(model_type: str) -> None:
    """
    Call `paddleocr.tools.export_model` to convert training checkpoint
    to Paddle inference format (model.pdmodel + model.pdiparams).
    """
    if model_type == "det":
        config_file = os.path.join(BASE_DIR, "configs", "det", "mc_ocr_det.yml")
        checkpoint  = os.path.join(BASE_DIR, "output", "det_ppocr_v4", "best_accuracy")
        save_dir    = DET_INFERENCE_DIR
    else:
        config_file = os.path.join(BASE_DIR, "configs", "rec", "mc_ocr_rec.yml")
        checkpoint  = os.path.join(BASE_DIR, "output", "rec_ppocr_v4", "best_accuracy")
        save_dir    = REC_INFERENCE_DIR

    cmd = [
        "python", "-m", "paddleocr.tools.export_model",
        "-c", config_file,
        "-o", f"Global.checkpoints={checkpoint}",
        f"Global.save_inference_dir={save_dir}",
    ]
    print(f"[export] Exporting {model_type} model → {save_dir}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step 2 – Convert Paddle Inference → ONNX via paddle2onnx
# ---------------------------------------------------------------------------

def convert_to_onnx(model_type: str) -> None:
    if model_type == "det":
        inference_dir = DET_INFERENCE_DIR
        onnx_out      = DET_ONNX_OUT
        input_names   = "x"
        output_names  = "sigmoid_0.tmp_0"
    else:
        inference_dir = REC_INFERENCE_DIR
        onnx_out      = REC_ONNX_OUT
        input_names   = "x"
        output_names  = "softmax_0.tmp_0"

    _ensure_dir(onnx_out)

    cmd = [
        "paddle2onnx",
        "--model_dir",         inference_dir,
        "--model_filename",    "model.pdmodel",
        "--params_filename",   "model.pdiparams",
        "--save_file",         onnx_out,
        "--opset_version",     "11",
        "--input_shape_dict",  '{"x": [-1, 3, -1, -1]}',
    ]
    print(f"[onnx] Converting {model_type} → {onnx_out}")
    subprocess.run(cmd, check=True)
    print(f"[onnx] ✓  {onnx_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def export(model_type: str) -> None:
    export_paddle_inference(model_type)
    convert_to_onnx(model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PP-OCRv4 to ONNX for Triton")
    parser.add_argument("--det", action="store_true", help="Export detection model")
    parser.add_argument("--rec", action="store_true", help="Export recognition model")
    parser.add_argument("--all", action="store_true", help="Export both models")
    args = parser.parse_args()

    if args.all or args.det:
        export("det")
    if args.all or args.rec:
        export("rec")
    if not any([args.det, args.rec, args.all]):
        parser.print_help()
