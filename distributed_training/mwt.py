# distributed_training/mwt.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4 fine-tuned on MC-OCR 2021
#
# Multi-Worker Training entry-point.
# Uses PaddlePaddle Fleet (distributed) under Kubeflow PaddleJob.
# Logs metrics and artefacts to MLflow, then uploads ONNX to MinIO.

import argparse
import os
import subprocess
import sys

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset() -> None:
    """Convert MC-OCR 2021 annotations to PaddleOCR SimpleDataSet format."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from distributed_training.utils.dataset import build_paddleocr_label_file
    from distributed_training.utils import config

    for split in ("train", "val"):
        n = build_paddleocr_label_file(
            annotation_dir=os.path.join(config.data_dir, "annotations"),
            image_root=os.path.join(config.data_dir, config.image_dir),
            output_file=os.path.join(config.data_dir, f"{split}_list.txt"),
            split=split,
        )
        print(f"[prepare] {split}: {n} samples written.")


# ---------------------------------------------------------------------------
# Fine-tune PP-OCRv4 using PaddleOCR's training pipeline
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Launch distributed PP-OCRv4 recognition fine-tuning via
    paddle.distributed.launch (FleetAPI).

    PaddleJob (Kubeflow) sets TF_CONFIG / PADDLE_TRAINERS_NUM automatically.
    """
    config_file = os.path.join(
        os.path.dirname(__file__), "configs", "rec", "mc_ocr_rec.yml"
    )

    cmd = [
        "python", "-m", "paddle.distributed.launch",
        "--log_dir", "./log",
        "-m", "paddleocr.tools.train",
        "-c", config_file,
    ]

    with mlflow.start_run(run_name="ppocr_v4_rec_finetune"):
        mlflow.log_params({
            "model":   "PP-OCRv4-rec",
            "dataset": "MC-OCR 2021",
        })

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            mlflow.set_tag("status", "failed")
            sys.exit(result.returncode)

        output_dir = os.path.join("output", "rec_ppocr_v4")
        if os.path.isdir(output_dir):
            mlflow.log_artifacts(output_dir, artifact_path="checkpoints")

        mlflow.set_tag("status", "success")
        print("[train] Fine-tuning complete. Artefacts logged to MLflow.")


# ---------------------------------------------------------------------------
# Evaluate on MC-OCR 2021 validation set
# ---------------------------------------------------------------------------

def evaluate() -> None:
    config_file = os.path.join(
        os.path.dirname(__file__), "configs", "rec", "mc_ocr_rec.yml"
    )
    cmd = [
        "python", "-m", "paddleocr.tools.eval",
        "-c", config_file,
        "-o", "Global.checkpoints=output/rec_ppocr_v4/best_accuracy",
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR PP-OCRv4 Training")
    parser.add_argument("--prepare", action="store_true", help="Convert MC-OCR 2021 annotations")
    parser.add_argument("--train",   action="store_true", help="Fine-tune PP-OCRv4")
    parser.add_argument("--evaluate",action="store_true", help="Evaluate on val set")
    args = parser.parse_args()

    if args.prepare:
        prepare_dataset()
    if args.train:
        train()
    if args.evaluate:
        evaluate()
    if not any([args.prepare, args.train, args.evaluate]):
        parser.print_help()
