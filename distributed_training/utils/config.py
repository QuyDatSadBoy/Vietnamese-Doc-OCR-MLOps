# distributed_training/utils/config.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4 fine-tuned on MC-OCR 2021

import os, sys, posixpath

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from constants import NUM_CLASSES, REC_IMAGE_HEIGHT, REC_IMAGE_WIDTH  # noqa

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
data_dir        = posixpath.join(".", "Dataset")
image_dir       = "images"
label_dir       = "labels"
train_list      = posixpath.join(data_dir, "train_list.txt")   # PaddleOCR format
val_list        = posixpath.join(data_dir, "val_list.txt")
char_dict_path  = posixpath.join(".", "distributed_training", "configs", "vi_dict.txt")

# ---------------------------------------------------------------------------
# PaddleOCR PP-OCRv4 hyper-parameters
# ---------------------------------------------------------------------------
num_epochs: int     = 100
batch_size: int     = 64
learning_rate: float = 0.001
max_text_length: int = 25      # PP-OCRv4 default max text length

# ---------------------------------------------------------------------------
# Recognition model input (PP-OCRv4 rec)
# ---------------------------------------------------------------------------
rec_image_height: int    = REC_IMAGE_HEIGHT   # 48
rec_image_width:  int    = REC_IMAGE_WIDTH    # 320
rec_image_channels: int  = 3                  # RGB
num_classes: int         = NUM_CLASSES

# ---------------------------------------------------------------------------
# Detection model input
# ---------------------------------------------------------------------------
det_short_side: int = 736    # resize short side to this before inference

# ---------------------------------------------------------------------------
# Pretrained checkpoints (PP-OCRv4 Vietnamese)
# ---------------------------------------------------------------------------
PRETRAINED_DET = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/vietnamese/"
    "det_PP-OCRv4_viet_server_train.tar"
)
PRETRAINED_REC = (
    "https://paddleocr.bj.bcebos.com/PP-OCRv4/vietnamese/"
    "rec_PPOCRv4_viet_mobile_train.tar"
)

# ---------------------------------------------------------------------------
# Output / versioning
# ---------------------------------------------------------------------------
output_dir: str  = "./output"
version: str     = "v1"
