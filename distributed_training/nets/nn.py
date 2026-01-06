# distributed_training/nets/nn.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4
#
# PaddleOCR model utilities:
#  - Build PaddleOCR inference engine (det + rec)
#  - CTC greedy decode using the Vietnamese character dictionary

import os
import sys
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from constants import IDX_TO_CHAR, VIETNAMESE_CHARS


# ---------------------------------------------------------------------------
# CTC greedy decoder  (mirrors PaddleOCR CTCLabelDecode)
# ---------------------------------------------------------------------------

def ctc_greedy_decode(logits: np.ndarray) -> str:
    """
    CTC greedy decode for PP-OCRv4 recognition output.

    Parameters
    ----------
    logits : np.ndarray  shape (time_steps, num_classes)
        Softmax output from the recognition model.
        Class 0 is the CTC blank token.

    Returns
    -------
    str  Decoded Vietnamese text.
    """
    best = np.argmax(logits, axis=-1)        # (T,)
    merged = []
    prev = -1
    for tok in best:
        if tok != prev:
            merged.append(int(tok))
        prev = tok
    return "".join(IDX_TO_CHAR[t] for t in merged if t != 0)


# ---------------------------------------------------------------------------
# PaddleOCR inference pipeline (uses paddleocr Python API if available)
# ---------------------------------------------------------------------------

def build_paddleocr_engine(
    det_model_dir: str = None,
    rec_model_dir: str = None,
    char_dict_path: str = None,
    use_gpu: bool = False,
):
    """
    Build a PaddleOCR inference engine for the fine-tuned PP-OCRv4 model.

    Parameters
    ----------
    det_model_dir  : path to exported Paddle detection inference model dir
    rec_model_dir  : path to exported Paddle recognition inference model dir
    char_dict_path : path to vi_dict.txt
    use_gpu        : whether to use GPU for inference

    Returns
    -------
    PaddleOCR instance or None if paddleocr is not installed.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return None

    kwargs = dict(
        lang="vi",
        use_gpu=use_gpu,
        show_log=False,
    )
    if det_model_dir:
        kwargs["det_model_dir"] = det_model_dir
    if rec_model_dir:
        kwargs["rec_model_dir"] = rec_model_dir
    if char_dict_path:
        kwargs["rec_char_dict_path"] = char_dict_path

    return PaddleOCR(**kwargs)
