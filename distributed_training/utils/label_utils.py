# distributed_training/utils/label_utils.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Pure-Python label encoding/decoding — no TensorFlow dependency.
# Used by both dataset.py and the test suite.

import numpy as np

from constants import CHAR_TO_IDX, IDX_TO_CHAR
from utils import config


def encode_label(text: str, max_length: int = config.max_text_length) -> np.ndarray:
    """Encode a UTF-8 string to a zero-padded integer array using CHAR_TO_IDX."""
    encoded = np.zeros(max_length, dtype=np.int32)
    for i, char in enumerate(text[:max_length]):
        encoded[i] = CHAR_TO_IDX.get(char, 0)
    return encoded


def decode_label(indices: np.ndarray) -> str:
    """Decode an integer array back to a UTF-8 string, stopping at first zero."""
    chars = []
    for idx in indices:
        if idx == 0:
            break
        chars.append(IDX_TO_CHAR.get(int(idx), ""))
    return "".join(chars)
