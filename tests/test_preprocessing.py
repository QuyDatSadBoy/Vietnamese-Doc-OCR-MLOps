# tests/test_preprocessing.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4

import numpy as np
import pytest


class TestRecPreprocess:
    """PP-OCRv4 recognition preprocessing tests."""

    def test_output_shape(self, sample_bgr_image):
        from utils.image_utils import preprocess_for_rec
        result = preprocess_for_rec(sample_bgr_image)
        assert result.shape == (1, 3, 48, 320), f"Got {result.shape}"

    def test_dtype_float32(self, sample_bgr_image):
        from utils.image_utils import preprocess_for_rec
        result = preprocess_for_rec(sample_bgr_image)
        assert result.dtype == np.float32

    def test_normalised_range(self, sample_bgr_image):
        from utils.image_utils import preprocess_for_rec
        result = preprocess_for_rec(sample_bgr_image)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <=  1.0 + 1e-5

    def test_narrow_crop_padded(self):
        from utils.image_utils import preprocess_for_rec
        narrow = np.zeros((200, 30, 3), dtype=np.uint8)
        result = preprocess_for_rec(narrow)
        assert result.shape == (1, 3, 48, 320)

    def test_wide_crop_cropped(self):
        from utils.image_utils import preprocess_for_rec
        wide = np.zeros((48, 1000, 3), dtype=np.uint8)
        result = preprocess_for_rec(wide)
        assert result.shape == (1, 3, 48, 320)


class TestDetPreprocess:
    """PP-OCRv4 detection preprocessing tests."""

    def test_output_has_batch_dim(self, sample_bgr_image):
        from utils.image_utils import preprocess_for_det
        tensor, rh, rw = preprocess_for_det(sample_bgr_image)
        assert tensor.ndim == 4          # (1, 3, H, W)
        assert tensor.shape[1] == 3

    def test_output_divisible_by_32(self, sample_bgr_image):
        from utils.image_utils import preprocess_for_det
        tensor, _, _ = preprocess_for_det(sample_bgr_image)
        _, _, h, w = tensor.shape
        assert h % 32 == 0, f"H={h} not divisible by 32"
        assert w % 32 == 0, f"W={w} not divisible by 32"

    def test_scale_factors_positive(self, sample_bgr_image):
        from utils.image_utils import preprocess_for_det
        _, rh, rw = preprocess_for_det(sample_bgr_image)
        assert rh > 0
        assert rw > 0

    def test_dtype_float32(self, sample_bgr_image):
        from utils.image_utils import preprocess_for_det
        tensor, _, _ = preprocess_for_det(sample_bgr_image)
        assert tensor.dtype == np.float32


class TestLabelUtils:
    def test_encode_decode_roundtrip(self):
        from utils.label_utils import encode_label, decode_label
        text = "xin chào"
        assert decode_label(encode_label(text)) == text

    def test_encode_truncates_to_max(self):
        from utils.label_utils import encode_label
        from utils import config
        long_text = "a" * (config.max_text_length + 10)
        assert len(encode_label(long_text)) == config.max_text_length

    def test_unknown_char_zero(self):
        from utils.label_utils import encode_label
        enc = encode_label("\x00\x01")
        assert all(v == 0 for v in enc)
