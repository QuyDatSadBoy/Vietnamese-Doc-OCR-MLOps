# tests/test_triton_client.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline

import base64, json, os
import numpy as np
import pytest
import cv2


class TestRecPreprocess:
    def test_output_shape(self, tmp_path, sample_bgr_image):
        img_path = str(tmp_path / "doc.jpg")
        cv2.imwrite(img_path, sample_bgr_image)
        from api.triton_client import preprocess_for_rec
        img = cv2.imread(img_path)
        result = preprocess_for_rec(img)
        assert result.shape == (1, 3, 48, 320)

    def test_normalised_range(self, tmp_path, sample_bgr_image):
        img_path = str(tmp_path / "doc.jpg")
        cv2.imwrite(img_path, sample_bgr_image)
        from api.triton_client import preprocess_for_rec
        img = cv2.imread(img_path)
        result = preprocess_for_rec(img)
        assert result.min() >= -1.05
        assert result.max() <=  1.05

    def test_file_not_found(self):
        from api.triton_client import ocr
        with pytest.raises(FileNotFoundError):
            ocr("/nonexistent/path.jpg")


class TestCTCDecode:
    def test_blank_only_empty(self):
        from api.triton_client import ctc_greedy_decode
        from constants import NUM_CLASSES
        logits = np.zeros((64, NUM_CLASSES), dtype=np.float32)
        logits[:, 0] = 1.0
        assert ctc_greedy_decode(logits) == ""

    def test_returns_string(self, sample_logits):
        from api.triton_client import ctc_greedy_decode
        assert isinstance(ctc_greedy_decode(sample_logits), str)

    def test_valid_chars_only(self, sample_logits):
        from api.triton_client import ctc_greedy_decode
        from constants import VIETNAMESE_CHARS
        valid = set(VIETNAMESE_CHARS)
        for ch in ctc_greedy_decode(sample_logits):
            assert ch in valid


class TestDetPreprocess:
    def test_output_divisible_32(self, sample_bgr_image):
        from api.triton_client import preprocess_for_det
        tensor, rh, rw = preprocess_for_det(sample_bgr_image)
        assert tensor.shape[2] % 32 == 0
        assert tensor.shape[3] % 32 == 0

    def test_scale_positive(self, sample_bgr_image):
        from api.triton_client import preprocess_for_det
        _, rh, rw = preprocess_for_det(sample_bgr_image)
        assert rh > 0 and rw > 0
