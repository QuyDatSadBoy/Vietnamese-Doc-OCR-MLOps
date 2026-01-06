# tests/test_model.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
# Model  : PaddleOCR PP-OCRv4

import numpy as np
import pytest


class TestCTCGreedyDecode:
    """Tests for the CTC greedy decoder (nets/nn.py)."""

    def test_blank_only_returns_empty(self):
        from nets.nn import ctc_greedy_decode
        from constants import NUM_CLASSES
        logits = np.zeros((64, NUM_CLASSES), dtype=np.float32)
        logits[:, 0] = 1.0   # all blank
        assert ctc_greedy_decode(logits) == ""

    def test_repeated_char_collapsed(self):
        """CTC collapses repeated same tokens → single character."""
        from nets.nn import ctc_greedy_decode
        from constants import NUM_CLASSES, CHAR_TO_IDX
        logits = np.zeros((10, NUM_CLASSES), dtype=np.float32)
        a_idx = CHAR_TO_IDX.get("a", 1)
        logits[:, a_idx] = 1.0
        assert ctc_greedy_decode(logits) == "a"

    def test_returns_string(self, sample_logits):
        from nets.nn import ctc_greedy_decode
        assert isinstance(ctc_greedy_decode(sample_logits), str)

    def test_output_only_valid_chars(self, sample_logits):
        from nets.nn import ctc_greedy_decode
        from constants import VIETNAMESE_CHARS
        valid = set(VIETNAMESE_CHARS)
        result = ctc_greedy_decode(sample_logits)
        for ch in result:
            assert ch in valid, f"Unexpected char: {repr(ch)}"

    def test_sequence_with_blanks(self):
        """Blank tokens between same chars should allow repetition."""
        from nets.nn import ctc_greedy_decode
        from constants import NUM_CLASSES, CHAR_TO_IDX
        logits = np.zeros((6, NUM_CLASSES), dtype=np.float32)
        a = CHAR_TO_IDX.get("a", 1)
        # a, blank, a  → "aa"
        logits[0, a] = 1.0
        logits[1, 0] = 1.0   # blank
        logits[2, a] = 1.0
        logits[3, 0] = 1.0
        logits[4, a] = 1.0
        logits[5, 0] = 1.0
        result = ctc_greedy_decode(logits)
        assert result == "aaa"


class TestPaddleOCREngine:
    """Smoke test for build_paddleocr_engine (returns None if not installed)."""

    def test_engine_returns_none_or_paddleocr(self):
        from nets.nn import build_paddleocr_engine
        engine = build_paddleocr_engine(use_gpu=False)
        # Must return None (paddleocr not installed in test env) or a PaddleOCR obj
        assert engine is None or hasattr(engine, "ocr")
