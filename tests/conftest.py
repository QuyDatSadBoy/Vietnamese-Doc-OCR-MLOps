# tests/conftest.py
# Author : Trần Quý Đạt | tranquydat.work@gmail.com
import sys, os
import numpy as np
import pytest

_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.join(_ROOT, "distributed_training")
for p in [_ROOT, _TRAINING]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """Random BGR uint8 image (200, 800, 3) — simulates a document crop."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(200, 800, 3), dtype=np.uint8)


@pytest.fixture
def sample_logits() -> np.ndarray:
    """Fake softmax logits (64, NUM_CLASSES) for CTC decode tests."""
    from constants import NUM_CLASSES
    rng = np.random.default_rng(42)
    raw = rng.random((64, NUM_CLASSES)).astype(np.float32)
    exp = np.exp(raw - raw.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)
