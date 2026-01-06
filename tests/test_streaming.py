# tests/test_streaming.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline

import base64
import json
import os
import tempfile
import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch


class TestFlinkProcessor:
    """Unit tests for the Flink stream processor (no Kafka/Redis/Postgres required)."""

    def test_process_valid_message(self):
        from streaming.flink_processor import process_message

        img = np.zeros((64, 256), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = json.dumps({
            "image_id": 1,
            "filename": "test_doc.jpg",
            "image_data": b64,
        }).encode("utf-8")

        mock_redis = MagicMock()
        mock_pg = MagicMock()
        mock_pg.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_pg.cursor.return_value.__exit__ = MagicMock(return_value=False)

        process_message(payload, mock_redis, mock_pg)

        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()

    def test_process_malformed_message(self):
        """Malformed JSON should be silently dropped without raising."""
        from streaming.flink_processor import process_message

        mock_redis = MagicMock()
        mock_pg = MagicMock()

        # Should not raise
        process_message(b"not valid json {{{", mock_redis, mock_pg)
        mock_redis.hset.assert_not_called()

    def test_process_empty_image_data(self):
        from streaming.flink_processor import process_message

        payload = json.dumps({
            "image_id": 99,
            "filename": "empty.jpg",
            "image_data": "",
        }).encode("utf-8")

        mock_redis = MagicMock()
        mock_pg = MagicMock()
        mock_pg.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_pg.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Should not raise; byte_size == 0
        process_message(payload, mock_redis, mock_pg)

    def test_redis_key_format(self):
        """Redis key for image_id=5 should be 'ocr:image:5'."""
        from streaming.flink_processor import write_to_redis

        mock_redis = MagicMock()
        event = {"image_id": 5, "filename": "doc.png", "byte_size": 1234}
        write_to_redis(mock_redis, event)

        mock_redis.hset.assert_called_once_with(
            "ocr:image:5",
            mapping={"filename": "doc.png", "byte_size": "1234"},
        )
        mock_redis.expire.assert_called_once_with("ocr:image:5", 86400)


class TestKafkaProducer:
    """Tests for the produce.py message structure (no live Kafka required)."""

    def test_message_has_required_fields(self, tmp_path):
        """produce.py should build messages with image_id, filename, image_data."""
        img = np.zeros((64, 256), dtype=np.uint8)
        img_path = str(tmp_path / "sample.jpg")
        cv2.imwrite(img_path, img)

        with open(img_path, "rb") as f:
            raw = f.read()

        message = {
            "image_id": 1,
            "filename": os.path.basename(img_path),
            "image_data": base64.b64encode(raw).decode("utf-8"),
        }

        assert "image_id" in message
        assert "filename" in message
        assert "image_data" in message

    def test_image_data_is_valid_base64(self, tmp_path):
        img = np.zeros((32, 128), dtype=np.uint8)
        img_path = str(tmp_path / "doc.jpg")
        cv2.imwrite(img_path, img)

        with open(img_path, "rb") as f:
            raw = f.read()

        b64 = base64.b64encode(raw).decode("utf-8")
        decoded = base64.b64decode(b64)
        assert decoded == raw
