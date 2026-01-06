# data_pipeline/operators/feature_extraction.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Airflow operator: extract pixel-level and HOG features from preprocessed
# document images and persist them to the PostgreSQL offline store via Feast.

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import psycopg2
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_URL = os.getenv(
    "OFFLINE_STORE_URL",
    "postgresql://feast:feast@localhost:5432/feast",
)


def _hog_features(image: np.ndarray) -> np.ndarray:
    """Compute a compact HOG descriptor (128-d) from a grayscale image."""
    win_size = (image.shape[1], image.shape[0])
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    descriptor = hog.compute(image)
    return descriptor.flatten()


def _insert_features(
    conn, image_id: str, hog: np.ndarray, timestamp: datetime
) -> None:
    """Insert one image's feature vector into PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ocr_image_features (image_id, hog_features, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (image_id) DO UPDATE
              SET hog_features = EXCLUDED.hog_features,
                  created_at   = EXCLUDED.created_at
            """,
            (image_id, hog.tolist(), timestamp),
        )
    conn.commit()


def extract_and_store_features(processed_dir: str) -> int:
    """
    Compute HOG features for every image in *processed_dir* and store to PostgreSQL.

    Returns
    -------
    int
        Number of images processed.
    """
    proc_path = Path(processed_dir)
    conn = psycopg2.connect(DB_URL)

    # Ensure table exists
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_image_features (
                image_id   TEXT PRIMARY KEY,
                hog_features  FLOAT8[],
                created_at TIMESTAMPTZ NOT NULL
            )
            """
        )
    conn.commit()

    count = 0
    ts = datetime.now(tz=timezone.utc)
    for img_file in proc_path.rglob("*.png"):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        hog = _hog_features(img)
        image_id = img_file.stem
        _insert_features(conn, image_id, hog, ts)
        count += 1

    conn.close()
    logger.info("Stored features for %d images to PostgreSQL", count)
    return count
