# feature_store/features/ocr_features.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Feast feature definitions for the Vietnamese OCR pipeline.
# Features are ingested from PostgreSQL (offline store) and synced
# to Redis (online store) via `feast materialize`.

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String, Int32, Array


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
image_entity = Entity(
    name="image_id",
    description="Unique identifier for a document image in the MC-OCR 2021 dataset.",
)

# ---------------------------------------------------------------------------
# Offline source (PostgreSQL table exported as Parquet for Feast ingestion)
# ---------------------------------------------------------------------------
image_source = FileSource(
    path="./feature_store/data/ocr_image_features.parquet",
    timestamp_field="created_at",
)

# ---------------------------------------------------------------------------
# Feature view
# ---------------------------------------------------------------------------
ocr_image_features = FeatureView(
    name="ocr_image_features",
    entities=[image_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="hog_features", dtype=Array(Float32)),
        Field(name="image_width", dtype=Int32),
        Field(name="image_height", dtype=Int32),
        Field(name="label_text", dtype=String),
    ],
    source=image_source,
    tags={"pipeline": "ocr", "dataset": "mc-ocr-2021"},
)
