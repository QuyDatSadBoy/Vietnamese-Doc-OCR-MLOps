# streaming/flink_processor.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Apache Flink stream processor:
#   Kafka (ocr-images) → decode image → extract metadata → Redis (online) + PostgreSQL (offline)

import base64
import json
import os
import time
import logging
from typing import Any

import redis
import psycopg2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ocr-images")
KAFKA_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "flink-ocr-processor")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_DSN = os.getenv(
    "OFFLINE_STORE_URL",
    "postgresql://feast:feast@localhost:5432/feast",
)

# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def get_pg_conn():
    return psycopg2.connect(POSTGRES_DSN)


def ensure_pg_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_stream_events (
                image_id   BIGINT PRIMARY KEY,
                filename   TEXT,
                byte_size  INT,
                received_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
    conn.commit()


def write_to_redis(r: redis.Redis, event: dict[str, Any]) -> None:
    """Cache event metadata in Redis (online store) with 24 h TTL."""
    key = f"ocr:image:{event['image_id']}"
    r.hset(key, mapping={
        "filename": event["filename"],
        "byte_size": str(event.get("byte_size", 0)),
    })
    r.expire(key, 86400)


def write_to_postgres(conn, event: dict[str, Any]) -> None:
    """Persist event metadata to PostgreSQL (offline store)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ocr_stream_events (image_id, filename, byte_size)
            VALUES (%s, %s, %s)
            ON CONFLICT (image_id) DO NOTHING
            """,
            (event["image_id"], event["filename"], event.get("byte_size", 0)),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Processor (pure-Python simulation of Flink's streaming job)
# ---------------------------------------------------------------------------

def process_message(raw: bytes, r: redis.Redis, pg_conn) -> None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Malformed message: %s", exc)
        return

    image_data = base64.b64decode(payload.get("image_data", ""))
    event = {
        "image_id": payload.get("image_id"),
        "filename": payload.get("filename", "unknown"),
        "byte_size": len(image_data),
    }

    write_to_redis(r, event)
    write_to_postgres(pg_conn, event)
    logger.info(
        "Processed image_id=%s  file=%s  size=%d bytes",
        event["image_id"], event["filename"], event["byte_size"],
    )


def run() -> None:
    """Start the Flink-style stream processor (Kafka consumer loop)."""
    from kafka import KafkaConsumer

    r = get_redis_client()
    pg_conn = get_pg_conn()
    ensure_pg_table(pg_conn)

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=KAFKA_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    logger.info("Flink processor listening on topic '%s' ...", KAFKA_TOPIC)

    for msg in consumer:
        process_message(msg.value, r, pg_conn)


if __name__ == "__main__":
    run()
