# streaming/produce.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Kafka producer: streams simulated document images to the `ocr-images` topic
# for downstream Flink processing and Redis/PostgreSQL ingestion.

import argparse
import base64
import json
import os
from time import sleep

from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic

TOPIC_NAME = "ocr-images"

parser = argparse.ArgumentParser(description="Fake stream data producer for OCR pipeline")
parser.add_argument("-m", "--mode", default="setup",
                    choices=["setup", "teardown"],
                    help="Setup or teardown the Kafka topic.")
parser.add_argument("-b", "--bootstrap_servers", default="localhost:9092",
                    help="Kafka bootstrap server address.")
parser.add_argument("-i", "--image_dir", default="./images",
                    help="Directory containing document images to stream.")
parser.add_argument("--interval", type=float, default=1.0,
                    help="Seconds between messages.")
args = parser.parse_args()

_image_id_counter = 1


def create_topic(admin: KafkaAdminClient, topic_name: str) -> None:
    try:
        topic = NewTopic(name=topic_name, num_partitions=1, replication_factor=1)
        admin.create_topics([topic])
        print(f"Topic '{topic_name}' created.")
    except Exception:
        print(f"Topic '{topic_name}' already exists — skipping creation.")


def delete_topic(admin: KafkaAdminClient, topic_name: str) -> None:
    try:
        admin.delete_topics([topic_name])
        print(f"Topic '{topic_name}' deleted.")
    except Exception as exc:
        print(f"Could not delete topic '{topic_name}': {exc}")


def create_streams(servers: str, image_dir: str, interval: float) -> None:
    global _image_id_counter

    producer = admin = None
    for attempt in range(10):
        try:
            producer = KafkaProducer(bootstrap_servers=servers)
            admin = KafkaAdminClient(bootstrap_servers=servers)
            print("Kafka producer ready.")
            break
        except Exception as exc:
            print(f"[Attempt {attempt + 1}/10] Cannot connect to Kafka: {exc}")
            sleep(10)

    if producer is None:
        raise RuntimeError("Failed to connect to Kafka after 10 attempts.")

    create_topic(admin, TOPIC_NAME)

    image_files = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Streaming {len(image_files)} document images to topic '{TOPIC_NAME}'...")
    idx = 0
    while True:
        img_path = image_files[idx % len(image_files)]
        idx += 1

        with open(img_path, "rb") as f:
            raw = f.read()

        message = {
            "image_id": _image_id_counter,
            "filename": os.path.basename(img_path),
            # Base64-encode binary so JSON serialisation works
            "image_data": base64.b64encode(raw).decode("utf-8"),
        }
        _image_id_counter += 1

        producer.send(TOPIC_NAME, value=json.dumps(message).encode("utf-8"))
        print(f"  Sent image_id={message['image_id']}  file={message['filename']}")
        sleep(interval)


if __name__ == "__main__":
    if args.mode == "setup":
        create_streams(args.bootstrap_servers, args.image_dir, args.interval)
    elif args.mode == "teardown":
        admin = KafkaAdminClient(bootstrap_servers=args.bootstrap_servers)
        delete_topic(admin, TOPIC_NAME)
