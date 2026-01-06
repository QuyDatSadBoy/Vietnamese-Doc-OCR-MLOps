# api/upload_model_to_minio.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Upload the ONNX OCR model repository to MinIO (S3-compatible storage)
# so that KServe / ModelMesh can pull and serve it.

import os
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

MODEL_REPO_PATH = "./model_repo/vn_doc_ocr/"
BUCKET_NAME = "modelmesh-models"
OBJECT_PREFIX = "ocr/vn_doc_ocr"


def upload_directory(client: Minio, local_dir: str, bucket: str, prefix: str) -> None:
    """Recursively upload all files under *local_dir* to MinIO."""
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            relative = os.path.relpath(local_path, local_dir)
            object_name = f"{prefix}/{relative}".replace(os.sep, "/")
            client.fput_object(bucket, object_name, local_path)
            print(f"  ✓  {local_path}  →  s3://{bucket}/{object_name}")


def main() -> None:
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )

    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created.")
    else:
        print(f"Bucket '{BUCKET_NAME}' already exists.")

    upload_directory(client, MODEL_REPO_PATH, BUCKET_NAME, OBJECT_PREFIX)
    print(f"\nModel repository successfully uploaded to s3://{BUCKET_NAME}/{OBJECT_PREFIX}")


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print(f"MinIO error: {exc}")
        raise
