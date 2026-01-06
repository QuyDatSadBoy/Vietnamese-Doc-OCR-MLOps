#!/bin/bash
# Author: Trần Quý Đạt | tranquydat.work@gmail.com
# Build and push the OCR training Docker image to DockerHub.

set -e

IMAGE=tranquydat/vn-doc-ocr-training:latest
VERSION_TAG=tranquydat/vn-doc-ocr-training:0.1.0

docker build -t "$IMAGE" -t "$VERSION_TAG" .
docker push "$IMAGE"
docker push "$VERSION_TAG"

echo "Image pushed: $IMAGE"
