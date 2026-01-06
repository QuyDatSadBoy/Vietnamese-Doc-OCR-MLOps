# distributed_training/Dockerfile
# Author : Trần Quý Đạt | tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model — MLOps Pipeline
# Model  : PaddleOCR 3.x + PaddleOCR-VL fine-tuning container (MC-OCR 2021)

FROM paddlepaddle/paddle:3.0.0-gpu-cuda12.3-cudnn9.0-trt8.6

LABEL maintainer="tranquydat.work@gmail.com"
LABEL description="PaddleOCR 3.x + PaddleOCR-VL distributed fine-tuning container (MC-OCR 2021)"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PaddleOCR 3.x and utilities
RUN pip install --no-cache-dir \
        paddleocr==3.0.1 \
        paddle2onnx==1.3.0 \
        mlflow==2.14.3 \
        tqdm==4.66.4 \
        python-dotenv==1.0.1 \
        opencv-python-headless==4.9.0.80

# Copy project
COPY . /workspace
WORKDIR /workspace

CMD ["python", "mwt.py", "--train"]
