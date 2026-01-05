<div align="center">

# 🇻🇳 Vietnamese Document OCR — End-to-End MLOps Pipeline

<p align="center">
  <img src="images/PipelineAllcode.png" alt="MLOps Pipeline Architecture" width="950"/>
</p>

<p align="center">
  <a href="https://github.com/PaddlePaddle/PaddleOCR"><img src="https://img.shields.io/badge/PaddleOCR-3.x%20%2B%20VL-orange?logo=paddlepaddle" alt="PaddleOCR"/></a>
  <a href="https://kserve.github.io/website/"><img src="https://img.shields.io/badge/Serving-KServe%20ModelMesh-blue" alt="KServe"/></a>
  <a href="https://airflow.apache.org/"><img src="https://img.shields.io/badge/Pipeline-Apache%20Airflow-017CEE?logo=apacheairflow" alt="Airflow"/></a>
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/Tracking-MLflow-0194E2?logo=mlflow" alt="MLflow"/></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

<p align="center">
  <b>Tác giả: Trần Quý Đạt</b> &nbsp;·&nbsp;
  <a href="mailto:tranquydat.work@gmail.com">tranquydat.work@gmail.com</a>
</p>

</div>

---

## Tổng quan

Dự án xây dựng một **MLOps pipeline hoàn chỉnh cấp production** cho bài toán Nhận dạng Ký tự Quang học (OCR) trên tài liệu hành chính tiếng Việt. Pipeline bao gồm toàn bộ vòng đời từ thu thập dữ liệu thô, xử lý stream thời gian thực, huấn luyện phân tán, đến triển khai mô hình với khả năng mở rộng tự động trên Kubernetes.

| | |
|---|---|
| **Dataset** | [MC-OCR 2021](https://aihub.vn/competitions/1) — Vietnamese administrative document OCR dataset |
| **Mô hình** | [PaddleOCR 3.x + PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) — State-of-the-art OCR với Vision-Language capabilities, nhận dạng dấu tiếng Việt và document parsing tốt nhất thực tế, **lựa chọn số 1 cho production** |

---

## Kiến trúc Pipeline

Pipeline được chia thành **6 giai đoạn** hoạt động liên tiếp:

```
MC-OCR 2021
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1 · DATA PIPELINE (Apache Airflow)                   │
│  Load Images → Preprocess (deskew/pad) → HOG Features       │
│                                        → PostgreSQL          │
└─────────────────────────────┬───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                                       ▼
┌─────────────────────┐              ┌────────────────────────┐
│  STAGE 2 · STREAM   │              │  STAGE 3 · FEATURE     │
│  Kafka + Flink      │  ──sync──►   │  STORE (Feast)         │
│  Producer → Topic   │              │  PostgreSQL (offline)  │
│  → Redis + Postgres │              │  Redis (online)        │
└─────────────────────┘              └────────────┬───────────┘
                                                  │
                                                  ▼
                              ┌───────────────────────────────┐
                              │  STAGE 4 · TRAINING (Kubeflow)│
                              │  Feast pull features          │
                              │  PaddleJob (2× GPU workers)   │
                              │  PaddleOCR 3.x fine-tuning    │
                              │  → MLflow Model Registry      │
                              └────────────────┬──────────────┘
                                               │
                                               ▼
                              ┌───────────────────────────────┐
                              │  STAGE 5 · CI/CD (Jenkins)    │
                              │  paddle2onnx export           │
                              │  pytest (25+ test cases)      │
                              │  Upload ONNX → MinIO (S3)     │
                              │  kubectl apply KServe         │
                              └────────────────┬──────────────┘
                                               │
                                               ▼
                              ┌───────────────────────────────┐
                              │  STAGE 6 · SERVING            │
                              │  KServe ModelMesh             │
                              │  NVIDIA Triton (ONNX Runtime) │
                              │  det → crop → rec → CTC decode│
                              └───────────────────────────────┘
```

---

## Technology Stack

| Thành phần | Công nghệ | Ghi chú |
|---|---|---|
| **OCR Model** | PaddleOCR 3.x + PaddleOCR-VL | PP-OCRv5 · Vision-Language · tốt nhất cho tiếng Việt |
| **Deep Learning** | PaddlePaddle 3.0 | Distributed training with Fleet |
| **Data Pipeline** | Apache Airflow 2.x | Batch ingest · preprocess · feature extraction |
| **Stream Processing** | Apache Kafka + Flink | Confluent 7.5 · real-time document ingestion |
| **Feature Store** | Feast 0.40 | PostgreSQL offline · Redis online |
| **Experiment Tracking** | MLflow 2.14 | Model registry · artifact tracking |
| **Distributed Training** | Kubeflow PaddleJob | 2 GPU workers · ReadWriteMany PVC |
| **CI/CD** | Jenkins LTS | Build → Test → ONNX export → Deploy |
| **Model Storage** | MinIO (S3-compatible) | `s3://modelmesh-models/ocr/` |
| **ONNX Export** | paddle2onnx 1.3 | Det + Rec → ONNX opset 11 |
| **Model Serving** | NVIDIA Triton 23.09 | ONNX Runtime · dynamic batching |
| **Serving Orch.** | KServe ModelMesh 0.11+ | HPA auto-scaling (2–3 replicas · 75% CPU) |
| **Infrastructure** | Kubernetes 1.27+ | Multi-pod · Model Mesh pattern |
| **Testing** | pytest 8.x | 25+ test cases · preprocessing/model/streaming |

---

## Cấu trúc thư mục

```
Vietnamese-Doc-OCR-MLOps/
│
├── constants.py                        # Bảng ký tự tiếng Việt (212 chars) · CTC blank · image dims
├── requirements.txt                    # Python dependencies (PaddleOCR 3.x stack)
├── docker-compose.yml                  # Local dev: MLflow + training container + Jenkins
├── Jenkinsfile                         # CI/CD: build → paddle2onnx → pytest → MinIO → KServe
├── .env.example                        # Template biến môi trường
├── mlops-pipeline.excalidraw           # Sơ đồ kiến trúc (editable)
│
├── data_pipeline/                      ← STAGE 1: Apache Airflow
│   ├── dags/
│   │   └── ocr_data_pipeline.py        # DAG hàng ngày: load → preprocess → feature extract
│   └── operators/
│       ├── load_image.py               # Copy ảnh MC-OCR 2021 vào thư mục xử lý
│       ├── preprocess_image.py         # Deskew + resize/pad về 64×256 grayscale
│       └── feature_extraction.py      # HOG features → PostgreSQL (upsert)
│
├── streaming/                          ← STAGE 2: Kafka + Flink
│   ├── produce.py                      # Producer: base64-encode ảnh → topic `ocr-images`
│   ├── flink_processor.py             # Consumer: Redis (online) + PostgreSQL (offline)
│   ├── docker-compose.yml             # Zookeeper · Kafka · Schema Registry · Connect · TimescaleDB
│   ├── Dockerfile                     # Producer container (python:3.8-slim)
│   ├── run.sh                         # Đăng ký Kafka connector
│   └── kafka_connector/
│       └── connect-timescaledb-sink.json
│
├── feature_store/                      ← STAGE 3: Feast
│   ├── feature_store.yaml             # Offline: PostgreSQL · Online: Redis
│   └── features/
│       └── ocr_features.py            # Entity, FeatureView, HOG schema (TTL 30 ngày)
│
├── distributed_training/               ← STAGE 4: Kubeflow PaddleJob
│   ├── mwt.py                          # Entry point: --prepare · --train · --evaluate
│   ├── export_onnx.py                  # paddle2onnx: det + rec PaddleInfer → ONNX
│   ├── Dockerfile                      # paddlepaddle/paddle:3.0.0-gpu base image
│   ├── build.sh                        # Build & push Docker image
│   ├── configs/
│   │   ├── vi_dict.txt                 # Từ điển ký tự tiếng Việt (212 ký tự)
│   │   ├── det/mc_ocr_det.yml          # DBNet++ · PPLCNetV3 · cosine LR · 500 epochs
│   │   └── rec/mc_ocr_rec.yml          # SVTR-LCNet · MobileNetV1 · CTC · 100 epochs
│   ├── nets/
│   │   └── nn.py                       # CTC greedy decode · PaddleOCR engine builder
│   └── utils/
│       ├── config.py                   # Hyperparams · paths · constants
│       ├── dataset.py                  # MC-OCR 2021 COCO format → PaddleOCR label file
│       ├── image_utils.py             # Det/rec preprocessing · augmentation
│       └── label_utils.py             # CTC encode/decode · Vietnamese char mapping
│
├── api/                                ← STAGE 5/6: Inference & Upload
│   ├── triton_client.py               # Full OCR: det → crop → rec → CTC decode (HTTP)
│   └── upload_model_to_minio.py       # Push model_repo/ lên MinIO S3
│
├── deployments/                        ← STAGE 6: Kubernetes manifests
│   ├── triton-servingruntime.yaml     # ServingRuntime: Triton 23.09 · ONNX · HPA
│   ├── triton-isvc.yaml               # InferenceService: vn_doc_ocr_det + vn_doc_ocr_rec
│   └── mwt.yaml                       # Kubeflow PaddleJob (2 workers · GPU · PVC 50Gi)
│
├── model_repo/                         # Triton model repository
│   ├── vn_doc_ocr_det/
│   │   ├── config.pbtxt               # ONNX backend · dynamic input (3, -1, -1)
│   │   └── 1/                         # ← model.onnx sau khi export
│   └── vn_doc_ocr_rec/
│       ├── config.pbtxt               # ONNX backend · fixed (3, 48, 320) · batch 8
│       └── 1/                         # ← model.onnx sau khi export
│
├── mlflow/
│   └── Dockerfile                     # MLflow tracking server
├── notebooks/
│   └── debug.ipynb                    # Dataset exploration · generator prototype
└── tests/
    ├── conftest.py                     # Fixtures: sample images · logits
    ├── test_preprocessing.py          # Rec/det shape · dtype · normalization · padding
    ├── test_model.py                  # CTC decode · blank collapse · valid chars
    ├── test_triton_client.py          # Triton client · preprocess · decode · FileNotFound
    └── test_streaming.py             # Flink message · Redis TTL · Kafka producer format
```

---

## Chi tiết Pipeline

### Stage 1 · Data Pipeline (Apache Airflow)

DAG `ocr_data_pipeline` chạy hàng ngày với 3 task tuần tự:

```
MC-OCR 2021 Source
    │
    ▼ [load_images]
    Sao chép .jpg/.jpeg/.png → raw directory
    │
    ▼ [preprocess_images]  
    Deskew (minimum-area rect) + resize/pad → 64×256 grayscale PNG
    │
    ▼ [extract_features]
    HOG features per image → upsert bảng ocr_image_features (PostgreSQL)
```

### Stage 2 · Stream Processing (Kafka + Flink)

Mô phỏng real-time document ingestion cho online serving:

```
produce.py
    │ base64-encode image → JSON message
    ▼
Kafka topic: ocr-images
    │
    ▼
flink_processor.py
    ├── Redis: key=ocr:image:{id}  →  online feature lookup  (TTL 24h)
    └── PostgreSQL: ocr_stream_events  →  offline analytics
```

### Stage 3 · Feature Store (Feast)

| Store | Backend | Mục đích |
|---|---|---|
| Offline | PostgreSQL | Batch retrieval cho training |
| Online | Redis | Low-latency lookup khi serving |

FeatureView `ocr_image_features`: HOG vector · image dimensions · label text · TTL 30 ngày.

### Stage 4 · Distributed Training (Kubeflow + MLflow)

```
Feast → pull training features
    │
    ▼ mwt.py --prepare
    MC-OCR 2021 COCO JSON → PaddleOCR label format (train_list.txt / val_list.txt)
    │
    ▼ kubectl apply -f deployments/mwt.yaml
    Kubeflow PaddleJob: 2 worker × 1 GPU
    ├── Detection: DBNet++ · PPLCNetV3 backbone · 500 epochs · cosine LR
    └── Recognition: SVTR-LCNet · MobileNetV1 · CTC loss · 100 epochs
    │
    ▼ MLflow tracking
    Params · metrics · checkpoints → Model Registry
```

### Stage 5 · CI/CD Pipeline (Jenkins)

Triggered tự động khi có code push:

```
[Build Training Image] → push tranquydat/vn-doc-ocr-training:latest
        │
        ▼
[Model Optimization - Export ONNX]
    python distributed_training/export_onnx.py --all
    ├── Det: PaddleInfer → ONNX (opset 11) → model_repo/vn_doc_ocr_det/1/model.onnx
    └── Rec: PaddleInfer → ONNX (opset 11) → model_repo/vn_doc_ocr_rec/1/model.onnx
        │
        ▼
[Model Testing]
    pytest tests/ -v --tb=short  (25+ test cases)
        │
        ▼
[Upload Model to MinIO]
    python api/upload_model_to_minio.py
    → s3://modelmesh-models/ocr/vn_doc_ocr_det/
    → s3://modelmesh-models/ocr/vn_doc_ocr_rec/
        │
        ▼
[Deploy to KServe]
    kubectl apply -f deployments/triton-servingruntime.yaml
    kubectl apply -f deployments/triton-isvc.yaml
```

Email alert khi fail → `tranquydat.work@gmail.com`

### Stage 6 · Model Serving (KServe ModelMesh + Triton)

```
User request (image file)
    │
    ▼ api/triton_client.py
    preprocess_for_det(image, short_side=736)
    │
    ▼ Triton HTTP: vn_doc_ocr_det
    DBNet++ → probability map
    │
    ▼ decode_prob_map() → text box coordinates
    Crop ROI images
    │
    ▼ Triton HTTP: vn_doc_ocr_rec  (dynamic batch [1,4,8])
    SVTR-LCNet → character logits
    │
    ▼ ctc_greedy_decode() → Vietnamese text
    Return: [{"box": (x1,y1,x2,y2), "text": "..."}]
```

KServe ModelMesh tự động scale: HPA · 2–3 replicas · ngưỡng CPU 75%.

---

## Điều kiện cài đặt

| Công cụ | Version |
|---|---|
| Python | ≥ 3.9 |
| Docker + Docker Compose | ≥ 24.x |
| Kubernetes | ≥ 1.27 (với KServe + Kubeflow đã cài) |
| NVIDIA GPU (cho training) | CUDA 12.3 · cuDNN 9.0 |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/tranquydat/vn-doc-ocr-mlops.git
cd vn-doc-ocr-mlops
pip install -r requirements.txt
```

### 2. Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env: MINIO_ENDPOINT, POSTGRES_HOST, REDIS_URL, MLFLOW_TRACKING_URI
```

### 3. Khởi động services (local)

```bash
# MLflow + Jenkins
docker compose up -d

# Kafka + Flink streaming stack
cd streaming && docker compose up -d
```

### 4. Chạy Data Pipeline (Airflow)

```bash
# Cài đặt Airflow variables trước:
# mc_ocr_source_dir, mc_ocr_raw_dir, mc_ocr_processed_dir
airflow dags trigger ocr_data_pipeline
```

### 5. Khởi tạo Feature Store

```bash
feast -c feature_store apply
feast -c feature_store materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
```

### 6. Distributed Training (Kubeflow)

```bash
# Chuẩn bị dữ liệu MC-OCR 2021
python distributed_training/mwt.py --prepare

# Triển khai Kubeflow PaddleJob (2 GPU workers)
kubectl apply -f deployments/mwt.yaml

# Theo dõi MLflow
open http://localhost:5000
```

### 7. Export ONNX & Upload MinIO

```bash
python distributed_training/export_onnx.py --all
python api/upload_model_to_minio.py
```

### 8. Deploy lên KServe

```bash
kubectl apply -f deployments/triton-servingruntime.yaml
kubectl apply -f deployments/triton-isvc.yaml
```

### 9. Chạy inference

```python
from api.triton_client import ocr

results = ocr("path/to/vietnamese_document.jpg")
for r in results:
    print(r["text"], "→", r["box"])
```

---

## Tests

```bash
pytest tests/ -v --tb=short
```

| Module | Nội dung |
|---|---|
| `test_preprocessing.py` | Rec/det preprocess shape · dtype · normalization · padding edge cases |
| `test_model.py` | CTC greedy decode · blank collapse · valid Vietnamese chars |
| `test_triton_client.py` | Triton client preprocess · decode · FileNotFoundError |
| `test_streaming.py` | Flink message processing · Redis TTL · Kafka producer format |

---

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `TRITON_URL` | `http://localhost:8000` | Triton HTTP endpoint |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `OFFLINE_STORE_URL` | `postgresql://feast:feast@localhost:5432/feast` | PostgreSQL DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka bootstrap servers |
| `KAFKA_TOPIC` | `ocr-images` | Kafka topic name |

---

## Tác giả

**Trần Quý Đạt**  
[tranquydat.work@gmail.com](mailto:tranquydat.work@gmail.com)

---

<div align="center">
  <sub>Built with ❤️ — PaddleOCR 3.x · KServe · Apache Airflow · Feast · MLflow · Jenkins · Kubernetes</sub>
</div>
# Vietnamese-Doc-OCR-MLOps
