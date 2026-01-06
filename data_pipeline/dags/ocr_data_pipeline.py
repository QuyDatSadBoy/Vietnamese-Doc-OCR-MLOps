# data_pipeline/dags/ocr_data_pipeline.py
# Author : Trần Quý Đạt
# Email  : tranquydat.work@gmail.com
# Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
#
# Apache Airflow DAG: ingest MC-OCR 2021 images → preprocess → extract features
# → persist to PostgreSQL (offline store) and Redis (online store via Feast).

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from data_pipeline.operators.load_image import load_images
from data_pipeline.operators.preprocess_image import preprocess_images
from data_pipeline.operators.feature_extraction import extract_and_store_features

# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "tranquydat",
    "email": ["tranquydat.work@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="ocr_data_pipeline",
    description="Ingest, preprocess, and extract features from MC-OCR 2021 dataset",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["ocr", "data-pipeline", "mlops"],
) as dag:

    t_load = PythonOperator(
        task_id="load_images",
        python_callable=load_images,
        op_kwargs={
            "source_dir": "{{ var.value.mc_ocr_source_dir }}",
            "dest_dir": "{{ var.value.mc_ocr_raw_dir }}",
        },
    )

    t_preprocess = PythonOperator(
        task_id="preprocess_images",
        python_callable=preprocess_images,
        op_kwargs={
            "input_dir": "{{ var.value.mc_ocr_raw_dir }}",
            "output_dir": "{{ var.value.mc_ocr_processed_dir }}",
        },
    )

    t_extract = PythonOperator(
        task_id="extract_features",
        python_callable=extract_and_store_features,
        op_kwargs={
            "processed_dir": "{{ var.value.mc_ocr_processed_dir }}",
        },
    )

    t_load >> t_preprocess >> t_extract
