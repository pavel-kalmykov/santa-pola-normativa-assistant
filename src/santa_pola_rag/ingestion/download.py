import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import boto3
import requests
from botocore.client import Config
from botocore.exceptions import ClientError

from santa_pola_rag.config import settings
from santa_pola_rag.ingestion.scraper import REQUEST_HEADERS, DocumentLink

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_s3_client():
    client = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint_url,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
    )
    _ensure_bucket(client)
    return client


def _ensure_bucket(client) -> None:
    try:
        client.head_bucket(Bucket=settings.minio_bucket)
    except ClientError:
        client.create_bucket(Bucket=settings.minio_bucket)


def _object_key(document: DocumentLink) -> str:
    url_hash = hashlib.sha1(document.url.encode("utf-8")).hexdigest()[:10]
    filename = f"{url_hash}_{Path(document.url).name}"
    return f"{document.category_slug}/{filename}"


def download_pdf(document: DocumentLink) -> bytes:
    """Fetch a PDF's bytes, using the MinIO bucket as a cache so re-runs
    never re-download (or, combined with Postgres staging, re-OCR) a PDF
    that has already been processed."""
    client = get_s3_client()
    key = _object_key(document)

    try:
        response = client.get_object(Bucket=settings.minio_bucket, Key=key)
        logger.debug(
            "Cache hit for %s (s3://%s/%s)", document.url, settings.minio_bucket, key
        )
        return response["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise

    response = requests.get(document.url, headers=REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    pdf_bytes = response.content

    client.put_object(
        Bucket=settings.minio_bucket,
        Key=key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )
    logger.info("Downloaded %s -> s3://%s/%s", document.url, settings.minio_bucket, key)
    return pdf_bytes
