import logging

from santa_pola_rag.config import PROJECT_ROOT, settings
from santa_pola_rag.ingestion.download import get_s3_client

logger = logging.getLogger(__name__)

RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw"


def run() -> int:
    client = get_s3_client()
    n_uploaded = 0

    for pdf_path in sorted(RAW_PDF_DIR.glob("*/*.pdf")):
        key = f"{pdf_path.parent.name}/{pdf_path.name}"
        client.put_object(
            Bucket=settings.minio_bucket,
            Key=key,
            Body=pdf_path.read_bytes(),
            ContentType="application/pdf",
        )
        n_uploaded += 1
        logger.info("Uploaded %s -> s3://%s/%s", pdf_path, settings.minio_bucket, key)

    return n_uploaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = run()
    print(
        f"Uploaded {n} PDFs from {RAW_PDF_DIR} to MinIO bucket '{settings.minio_bucket}'"
    )
