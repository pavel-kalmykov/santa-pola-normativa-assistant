import argparse
import logging
import os
from dataclasses import asdict

import dlt
import psycopg2
from dlt.destinations import postgres

from santa_pola_rag.config import settings
from santa_pola_rag.ingestion.download import download_pdf
from santa_pola_rag.ingestion.extract import extract_pages, render_page_png
from santa_pola_rag.ingestion.ocr import ocr_page
from santa_pola_rag.ingestion.scraper import (
    CATEGORIES,
    DocumentLink,
    discover_documents,
)

# dlt's default extract concurrency (5 workers) deterministically hung the
# pipeline mid-run (near-zero CPU, no progress) whenever two large scanned
# PDFs happened to be OCR'd at the same time via OpenRouter; the same page
# OCR'd in isolation succeeds in seconds. Root cause not fully isolated
# (client/connection-pool or provider-side concurrency limit), so ingestion
# is kept single-threaded rather than shipping a pipeline that can silently
# stall for hours on paid API calls.
os.environ.setdefault("EXTRACT__WORKERS", "1")

logger = logging.getLogger(__name__)


def _existing_pages(document_url: str) -> dict[int, dict]:
    """Pages already staged for this document, keyed by page number. OCR is
    a paid API call: re-running the pipeline (e.g. after losing the local
    PDF cache) must not blindly re-OCR pages already sitting in Postgres."""
    try:
        conn = psycopg2.connect(settings.postgres_dsn)
    except psycopg2.OperationalError:
        return {}
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "select page_number, page_count, text, source, char_count "
                    "from santa_pola_raw.pages where document_url = %s",
                    (document_url,),
                )
            except psycopg2.errors.UndefinedTable:
                return {}
            return {
                page_number: {
                    "page_count": page_count,
                    "text": text,
                    "source": source,
                    "char_count": char_count,
                }
                for page_number, page_count, text, source, char_count in cur.fetchall()
            }
    finally:
        conn.close()


@dlt.resource(name="documents", write_disposition="merge", primary_key="url")
def documents_resource(category_slugs: list[str]):
    for category_slug in category_slugs:
        for document in discover_documents(category_slug):
            yield asdict(document)


@dlt.transformer(
    data_from=documents_resource,
    name="pages",
    write_disposition="merge",
    primary_key=["document_url", "page_number"],
)
def pages_resource(document: dict, force: bool = False):
    existing = {} if force else _existing_pages(document["url"])
    pdf_bytes = download_pdf(DocumentLink(**document))
    pages = extract_pages(pdf_bytes, label=document["title"])

    for page in pages:
        cached = existing.get(page.page_number)
        if cached is not None:
            logger.debug(
                "Reusing staged text for page %d/%d of '%s' (source=%s)",
                page.page_number,
                page.page_count,
                document["title"],
                cached["source"],
            )
            yield {
                "document_url": document["url"],
                "category_slug": document["category_slug"],
                "page_number": page.page_number,
                **cached,
            }
            continue

        if page.needs_ocr:
            try:
                png = render_page_png(pdf_bytes, page.page_number)
                text = ocr_page(
                    png,
                    document["title"],
                    document["category_name"],
                    page.page_number,
                    page.page_count,
                )
                source = "vision_ocr"
            except Exception:
                # A single unresponsive OCR call must not block/lose an
                # entire category's worth of already-paid ingestion work.
                logger.exception(
                    "OCR failed for page %d/%d of '%s', skipping",
                    page.page_number,
                    page.page_count,
                    document["title"],
                )
                text = page.text
                source = "ocr_failed"
        else:
            text = page.text
            source = "pdf_text"

        yield {
            "document_url": document["url"],
            "category_slug": document["category_slug"],
            "page_number": page.page_number,
            "page_count": page.page_count,
            "text": text,
            "source": source,
            "char_count": len(text),
        }


def build_pipeline() -> dlt.Pipeline:
    return dlt.pipeline(
        pipeline_name="santa_pola_ingestion",
        destination=postgres(credentials=settings.postgres_dsn),
        dataset_name="santa_pola_raw",
    )


def run(
    category_slugs: list[str] | None = None, force: bool = False
) -> list[dlt.common.pipeline.LoadInfo]:
    """Run ingestion one category at a time so each category's (paid) OCR work
    is committed to Postgres as soon as it finishes, instead of losing an
    entire multi-hour run to a single late failure.

    By default, pages already staged in Postgres are reused instead of being
    re-OCR'd, so re-running after e.g. losing the local PDF cache doesn't
    silently re-pay for OCR. Pass force=True to always re-process every page.
    """
    category_slugs = category_slugs or list(CATEGORIES)
    unknown = set(category_slugs) - set(CATEGORIES)
    if unknown:
        raise ValueError(
            f"Unknown categories: {sorted(unknown)}. Known: {sorted(CATEGORIES)}"
        )

    pipeline = build_pipeline()
    load_infos = []
    for category_slug in category_slugs:
        documents = documents_resource([category_slug])
        load_info = pipeline.run([documents, documents | pages_resource(force=force)])
        logger.info("Ingestion run finished for '%s': %s", category_slug, load_info)
        load_infos.append(load_info)
    return load_infos


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Santa Pola ordinances ingestion")
    parser.add_argument(
        "--categories",
        help=(
            "Comma-separated category slugs to ingest "
            f"(default: all of {', '.join(CATEGORIES)})"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download, re-extract and re-OCR every page even if already staged",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    category_slugs = args.categories.split(",") if args.categories else None
    for info in run(category_slugs=category_slugs, force=args.force):
        print(info)
