import io
import logging
import os
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import fitz  # PyMuPDF

# Docling's layout/table-structure model hits a real torch.compile (inductor
# backend) crash on at least one production PDF's specific tensor shapes;
# must be set before torch is imported. Costs some inference speed, not
# worth it for a one-off ingestion run over 2,800-odd pages either way.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.base_models import DocumentStream, InputFormat  # noqa: E402
from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: E402
from docling.document_converter import (  # noqa: E402
    DocumentConverter,
    PdfFormatOption,
)

logger = logging.getLogger(__name__)

# dlt's transformer runs pages_resource on a worker thread pool, and MuPDF is
# not safe to call concurrently from multiple threads: two large PDFs being
# rendered at once reliably deadlocked (near-zero CPU, no progress) during a
# real ingestion run. Serializing all fitz access avoids it at a small cost
# to parallelism.
_FITZ_LOCK = threading.Lock()

# Docling's own thread-safety under a worker pool is unverified, and the
# fitz deadlock above is exactly the kind of failure that's expensive to
# debug after the fact; serializing costs little since a single convert()
# call already covers a whole PDF (not per-page), and is not worth the risk.
_DOCLING_LOCK = threading.Lock()

# Below this many extracted characters, a page is treated as scanned/image-only
# and routed to vision OCR instead of trusting the (likely garbage) text layer.
MIN_EXTRACTABLE_CHARS = 20

# Some PDFs embed subset fonts with a broken/missing ToUnicode CMap: PyMuPDF
# then "extracts" plenty of characters, but they are raw glyph codes (mostly
# ASCII control characters), not real text. A high ratio of control
# characters is the tell that the text layer is unusable and OCR is needed.
MAX_CONTROL_CHAR_RATIO = 0.1


def _is_garbled(text: str) -> bool:
    if not text:
        return True
    control_chars = sum(
        1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\t\r"
    )
    return control_chars / len(text) > MAX_CONTROL_CHAR_RATIO


@dataclass
class PageContent:
    page_number: int  # 1-indexed
    text: str
    needs_ocr: bool
    page_count: int


@lru_cache(maxsize=1)
def _docling_converter() -> DocumentConverter:
    # OCR stays off here: a scanned/image-only page is caught by the same
    # needs_ocr heuristic below (Docling emits little to no text for it,
    # same as PyMuPDF did) and routed to Gemini Vision OCR elsewhere in the
    # pipeline, which reads real handwriting/stamps far better than
    # Docling's bundled rapidocr. This call is only for structured
    # extraction of a page's real text and table layout.
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def extract_pages(pdf_bytes: bytes, label: str = "") -> list[PageContent]:
    stream = DocumentStream(name=f"{label or 'document'}.pdf", stream=io.BytesIO(pdf_bytes))
    with _DOCLING_LOCK:
        document = _docling_converter().convert(stream).document
    page_count = document.num_pages()

    pages = []
    for page_number in range(1, page_count + 1):
        text = document.export_to_markdown(page_no=page_number).strip()
        needs_ocr = len(text) < MIN_EXTRACTABLE_CHARS or _is_garbled(text)
        pages.append(
            PageContent(
                page_number=page_number,
                text=text,
                needs_ocr=needs_ocr,
                page_count=page_count,
            )
        )
    n_ocr = sum(1 for p in pages if p.needs_ocr)
    logger.info("%s: %d page(s), %d need OCR", label, len(pages), n_ocr)
    return pages


def render_page_png(pdf_bytes: bytes, page_number: int, zoom: float = 2.0) -> bytes:
    """Render a 1-indexed page to PNG bytes for vision OCR."""
    with _FITZ_LOCK, fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_number - 1]
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)
        return pixmap.tobytes("png")
