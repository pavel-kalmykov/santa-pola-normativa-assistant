import logging
import threading
import unicodedata
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# dlt's transformer runs pages_resource on a worker thread pool, and MuPDF is
# not safe to call concurrently from multiple threads: two large PDFs being
# rendered at once reliably deadlocked (near-zero CPU, no progress) during a
# real ingestion run. Serializing all fitz access avoids it at a small cost
# to parallelism.
_FITZ_LOCK = threading.Lock()

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


def extract_pages(pdf_bytes: bytes, label: str = "") -> list[PageContent]:
    pages = []
    with _FITZ_LOCK, fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            needs_ocr = len(text) < MIN_EXTRACTABLE_CHARS or _is_garbled(text)
            pages.append(
                PageContent(
                    page_number=index + 1,
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
