import base64
import logging

from openai import OpenAI

from santa_pola_rag.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VISION_MODEL = "google/gemini-2.5-flash"

# A real page never needs more than a few thousand characters of transcription.
# Without a cap, a degenerate repetition loop on a problematic scan drove the
# model to generate 1M+ characters in one response, taking minutes and making
# the whole ingestion run look hung. max_tokens bounds generation length up
# front; MAX_DESCRIPTION_CHARS is a defense-in-depth truncation of whatever
# comes back.
MAX_OUTPUT_TOKENS = 2048
MAX_DESCRIPTION_CHARS = 8000

SYSTEM_PROMPT = (
    "You are a precise OCR and document-description assistant working on scanned "
    "pages from Spanish municipal ordinances and public notices. Transcribe all "
    "readable text verbatim, preserving numbers, dates and legal references "
    "exactly. If the page is a diagram, map or table rather than plain text, "
    "describe its structure and content factually instead of inventing details "
    "you cannot read clearly. Respond in Spanish, the language of the source "
    "document."
)

# The OpenAI SDK has no default timeout: a stalled OpenRouter connection would
# otherwise hang the ingestion pipeline indefinitely with no data committed.
# Kept short and with a single retry (rather than the SDK's default backoff
# schedule) so one unresponsive page fails fast instead of blocking the whole
# ingestion run for minutes.
_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=settings.openrouter_api_key,
    timeout=30.0,
    max_retries=1,
)


def ocr_page(
    image_png: bytes,
    document_title: str,
    category_name: str,
    page_number: int,
    page_count: int,
) -> str:
    """Describe/transcribe a scanned page using a vision LLM, with document context."""
    image_b64 = base64.b64encode(image_png).decode("ascii")
    context = (
        f"Document: '{document_title}' (category: {category_name}). "
        f"This is page {page_number} of {page_count}."
    )

    response = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    description = response.choices[0].message.content or ""
    if len(description) > MAX_DESCRIPTION_CHARS:
        logger.warning(
            "OCR output for page %d/%d of '%s' was %d chars, truncating to %d "
            "(likely a degenerate/repetitive generation)",
            page_number,
            page_count,
            document_title,
            len(description),
            MAX_DESCRIPTION_CHARS,
        )
        description = description[:MAX_DESCRIPTION_CHARS]
    logger.info(
        "OCR'd page %d/%d of '%s' -> %d chars",
        page_number,
        page_count,
        document_title,
        len(description),
    )
    return description
