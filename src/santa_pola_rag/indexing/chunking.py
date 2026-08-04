import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# A tax ordinance's tariff table (labels plus per-category euro amounts) ran
# 1,000-1,300 chars in real cases and, at the old 800/150 setting, regularly
# split with the labels in one chunk and the amounts in the next: the amounts
# chunk alone reads as bare digits with almost no lexical/semantic signal, so
# neither BM25 nor the embedding ever surfaced it and the agent reported the
# figures as unavailable. Confirmed against a real user question ("how much
# to get my car back from the pound") where the answer chunk never appeared
# even in the top 50 raw candidates from either search channel. 1200/200
# keeps tables like that in a single chunk without materially changing how
# the rest of the (mostly prose) corpus splits.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


@dataclass
class Chunk:
    chunk_id: str
    document_url: str
    category_slug: str
    title: str
    page_number: int
    page_count: int
    source: str
    text: str


def chunk_page(
    document_url: str,
    category_slug: str,
    title: str,
    page_number: int,
    page_count: int,
    source: str,
    text: str,
) -> list[Chunk]:
    """Split a single page's text into overlapping chunks, keeping page-level citation."""
    text = text.strip()
    if not text:
        return []

    pieces = _splitter.split_text(text)
    chunks = []
    for piece_index, piece in enumerate(pieces):
        raw_id = f"{document_url}#{page_number}#{piece_index}"
        chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_url=document_url,
                category_slug=category_slug,
                title=title,
                page_number=page_number,
                page_count=page_count,
                source=source,
                # The document title is prepended to every chunk's indexed/
                # embedded text, not just kept as separate metadata: a
                # mid-page fragment (e.g. one row of a multi-page tariff
                # table) often never repeats the document's own subject on
                # its own, so neither BM25 nor the embedding had any signal
                # tying it back to what it's actually about. Confirmed with
                # a real case: a chunk listing "Peluquerías... 249,15 EUR"
                # never surfaced for natural phrasings of "cuanto cuesta la
                # licencia de apertura de una peluqueria" because nothing in
                # that chunk said "licencia de apertura" at all.
                text=f"{title}\n\n{piece}",
            )
        )
    return chunks
