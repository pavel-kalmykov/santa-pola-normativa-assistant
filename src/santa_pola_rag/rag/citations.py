import re

_FOOTNOTE_LINE_RE = re.compile(
    r"^\[(?P<num>\d+)\]\s*(?P<title>.+),\s*p\.\s*(?P<page>\d+),\s*(?P<url>\S+)\s*$",
    re.MULTILINE,
)
_INLINE_MARKER_RE = re.compile(r"\[(\d+)\]")
# The model isn't asked to produce HTML, but was observed doing so anyway
# once (wrapping the whole citation list in a stray <p>...</p>), which broke
# _FOOTNOTE_LINE_RE's start-of-line anchor for the first entry and glued the
# closing tag onto the last URL. Stripped before any other parsing so this
# class of unrequested markup can't corrupt citation numbering again.
_STRAY_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


_FOOTNOTES_HEADING = {
    "es": "Fuentes",
    "en": "Sources",
    "fr": "Sources",
    "de": "Quellen",
    "ca": "Fonts",
    "it": "Fonti",
    "pt": "Fontes",
    "nl": "Bronnen",
}


def has_footnote_list(answer: str) -> bool:
    """Whether the answer contains at least one parseable "[n] Title, p.
    <page>, <url>" citation line, the single source of truth for what
    counts as "cited" (observability/query_log.py's citation-rate metric
    checks this instead of duplicating the format)."""
    return _FOOTNOTE_LINE_RE.search(answer) is not None


def render_footnotes(answer: str, message_id: int, language_code: str | None) -> str:
    """Turn the model's plain-text citation list ("[1] Title, p. 4, url" one
    per line at the end of the answer, "[1]" inline elsewhere) into an HTML
    footnote block with real bidirectional links: clicking an inline marker
    jumps to its footnote, and back. Numbers are recomputed from the actual
    (title, page, url) tuples rather than trusted from the model, so a
    source the model accidentally cited under two different numbers still
    collapses into a single footnote. message_id namespaces the anchor ids
    so multiple chat messages on the same page never collide.

    Returns the answer unchanged if it contains no parseable footnote list,
    or if the list is present but never actually referenced inline (nothing
    safe to link, so nothing is rewritten).
    """
    answer = _STRAY_HTML_TAG_RE.sub("", answer)
    footnote_lines = list(_FOOTNOTE_LINE_RE.finditer(answer))
    if not footnote_lines:
        return answer

    raw_sources = {
        m.group("num"): (m.group("title").strip(), m.group("page"), m.group("url"))
        for m in footnote_lines
    }
    body = _FOOTNOTE_LINE_RE.sub("", answer).rstrip()

    canonical_by_source: dict[tuple[str, str, str], int] = {}
    occurrences_by_canonical: dict[int, int] = {}

    def replace_marker(match: re.Match) -> str:
        source = raw_sources.get(match.group(1))
        if source is None:
            return match.group(0)
        canonical = canonical_by_source.setdefault(source, len(canonical_by_source) + 1)
        # Every inline mention needs its own anchor id (duplicate ids are
        # invalid HTML and make the browser's back-navigation ambiguous), so
        # a source cited three times gets three distinct ref ids; the
        # footnote's single backlink always targets the first one.
        occurrence = occurrences_by_canonical.setdefault(canonical, 0) + 1
        occurrences_by_canonical[canonical] = occurrence
        return (
            f'<sup><a href="#sp-fn-{message_id}-{canonical}" '
            f'id="sp-ref-{message_id}-{canonical}-{occurrence}">[{canonical}]</a></sup>'
        )

    linked_body = _INLINE_MARKER_RE.sub(replace_marker, body)
    if not canonical_by_source:
        return answer

    heading = _FOOTNOTES_HEADING.get((language_code or "en").lower(), "Sources")
    items = "\n".join(
        f'<li id="sp-fn-{message_id}-{n}">{title}, p. {page} '
        f'&mdash; <a href="{url}">{url}</a> '
        f'<a href="#sp-ref-{message_id}-{n}-1">&#8617;</a></li>'
        for (title, page, url), n in sorted(
            canonical_by_source.items(), key=lambda kv: kv[1]
        )
    )
    return f"{linked_body}\n\n---\n**{heading}:**\n<ol>\n{items}\n</ol>"
