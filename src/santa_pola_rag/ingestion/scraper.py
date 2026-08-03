import logging
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://santapola.es"

# Municipal ordinances, tax ordinances, bylaws and public notices: the "hard
# regulation" categories a resident or business actually needs to cite.
# Listed at https://santapola.es/ayuntamiento/, easy to extend with more
# slugs from that same index later.
CATEGORIES = {
    "ordenanzas-fiscales": "Ordenanzas Fiscales",
    "reglamentos-otras-ordenanzas": "Reglamentos y otras Ordenanzas",
    "normativas": "Normativas",
    "bandos": "Bandos",
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; santa-pola-rag-ingestion/0.1; "
        "+https://github.com/pavel-kalmykov/santa-pola-normativa-assistant)"
    )
}


@dataclass
class DocumentLink:
    category_slug: str
    category_name: str
    title: str
    url: str
    listing_page: str


def _listing_url(category_slug: str) -> str:
    return f"{BASE_URL}/ayuntamiento/{category_slug}/"


def discover_documents(category_slug: str) -> list[DocumentLink]:
    """Scrape a santapola.es/ayuntamiento/<category> listing page for PDF links."""
    category_name = CATEGORIES[category_slug]
    listing_url = _listing_url(category_slug)

    response = requests.get(listing_url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    documents = []
    seen_urls = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not href.lower().split("?")[0].endswith(".pdf"):
            continue

        absolute_url = urljoin(listing_url, href)
        if absolute_url in seen_urls:
            continue
        seen_urls.add(absolute_url)

        title = anchor.get("title") or anchor.get_text(strip=True) or absolute_url
        documents.append(
            DocumentLink(
                category_slug=category_slug,
                category_name=category_name,
                title=title.strip(),
                url=absolute_url,
                listing_page=listing_url,
            )
        )

    logger.info("Discovered %d PDF(s) in category '%s'", len(documents), category_slug)
    return documents


def discover_all_documents() -> list[DocumentLink]:
    all_documents = []
    for category_slug in CATEGORIES:
        all_documents.extend(discover_documents(category_slug))
    return all_documents
