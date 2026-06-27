"""PDF download + text extraction. Generic by design — never raises on remote errors."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Status codes the publisher uses to say "you don't have access here."
# These are expected on paywalled content and are reported as a summary,
# not per-paper warnings.
PAYWALL_STATUSES = {401, 402, 403, 429}


@dataclass
class DownloadResult:
    """Outcome of a single PDF fetch — lets the pipeline summarize at the end."""

    path: Optional[str] = None  # local file path on success
    paywalled: bool = False  # publisher returned 401/402/403/429
    errored: bool = False  # network/timeout/other — actionable
    status: Optional[int] = None  # last seen HTTP status, if any


def download_pdf(
    url: str,
    save_dir: str,
    user_agent: str = "mmore-paper-discovery/1.0",
    timeout: int = 30,
    proxy_prefix: Optional[str] = None,
) -> DownloadResult:
    """Returns a DownloadResult describing what happened.

    1. Optionally wrap URL through an EZproxy prefix.
    2. GET the URL.
    3. If response looks like a PDF, save it.
    4. Otherwise parse the HTML for the first PDF-looking <a href>, follow it.

    The polite default User-Agent identifies this tool honestly. Publishers
    that block automated tools will return 401/402/403/429 — that is their
    call, and we surface it as paywalled, not as an error.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent}
    fetch_url = _proxify(url, proxy_prefix)

    try:
        r = requests.get(
            fetch_url, headers=headers, timeout=timeout, allow_redirects=True
        )
    except requests.RequestException as e:
        logger.debug("download_pdf network error for %s: %s", url, e)
        return DownloadResult(errored=True)

    if r.status_code in PAYWALL_STATUSES:
        return DownloadResult(paywalled=True, status=r.status_code)

    if r.status_code != 200:
        logger.debug("download_pdf got %s for %s", r.status_code, url)
        return DownloadResult(errored=True, status=r.status_code)

    if _looks_like_pdf(r):
        return DownloadResult(path=_save_pdf(r.content, url, save_dir))

    pdf_url = _find_pdf_link(r.text, base=url)
    if not pdf_url:
        return DownloadResult(status=r.status_code)

    try:
        r2 = requests.get(
            _proxify(pdf_url, proxy_prefix),
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.RequestException as e:
        logger.debug("download_pdf follow-link error for %s: %s", pdf_url, e)
        return DownloadResult(errored=True)

    if r2.status_code in PAYWALL_STATUSES:
        return DownloadResult(paywalled=True, status=r2.status_code)

    if r2.status_code == 200 and _looks_like_pdf(r2):
        return DownloadResult(path=_save_pdf(r2.content, pdf_url, save_dir))
    return DownloadResult(status=r2.status_code)


def _proxify(url: str, prefix: Optional[str]) -> str:
    """Wrap a URL through an EZproxy-style prefix for institutional access.

    EPFL example:
        prefix = "https://login.proxy.epfl.ch"
        url    = "https://onlinelibrary.wiley.com/doi/pdf/10.1111/cogs.13256"
        ->     "https://login.proxy.epfl.ch/login?url=https%3A%2F%2F..."

    No-op when prefix is None or URL is already proxied.
    """
    if not prefix or prefix in url:
        return url
    return f"{prefix.rstrip('/')}/login?url={quote(url, safe='')}"


def _looks_like_pdf(response: requests.Response) -> bool:
    ctype = response.headers.get("Content-Type", "").lower()
    if "pdf" in ctype:
        return True
    if response.url.lower().endswith(".pdf"):
        return True
    return response.content[:5] == b"%PDF-"


def expected_pdf_path(url: str, save_dir: str) -> Path:
    """Where a PDF for `url` would be cached. Pure - no I/O.

    Used both by `_save_pdf` (on write) and by the pipeline's cache check
    (on read), so the two stay in sync.
    """
    name = Path(url.split("?", 1)[0]).name or "paper"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return Path(save_dir) / name


def _save_pdf(content: bytes, url: str, save_dir: str) -> str:
    path = expected_pdf_path(url, save_dir)
    path.write_bytes(content)
    return str(path)


def _find_pdf_link(html: str, base: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if href.endswith(".pdf") or "/pdf" in href or "/epdf" in href:
            return urljoin(base, a["href"])
    return None


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed — install `mmore[paper_discovery]`")
        return ""

    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning("extract_text failed for %s: %s", pdf_path, e)
        return ""
