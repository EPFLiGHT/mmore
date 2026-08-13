"""Download PDFs and pull text out of them. Never raises on remote errors."""

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Status codes the publisher uses to say "you don't have access here."
# These are expected on paywalled content and are reported as a summary,
# not per-paper warnings.
PAYWALL_STATUSES = {401, 402, 403, 429}

# Substrings that mark an HTML body as a sign-in page rather than content.
# A proxy that needs authentication answers 200 OK with one of these, which
# would otherwise be counted as a silent skip.
LOGIN_PAGE_MARKERS = (
    "shibboleth",
    "tequila",
    'name="saml',
    'type="password"',
    "discovery service",
    "institutional login",
)


@dataclass
class DownloadResult:
    """Outcome of a single PDF fetch. The pipeline tallies these at the end."""

    path: str | None = None  # local file path on success
    paywalled: bool = False  # publisher returned 401/402/403/429
    errored: bool = False  # network/timeout/other, actionable
    status: int | None = None  # last seen HTTP status, if any
    login_page: bool = False  # got an auth page instead of the PDF


def download_pdf(
    url: str,
    save_dir: str,
    user_agent: str = "mmore-paper-discovery/1.0",
    timeout: int = 30,
    proxy_prefix: str | None = None,
) -> DownloadResult:
    """Fetch one PDF, following a landing page if that is what we get.

    Args:
      url: Where the PDF is, or a page that links to it.
      save_dir: Directory to cache the file in.
      user_agent: Sent on the request. The default identifies this tool
        honestly rather than posing as a browser.
      timeout: Per-request timeout in seconds.
      proxy_prefix: Optional EZproxy host to route through.

    Returns:
      A `DownloadResult` saying what happened, including whether the
      publisher refused us and whether we hit a sign-in page.
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

    if _looks_like_login_page(r):
        logger.debug("download_pdf got a sign-in page for %s", url)
        return DownloadResult(status=r.status_code, login_page=True)

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
    if _looks_like_login_page(r2):
        return DownloadResult(status=r2.status_code, login_page=True)
    return DownloadResult(status=r2.status_code)


def _proxify(url: str, prefix: str | None) -> str:
    """Route a URL through an EZproxy host.

        "https://ezproxy.example.edu" + "https://wiley.com/x.pdf"
        -> "https://ezproxy.example.edu/login?url=https%3A%2F%2F..."

    Does nothing without a prefix, or if the URL is already routed. Only
    relevant where the institution runs EZproxy, since VPN and IP
    recognition need no rewriting.
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


def _looks_like_login_page(response: requests.Response) -> bool:
    """True when an HTML body is a sign-in form rather than content.

    A proxy that cannot authenticate us answers 200 OK with its login page.
    Detecting it is what stops the pipeline recording a silent skip.
    """
    if "html" not in response.headers.get("Content-Type", "").lower():
        return False
    body = response.text[:20000].lower()
    return any(marker in body for marker in LOGIN_PAGE_MARKERS)


def expected_pdf_path(url: str, save_dir: str) -> Path:
    """Where a PDF for `url` would be cached. No I/O.

    Shared by the writer and the pipeline's cache check so the two agree.
    """
    name = Path(url.split("?", 1)[0]).name or "paper"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return Path(save_dir) / name


def _save_pdf(content: bytes, url: str, save_dir: str) -> str:
    path = expected_pdf_path(url, save_dir)
    path.write_bytes(content)
    return str(path)


def _find_pdf_link(html: str, base: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        # bs4 types an attribute as str | list[str]; join covers the rare
        # multi-valued case so the rest of the function sees one string.
        raw = a["href"]
        href = " ".join(raw) if isinstance(raw, list) else str(raw)
        lowered = href.lower()
        if lowered.endswith(".pdf") or "/pdf" in lowered or "/epdf" in lowered:
            return urljoin(base, href)
    return None


def extract_text(pdf_path: str, mode: str = "fast") -> str:
    """Pull text out of a PDF using mmore's own `PDFProcessor`.

    Args:
      pdf_path: Local path to the PDF.
      mode: `"fast"` uses the PyMuPDF path and loads no models. `"full"`
        uses marker and surya, which handles complex layouts better but
        downloads weights on first use. Unknown values fall back to fast.

    Returns:
      The extracted text, or an empty string if extraction failed.
    """
    try:
        from ..process.processors.base import ProcessorConfig
        from ..process.processors.pdf_processor import PDFProcessor
    except ImportError as e:
        logger.error(
            "mmore.process not installed - install `mmore[paper_discovery]` "
            "(which depends on `mmore[process]`) to enable PDF extraction. (%s)",
            e,
        )
        return ""

    if mode not in {"fast", "full"}:
        logger.warning("Unknown pdf_extractor mode %r; falling back to 'fast'", mode)
        mode = "fast"

    try:
        processor = PDFProcessor(
            ProcessorConfig(custom_config={"extract_images": False})
        )
        sample = (
            processor.process(pdf_path)
            if mode == "full"
            else processor.process_fast(pdf_path)
        )
        return sample.text or ""
    except Exception as e:
        logger.warning("extract_text failed for %s: %s", pdf_path, e)
        return ""
