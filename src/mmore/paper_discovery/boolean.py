"""Stage 1: turn synonyms and categories into boolean queries.

Offline and deterministic, no network calls.
"""

import json
import logging
import re
from pathlib import Path

from .schema import CategoryQuery, SynonymEntry

logger = logging.getLogger(__name__)


def _sanitize_term(term: str) -> str:
    """Strip characters that would break a quoted search term.

    Sources read `"..."` as a literal phrase, so a `"` inside a term closes
    the phrase early and corrupts the query. Also collapses whitespace.
    """
    return re.sub(r"\s+", " ", term.replace('"', "")).strip()


def load_synonyms(path: str | Path) -> list[SynonymEntry]:
    """Read a synonym file.

    Args:
      path: A `.jsonl` file, one object per line, each with a `word` and a
        `synonyms` value. Synonyms can be a list, or one string separated
        by commas or semicolons.

            {"word": "Foundation model", "synonyms": ["LLM", "GPT"]}
            {"word": "Crisis response", "synonyms": "aid; disaster relief"}

    Returns:
      One entry per line. Blank lines and rows without a `word` are
      skipped. Terms are sanitized, and lookups later are case-insensitive.
    """
    path = Path(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    entries: list[SynonymEntry] = []
    for row in rows:
        word = row.get("word")
        raw = row.get("synonyms") or ""
        if isinstance(raw, str):
            synonyms_raw = [s.strip() for s in re.split(r"[,;]", raw) if s.strip()]
        else:
            synonyms_raw = [s.strip() for s in raw if s and s.strip()]

        clean_synonyms = [t for t in (_sanitize_term(s) for s in synonyms_raw) if t]
        clean_word = _sanitize_term(word) if word else ""
        if clean_word:
            entries.append(SynonymEntry(word=clean_word, synonyms=clean_synonyms))
    return entries


def _or_group(entry: SynonymEntry) -> str:
    """Render one entry as `("term" OR "term" OR ...)`, sorted for stable output."""
    terms = {entry.word, *entry.synonyms}
    # Sanitized again here so hand-built entries are safe too.
    quoted = sorted(f'"{_sanitize_term(t)}"' for t in terms if _sanitize_term(t))
    return "(" + " OR ".join(quoted) + ")"


def build_boolean_queries(
    synonyms: list[SynonymEntry],
    categories: dict[str, list[str]],
) -> list[CategoryQuery]:
    """Build one query per category.

    Each category becomes an AND of OR-groups, so a paper matches when it
    mentions at least one term from every group.

    Args:
      synonyms: Entries from `load_synonyms`.
      categories: Category name to the words it searches for. Words must
        exist in `synonyms`, matched case-insensitively.

    Returns:
      One query per category that resolved at least one word. Categories
      whose words are all unknown are logged and skipped, so a typo never
      fails the run.
    """
    by_word = {e.word.lower(): e for e in synonyms}
    queries: list[CategoryQuery] = []

    for cat_name, words in categories.items():
        groups: list[str] = []
        for w in words:
            entry = by_word.get(w.lower())
            if entry is None:
                logger.warning(
                    "Category %r references unknown word %r, skipping", cat_name, w
                )
                continue
            groups.append(_or_group(entry))

        if not groups:
            continue

        queries.append(
            CategoryQuery(
                combination_title=cat_name,
                boolean_combination=" AND ".join(groups),
            )
        )

    return queries
