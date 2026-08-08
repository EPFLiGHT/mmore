"""Stage 1: build boolean queries from a synonym table + category map.

Offline, deterministic - no network calls.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Union

from .schema import CategoryQuery, SynonymEntry

logger = logging.getLogger(__name__)


def _sanitize_term(term: str) -> str:
    """Drop characters that would break a quoted boolean term.

    Source APIs (arXiv, OpenAlex, Europe PMC) interpret `"..."` as a
    literal phrase, so an embedded `"` inside the term closes the
    phrase early and produces malformed queries. We just strip them.
    Whitespace is normalized too.
    """
    return re.sub(r"\s+", " ", term.replace('"', "")).strip()


def load_synonyms(path: Union[str, Path]) -> List[SynonymEntry]:
    """Load synonyms from a JSONL file (one object per line).

    Args:
      path: Path to a `.jsonl` file. Each non-empty line must be a JSON
            object with a `word` key and a `synonyms` key holding either
            a list of strings or a single string with terms separated by
            `,` or `;`.

            Example:
                {"word": "Foundation model", "synonyms": ["LLM", "GPT"]}
                {"word": "Crisis response", "synonyms": ["humanitarian aid"]}

    Returns:
      List of `SynonymEntry`, one per row (rows missing `word` are
      skipped). Term whitespace is normalized and `"` characters are
      stripped. Case is preserved on the stored value but lookups in
      `build_boolean_queries` are case-insensitive.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    entries: List[SynonymEntry] = []
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
    """Turn a SynonymEntry into an OR-group of quoted strings."""
    terms = {entry.word, *entry.synonyms}
    # `load_synonyms` already sanitizes, but apply again here so direct
    # callers that build SynonymEntry by hand stay safe.
    quoted = sorted(f'"{_sanitize_term(t)}"' for t in terms if _sanitize_term(t))
    return "(" + " OR ".join(quoted) + ")"


def build_boolean_queries(
    synonyms: List[SynonymEntry],
    categories: Dict[str, List[str]],
) -> List[CategoryQuery]:
    """Compose category-level AND-of-OR-groups boolean queries.

    Args:
      synonyms:   Output of `load_synonyms` - the dictionary of canonical
                  words and their synonyms.
      categories: Mapping `{category_name -> [word, ...]}`. The list
                  values are canonical `word`s that MUST appear as
                  `SynonymEntry.word` in `synonyms` (lookup is
                  case-insensitive, whitespace preserved). Example:
                      {
                        "Humanitarian AI": ["Foundation model",
                                            "Crisis response"]
                      }

    Returns:
      One `CategoryQuery` per category that resolved to at least one
      synonym group. Categories whose words all reference unknown
      `word`s are logged and skipped - the full run never fails.
    """

    by_word = {e.word.lower(): e for e in synonyms}
    queries: List[CategoryQuery] = []

    for cat_name, words in categories.items():
        groups: List[str] = []
        for w in words:
            entry = by_word.get(w.lower())
            if entry is None:
                logger.warning(
                    "Category %r references unknown word %r - skipping",
                    cat_name,
                    w,
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
