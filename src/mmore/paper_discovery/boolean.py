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


def load_synonyms(path: Union[str, Path]) -> List[SynonymEntry]:
    """Load synonyms from a JSON file.

    Args:
      path: JSON file containing a list of objects. Each object must have a
            `word` (or `WORD`) key, plus a `synonyms` (or
            `SYNONYMS AND NEAR SYNONYMS`) key holding either:
              - a list of strings, or
              - a single string with terms separated by `,` or `;`.

            Example:
                [
                  {"word": "Foundation model",
                   "synonyms": ["LLM", "large language model", "GPT"]},
                  {"word": "Crisis response",
                   "synonyms": "humanitarian aid; disaster response"}
                ]

    Returns:
      List of `SynonymEntry`, one per row (rows missing `word` are skipped).
      Term whitespace is stripped; case is preserved on the stored value but
      lookups in `build_boolean_queries` are case-insensitive.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for row in data:
        word = row.get("word") or row.get("WORD")
        raw = row.get("synonyms") or row.get("SYNONYMS AND NEAR SYNONYMS") or ""
        if isinstance(raw, str):
            synonyms = [s.strip() for s in re.split(r"[,;]", raw) if s.strip()]
        else:
            synonyms = [s.strip() for s in raw if s and s.strip()]
        if word:
            entries.append(SynonymEntry(word=word.strip(), synonyms=synonyms))
    return entries


def _or_group(entry: SynonymEntry) -> str:
    """Turn a SynonymEntry into an OR-group of quoted strings."""
    terms = {entry.word, *entry.synonyms}
    quoted = sorted(f'"{t}"' for t in terms if t)
    return "(" + " OR ".join(quoted) + ")"


def build_boolean_queries(
    synonyms: List[SynonymEntry],
    categories: Dict[str, List[str]],
) -> List[CategoryQuery]:
    """Compose category-level AND-of-OR-groups boolean queries.

    Args:
      synonyms:   Output of `load_synonyms` - the dictionary of canonical
                  words and their synonyms.
      categories: Mapping `{category_name -> [word, ...]}`. The list values
                  are canonical `word`s that MUST appear as `SynonymEntry.word`
                  in `synonyms` (lookup is case-insensitive, whitespace
                  preserved). Example:
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
