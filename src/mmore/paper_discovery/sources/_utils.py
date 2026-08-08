"""Helpers shared by the source adapters."""


def coerce_year(value: object) -> int | None:
    """Best-effort publication year from whatever a source returned.

    Handles the shapes the adapters actually see: an int (OpenAlex), a
    bare year string (Google Scholar), and an ISO date string whose first
    four characters are the year (arXiv, Europe PMC).

    Args:
      value: Raw field value from a source API response.

    Returns:
      The year as an int, or None if the value is missing or unparseable.
    """
    if value is None:
        return None
    try:
        return int(str(value).strip()[:4])
    except ValueError:
        return None


def first_year(entry: dict, *keys: str) -> int | None:
    """First key in `keys` that yields a usable year via `coerce_year`.

    Args:
      entry: Source API record.
      keys:  Field names to try, in priority order.

    Returns:
      The first parseable year, or None if no key yields one.
    """
    for key in keys:
        year = coerce_year(entry.get(key))
        if year is not None:
            return year
    return None
