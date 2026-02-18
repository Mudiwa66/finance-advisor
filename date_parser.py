"""
Date range parser for natural language time expressions.

parse_date_range(text) → (start_date, end_date) | None
Both dates are datetime objects (start at midnight, end at 23:59:59).
"""

import re
from calendar import monthrange
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

QUARTER_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
}

_MONTH_RE = "(" + "|".join(MONTH_NAMES) + ")"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 0, 0, 0)


def _end(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 23, 59, 59)


def _month_range(year: int, month: int) -> tuple[datetime, datetime]:
    last_day = monthrange(year, month)[1]
    return _start(year, month, 1), _end(year, month, last_day)


def _quarter_range(year: int, quarter: int) -> tuple[datetime, datetime]:
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    last_day = monthrange(year, end_month)[1]
    return _start(year, start_month, 1), _end(year, end_month, last_day)


def _year_range(year: int) -> tuple[datetime, datetime]:
    return _start(year, 1, 1), _end(year, 12, 31)


def _most_recent_month(month: int, today: datetime) -> int:
    """Return the year of the most recent occurrence of a given month number."""
    return today.year if month <= today.month else today.year - 1


def _current_quarter(today: datetime) -> int:
    return (today.month - 1) // 3 + 1


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_date_range(
    text: str,
    _today: datetime | None = None,  # injectable for testing
) -> tuple[datetime, datetime] | None:
    """
    Parse a natural language date expression into a (start, end) datetime tuple.
    Returns None if no recognisable date range is found.
    """
    today = _today or datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    t = text.lower().strip()

    # ---- Exact days -------------------------------------------------------

    if re.search(r"\btoday\b", t):
        return _start(today.year, today.month, today.day), _end(today.year, today.month, today.day)

    if re.search(r"\byesterday\b", t):
        d = today - timedelta(days=1)
        return _start(d.year, d.month, d.day), _end(d.year, d.month, d.day)

    # ---- Relative N days --------------------------------------------------

    m = re.search(r"\b(?:past|last)\s+(\d+)\s+days?\b", t)
    if m:
        n = int(m.group(1))
        start = today - timedelta(days=n - 1)
        return _start(start.year, start.month, start.day), _end(today.year, today.month, today.day)

    # ---- Weeks ------------------------------------------------------------

    if re.search(r"\b(?:past|last)\s+week\b", t):
        start = today - timedelta(days=6)
        return _start(start.year, start.month, start.day), _end(today.year, today.month, today.day)

    if re.search(r"\bthis\s+week\b", t):
        # Monday of current week → today
        start = today - timedelta(days=today.weekday())
        return _start(start.year, start.month, start.day), _end(today.year, today.month, today.day)

    # ---- Months (relative) ------------------------------------------------

    if re.search(r"\bthis\s+month\b", t):
        return _month_range(today.year, today.month)

    if re.search(r"\blast\s+month\b", t):
        if today.month == 1:
            return _month_range(today.year - 1, 12)
        return _month_range(today.year, today.month - 1)

    # ---- Month + relative year: "january last year", "march this year" ----
    # Must come before standalone "this year" / "last year" checks

    m = re.search(_MONTH_RE + r"\s+last\s+year\b", t)
    if m:
        return _month_range(today.year - 1, MONTH_NAMES[m.group(1)])

    m = re.search(_MONTH_RE + r"\s+this\s+year\b", t)
    if m:
        return _month_range(today.year, MONTH_NAMES[m.group(1)])

    # ---- Years (relative) -------------------------------------------------

    if re.search(r"\bthis\s+year\b", t):
        return _year_range(today.year)

    if re.search(r"\blast\s+year\b", t):
        if not re.search(_MONTH_RE, t):
            return _year_range(today.year - 1)

    # ---- Quarters (with explicit year) ------------------------------------

    # "Q1 2024" / "first quarter 2024" / "2024 Q1"
    m = re.search(r"\bq([1-4])\s+(\d{4})\b", t)
    if m:
        return _quarter_range(int(m.group(2)), int(m.group(1)))

    m = re.search(r"\b(\d{4})\s+q([1-4])\b", t)
    if m:
        return _quarter_range(int(m.group(1)), int(m.group(2)))

    m = re.search(r"\b(first|second|third|fourth)\s+quarter\s+(\d{4})\b", t)
    if m:
        return _quarter_range(int(m.group(2)), QUARTER_WORDS[m.group(1)])

    m = re.search(r"\b(\d{4})\s+(first|second|third|fourth)\s+quarter\b", t)
    if m:
        return _quarter_range(int(m.group(1)), QUARTER_WORDS[m.group(2)])

    # ---- Quarters (relative / no year) ------------------------------------

    if re.search(r"\blast\s+quarter\b", t):
        cq = _current_quarter(today)
        if cq == 1:
            return _quarter_range(today.year - 1, 4)
        return _quarter_range(today.year, cq - 1)

    if re.search(r"\bthis\s+quarter\b", t):
        return _quarter_range(today.year, _current_quarter(today))

    m = re.search(r"\bq([1-4])\b", t)
    if m:
        q = int(m.group(1))
        cq = _current_quarter(today)
        year = today.year if q <= cq else today.year - 1
        return _quarter_range(year, q)

    m = re.search(r"\b(first|second|third|fourth)\s+quarter\b", t)
    if m:
        q = QUARTER_WORDS[m.group(1)]
        cq = _current_quarter(today)
        year = today.year if q <= cq else today.year - 1
        return _quarter_range(year, q)

    # ---- Month + explicit year: "march 2024", "2024 march" ----------------

    m = re.search(_MONTH_RE + r"\s+(20\d{2})\b", t)
    if m:
        return _month_range(int(m.group(2)), MONTH_NAMES[m.group(1)])

    m = re.search(r"\b(20\d{2})\s+" + _MONTH_RE, t)
    if m:
        return _month_range(int(m.group(1)), MONTH_NAMES[m.group(2)])

    # ---- Explicit year only: "2024" ---------------------------------------

    m = re.search(r"\b(20\d{2})\b", t)
    if m:
        return _year_range(int(m.group(1)))

    # ---- Month name only: "march" → most recent occurrence ----------------

    m = re.search(_MONTH_RE + r"\b", t)
    if m:
        month = MONTH_NAMES[m.group(1)]
        year = _most_recent_month(month, today)
        return _month_range(year, month)

    return None


# ---------------------------------------------------------------------------
# CLI / test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fixed reference date: Feb 18, 2026
    TODAY = datetime(2026, 2, 18)

    cases = [
        # (input, expected_description)
        ("march",               "Mar 2025"),
        ("january",             "Jan 2026"),
        ("february",            "Feb 2026"),
        ("last month",          "Jan 2026"),
        ("this month",          "Feb 2026"),
        ("past week",           "Feb 12–18 2026"),
        ("last week",           "Feb 12–18 2026"),
        ("last 7 days",         "Feb 12–18 2026"),
        ("past 30 days",        "Jan 20–Feb 18 2026"),
        ("yesterday",           "Feb 17 2026"),
        ("today",               "Feb 18 2026"),
        ("Q1 2024",             "Jan 1–Mar 31 2024"),
        ("Q2 2024",             "Apr 1–Jun 30 2024"),
        ("Q1",                  "Q1 2026 (current)"),
        ("Q3",                  "Q3 2025 (most recent)"),
        ("last quarter",        "Q4 2025"),
        ("this quarter",        "Q1 2026"),
        ("first quarter 2024",  "Jan 1–Mar 31 2024"),
        ("2024",                "Full year 2024"),
        ("last year",           "Full year 2025"),
        ("this year",           "Full year 2026"),
        ("march 2024",          "Mar 2024"),
        ("january last year",   "Jan 2025"),
        ("march this year",     "Mar 2026"),
        ("unknown text",        "None"),
    ]

    print(f"Reference date: {TODAY.date()}\n")
    print(f"{'Input':<25} {'Result'}")
    print("-" * 55)

    for text, note in cases:
        result = parse_date_range(text, _today=TODAY)
        if result:
            start, end = result
            if start.date() == end.date():
                display = str(start.date())
            elif start.month == end.month and start.year == end.year:
                display = f"{start.date()} → {end.date()}"
            else:
                display = f"{start.date()} → {end.date()}"
        else:
            display = "None"
        print(f"{text:<25} {display:<30}  # {note}")
