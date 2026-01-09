# services/validators.py
# ------------------------------------------------------------
# Validators & normalization utilities (production-ready POC)
#
# Purpose:
# - Centralize input validation for Streamlit pages
# - Prevent common runtime issues (bad depot IDs, bad dates, unsafe limits)
# - Provide consistent, buyer-friendly error messages
#
# Principles:
# - Prefer deterministic validation (no network calls)
# - Keep outputs simple: (value, error_message)
# - Provide a small "require()" helper to raise Streamlit-friendly exceptions
# ------------------------------------------------------------

from __future__ import annotations

import re
import datetime as dt
from typing import Any, Optional, Sequence, Tuple


# -----------------------------
# Exceptions
# -----------------------------
class ValidationError(ValueError):
    """Raised when user input fails validation."""


# -----------------------------
# Basic parsing
# -----------------------------
def normalize_str(x: Any, *, strip: bool = True, empty_to_none: bool = True) -> Optional[str]:
    if x is None:
        return None
    s = str(x)
    if strip:
        s = s.strip()
    if empty_to_none and s == "":
        return None
    return s


def parse_int(x: Any) -> Optional[int]:
    s = normalize_str(x)
    if s is None:
        return None
    try:
        return int(s)
    except Exception:
        return None


def parse_float(x: Any) -> Optional[float]:
    s = normalize_str(x)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_iso_date(x: Any) -> Optional[str]:
    s = normalize_str(x)
    if s is None:
        return None
    try:
        dt.date.fromisoformat(s)
        return s
    except Exception:
        return None


def parse_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


# -----------------------------
# Domain validators
# -----------------------------
def validate_depot(depot: Any, *, allow_none: bool = True) -> Tuple[Optional[int], Optional[str]]:
    if depot is None or (isinstance(depot, str) and depot.strip() == ""):
        return (None, None) if allow_none else (None, "Depot is required.")

    d = parse_int(depot)
    if d is None:
        return None, "Depot must be an integer (e.g., 172)."
    if d <= 0:
        return None, "Depot must be a positive integer."
    return d, None


def validate_item(
    item: Any,
    *,
    allowed: Optional[Sequence[str]] = None,
    allow_all: bool = True,
    normalize_case: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    s = normalize_str(item)
    if s is None:
        return None, "Item is required."

    if normalize_case and allowed:
        # Try to map case-insensitively to an allowed value (keeps canonical casing)
        m = {a.lower(): a for a in allowed}
        s = m.get(s.lower(), s)

    if allow_all and s == "All":
        return "All", None
    if allowed is not None and s not in allowed:
        return None, f"Item must be one of: {', '.join(allowed)}."
    return s, None


def validate_horizon(
    horizon: Any,
    *,
    allow_none: bool = True,
    min_h: int = 1,
    max_h: int = 3,
) -> Tuple[Optional[int], Optional[str]]:
    if horizon is None or (isinstance(horizon, str) and horizon.strip() == ""):
        return (None, None) if allow_none else (None, "Horizon is required.")

    h = parse_int(horizon)
    if h is None:
        return None, f"Horizon must be an integer between {min_h} and {max_h}."
    if h < min_h or h > max_h:
        return None, f"Horizon must be between {min_h} and {max_h}."
    return h, None


def validate_run_id(run_id: Any, *, allow_none: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    run_id is expected to be an ISO date string (Monday run date).
    """
    s = parse_iso_date(run_id)
    if s is None:
        if allow_none:
            return None, None
        return None, "Run date (run_id) must be a valid date in YYYY-MM-DD format."
    return s, None


def validate_week_start(week_start: Any, *, allow_none: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Week start used for fiscal week start dates.
    """
    s = parse_iso_date(week_start)
    if s is None:
        return (None, None) if allow_none else (None, "Week start must be a valid date in YYYY-MM-DD format.")
    return s, None


def validate_date_range(
    start_date: Any,
    end_date: Any,
    *,
    allow_none: bool = True,
    max_days: Optional[int] = None,
) -> Tuple[Tuple[Optional[str], Optional[str]], Optional[str]]:
    """
    Validates (start_date, end_date) in ISO format, ensures start <= end.
    Optionally enforces max_days window.
    """
    s = parse_iso_date(start_date)
    e = parse_iso_date(end_date)

    if s is None and e is None:
        return ((None, None), None) if allow_none else ((None, None), "Start and end dates are required.")

    if s is None:
        return (None, None), "Start date must be a valid date in YYYY-MM-DD format."
    if e is None:
        return (None, None), "End date must be a valid date in YYYY-MM-DD format."

    sd = dt.date.fromisoformat(s)
    ed = dt.date.fromisoformat(e)
    if sd > ed:
        return (None, None), "Start date must be on or before end date."

    if max_days is not None and (ed - sd).days > int(max_days):
        return (None, None), f"Date range is too large. Please choose {max_days} days or fewer."

    return (s, e), None


def validate_limit(
    limit: Any,
    *,
    default: int = 2000,
    min_v: int = 1,
    max_v: int = 50000,
) -> Tuple[int, Optional[str]]:
    """
    Validates an integer limit for table queries.
    """
    if limit is None or (isinstance(limit, str) and limit.strip() == ""):
        return int(default), None

    v = parse_int(limit)
    if v is None:
        return int(default), f"Limit must be an integer between {min_v} and {max_v}. Using {default:,}."
    if v < min_v:
        return int(min_v), f"Limit increased to minimum {min_v}."
    if v > max_v:
        return int(max_v), f"Limit capped at {max_v:,}."
    return int(v), None


# -----------------------------
# Text extraction helpers (assistant routing support)
# -----------------------------
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_DEPOT_RE = re.compile(r"\bdepot\b\D*(\d{1,6})\b", re.IGNORECASE)
_HORIZON_RE = re.compile(r"\b(?:horizon|week)\s*([1-3])\b", re.IGNORECASE)
_TOPN_RE = re.compile(r"\btop\s*(\d{1,3})\b", re.IGNORECASE)


def extract_first_iso_date(text: Any) -> Optional[str]:
    t = normalize_str(text)
    if not t:
        return None
    m = _DATE_RE.search(t)
    return parse_iso_date(m.group(1)) if m else None


def extract_depot_from_text(text: Any) -> Optional[int]:
    t = normalize_str(text)
    if not t:
        return None
    m = _DEPOT_RE.search(t)
    return parse_int(m.group(1)) if m else None


def extract_horizon_from_text(text: Any) -> Optional[int]:
    t = normalize_str(text)
    if not t:
        return None
    m = _HORIZON_RE.search(t)
    return parse_int(m.group(1)) if m else None


def extract_top_n_from_text(text: Any, *, max_v: int = 50) -> Optional[int]:
    t = normalize_str(text)
    if not t:
        return None
    m = _TOPN_RE.search(t)
    if not m:
        return None
    n = parse_int(m.group(1))
    if n is None:
        return None
    if n < 1:
        return 1
    if n > int(max_v):
        return int(max_v)
    return n


# -----------------------------
# Streamlit-friendly wrapper
# -----------------------------
def require(value: Any, error: Optional[str]) -> Any:
    """
    Convenience: raise a ValidationError if error is present, else return value.
    Streamlit pages can catch ValidationError and show st.error().
    """
    if error:
        raise ValidationError(error)
    return value
