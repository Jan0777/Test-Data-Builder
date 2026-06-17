"""
Content-based semantic type detection.
Classifies a column's semantic type by inspecting its VALUES (sampled),
with the column name used only as a secondary tiebreaker.
"""
from __future__ import annotations
import re
import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Compiled regex patterns ──────────────────────────────────────────────────
_EMAIL_RE = re.compile(r'^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,10}$')
_PHONE_RE = re.compile(r'^[\+]?[\d\s\-\(\)\.]{7,20}$')
_URL_RE = re.compile(r'^https?://')
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)
_IP_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$')
_POSTCODE_RE = re.compile(r'^\d{5}(?:-\d{4})?$|^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$', re.IGNORECASE)

# Small reference sets for membership checks
_COUNTRY_NAMES = frozenset({
    "united states", "usa", "u.s.a.", "canada", "united kingdom", "uk", "germany",
    "france", "italy", "spain", "china", "japan", "india", "brazil", "australia",
    "mexico", "russia", "south korea", "netherlands", "sweden", "norway", "denmark",
    "switzerland", "austria", "belgium", "poland", "argentina", "south africa",
    "new zealand", "singapore", "hong kong", "taiwan", "indonesia", "malaysia",
    "thailand", "philippines", "vietnam", "turkey", "egypt", "nigeria", "kenya",
})
_COUNTRY_CODES = frozenset({
    "US", "CA", "GB", "DE", "FR", "IT", "ES", "CN", "JP", "IN", "BR", "AU",
    "MX", "RU", "KR", "NL", "SE", "NO", "DK", "CH", "AT", "BE", "PL", "AR",
    "ZA", "NZ", "SG", "HK", "TW", "ID", "MY", "TH", "PH", "VN", "TR", "EG",
    "NG", "KE", "AE", "SA", "IL", "CL", "CO", "PE",
})
_CURRENCY_CODES = frozenset({
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "HKD", "SGD",
    "SEK", "NOK", "DKK", "NZD", "MXN", "BRL", "INR", "ZAR", "KRW", "TRY",
    "AED", "SAR", "ILS", "PLN", "CZK", "HUF", "RUB", "THB", "IDR", "MYR",
})
_US_STATES = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
})
_COMPANY_SUFFIXES = frozenset({
    "inc", "inc.", "llc", "llc.", "ltd", "ltd.", "corp", "corp.", "co.", "co",
    "plc", "gmbh", "ag", "sa", "bv", "nv", "pty", "pty.", "srl",
})


# ── Public API ───────────────────────────────────────────────────────────────

def detect_semantic(
    name: str,
    series: pd.Series,
    sample_size: int = 200,
) -> Tuple[str, Optional[str]]:
    """
    Detect the semantic type of a column by inspecting its values.

    Returns
    -------
    (semantic_type, faker_method)
        semantic_type : one of the allowed ColumnSpec.semantic_type literals
        faker_method  : Faker method name, or None
    """
    clean = series.dropna().astype(str)
    if len(clean) == 0:
        return "none", None

    sample = clean.head(sample_size)
    n = len(sample)

    # ── Regex / pattern checks ───────────────────────────────────────────
    if _match_rate(sample, _EMAIL_RE) >= 0.80:
        logger.debug("  semantic: email (content)")
        return "email", "email"

    if _match_rate(sample, _UUID_RE) >= 0.80:
        logger.debug("  semantic: id/uuid (content)")
        return "id", None  # keep as empirical categorical / pattern

    if _match_rate(sample, _URL_RE) >= 0.80:
        logger.debug("  semantic: url (content)")
        return "url", "url"

    if _match_rate(sample, _IP_RE) >= 0.80:
        logger.debug("  semantic: id/ip (content)")
        return "id", None

    if _match_rate(sample, _PHONE_RE) >= 0.80:
        # Extra guard: phone-like means mostly digits/spaces/dashes, not sentences
        avg_len = sample.str.len().mean()
        if avg_len <= 20:
            logger.debug("  semantic: phone (content)")
            return "phone", "phone_number"

    if _match_rate(sample, _POSTCODE_RE) >= 0.75:
        logger.debug("  semantic: zip (content)")
        return "zip", "postcode"

    # ── Value-set membership checks ──────────────────────────────────────
    lower_vals = sample.str.lower()

    country_rate = _set_match_rate(sample, _COUNTRY_NAMES) or _set_match_rate(sample, _COUNTRY_CODES)
    if country_rate >= 0.70:
        logger.debug("  semantic: country (content)")
        return "country", "country"

    if _set_match_rate(sample, _CURRENCY_CODES) >= 0.70:
        logger.debug("  semantic: currency (content)")
        return "currency", None

    state_rate = _set_match_rate(sample, _US_STATES)
    if state_rate >= 0.60:
        logger.debug("  semantic: state (content)")
        return "state", "state"

    # Company suffix heuristic
    last_word_rate = sum(
        1 for v in sample
        if v.split()[-1].lower().rstrip(".") in _COMPANY_SUFFIXES
    ) / n
    if last_word_rate >= 0.50:
        logger.debug("  semantic: company (content)")
        return "company", "company"

    # ── Name heuristic: two-token, first token capitalized ──────────────
    name_like = sum(
        1 for v in sample
        if _looks_like_person_name(v)
    ) / n
    if name_like >= 0.70:
        logger.debug("  semantic: name (content)")
        return "name", "name"

    # ── Column-name fallback ─────────────────────────────────────────────
    return _guess_from_name(name)


def _match_rate(sample: pd.Series, pattern: re.Pattern) -> float:
    if len(sample) == 0:
        return 0.0
    return sum(1 for v in sample if pattern.match(str(v))) / len(sample)


def _set_match_rate(sample: pd.Series, ref_set: frozenset) -> float:
    if len(sample) == 0:
        return 0.0
    hits = sum(1 for v in sample if v in ref_set or v.lower() in ref_set)
    return hits / len(sample)


def _looks_like_person_name(s: str) -> bool:
    parts = s.strip().split()
    if len(parts) < 2 or len(parts) > 4:
        return False
    return all(p[0].isupper() and p.replace("-", "").replace("'", "").isalpha() for p in parts)


def _guess_from_name(col_name: str) -> Tuple[str, Optional[str]]:
    """Column-name-based fallback — only used when content sniffing is inconclusive."""
    n = col_name.lower()
    if any(k in n for k in ("email", "e_mail", "mail")):
        return "email", "email"
    if "first_name" in n or n == "firstname":
        return "name", "first_name"
    if "last_name" in n or "surname" in n or n == "lastname":
        return "name", "last_name"
    if any(k in n for k in ("fullname", "full_name")) or n == "name":
        return "name", "name"
    if any(k in n for k in ("phone", "mobile", "cell", "tel")):
        return "phone", "phone_number"
    if any(k in n for k in ("street", "address", "addr")):
        return "address", "street_address"
    if "city" in n:
        return "city", "city"
    if "state" in n and "zip" not in n and "postal" not in n:
        return "state", "state"
    if any(k in n for k in ("zip", "postal", "postcode")):
        return "zip", "postcode"
    if "country" in n:
        return "country", "country"
    if any(k in n for k in ("price", "salary", "wage", "amount", "cost", "revenue", "income")):
        return "currency", None
    if any(k in n for k in ("category", "status", "type", "tier", "level")):
        return "category", None
    if any(k in n for k in ("date", "time", "created", "updated", "at")):
        return "date", None
    if any(k in n for k in ("url", "website", "link", "href")):
        return "url", "url"
    if any(k in n for k in ("company", "employer", "organization", "org")):
        return "company", "company"
    if n in ("id", "_id") or n.endswith("_id") or n.endswith("id"):
        return "id", None
    return "none", None
