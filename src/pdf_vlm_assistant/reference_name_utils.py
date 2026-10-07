from __future__ import annotations

import re
from typing import Any


LEADING_REFERENCE_LABEL_RE = re.compile(
    r"^\s*(?:\[\d+\]|\(?\d+\)?|(?:\d+|[IVXLC]+)[.)])\s*",
    re.IGNORECASE,
)
NO_AUTHOR_PREFIX_RE = re.compile(r"^(?:s\.?\s*a\.?|anon(?:ymous)?|unknown)\b", re.IGNORECASE)
QUOTED_TITLE_RE = re.compile(r'["“”\'‘’「」『』](?P<title>.+?)["“”\'‘’「」『』]')
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PAREN_YEAR_RE = re.compile(r"\((?P<year>(?:19|20)\d{2})\)")
DOI_OR_URL_RE = re.compile(r"(?:https?://\S+|\bdoi\s*:?\s*\S+)", re.IGNORECASE)
INITIAL_DOT_PROTECT_RE = re.compile(r"(?<=\b[A-Z])\.(?=(?:\s*[A-Z]\.)|\s*[,;&])")
VENUE_PREFIXES = (
    "acm ",
    "bmc ",
    "conference",
    "elsevier",
    "ieee ",
    "in proceedings",
    "journal ",
    "lecture notes",
    "nature",
    "ophthalmology",
    "phys ",
    "proc ",
    "proceedings",
    "rev ",
    "science",
    "springer",
)


def normalize_reference_text(text: str) -> str:
    return " ".join(str(text or "").replace("\u00a0", " ").split())


def join_unique_nonempty_text(parts: list[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = normalize_reference_text(part)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return "\n".join(ordered)


def best_reference_display_name(reference: dict[str, Any]) -> str:
    explicit = normalize_reference_text(reference.get("title") or "")
    if explicit:
        return explicit
    inferred = infer_reference_title(reference)
    if inferred:
        return inferred
    for value in reference.get("aliases") or []:
        alias = normalize_reference_text(value)
        if alias:
            return alias
    return ""


def infer_reference_title(reference: dict[str, Any]) -> str:
    explicit = normalize_reference_text(reference.get("title") or "")
    if explicit:
        return explicit

    candidates: list[str] = []
    raw_variants = [
        normalize_reference_text(value)
        for value in (reference.get("raw_variants") or [])
        if normalize_reference_text(value)
    ]
    aliases = [
        normalize_reference_text(value)
        for value in (reference.get("aliases") or [])
        if normalize_reference_text(value)
    ]
    authors_text = normalize_reference_text(reference.get("authors_text") or "")
    year = normalize_reference_text(reference.get("year") or "")

    for alias in aliases:
        candidates.append(alias)
    for raw_text in raw_variants[:3]:
        quoted = _extract_quoted_title(raw_text)
        if quoted:
            candidates.append(quoted)
        candidates.extend(_extract_title_candidates_from_raw(raw_text, year=year))
    if authors_text:
        candidates.extend(_extract_title_candidates_from_author_blob(authors_text))
    if not candidates:
        signature_fallback = _extract_title_from_reference_signature(reference.get("reference_id") or "")
        if signature_fallback:
            candidates.append(signature_fallback)

    best = ""
    best_score = float("-inf")
    for candidate in candidates:
        cleaned = clean_reference_candidate(candidate)
        score = score_reference_title_candidate(cleaned)
        if score > best_score:
            best = cleaned
            best_score = score
    return best if best_score >= 3.0 else ""


def build_reference_search_text(reference: dict[str, Any]) -> str:
    title = best_reference_display_name(reference)
    aliases = [
        normalize_reference_text(value)
        for value in (reference.get("aliases") or [])
        if normalize_reference_text(value)
    ]
    raw_variants = [
        normalize_reference_text(value)
        for value in (reference.get("raw_variants") or [])
        if normalize_reference_text(value)
    ]
    return join_unique_nonempty_text(
        [
            title,
            f"authors {normalize_reference_text(reference.get('authors_text') or '')}"
            if normalize_reference_text(reference.get("authors_text") or "")
            else "",
            f"year {normalize_reference_text(reference.get('year') or '')}"
            if normalize_reference_text(reference.get("year") or "")
            else "",
            f"venue {normalize_reference_text(reference.get('venue_text') or '')}"
            if normalize_reference_text(reference.get("venue_text") or "")
            else "",
            f"doi {normalize_reference_text(reference.get('doi') or '')}"
            if normalize_reference_text(reference.get("doi") or "")
            else "",
            f"pmid {normalize_reference_text(reference.get('pmid') or '')}"
            if normalize_reference_text(reference.get("pmid") or "")
            else "",
            *aliases,
            *raw_variants[:6],
        ]
    )


def clean_reference_candidate(text: str) -> str:
    cleaned = normalize_reference_text(text)
    cleaned = LEADING_REFERENCE_LABEL_RE.sub("", cleaned)
    cleaned = cleaned.strip(" .;,:()[]{}<>\"'“”‘’")
    cleaned = DOI_OR_URL_RE.sub("", cleaned).strip(" .;,:()[]{}<>\"'“”‘’")
    cleaned = re.sub(r"[※§¶†‡]+", "", cleaned)
    cleaned = normalize_reference_text(cleaned)
    return cleaned


def score_reference_title_candidate(text: str) -> float:
    candidate = clean_reference_candidate(text)
    if not candidate:
        return float("-inf")

    alpha_tokens = count_alpha_tokens(candidate)
    if alpha_tokens < 3:
        return float("-inf")

    lowered = candidate.casefold()
    score = float(alpha_tokens)
    score += min(len(candidate) / 28.0, 4.0)

    if looks_like_author_segment(candidate):
        score -= 16.0
    if looks_like_venue_segment(candidate):
        score -= 10.0
    if "accessed" in lowered or "retrieved" in lowered:
        score -= 5.0
    if candidate.count(",") >= 5:
        score -= 3.0
    if len(candidate) > 220:
        score -= 5.0
    return score


def count_alpha_tokens(text: str) -> int:
    return sum(1 for token in re.findall(r"[^\W\d_]+", str(text or ""), flags=re.UNICODE) if token)


def looks_like_author_segment(text: str) -> bool:
    candidate = normalize_reference_text(text)
    if not candidate:
        return False
    lowered = candidate.casefold()
    if "et al" in lowered or ";" in candidate or " & " in candidate:
        return True
    tokens = re.findall(r"[A-Za-z][A-Za-z.\-]*", candidate)
    if len(tokens) < 2:
        return False
    short_tokens = sum(1 for token in tokens if len(token.strip(".-")) <= 2)
    punctuation_score = candidate.count(",") + candidate.count(";")
    if candidate.count(",") >= 1 and short_tokens >= 1 and len(tokens) <= 8:
        return True
    return punctuation_score >= 2 and short_tokens >= max(1, len(tokens) // 4)


def looks_like_venue_segment(text: str) -> bool:
    candidate = normalize_reference_text(text)
    if not candidate:
        return False
    lowered = candidate.casefold()
    if DOI_OR_URL_RE.search(candidate):
        return True
    if lowered.startswith(VENUE_PREFIXES):
        return True
    if re.search(r"\b\d+\s*[:;,]\s*\d+\b", candidate):
        return True
    return False


def split_reference_sentences(text: str) -> list[str]:
    normalized = normalize_reference_text(text)
    if not normalized:
        return []
    protected = INITIAL_DOT_PROTECT_RE.sub("∯", normalized)
    parts = [
        normalize_reference_text(part.replace("∯", ".")).strip(" .;,:")
        for part in re.split(r"\.\s+", protected)
        if normalize_reference_text(part.replace("∯", ".")).strip(" .;,:")
    ]
    return parts


def _extract_quoted_title(text: str) -> str:
    match = QUOTED_TITLE_RE.search(str(text or ""))
    if not match:
        return ""
    return clean_reference_candidate(match.group("title"))


def _extract_title_candidates_from_author_blob(text: str) -> list[str]:
    parts = split_reference_sentences(text)
    if len(parts) >= 2 and looks_like_author_segment(parts[0]):
        for part in parts[1:]:
            if looks_like_author_segment(part) or looks_like_venue_segment(part):
                continue
            if count_alpha_tokens(part) >= 3:
                return [part]
    return []


def _extract_title_candidates_from_raw(text: str, year: str) -> list[str]:
    normalized = clean_reference_candidate(text)
    if not normalized:
        return []

    candidates: list[str] = []
    parts = split_reference_sentences(normalized)
    if parts:
        if NO_AUTHOR_PREFIX_RE.match(parts[0]) and len(parts) >= 2:
            for part in parts[1:]:
                if looks_like_venue_segment(part):
                    continue
                if count_alpha_tokens(part) >= 3:
                    candidates.append(part)
                    break
        if looks_like_author_segment(parts[0]) and len(parts) >= 2:
            for part in parts[1:]:
                if looks_like_author_segment(part) or looks_like_venue_segment(part):
                    continue
                if count_alpha_tokens(part) >= 3:
                    candidates.append(part)
                    break
        if not looks_like_author_segment(parts[0]) and not looks_like_venue_segment(parts[0]):
            candidates.append(parts[0])

    if year:
        paren_match = re.search(rf"\({re.escape(year)}\)\s*(.+?)(?:\.\s|$)", normalized)
        if paren_match:
            candidates.append(paren_match.group(1))
        year_match = re.search(rf"\b{re.escape(year)}\b[.,]?\s*(.+?)(?:\.\s|$)", normalized)
        if year_match and not looks_like_venue_segment(year_match.group(1)):
            candidates.append(year_match.group(1))
    else:
        year_match = PAREN_YEAR_RE.search(normalized) or YEAR_RE.search(normalized)
        if year_match:
            remainder = normalized[year_match.end() :].lstrip(" .,:;()[]")
            if remainder:
                candidates.append(remainder.split(". ", 1)[0])

    return candidates


def _extract_title_from_reference_signature(reference_id: str) -> str:
    parts = str(reference_id or "").split("|")
    if len(parts) < 3:
        return ""
    title_part = normalize_reference_text(parts[2].replace("-", " "))
    if not title_part or title_part.startswith("doi "):
        return ""
    return title_part
