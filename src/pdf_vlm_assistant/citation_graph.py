from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .paper_graph import build_node_metadata_text, count_alpha_tokens, normalize_bbox, normalize_text, page_span, tokenize_for_similarity, to_int, unique_preserve_order, update_section_stack


REFERENCE_SECTION_TOKENS = (
    "references",
    "bibliography",
    "works cited",
    "literature cited",
    "key references",
)
REFERENCE_SUBTYPES = {"ref_text", "reference"}
REFERENCE_ITEM_TYPES = {"text", "paragraph", "list", "list_item"}
REFERENCE_LABEL_RE = re.compile(r"^\s*(?:\[(?P<bracket>\d+[A-Za-z]?)\]|(?P<num>\d+[A-Za-z]?)[.)])\s*")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
PMID_RE = re.compile(r"\bPMID\s*[:：]?\s*(\d+)\b", re.IGNORECASE)
QUOTED_TITLE_RE = re.compile(r"[\"“”'](?P<title>[^\"“”']{8,300})[\"“”']")
REFERENCE_START_RE = re.compile(r"^(?:\[\d+[A-Za-z]?\]|\d+[A-Za-z]?[.)])\s+")
NON_CITATION_PREFIXES = (
    "disclaimer",
    "publisher's note",
    "publisher’s note",
    "copyright",
)
NON_CITATION_CONTAINS = (
    "licensee mdpi",
    "creative commons",
    "all publications are solely",
    "the statements, opinions and data contained",
)
DOI_PREFIX_RE = re.compile(r"10\.\d{4,9}/", re.IGNORECASE)
DOI_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._;()/:")
GENERIC_DOI_SUFFIXES = {
    "journal",
    "pnas",
    "physr",
    "ryai",
    "saso",
}
DOI_TRAILING_PROSE_PREFIXES = (
    "accessed",
    "available",
    "copyright",
    "posted",
    "preprint",
    "published",
    "retrieved",
    "thisversion",
    "versionposted",
)


def build_citation_nodes_for_paper(
    paper_dir: Path,
    items: list[dict[str, Any]],
    paper_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    paper_id = paper_dir.name
    paper_title = paper_metadata["paper_title"]
    nodes: list[dict[str, Any]] = []
    section_stack: list[str] = []
    seen_entries: set[str] = set()
    citation_index = 0

    for item_index, item in enumerate(items):
        raw_text = extract_reference_source_text(item)
        if raw_text and is_section_heading_like(item, raw_text):
            update_section_stack(section_stack, raw_text, to_int(item.get("text_level")))
            continue

        if not is_reference_item(item, section_stack):
            continue

        section_path = list(section_stack) or ["References"]
        section_title = section_path[-1] if section_path else "References"
        section_depth = len(section_path) if section_path else 1
        item_type = str(item.get("type") or "text")
        sub_type = str(item.get("sub_type") or "")
        page_idx = to_int(item.get("page_idx") or item.get("page_no") or item.get("page"))
        bbox = normalize_bbox(item.get("bbox"))

        for citation_text in extract_reference_entries_from_item(item):
            label, cleaned_text = strip_reference_label(citation_text)
            normalized_citation = normalize_text(cleaned_text)
            if not looks_like_citation_text(normalized_citation):
                continue
            dedupe_key = build_reference_dedupe_key(
                doi=extract_doi_from_text(normalized_citation),
                title="",
                fallback_text=normalized_citation,
            )
            if dedupe_key in seen_entries:
                continue
            seen_entries.add(dedupe_key)

            fields = parse_citation_fields(normalized_citation)
            metadata_text = build_node_metadata_text(
                paper_title=paper_title,
                authors_text=paper_metadata["paper_author_text"],
                institution_text=paper_metadata["paper_institution_text"],
                section_title=section_title,
                section_group="references",
            )
            citation_work_text = build_citation_work_text(fields, normalized_citation)
            search_text = join_unique_nonempty(
                [
                    metadata_text,
                    f"cited authors {fields['citation_author_text']}" if fields["citation_author_text"] else "",
                    f"cited title {fields['citation_title']}" if fields["citation_title"] else "",
                    f"cited year {fields['citation_year']}" if fields["citation_year"] else "",
                    f"cited doi {fields['citation_doi']}" if fields["citation_doi"] else "",
                    f"cited pmid {fields['citation_pmid']}" if fields["citation_pmid"] else "",
                    citation_work_text,
                    normalized_citation,
                ]
            )
            citation_index += 1
            nodes.append(
                {
                    "node_id": f"{paper_id}:citation:{citation_index:04d}",
                    "paper_id": paper_id,
                    "paper_dir": str(paper_dir),
                    "paper_title": paper_title,
                    "paper_authors": paper_metadata["paper_authors"],
                    "paper_author_text": paper_metadata["paper_author_text"],
                    "paper_institutions": paper_metadata["paper_institutions"],
                    "paper_institution_text": paper_metadata["paper_institution_text"],
                    "paper_profile_text": paper_metadata["paper_profile_text"],
                    "order_in_paper": citation_index,
                    "modality": "text",
                    "node_kind": "citation",
                    "source_item_type": item_type,
                    "source_sub_type": sub_type,
                    "section_path": section_path,
                    "section_group": "references",
                    "section_depth": section_depth,
                    "section_title": section_title,
                    "page_start": page_idx,
                    "page_end": page_idx,
                    "page_span": page_span(page_idx, page_idx),
                    "page_indices": [page_idx] if page_idx is not None else [],
                    "bbox": bbox,
                    "text": normalized_citation,
                    "text_char_count": len(normalized_citation),
                    "feature_text": citation_work_text or normalized_citation,
                    "metadata_text": metadata_text,
                    "cross_paper_feature_text": citation_work_text or normalized_citation,
                    "search_text": search_text,
                    "image_path": None,
                    "caption": [],
                    "footnote": [],
                    "reference_labels": [label] if label else [],
                    "source_item_indices": [item_index],
                    "source_item_types": [item_type],
                    "citation_raw_text": normalize_text(citation_text),
                    "citation_label": label,
                    "citation_entry_index": citation_index,
                    "citation_authors": fields["citation_authors"],
                    "citation_author_text": fields["citation_author_text"],
                    "citation_lead_author": fields["citation_lead_author"],
                    "citation_title": fields["citation_title"],
                    "citation_year": fields["citation_year"],
                    "citation_doi": fields["citation_doi"],
                    "citation_pmid": fields["citation_pmid"],
                    "citation_venue": fields["citation_venue"],
                    "citation_title_key": fields["citation_title_key"],
                    "citation_match_key": fields["citation_match_key"],
                    "citation_work_text": citation_work_text,
                }
            )
    return nodes


def build_reference_dedupe_key(doi: str, title: str, fallback_text: str) -> str:
    normalized_doi = normalize_doi(doi)
    if normalized_doi and not is_suspicious_doi(normalized_doi):
        return f"doi:{normalized_doi}"
    title_key = build_title_key(title)
    if title_key:
        return f"title:{title_key}"
    return normalize_text(fallback_text).lower()


def extract_doi_from_text(text: str) -> str:
    match = DOI_RE.search(text or "")
    if not match:
        return ""
    return normalize_doi(match.group(0))


def extract_reference_entries_from_item(item: dict[str, Any]) -> list[str]:
    list_items = item.get("list_items")
    if isinstance(list_items, list) and list_items:
        entries: list[str] = []
        for list_item in list_items:
            text = normalize_text(flatten_text(list_item))
            if not text:
                continue
            entries.extend(split_reference_entries(text))
        return unique_preserve_order([entry for entry in entries if entry])

    raw_text = extract_reference_source_text(item)
    if not raw_text:
        return []
    return unique_preserve_order(split_reference_entries(raw_text))


def split_reference_entries(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    lines = [normalize_text(line) for line in normalized.splitlines() if normalize_text(line)]
    if not lines:
        return []
    if len(lines) == 1:
        return [lines[0]]

    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and REFERENCE_START_RE.match(line):
            entries.append(normalize_text(" ".join(current)))
            current = [line]
            continue
        current.append(line)
    if current:
        entries.append(normalize_text(" ".join(current)))
    return [entry for entry in entries if entry]


def strip_reference_label(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    match = REFERENCE_LABEL_RE.match(normalized)
    if not match:
        return "", normalized
    label = match.group("bracket") or match.group("num") or ""
    return label, normalize_text(normalized[match.end() :])


def looks_like_citation_text(text: str) -> bool:
    lowered = normalize_text(text).lower()
    if not lowered:
        return False
    if any(lowered.startswith(prefix) for prefix in NON_CITATION_PREFIXES):
        return False
    if any(fragment in lowered for fragment in NON_CITATION_CONTAINS):
        return False
    if count_alpha_tokens(lowered) < 4:
        return False
    if DOI_RE.search(lowered):
        return True
    if QUOTED_TITLE_RE.search(text) and YEAR_RE.search(text):
        return True
    return bool(YEAR_RE.search(text) and ("," in text or "." in text))


def parse_citation_fields(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    doi = normalize_doi(normalized)
    pmid = normalize_pmid(normalized)
    year_match = YEAR_RE.search(normalized)
    year = year_match.group(0) if year_match else ""
    author_text = extract_citation_author_text(normalized, year_match)
    title = extract_citation_title(normalized, year_match)
    venue = extract_citation_venue(normalized, title, year_match)
    authors = split_citation_authors(author_text)
    lead_author = first_author_surname(author_text)
    title_key = build_title_key(title or normalized)
    match_key = build_citation_match_key(
        doi=doi,
        lead_author=lead_author,
        year=year,
        title_key=title_key,
    )
    return {
        "citation_authors": authors,
        "citation_author_text": author_text,
        "citation_lead_author": lead_author,
        "citation_title": title,
        "citation_year": year,
        "citation_doi": doi,
        "citation_pmid": pmid,
        "citation_venue": venue,
        "citation_title_key": title_key,
        "citation_match_key": match_key,
    }


def build_citation_work_text(fields: dict[str, Any], fallback_text: str) -> str:
    return join_unique_nonempty(
        [
            fields.get("citation_title") or "",
            f"authors {fields.get('citation_author_text')}" if fields.get("citation_author_text") else "",
            f"year {fields.get('citation_year')}" if fields.get("citation_year") else "",
            f"doi {fields.get('citation_doi')}" if fields.get("citation_doi") else "",
            f"pmid {fields.get('citation_pmid')}" if fields.get("citation_pmid") else "",
            fields.get("citation_venue") or "",
            fallback_text,
        ]
    )


def build_citation_match_key(
    doi: str,
    lead_author: str,
    year: str,
    title_key: str,
) -> str:
    normalized_doi = normalize_doi(doi)
    if normalized_doi and not is_suspicious_doi(normalized_doi):
        return f"doi:{normalized_doi}"
    if title_key and (lead_author or year):
        return "|".join(part for part in [lead_author, year, title_key] if part)
    return title_key


def extract_citation_author_text(text: str, year_match: re.Match[str] | None) -> str:
    if year_match:
        candidate = normalize_text(text[: year_match.start()].rstrip(" ,.;:()"))
        if count_alpha_tokens(candidate) >= 2:
            return candidate
    quoted_match = QUOTED_TITLE_RE.search(text)
    if quoted_match:
        candidate = normalize_text(text[: quoted_match.start()].rstrip(" ,.;:()"))
        if count_alpha_tokens(candidate) >= 2:
            return candidate
    first_sentence = normalize_text(text.split(".", 1)[0])
    if count_alpha_tokens(first_sentence) >= 2:
        return first_sentence
    return ""


def extract_citation_title(text: str, year_match: re.Match[str] | None) -> str:
    quoted_match = QUOTED_TITLE_RE.search(text)
    if quoted_match:
        title = normalize_text(quoted_match.group("title"))
        if count_alpha_tokens(title) >= 3:
            return title

    working = text
    if year_match:
        working = text[year_match.end() :]
    working = normalize_text(working.lstrip(" .,:;()[]"))
    if not working:
        return ""

    first_sentence = normalize_text(working.split(".", 1)[0].strip(" \"“”'"))
    if count_alpha_tokens(first_sentence) >= 3:
        return first_sentence
    return ""


def extract_citation_venue(text: str, title: str, year_match: re.Match[str] | None) -> str:
    working = normalize_text(text)
    if title:
        title_index = working.find(title)
        if title_index >= 0:
            working = normalize_text(working[title_index + len(title) :].lstrip(" .,:;()"))
    elif year_match:
        working = normalize_text(working[year_match.end() :].lstrip(" .,:;()"))
    venue = normalize_text(working.split(".", 1)[0])
    if venue == title or count_alpha_tokens(venue) < 2:
        return ""
    return venue


def split_citation_authors(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    working = re.sub(r"\bet al\b\.?", "", normalized, flags=re.IGNORECASE)
    working = re.sub(r"\band\b", "|", working, flags=re.IGNORECASE)
    working = working.replace("&", "|").replace(";", "|")
    raw_parts = [part.strip(" ,") for part in working.split("|") if part.strip(" ,")]
    if len(raw_parts) == 1:
        raw_parts = [part.strip(" ,") for part in raw_parts[0].split(",") if part.strip(" ,")]
    authors: list[str] = []
    for part in raw_parts:
        candidate = normalize_text(part)
        if count_alpha_tokens(candidate) < 1:
            continue
        authors.append(candidate)
    return unique_preserve_order(authors)


def first_author_surname(text: str) -> str:
    authors = split_citation_authors(text)
    if not authors:
        return ""
    first_author = authors[0]
    tokens = [token.lower() for token in re.findall(r"[A-Za-z]+", first_author)]
    if not tokens:
        return ""
    if len(tokens[0]) == 1 and len(tokens) > 1:
        return tokens[-1]
    return tokens[0]


def build_title_key(text: str) -> str:
    tokens = tokenize_for_similarity(text)
    return "-".join(tokens[:12])


def normalize_doi(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    for match in DOI_PREFIX_RE.finditer(normalized):
        candidate = collect_doi_candidate(normalized, match.start())
        if candidate:
            return candidate
    return ""


def collect_doi_candidate(text: str, start_index: int) -> str:
    chars: list[str] = []
    index = start_index
    while index < len(text):
        char = text[index]
        if char in DOI_ALLOWED_CHARS:
            chars.append(char)
            index += 1
            continue
        if char.isspace():
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if (
                chars
                and next_index < len(text)
                and text[next_index] in DOI_ALLOWED_CHARS
                and chars[-1] in DOI_ALLOWED_CHARS
            ):
                index = next_index
                continue
        break
    candidate = "".join(chars).rstrip(".,;)").lower()
    return trim_doi_trailing_prose(candidate)


def trim_doi_trailing_prose(doi: str) -> str:
    normalized = normalize_text(doi).lower().rstrip(".,;)")
    if "/" not in normalized:
        return normalized

    prefix, suffix = normalized.split("/", 1)
    segments = suffix.split(";")
    kept_segments: list[str] = []
    for segment in segments:
        cleaned = re.sub(r"[^a-z0-9]+", "", segment)
        if cleaned and cleaned.startswith(DOI_TRAILING_PROSE_PREFIXES):
            break
        kept_segments.append(segment)
    trimmed_suffix = ";".join(part for part in kept_segments if part).strip(" .,:;()[]")
    return f"{prefix}/{trimmed_suffix}" if trimmed_suffix else normalized


def normalize_pmid(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    match = PMID_RE.search(normalized)
    if not match:
        return ""
    return match.group(1)


def is_suspicious_doi(doi: str) -> bool:
    normalized = normalize_text(doi).lower()
    if not normalized or "/" not in normalized:
        return True
    suffix = normalized.split("/", 1)[1].strip(" .,:;()[]")
    if not suffix:
        return True
    if suffix in GENERIC_DOI_SUFFIXES:
        return True
    if re.fullmatch(r"[a-z]+", suffix) and len(suffix) <= 8:
        return True
    return False


def is_reference_item(item: dict[str, Any], section_stack: list[str]) -> bool:
    item_type = str(item.get("type") or "").strip().lower()
    sub_type = str(item.get("sub_type") or "").strip().lower()
    if sub_type in REFERENCE_SUBTYPES:
        return True
    if item_type not in REFERENCE_ITEM_TYPES:
        return False
    section_title = normalize_text(section_stack[-1] if section_stack else "")
    return is_reference_section_title(section_title)


def is_reference_section_title(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(token in lowered for token in REFERENCE_SECTION_TOKENS)


def extract_reference_source_text(item: dict[str, Any]) -> str:
    return normalize_text(
        flatten_text(
            item.get("list_items")
            or item.get("text")
            or item.get("content")
        )
    )


def is_section_heading_like(item: dict[str, Any], text: str) -> bool:
    text_level = to_int(item.get("text_level"))
    if text_level is None or text_level <= 1:
        return False
    normalized = normalize_text(text)
    return bool(normalized and len(normalized) <= 220)


def join_unique_nonempty(parts: list[str]) -> str:
    seen: set[str] = set()
    values: list[str] = []
    for part in parts:
        normalized = normalize_text(str(part))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return "\n".join(values)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return "\n".join(flatten_text(item) for item in value.values() if item is not None)
    return str(value).replace("\x00", "")
