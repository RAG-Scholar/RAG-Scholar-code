from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



TEXT_ITEM_TYPES = {"text", "title", "paragraph", "list", "list_item"}
IMAGE_ITEM_TYPES = {"image", "chart", "table"}
DEFAULT_SECTION_NAME = "Front Matter"
LOW_SIGNAL_PATTERNS = (
    re.compile(r"^\s*https?://", re.IGNORECASE),
    re.compile(r"\bdoi\.org\b", re.IGNORECASE),
    re.compile(r"^\s*citation\s*:", re.IGNORECASE),
    re.compile(r"^\s*editor\s*:", re.IGNORECASE),
    re.compile(r"^\s*copyright\s*:", re.IGNORECASE),
    re.compile(r"^\s*academic editor\s*:", re.IGNORECASE),
    re.compile(r"^\s*(received|revised|accepted|published)\s*:", re.IGNORECASE),
    re.compile(r"^\s*funding\s*:", re.IGNORECASE),
    re.compile(r"^\s*competing interests\s*:", re.IGNORECASE),
    re.compile(r"^\s*(\* )?e-?mail\s*:", re.IGNORECASE),
    re.compile(r"^\s*correspondence\s*:", re.IGNORECASE),
    re.compile(r"^\s*o\s+check\s+for\s+updates\s*$", re.IGNORECASE),
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;。！？])\s+(?=[A-Z0-9\"'(\[])")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|\d+(?:\.\d+)?")
CAPTION_LABEL_RE = re.compile(
    r"^\s*((figure|fig\.?|table|chart)\s*\.?\s*\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
FRONT_MATTER_STOP_PREFIXES = (
    "abstract",
    "introduction",
    "background",
    "keywords",
    "correspondence",
    "received",
    "revised",
    "accepted",
    "published",
    "citation",
    "editor",
    "academic editor",
    "copyright",
    "funding",
    "competing interests",
    "availability of data",
)
INSTITUTION_STRONG_HINTS = (
    "university",
    "universit",
    "college",
    "institute",
    "institut",
    "hospital",
    "laboratory",
    "lab ",
    "centre",
    "center",
    "academy",
    "medical center",
    "research center",
    "polytechnic",
    "cnrs",
)
INSTITUTION_WEAK_HINTS = (
    "school",
    "department",
    "departments",
    "faculty",
    "clinic",
    "research",
    "clinical",
)
INSTITUTION_SENTENCE_PATTERNS = (
    " this study ",
    " this review ",
    " to evaluate ",
    " followed by ",
    " while ",
    " could provide ",
    " challenge ",
    " aims to ",
    " our analyses ",
    " we show ",
    " we present ",
)


@dataclass(slots=True)
class GraphBuildConfig:
    min_chars: int = 350
    target_chars: int = 700
    max_chars: int = 1100
    same_page_window: int = 2
    same_section_window: int = 3
    nearby_text_k: int = 2
    feature_dim: int = 512
    max_image_feature_chars: int = 1500
    enable_cross_paper_edges: bool = False
    cross_paper_title_top_k: int = 3
    cross_paper_title_min_similarity: float = 0.16
    cross_paper_title_top_terms: int = 10
    cross_paper_node_top_terms: int = 12
    cross_paper_text_top_k: int = 1
    cross_paper_text_min_similarity: float = 0.18
    cross_paper_text_max_pair_edges: int = 20
    cross_paper_text_section_mismatch_penalty: float = 0.08
    cross_paper_text_min_chars: int = 120
    cross_paper_visual_top_k: int = 1
    cross_paper_visual_min_similarity: float = 0.16
    cross_paper_visual_max_pair_edges: int = 8
    cross_paper_visual_min_chars: int = 20
    cross_paper_min_shared_terms: int = 2


@dataclass(slots=True)
class ChunkBuffer:
    section_path: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    page_indices: list[int] = field(default_factory=list)
    source_item_indices: list[int] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)

    def clear(self) -> None:
        self.section_path.clear()
        self.paragraphs.clear()
        self.page_indices.clear()
        self.source_item_indices.clear()
        self.source_types.clear()
        self.bboxes.clear()

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs).strip()

    @property
    def char_count(self) -> int:
        return len(self.text)


def discover_paper_dirs(segments_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in segments_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def append_text_piece(
    buffer: ChunkBuffer,
    section_stack: list[str],
    page_idx: int | None,
    item_index: int,
    item_type: str,
    bbox: list[float] | None,
    piece: str,
    config: GraphBuildConfig,
    flush_buffer: Any,
) -> None:
    piece = normalize_text(piece)
    if not piece:
        return

    if buffer.paragraphs and buffer.section_path != (section_stack or [DEFAULT_SECTION_NAME]):
        flush_buffer()

    if (
        buffer.paragraphs
        and page_idx is not None
        and buffer.page_indices
        and buffer.page_indices[-1] != page_idx
        and buffer.char_count >= config.min_chars
    ):
        flush_buffer()

    if buffer.paragraphs and buffer.char_count + len(piece) + 2 > config.max_chars:
        flush_buffer()

    if not buffer.paragraphs:
        buffer.section_path = list(section_stack) or [DEFAULT_SECTION_NAME]

    buffer.paragraphs.append(piece)
    if page_idx is not None:
        buffer.page_indices.append(page_idx)
    buffer.source_item_indices.append(item_index)
    buffer.source_types.append(item_type)
    if bbox:
        buffer.bboxes.append(bbox)

    if buffer.char_count >= config.target_chars:
        flush_buffer()


def tokenize_for_similarity(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) < 3:
            continue
        tokens.append(token)
    return tokens


def classify_section_path(section_path: list[str]) -> str:
    section_title = normalize_text(section_path[-1] if section_path else DEFAULT_SECTION_NAME).lower()
    if section_title == DEFAULT_SECTION_NAME.lower():
        return "front_matter"
    if "abstract" in section_title:
        return "abstract"
    if "introduction" in section_title or "background" in section_title:
        return "introduction"
    if any(token in section_title for token in ("method", "materials", "patients", "cohort", "study design")):
        return "methods"
    if "result" in section_title or "finding" in section_title:
        return "results"
    if "discussion" in section_title:
        return "discussion"
    if any(token in section_title for token in ("conclusion", "summary", "final remarks")):
        return "conclusion"
    return "other"


def normalize_content_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("content_list", "data", "items", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Unsupported MinerU content_list JSON structure.")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def extract_paper_title(items: list[dict[str, Any]]) -> str | None:
    for item in items:
        if to_int(item.get("text_level")) == 1:
            title = normalize_text(extract_item_text(item))
            if title:
                return title
    for item in items:
        text = normalize_text(extract_item_text(item))
        if text and len(text) > 20:
            return text
    return None


def extract_paper_metadata(
    items: list[dict[str, Any]],
    paper_id: str,
) -> dict[str, Any]:
    paper_title = extract_paper_title(items) or paper_id
    title_index = find_title_index(items, paper_title)
    front_matter_lines = collect_front_matter_lines(items, title_index)
    authors_text = extract_authors_text(front_matter_lines)
    paper_authors = split_author_text(authors_text)
    paper_institutions = extract_institution_lines(front_matter_lines, authors_text)
    institution_text = "; ".join(paper_institutions)
    paper_profile_text = build_paper_profile_text(
        paper_title=paper_title,
        authors_text=authors_text,
        institution_text=institution_text,
    )
    metadata = {
        "paper_title": paper_title,
        "paper_authors": paper_authors,
        "paper_author_text": authors_text,
        "paper_institutions": paper_institutions,
        "paper_institution_text": institution_text,
        "paper_profile_text": paper_profile_text,
        "metadata_sources": ["mineru"],
    }
    metadata["paper_profile_text"] = build_paper_profile_text(
        paper_title=metadata["paper_title"],
        authors_text=metadata.get("paper_author_text") or "",
        institution_text=metadata.get("paper_institution_text") or "",
    )
    return metadata


def find_title_index(items: list[dict[str, Any]], paper_title: str) -> int:
    normalized_title = normalize_text(paper_title)
    for index, item in enumerate(items):
        text = normalize_text(extract_item_text(item))
        if text and text == normalized_title:
            return index
    return -1


def collect_front_matter_lines(items: list[dict[str, Any]], title_index: int) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items):
        if title_index >= 0 and index <= title_index:
            continue
        item_type = str(item.get("type") or "")
        if item_type not in TEXT_ITEM_TYPES:
            continue
        text = normalize_text(extract_item_text(item))
        if not text:
            continue
        text_level = to_int(item.get("text_level"))
        if text_level is not None and text_level >= 2:
            break
        if is_front_matter_stop_line(text):
            continue
        lines.append(text)
        if len(lines) >= 8:
            break
    return lines


def is_front_matter_stop_line(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(lowered.startswith(prefix) for prefix in FRONT_MATTER_STOP_PREFIXES)


def extract_authors_text(lines: list[str]) -> str:
    for line in lines:
        if looks_like_institution_line(line):
            continue
        lowered = line.lower()
        if any(lowered.startswith(prefix) for prefix in FRONT_MATTER_STOP_PREFIXES):
            continue
        if len(line) > 260:
            continue
        if looks_like_author_line(line):
            return line
    return ""


def looks_like_author_line(text: str) -> bool:
    cleaned = re.sub(r"[\d*†‡§¶]+", " ", normalize_text(text))
    if not cleaned:
        return False
    if looks_like_institution_line(cleaned):
        return False
    if "|" in cleaned or ";" in cleaned:
        return True
    tokens = [token for token in re.split(r"\s+", cleaned) if token]
    capitalized = sum(1 for token in tokens if token[:1].isupper())
    return capitalized >= 4 and len(tokens) <= 28


def strip_affiliation_leader(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    match = re.match(
        r"^\s*((?:\d+|[A-Za-z])(?:\s*,\s*(?:\d+|[A-Za-z]))*)\s+(.*)$",
        normalized,
    )
    if not match:
        return normalized
    rest = normalize_text(match.group(2))
    lowered_rest = rest.lower()
    if any(hint in lowered_rest for hint in INSTITUTION_STRONG_HINTS + INSTITUTION_WEAK_HINTS):
        return rest
    return normalized


def has_institution_strong_hint(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(hint in lowered for hint in INSTITUTION_STRONG_HINTS)


def has_institution_weak_hint(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(hint in lowered for hint in INSTITUTION_WEAK_HINTS)


def is_sentence_like_affiliation(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    lowered = f" {normalized.lower()} "
    if len(normalized) > 220:
        return True
    if any(marker in lowered for marker in (" http://", " https://", " doi ", " figure ", " table ")):
        return True
    if any(pattern in lowered for pattern in INSTITUTION_SENTENCE_PATTERNS):
        return True
    if normalized.count(".") >= 3:
        return True
    if re.search(
        r"\b(?:aims?|provide[sd]?|demonstrat(?:e|es|ed)|improv(?:e|es|ed)|evaluat(?:e|ed|es)|challenge(?:s|d)?|offer(?:s|ed)?)\b",
        lowered,
    ):
        return True
    return False


def looks_like_institution_line(text: str) -> bool:
    normalized = strip_affiliation_leader(text)
    lowered = normalized.lower()
    if not lowered:
        return False
    if any(lowered.startswith(prefix) for prefix in FRONT_MATTER_STOP_PREFIXES):
        return False
    if is_sentence_like_affiliation(normalized) and not has_institution_strong_hint(normalized):
        return False
    if has_institution_strong_hint(normalized):
        return True
    if has_institution_weak_hint(normalized):
        return count_alpha_tokens(normalized) <= 16 or "," in normalized or "(" in normalized
    return False


def split_author_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    working = re.sub(r"\s*\([^)]*@[^)]*\)\s*", " ", normalized)
    working = working.replace("\\", " ")
    working = re.sub(r"\band\b", "|", working, flags=re.IGNORECASE)
    if "|" in working:
        raw_parts = working.split("|")
    elif ";" in working:
        raw_parts = working.split(";")
    else:
        raw_parts = working.split(",")

    authors: list[str] = []
    for part in raw_parts:
        candidate = re.sub(r"[\d*†‡§¶]+", " ", part)
        candidate = normalize_text(candidate)
        if count_alpha_tokens(candidate) < 2:
            continue
        authors.append(candidate)
    return unique_preserve_order(authors)


def extract_institution_lines(lines: list[str], authors_text: str) -> list[str]:
    institutions: list[str] = []
    author_line_seen = not authors_text
    collecting = False
    for line in lines:
        if authors_text and not author_line_seen:
            if normalize_text(line) == normalize_text(authors_text):
                author_line_seen = True
            continue
        if not looks_like_institution_line(line):
            if collecting:
                break
            continue
        institutions.append(strip_affiliation_leader(line))
        collecting = True
    return unique_preserve_order(institutions)


def build_paper_profile_text(
    paper_title: str,
    authors_text: str,
    institution_text: str,
) -> str:
    return join_unique_nonempty(
        [
            paper_title,
            f"authors {authors_text}" if authors_text else "",
            f"institutions {institution_text}" if institution_text else "",
        ]
    )


def build_node_metadata_text(
    paper_title: str,
    authors_text: str,
    institution_text: str,
    section_title: str,
    section_group: str,
) -> str:
    return join_unique_nonempty(
        [
            f"title {paper_title}",
            f"authors {authors_text}" if authors_text else "",
            f"institutions {institution_text}" if institution_text else "",
            f"section {section_title}" if section_title else "",
            f"section group {section_group}" if section_group else "",
        ]
    )


def build_node_search_text(metadata_text: str, content_text: str) -> str:
    return join_unique_nonempty([metadata_text, content_text])


def join_unique_nonempty(parts: list[str]) -> str:
    unique_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = normalize_text(part)
        if not value or value in seen:
            continue
        unique_parts.append(value)
        seen.add(value)
    return "\n".join(unique_parts)


def count_alpha_tokens(text: str) -> int:
    return sum(1 for token in re.split(r"\s+", normalize_text(text)) if re.search(r"[A-Za-z]", token))


def extract_item_text(item: dict[str, Any]) -> str:
    return flatten_text(item.get("text") or item.get("content")).strip()


def build_image_feature_text(item: dict[str, Any], max_chars: int) -> str:
    parts = []
    parts.extend(extract_caption_list(item))
    parts.extend(extract_footnote_list(item))
    if item.get("type") == "table":
        table_body = strip_html(flatten_text(item.get("table_body")))
        if table_body:
            parts.append(table_body)
    text = normalize_text("\n".join(part for part in parts if part).strip())
    return text[:max_chars]


def extract_caption_list(item: dict[str, Any]) -> list[str]:
    return to_text_list(
        item.get("image_caption")
        or item.get("chart_caption")
        or item.get("table_caption")
    )


def extract_footnote_list(item: dict[str, Any]) -> list[str]:
    return to_text_list(
        item.get("image_footnote")
        or item.get("chart_footnote")
        or item.get("table_footnote")
    )


def extract_reference_labels_from_item(item: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for caption in extract_caption_list(item):
        match = CAPTION_LABEL_RE.match(caption)
        if not match:
            continue
        base_label = normalize_reference_label(match.group(1))
        if not base_label:
            continue
        labels.append(base_label)
        if base_label.startswith("figure "):
            labels.append(base_label.replace("figure ", "fig "))
    return unique_preserve_order(labels)


def normalize_reference_label(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    text = text.replace("fig.", "fig").replace("figure.", "figure")
    text = re.sub(r"^fig\s+", "figure ", text)
    text = re.sub(r"^chart\s+", "figure ", text)
    return text


def is_section_heading(item: dict[str, Any], text: str) -> bool:
    text_level = to_int(item.get("text_level"))
    if text_level is None:
        return False
    text = normalize_text(text)
    if not text:
        return False
    if is_low_signal_text(text):
        return False
    return len(text) <= 220


def update_section_stack(section_stack: list[str], heading_text: str, text_level: int | None) -> None:
    if text_level is None:
        return
    normalized_heading = normalize_text(heading_text)
    if not normalized_heading:
        return
    if text_level <= 1:
        return
    depth = max(1, text_level - 1)
    del section_stack[depth - 1 :]
    section_stack.append(normalized_heading)


def split_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    chunks = re.split(r"\n\s*\n+", text)
    paragraphs: list[str] = []
    for chunk in chunks:
        normalized = re.sub(r"[ \t]+", " ", chunk).strip()
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        if normalized:
            paragraphs.append(normalized)
    return paragraphs


def split_long_text(text: str, max_chars: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
    if len(sentences) <= 1:
        return hard_split_text(text, max_chars)

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_length = 0
            chunks.extend(hard_split_text(sentence, max_chars))
            continue

        added_length = len(sentence) + (1 if current else 0)
        if current and current_length + added_length > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_length = len(sentence)
            continue

        current.append(sentence)
        current_length += added_length

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def hard_split_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        added_length = len(word) + (1 if current else 0)
        if current and current_length + added_length > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
            current_length = len(word)
            continue
        current.append(word)
        current_length += added_length

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def resolve_image_path(raw_path: str, parsed_root: Path) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    direct = (parsed_root / candidate).resolve()
    if direct.exists():
        return direct
    nested = (parsed_root / "images" / candidate.name).resolve()
    if nested.exists():
        return nested
    return direct


def _split_portable_path(value: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", value) if part]


def _resolve_manifest_content_list(root: Path, manifest: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    content_list_value = str(manifest.get("content_list_path") or "").strip()
    if content_list_value:
        content_list_path = Path(content_list_value).expanduser()
        candidates.append(content_list_path)
        parts = _split_portable_path(content_list_value)
        if ".mineru_work" in parts:
            marker = parts.index(".mineru_work")
            suffix = parts[marker + 1 :]
            for ancestor in [root, *root.parents]:
                candidates.append((ancestor / ".mineru_work").joinpath(*suffix))

    mineru_raw_dir = str(manifest.get("mineru_raw_dir") or "").strip()
    if mineru_raw_dir:
        raw_parts = _split_portable_path(mineru_raw_dir)
        raw_hash = raw_parts[-1] if raw_parts else ""
        if raw_hash:
            for ancestor in [root, *root.parents]:
                auto_dir = ancestor / ".mineru_work" / raw_hash / "input" / "auto"
                candidates.append(auto_dir / "input_content_list.json")
                candidates.append(auto_dir / "input_content_list_v2.json")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def find_content_list(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*_content_list.json"))
    if not candidates:
        candidates = sorted(root.rglob("*content_list*.json"))
    if candidates:
        return candidates[0]

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return _resolve_manifest_content_list(root, manifest)


def load_segment_items(root: Path) -> tuple[list[dict[str, Any]], Path, Path | None]:
    content_list_path = find_content_list(root)
    if content_list_path is not None and content_list_path.exists():
        payload = json.loads(content_list_path.read_text(encoding="utf-8"))
        return normalize_content_payload(payload), content_list_path.parent, content_list_path

    paragraphs_path = root / "paragraphs.jsonl"
    images_path = root / "images.jsonl"
    text_items = load_jsonl_records(paragraphs_path)
    image_items = load_jsonl_records(images_path)
    if not text_items and not image_items:
        raise FileNotFoundError("content_list not found, and no paragraphs/images fallback found")

    items: list[dict[str, Any]] = []
    for record in text_items + image_items:
        item = dict(record)
        image_filename = str(item.get("image_filename") or "").strip()
        if image_filename:
            item["img_path"] = str(Path("images") / image_filename)
        elif item.get("image_path"):
            raw_image_path = str(item.get("image_path") or "").strip()
            image_name = Path(raw_image_path).name
            if image_name:
                portable_candidate = root / "images" / image_name
                item["img_path"] = str(Path("images") / image_name) if portable_candidate.exists() else raw_image_path
        items.append(item)

    items.sort(
        key=lambda row: (
            to_int(row.get("item_index")),
            to_int(row.get("block_id")),
            to_int(row.get("page_idx") or row.get("page_no") or row.get("page")),
        )
    )
    source_path = paragraphs_path if paragraphs_path.exists() else images_path if images_path.exists() else None
    return items, root, source_path


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


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_text_list(value: Any) -> list[str]:
    text = flatten_text(value).strip()
    if not text:
        return []
    return [normalize_text(line) for line in text.splitlines() if normalize_text(line)]


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    bbox: list[float] = []
    for part in value:
        try:
            bbox.append(float(part))
        except (TypeError, ValueError):
            return None
    return bbox


def merge_bboxes(bboxes: list[list[float]]) -> list[float] | None:
    valid = [bbox for bbox in bboxes if bbox and len(bbox) == 4]
    if not valid:
        return None
    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]


def page_range(page_indices: list[int]) -> tuple[int | None, int | None]:
    valid = [page for page in page_indices if page is not None]
    if not valid:
        return None, None
    return min(valid), max(valid)


def page_span(page_start: int | None, page_end: int | None) -> int:
    if page_start is None or page_end is None:
        return 0
    return page_end - page_start + 1


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def unique_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    unique_values: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def is_low_signal_text(text: str) -> bool:
    if not text:
        return True
    normalized = normalize_text(text)
    if not normalized:
        return True
    if len(normalized) <= 1:
        return True
    return any(pattern.search(normalized) for pattern in LOW_SIGNAL_PATTERNS)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
