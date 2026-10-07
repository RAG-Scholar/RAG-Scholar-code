from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .document_entity_retrieval import (
    DocumentEntityRetriever,
    DocumentEntityRetrieverConfig,
    detect_query_entity_types,
    is_visual_query,
)

TOKEN_ESTIMATE_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")
STRUCTURED_ENTITY_TYPES = {"author", "institution", "reference", "venue"}
SEMANTIC_ENTITY_TYPES = {"method", "dataset", "task", "metric", "model"}
@dataclass(slots=True)
class QueryContextBundleConfig:
    graph_dir: Path = Path("outputs") / "graph"
    retrieval_mode: str = "hierarchical"
    enable_cross_paper_edges: bool = True
    top_k_papers: int = 5
    top_k_sections: int = 2
    top_k_chunks: int = 2
    top_k_figures: int = 1
    top_k_entities: int = 5
    top_k_cross_paper: int = 2
    max_total_chars: int = 6500
    max_total_tokens: int | None = None
    preview_chars: int = 340
    section_summary_chars: int = 220
    chunk_chars: int = 1100
    figure_chars: int = 1100
    include_cross_paper_neighbors: bool = True
    include_figures: bool = True
    retrieval_backend: str = "auto"
    gnn_index_dir: Path | None = None
    gnn_device: str = "cpu"
    gnn_encoder_model: str | None = None
    hybrid_gnn_weight: float = 0.3
    hybrid_candidate_pool: int = 20


def build_query_context_bundle(
    query: str,
    config: QueryContextBundleConfig,
    retriever: DocumentEntityRetriever | None = None,
) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("query must not be empty")

    retriever = retriever or DocumentEntityRetriever(
        DocumentEntityRetrieverConfig(
            graph_dir=config.graph_dir,
            retrieval_backend=config.retrieval_backend,
            gnn_index_dir=config.gnn_index_dir,
            gnn_device=config.gnn_device,
            gnn_encoder_model=config.gnn_encoder_model,
            hybrid_gnn_weight=config.hybrid_gnn_weight,
            hybrid_candidate_pool=config.hybrid_candidate_pool,
            top_k_entities=config.top_k_entities,
            enable_cross_paper_edges=config.enable_cross_paper_edges,
            top_k_papers=config.top_k_papers,
            top_k_sections=config.top_k_sections,
            top_k_chunks=config.top_k_chunks,
            top_k_figures=config.top_k_figures,
            preview_chars=max(
                int(config.preview_chars),
                int(config.section_summary_chars),
                int(config.chunk_chars),
                int(config.figure_chars),
            ),
        )
    )
    retrieval_payload = retriever.retrieve(
        query=query,
        top_k_papers=config.top_k_papers,
        top_k_sections=config.top_k_sections,
        top_k_chunks=config.top_k_chunks,
        top_k_figures=config.top_k_figures,
        retrieval_mode=config.retrieval_mode,
    )
    for paper in retrieval_payload.get("results", []):
        for section in paper.get("top_sections", []):
            for chunk in section.get("top_chunks", []):
                source = retriever.chunk_by_node_id.get(chunk["node_id"], {})
                chunk["text"] = source.get("text") or chunk.get("preview") or ""

    query_mode = str(retrieval_payload.get("query_mode") or "ranked")
    selection_mode = str(retrieval_payload.get("selection_mode") or "")
    retrieval_mode = str(retrieval_payload.get("retrieval_mode") or config.retrieval_mode or "hierarchical")
    exhaustive_mode = query_mode == "exhaustive_list"

    # Build all bounded local candidates before applying the serialized budget.
    remaining_chars = math.inf
    selected_papers: list[dict[str, Any]] = []
    selected_figure_paths: list[str] = []
    for paper_rank, paper in enumerate(retrieval_payload.get("results") or [], start=1):
        paper_item, consumed_chars = compact_paper_context(
            paper=paper,
            paper_rank=paper_rank,
            config=config,
            remaining_chars=remaining_chars,
            query=query,
            roster_only=exhaustive_mode,
        )
        if not paper_item:
            continue
        selected_papers.append(paper_item)
        remaining_chars = max(0, remaining_chars - consumed_chars)
        for figure_path in paper_item.get("selected_figure_paths") or []:
            if figure_path not in selected_figure_paths:
                selected_figure_paths.append(figure_path)
        if not exhaustive_mode and remaining_chars <= 320:
            break

    selected_papers = enforce_bundle_budget(
        selected_papers, int(config.max_total_chars), query,
        max_total_tokens=config.max_total_tokens, preserve_paper_count=exhaustive_mode,
    )
    refresh_selected_figure_paths(selected_papers)
    markdown = render_context_bundle_markdown(query=query, papers=selected_papers)
    selected_figure_paths = collect_selected_figure_paths(selected_papers)
    summary = summarize_bundle(selected_papers, markdown)
    bundle = {
        "bundle_version": "query_context_bundle_v1",
        "query": query,
        "graph_dir": str(config.graph_dir.expanduser().resolve()),
        "retrieval_backend": retriever.retrieval_backend,
        "gnn_index_dir": str(retriever.config.gnn_index_dir) if retriever.config.gnn_index_dir else None,
        "hybrid": retriever.hybrid.metadata() if retriever.hybrid is not None else None,
        "retrieval_mode": retrieval_mode,
        "query_mode": query_mode,
        "selection_mode": selection_mode,
        "authoritative_candidate_ids": list(retrieval_payload.get("authoritative_candidate_ids") or []),
        "constraint_groups": list(retrieval_payload.get("constraint_groups") or []),
        "selection_config": {
                "retrieval_mode": retrieval_mode,
            "enable_cross_paper_edges": bool(config.enable_cross_paper_edges),
            "top_k_papers": int(config.top_k_papers),
            "top_k_sections": int(config.top_k_sections),
            "top_k_chunks": int(config.top_k_chunks),
            "top_k_figures": int(config.top_k_figures),
            "top_k_entities": int(config.top_k_entities),
            "top_k_cross_paper": int(config.top_k_cross_paper),
            "max_total_chars": int(config.max_total_chars),
            "max_total_tokens": (
                int(config.max_total_tokens)
                if config.max_total_tokens is not None
                else None
            ),
            "include_cross_paper_neighbors": bool(
                config.include_cross_paper_neighbors and config.enable_cross_paper_edges
            ),
            "include_figures": bool(config.include_figures),
        },
        "summary": summary,
        "selected_figure_paths": selected_figure_paths,
        "papers": selected_papers,
        "llm_context_markdown": markdown,
    }
    return bundle


def build_query_match_evidence(
    rows: list[dict[str, Any]],
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    aligned_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        entity_type = str(row.get("entity_type") or "")
        name = build_entity_label(row)
        matched_text = trim_text(str(row.get("matched_text") or ""), 120)
        key = (entity_type, name.casefold(), matched_text.casefold())
        if not name or key in seen_keys:
            continue
        seen_keys.add(key)
        item = clone_json_like(row)
        item["name"] = name
        item["matched_text"] = matched_text
        if entity_name_appears_in_query(name, query) or entity_name_appears_in_query(matched_text, query):
            aligned_rows.append(item)
        elif str(item.get("match_source") or "").strip().lower() == "exact":
            exact_rows.append(item)
    return (aligned_rows + exact_rows)[:limit]


def entity_name_appears_in_query(name: str, query: str) -> bool:
    normalized_name = normalize_query_text(name)
    normalized_query = normalize_query_text(query)
    return bool(normalized_name) and len(normalized_name) >= 4 and normalized_name in normalized_query


def normalize_query_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text or "").lower())


def compact_paper_context(
    paper: dict[str, Any],
    paper_rank: int,
    config: QueryContextBundleConfig,
    remaining_chars: int,
    query: str,
    roster_only: bool = False,
) -> tuple[dict[str, Any] | None, int]:
    if remaining_chars <= 0:
        return None, 0

    used_chars = 0
    paper_item: dict[str, Any] = {
        "paper_rank": int(paper_rank),
        "paper_id": str(paper.get("paper_id") or ""),
        "paper_title": str(paper.get("paper_title") or ""),
        "roster_only": bool(roster_only),
        "paper_author_text": trim_text(str(paper.get("paper_author_text") or ""), 120 if roster_only else 220),
        "paper_institution_text": trim_text(
            str(paper.get("paper_institution_text") or ""),
            120 if roster_only else 220,
        ),
        "rank_score": float(paper.get("rank_score") or 0.0),
        "why_selected": [],
        "query_match_evidence": [],
        "matched_entities": [],
        "cross_paper_neighbors": [],
        "sections": [],
        "selected_figure_paths": [],
    }
    used_chars += measure_text(paper_item["paper_title"])
    used_chars += measure_text(paper_item["paper_author_text"])
    used_chars += measure_text(paper_item["paper_institution_text"])

    why_selected = [
        trim_text(str(row.get("detail") or ""), 120 if roster_only else 220)
        for row in (paper.get("why_matched") or [])[: (1 if roster_only else 3)]
        if str(row.get("detail") or "").strip() and row.get("kind") != "hybrid_rank_fusion"
    ]
    paper_item["why_selected"] = why_selected
    used_chars += sum(measure_text(row) for row in why_selected)

    matched_entities = []
    entity_limit = min(3, int(config.top_k_entities)) if roster_only else int(config.top_k_entities)
    for row in (paper.get("matched_entities") or [])[:entity_limit]:
        label = build_entity_label(row)
        if not label:
            continue
        matched_entities.append(
            {
                "entity_type": str(row.get("entity_type") or ""),
                "name": label,
                "edge_type": str(row.get("edge_type") or ""),
                "match_source": str(row.get("match_source") or ""),
                "matched_text": trim_text(str(row.get("matched_text") or ""), 120),
                "score": float(row.get("score") or 0.0),
                "via_section_id": str(row.get("via_section_id") or ""),
            }
        )
    paper_item["matched_entities"] = matched_entities
    used_chars += sum(measure_text(row["name"]) for row in matched_entities)
    paper_item["query_match_evidence"] = build_query_match_evidence(
        rows=matched_entities,
        query=query,
        limit=2 if roster_only else min(4, max(2, int(config.top_k_entities))),
    )
    used_chars += sum(
        measure_text(row.get("name") or "") + measure_text(row.get("matched_text") or "")
        for row in (paper_item.get("query_match_evidence") or [])
    )

    if roster_only:
        exact_query_rows = [
            row for row in (paper_item.get("query_match_evidence") or [])
            if str(row.get("match_source") or "").strip().lower() == "exact"
        ]
        if exact_query_rows:
            paper_item["query_match_evidence"] = exact_query_rows[:1]
        elif paper_item.get("query_match_evidence"):
            paper_item["query_match_evidence"] = list(paper_item["query_match_evidence"][:1])
        else:
            paper_item["query_match_evidence"] = list((paper_item.get("matched_entities") or [])[:1])
        paper_item["matched_entities"] = list(paper_item.get("query_match_evidence") or [])
        kept_types = {
            str(row.get("entity_type") or "")
            for row in (paper_item.get("query_match_evidence") or [])
            if str(row.get("entity_type") or "")
        }
        if "author" not in kept_types:
            paper_item["paper_author_text"] = ""
        if "institution" not in kept_types:
            paper_item["paper_institution_text"] = ""
        paper_item["why_selected"] = []

    if config.enable_cross_paper_edges and config.include_cross_paper_neighbors and not roster_only:
        cross_rows = []
        for row in (paper.get("cross_paper_support") or [])[: config.top_k_cross_paper]:
            evidence_names = [trim_text(str(name or ""), 80) for name in (row.get("evidence_entity_names") or []) if str(name or "").strip()]
            cross_rows.append(
                {
                    "source_paper_id": str(row.get("source_paper_id") or ""),
                    "source_paper_title": trim_text(str(row.get("source_paper_title") or ""), 180),
                    "edge_type": str(row.get("edge_type") or ""),
                    "boost": float(row.get("boost") or 0.0),
                    "evidence_entity_names": evidence_names[:4],
                }
            )
        paper_item["cross_paper_neighbors"] = cross_rows
        used_chars += sum(
            measure_text(str(row.get("source_paper_title") or "")) + sum(measure_text(name) for name in row.get("evidence_entity_names") or [])
            for row in cross_rows
        )

    if not roster_only:
        for section in paper.get("top_sections") or []:
            if used_chars >= remaining_chars and paper_item["sections"]:
                break
            section_item, section_chars = compact_section_context(
                section=section,
                config=config,
                remaining_chars=max(0, remaining_chars - used_chars),
            )
            if not section_item:
                continue
            paper_item["sections"].append(section_item)
            paper_item["selected_figure_paths"].extend(section_item.pop("selected_figure_paths", []))
            used_chars += section_chars

    if not paper_item["sections"] and not paper_item["matched_entities"] and not paper_item["why_selected"]:
        return None, 0
    return paper_item, used_chars


def compact_section_context(
    section: dict[str, Any],
    config: QueryContextBundleConfig,
    remaining_chars: int,
) -> tuple[dict[str, Any] | None, int]:
    if remaining_chars <= 0:
        return None, 0

    used_chars = 0
    section_item: dict[str, Any] = {
        "section_id": str(section.get("section_id") or ""),
        "section_title": trim_text(str(section.get("section_title") or ""), 120),
        "section_group": str(section.get("section_group") or ""),
        "rank_score": float(section.get("rank_score") or 0.0),
        "section_summary": trim_text(str(section.get("section_summary") or ""), config.section_summary_chars),
        "section_entities": [],
        "chunks": [],
        "figures": [],
        "selected_figure_paths": [],
    }
    used_chars += measure_text(section_item["section_title"])
    used_chars += measure_text(section_item["section_summary"])

    section_entities = []
    for row in (section.get("matched_entities") or [])[:6]:
        label = build_entity_label(row)
        if not label:
            continue
        section_entities.append(label)
    section_item["section_entities"] = section_entities
    used_chars += sum(measure_text(row) for row in section_entities)

    for chunk in section.get("top_chunks") or []:
        if used_chars >= remaining_chars and section_item["chunks"]:
            break
        text = trim_text(str(chunk.get("text") or chunk.get("preview") or ""), config.chunk_chars)
        if not text:
            continue
        payload = {
            "node_id": str(chunk.get("node_id") or ""),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text": text,
            "rank_score": float(chunk.get("rank_score") or 0.0),
        }
        section_item["chunks"].append(payload)
        used_chars += measure_text(text)

    if config.include_figures:
        for figure in section.get("top_figures") or []:
            if used_chars >= remaining_chars and (section_item["figures"] or section_item["chunks"]):
                break
            caption_text = build_figure_text(figure, config.figure_chars)
            if not caption_text:
                continue
            payload = {
                "node_id": str(figure.get("node_id") or ""),
                "page_start": figure.get("page_start"),
                "page_end": figure.get("page_end"),
                "caption": caption_text,
                "rank_score": float(figure.get("rank_score") or 0.0),
                "image_path": str(figure.get("image_path") or ""),
            }
            section_item["figures"].append(payload)
            if payload["image_path"]:
                section_item["selected_figure_paths"].append(payload["image_path"])
            used_chars += measure_text(caption_text)

    if not section_item["section_summary"] and not section_item["chunks"] and not section_item["figures"]:
        return None, 0
    return section_item, used_chars


def summarize_bundle(papers: list[dict[str, Any]], markdown: str) -> dict[str, Any]:
    section_count = sum(len(paper.get("sections") or []) for paper in papers)
    chunk_count = sum(
        len(section.get("chunks") or [])
        for paper in papers
        for section in (paper.get("sections") or [])
    )
    figure_count = sum(
        len(section.get("figures") or [])
        for paper in papers
        for section in (paper.get("sections") or [])
    )
    char_count = len(markdown)
    return {
        "paper_count": len(papers),
        "section_count": section_count,
        "chunk_count": chunk_count,
        "figure_count": figure_count,
        "approx_char_count": char_count,
        "approx_token_count": estimate_token_count(markdown),
    }


def enforce_bundle_budget(
    papers: list[dict[str, Any]],
    max_total_chars: int,
    query: str,
    max_total_tokens: int | None = None,
    preserve_paper_count: bool = False,
) -> list[dict[str, Any]]:
    if max_total_chars <= 0 and (max_total_tokens is None or int(max_total_tokens) <= 0):
        return papers
    safe_papers = clone_json_like(papers)
    markdown = render_context_bundle_markdown(query=query, papers=safe_papers)
    while safe_papers and bundle_exceeds_budget(
        markdown=markdown,
        max_total_chars=max_total_chars,
        max_total_tokens=max_total_tokens,
    ):
        if not shrink_bundle_once(safe_papers, query, preserve_paper_count=preserve_paper_count):
            break
        markdown = render_context_bundle_markdown(query=query, papers=safe_papers)
    return safe_papers


def bundle_exceeds_budget(
    markdown: str,
    max_total_chars: int,
    max_total_tokens: int | None,
) -> bool:
    if max_total_chars > 0 and len(markdown) > max_total_chars:
        return True
    if max_total_tokens is not None and int(max_total_tokens) > 0:
        return estimate_token_count(markdown) > int(max_total_tokens)
    return False


def shrink_bundle_once(papers: list[dict[str, Any]], query: str, preserve_paper_count: bool = False) -> bool:
    visual_query = is_visual_query(query)
    for paper in reversed(papers):
        matched_entities = paper.get("matched_entities") or []
        query_evidence = paper.get("query_match_evidence") or []
        protected_keys = {build_entity_row_key(row) for row in query_evidence}
        for index in range(len(matched_entities) - 1, -1, -1):
            row = matched_entities[index]
            if build_entity_row_key(row) in protected_keys:
                continue
            if str(row.get("match_source") or "").strip().lower() == "exact":
                continue
            matched_entities.pop(index)
            return True

    for paper in reversed(papers):
        if paper.get("paper_institution_text"):
            paper["paper_institution_text"] = ""
            return True

    for paper in reversed(papers):
        if paper.get("paper_author_text"):
            paper["paper_author_text"] = ""
            return True

    for paper in reversed(papers):
        why_selected = paper.get("why_selected") or []
        if why_selected:
            why_selected.pop()
            return True

    for paper in reversed(papers):
        query_evidence = paper.get("query_match_evidence") or []
        if len(query_evidence) > 1:
            query_evidence.pop()
            return True

    for paper in reversed(papers):
        cross_neighbors = paper.get("cross_paper_neighbors") or []
        if cross_neighbors:
            cross_neighbors.pop()
            return True

    for paper in reversed(papers):
        for section in reversed(paper.get("sections") or []):
            chunks = section.get("chunks") or []
            if len(chunks) > 1:
                chunks.pop()
                return True

    for paper in reversed(papers):
        for section in reversed(paper.get("sections") or []):
            section_entities = section.get("section_entities") or []
            if section_entities:
                section_entities.pop()
                return True

    for paper in reversed(papers):
        for section in reversed(paper.get("sections") or []):
            if section.get("section_summary"):
                section["section_summary"] = ""
                return True

    for paper in reversed(papers):
        sections = paper.get("sections") or []
        for index in range(len(sections) - 1, -1, -1):
            section = sections[index]
            if (
                not (section.get("chunks") or [])
                and not (section.get("figures") or [])
                and not (section.get("section_entities") or [])
                and not str(section.get("section_summary") or "").strip()
            ):
                sections.pop(index)
                return True

    for paper in reversed(papers):
        sections = paper.get("sections") or []
        if len(sections) > 1:
            sections.pop()
            return True

    for paper in reversed(papers):
        for section in reversed(paper.get("sections") or []):
            chunks = section.get("chunks") or []
            if chunks:
                chunks.pop()
                return True

    if preserve_paper_count:
        return False

    if visual_query and len(papers) > 1:
        papers.pop()
        return True

    for paper in reversed(papers):
        for section in reversed(paper.get("sections") or []):
            figures = section.get("figures") or []
            if not figures:
                continue
            if visual_query and len(figures) == 1 and len(papers) == 1 and len(paper.get("sections") or []) == 1:
                continue
            figures.pop()
            return True

    if len(papers) > 1:
        papers.pop()
        return True
    return False


def collect_selected_figure_paths(papers: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for paper in papers:
        paper_paths: list[str] = []
        for section in paper.get("sections") or []:
            for image_path in section.get("selected_figure_paths") or []:
                image_path = str(image_path or "")
                if image_path and image_path not in paper_paths:
                    paper_paths.append(image_path)
            for figure in section.get("figures") or []:
                image_path = str(figure.get("image_path") or "")
                if image_path and image_path not in paper_paths:
                    paper_paths.append(image_path)
        if not paper_paths:
            for image_path in paper.get("selected_figure_paths") or []:
                image_path = str(image_path or "")
                if image_path and image_path not in paper_paths:
                    paper_paths.append(image_path)
        for image_path in paper_paths:
            if image_path not in paths:
                paths.append(image_path)
    return paths


def refresh_selected_figure_paths(papers: list[dict[str, Any]]) -> None:
    for paper in papers:
        for section in paper.get("sections") or []:
            section_paths: list[str] = []
            for figure in section.get("figures") or []:
                image_path = str(figure.get("image_path") or "")
                if image_path and image_path not in section_paths:
                    section_paths.append(image_path)
            section["selected_figure_paths"] = section_paths
        paper["selected_figure_paths"] = collect_selected_figure_paths([paper])


def clone_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clone_json_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_json_like(item) for item in value]
    return value


def render_context_bundle_markdown(query: str, papers: list[dict[str, Any]]) -> str:
    lines = [
        "# Query",
        query.strip(),
        "",
        "# Context Bundle",
    ]
    for paper in papers:
        lines.extend(render_paper_markdown(paper))
    return "\n".join(line for line in lines if line is not None).strip()


def render_paper_markdown(paper: dict[str, Any]) -> list[str]:
    if bool(paper.get("roster_only")):
        lines = [
            "",
            f"## Candidate {paper.get('paper_rank')}: {trim_text(str(paper.get('paper_title') or ''), 140)}",
            f"- paper_id: {paper.get('paper_id') or ''}",
        ]
        if paper.get("paper_author_text"):
            lines.append(f"- authors: {paper['paper_author_text']}")
        if paper.get("paper_institution_text"):
            lines.append(f"- institutions: {paper['paper_institution_text']}")
        evidence_rows = paper.get("query_match_evidence") or paper.get("matched_entities") or []
        if evidence_rows:
            detail = "; ".join(
                format_entity_evidence_line(row, include_match_text=False, compact=True)
                for row in evidence_rows[:1]
            )
            if detail:
                lines.append(f"- exact_match: {detail}")
        elif paper.get("why_selected"):
            lines.append(f"- reason: {paper['why_selected'][0]}")
        return lines

    lines = [
        "",
        f"## Paper {paper.get('paper_rank')}: {paper.get('paper_title') or ''}",
        f"- paper_id: {paper.get('paper_id') or ''}",
        f"- rank_score: {float(paper.get('rank_score') or 0.0):.4f}",
    ]
    if paper.get("paper_author_text"):
        lines.append(f"- authors: {paper['paper_author_text']}")
    if paper.get("paper_institution_text"):
        lines.append(f"- institutions: {paper['paper_institution_text']}")
    if paper.get("why_selected"):
        lines.append("- why_selected:")
        for row in paper["why_selected"]:
            lines.append(f"  - {row}")
    if paper.get("query_match_evidence"):
        lines.append("- query_match_evidence:")
        for row in paper["query_match_evidence"]:
            lines.append(f"  - {format_entity_evidence_line(row, include_match_text=True)}")
    if paper.get("matched_entities"):
        query_evidence_keys = {
            build_entity_row_key(row)
            for row in (paper.get("query_match_evidence") or [])
        }
        extra_matched_lines = []
        for row in paper["matched_entities"]:
            if build_entity_row_key(row) in query_evidence_keys:
                continue
            extra_matched_lines.append(
                f"  - {format_entity_evidence_line(row, include_match_text=True)}"
            )
        if extra_matched_lines:
            lines.append("- matched_entities:")
            lines.extend(extra_matched_lines)
    if paper.get("cross_paper_neighbors"):
        lines.append("- cross_paper_neighbors:")
        for row in paper["cross_paper_neighbors"]:
            detail = f"{row['edge_type']} <- {row['source_paper_title'] or row['source_paper_id']}"
            evidence = ", ".join(row.get("evidence_entity_names") or [])
            if evidence:
                detail += f" | evidence: {evidence}"
            lines.append(f"  - {detail}")
    for section in paper.get("sections") or []:
        lines.extend(render_section_markdown(section))
    return lines


def render_section_markdown(section: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"### Section: {section.get('section_title') or ''} [{section.get('section_group') or ''}]",
        f"- section_id: {section.get('section_id') or ''}",
        f"- rank_score: {float(section.get('rank_score') or 0.0):.4f}",
    ]
    if section.get("section_summary"):
        lines.append(f"- summary: {section['section_summary']}")
    if section.get("section_entities"):
        lines.append("- section_entities: " + ", ".join(section["section_entities"]))
    if section.get("chunks"):
        lines.append("- chunks:")
        for row in section["chunks"]:
            page = format_page_range(row.get("page_start"), row.get("page_end"))
            prefix = f"[{page}] " if page else ""
            lines.append(f"  - [{row['node_id']}] {prefix}{row['text']}")
    if section.get("figures"):
        lines.append("- figures:")
        for row in section["figures"]:
            page = format_page_range(row.get("page_start"), row.get("page_end"))
            prefix = f"[{page}] " if page else ""
            image_path = str(row.get("image_path") or "")
            if image_path:
                lines.append(f"  - [{row['node_id']}] {prefix}{row['caption']}")
            else:
                lines.append(f"  - {prefix}{row['caption']}")
    return lines


def build_figure_text(figure: dict[str, Any], max_chars: int) -> str:
    captions = [str(value).strip() for value in (figure.get("caption") or []) if str(value).strip()]
    if captions:
        return trim_text(" ".join(captions), max_chars)
    return trim_text(str(figure.get("preview") or ""), max_chars)


def build_entity_label(row: dict[str, Any]) -> str:
    return trim_text(
        str(row.get("name") or row.get("display_name") or row.get("canonical_name") or ""),
        120,
    )


def build_entity_row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("entity_type") or ""),
        build_entity_label(row).casefold(),
        str(row.get("edge_type") or "").casefold(),
        trim_text(str(row.get("matched_text") or ""), 120).casefold(),
    )


def format_entity_evidence_line(row: dict[str, Any], include_match_text: bool, compact: bool = False) -> str:
    detail = f"{row.get('entity_type') or ''}:{build_entity_label(row)}"
    if compact:
        matched_text = trim_text(str(row.get("matched_text") or ""), 80)
        if include_match_text and matched_text and normalize_query_text(matched_text) != normalize_query_text(build_entity_label(row)):
            detail += f" <- {matched_text}"
        return detail
    edge_type = str(row.get("edge_type") or "").strip()
    match_source = str(row.get("match_source") or "").strip()
    matched_text = trim_text(str(row.get("matched_text") or ""), 120)
    if edge_type:
        detail += f" | edge={edge_type}"
    if match_source:
        detail += f" | match={match_source}"
    if include_match_text and matched_text and normalize_query_text(matched_text) != normalize_query_text(build_entity_label(row)):
        detail += f" | matched_text={matched_text}"
    return detail


def trim_text(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return value[:max_chars]
    return value[: max_chars - 1].rstrip() + "…"


def measure_text(text: str) -> int:
    return len(str(text or ""))


def estimate_token_count(text: str) -> int:
    parts = TOKEN_ESTIMATE_RE.findall(str(text or ""))
    return max(1, int(math.ceil(len(parts) * 0.9))) if parts else 0


def format_page_range(page_start: Any, page_end: Any) -> str:
    """Render zero-based MinerU page indices as one-based PDF page numbers."""
    page_start = int(page_start) + 1 if page_start not in {None, ""} else None
    page_end = int(page_end) + 1 if page_end not in {None, ""} else None
    if page_start in {None, ""} and page_end in {None, ""}:
        return ""
    if page_start == page_end or page_end in {None, ""}:
        return f"p.{page_start}"
    if page_start in {None, ""}:
        return f"p.{page_end}"
    return f"p.{page_start}-{page_end}"
