from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_retrieval import preview_text, tfidf_scores
from .reference_name_utils import best_reference_display_name, infer_reference_title


PAPER_EDGE_TYPES = {
    "paper_has_author",
    "paper_has_institution",
    "paper_published_in",
    "paper_cites_reference",
    "paper_uses_method",
    "paper_uses_dataset",
    "paper_addresses_task",
    "paper_reports_metric",
    "paper_uses_model",
}
SECTION_EDGE_TYPES = {
    "section_uses_method",
    "section_uses_dataset",
    "section_addresses_task",
    "section_reports_metric",
    "section_uses_model",
}
RETRIEVAL_MODES = {"hierarchical", "flat_chunk", "paper_then_chunk"}
ENTITY_NODE_TYPES = {
    "author",
    "institution",
    "venue",
    "reference",
    "method",
    "dataset",
    "task",
    "metric",
    "model",
}
STRUCTURED_QUERY_FILTER_TYPES = {
    "author",
    "institution",
    "reference",
    "venue",
}
SEMANTIC_QUERY_ENTITY_TYPES = {
    "method",
    "dataset",
    "task",
    "metric",
    "model",
}
CROSS_PAPER_EDGE_TYPE_WEIGHTS = {
    "paper_shares_author": 1.2,
    "paper_shares_institution": 0.95,
    "paper_shares_method": 1.05,
    "paper_shares_dataset": 1.1,
    "paper_shares_task": 1.0,
    "paper_shares_metric": 0.9,
    "paper_shares_model": 1.0,
    "paper_bibliographic_coupling": 0.82,
    "paper_direct_citation": 1.08,
    "paper_cites_paper": 1.08,
}
CROSS_PAPER_EDGE_TO_ENTITY_TYPES = {
    "paper_shares_author": {"author"},
    "paper_shares_institution": {"institution"},
    "paper_shares_method": {"method"},
    "paper_shares_dataset": {"dataset"},
    "paper_shares_task": {"task"},
    "paper_shares_metric": {"metric"},
    "paper_shares_model": {"model"},
    "paper_bibliographic_coupling": {"reference"},
    "paper_direct_citation": {"reference"},
    "paper_cites_paper": {"reference"},
}
QUERY_QUOTE_RE = re.compile(r'["“”\'‘’](.+?)["“”\'‘’]')
QUERY_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+\-/]*")
LOOKUP_COMPACT_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
TITLE_REFERENCE_CUE_RE = re.compile(
    r"(?:this\s+paper|these\s+papers?|the\s+paper)",
    re.IGNORECASE,
)
TITLEISH_ENGLISH_SPAN_RE = re.compile(r"[A-Z0-9][A-Za-z0-9][A-Za-z0-9 .:+\-/,&()']{4,}")
EXHAUSTIVE_LIST_QUERY_PATTERNS = (
    re.compile(r"\bwhich\s+papers\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+papers\b", re.IGNORECASE),
    re.compile(r"\blist\s+(?:all\s+)?(?:the\s+)?papers\b", re.IGNORECASE),
)
TITLE_QUERY_PATTERNS = (
    re.compile(r"(?:paper|paper title|title)\s*(?:is|=|:)?\s*(.+?)\s*(?:paper)?[?]?\s*$", re.IGNORECASE),
)
QUERY_TYPE_HINTS = {
    "author": ("authors", "author"),
    "institution": ("institution", "institutions", "affiliation"),
    "venue": ("venue", "journal", "conference"),
    "reference": ("reference", "citation", "cites"),
    "method": ("method",),
    "dataset": ("dataset",),
    "task": ("task",),
    "metric": ("metric", "score"),
    "model": ("model",),
}


ENGLISH_VISUAL_QUERY_RE = re.compile(
    r"\b(?:image|images|figure|figures|fig|diagram|diagrams|table|tables|chart|charts|plot|plots|visual|visuals|visualization|visualizations|visualisation|visualisations|pipeline|workflow|overview|architecture|layout|matrix|schematic|illustration|tsne|t-sne|scatter|heatmap)\b",
    re.IGNORECASE,
)
VISUAL_ASSET_KIND_PREFERENCE = {
    "table": {"table"},
    "plot": {"chart", "image"},
    "chart": {"chart", "image"},
    "graph": {"chart", "image"},
    "curve": {"chart", "image"},
    "heatmap": {"chart", "image"},
    "scatter": {"chart", "image"},
    "tsne": {"chart", "image"},
    "t-sne": {"chart", "image"},
    "pipeline": {"image", "chart"},
    "diagram": {"image", "chart"},
    "schematic": {"image", "chart"},
    "illustration": {"image", "chart"},
    "figure": {"image", "chart"},
}
FIGURE_CAPTION_WEIGHT = 0.85
FIGURE_REFERENCE_LABEL_WEIGHT = 0.3
FIGURE_KIND_BIAS = 0.22
FIGURE_TABLE_PENALTY = 0.3
FIGURE_CAPTION_PREFIX_BONUS = 0.12
FIGURE_QUERY_STOPWORDS = {
    "show",
    "me",
    "the",
    "a",
    "an",
    "of",
    "and",
    "for",
    "with",
    "using",
    "use",
    "that",
    "this",
    "these",
    "those",
    "figure",
    "figures",
    "visualization",
    "visualizations",
    "visual",
    "plot",
    "plots",
    "chart",
    "charts",
    "diagram",
    "illustration",
    "comparison",
    "comparing",
}


SECTION_QUERY_METADATA_TYPES = {"author", "institution", "venue"}
SECTION_ENTITY_MIN_SCORE = 0.45
CONTENT_SECTION_ROLES = {
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "other_content",
}
SECTION_TITLE_REFERENCE_RE = re.compile(
    r"\b(references?|bibliograph(?:y|ies)?|works cited|cited works|citations?)\b",
    re.IGNORECASE,
)
SECTION_TITLE_ADMIN_RE = re.compile(
    r"\b("
    r"acknowledg(?:e)?ments?|"
    r"funding|"
    r"author contributions?|"
    r"conflicts? of interest|"
    r"competing interests?|"
    r"declarations?|"
    r"ethics?|"
    r"ethical approval|"
    r"consent|"
    r"data availability|"
    r"supporting information|"
    r"supplementary materials?"
    r")\b",
    re.IGNORECASE,
)
SECTION_TITLE_METHOD_RE = re.compile(
    r"\b("
    r"methods?|materials?|patients?|cohort|study design|"
    r"approach|approaches|algorithm|algorithms|implementation|"
    r"preprocessing|pipeline|workflow|protocol|"
    r"experimental setup|model selection|construction|generation|retrieval|indexing|judgment|exploration|encoding|reasoning|architecture|framework"
    r")\b",
    re.IGNORECASE,
)
SECTION_TITLE_RESULT_RE = re.compile(
    r"\b(results?|findings?|evaluation|experiments?|analysis|benchmark|ablation|performance|comparative|comparison|quality|impact)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DocumentEntityRetrieverConfig:
    graph_dir: Path = Path("outputs") / "graph"
    enable_cross_paper_edges: bool = True
    top_k_papers: int = 5
    top_k_sections: int = 2
    top_k_chunks: int = 2
    top_k_figures: int = 1
    top_k_entities: int = 5
    min_score: float = 0.02
    preview_chars: int = 260
    retrieval_backend: str = "auto"
    gnn_index_dir: Path | None = None
    gnn_device: str = "cpu"
    gnn_encoder_model: str | None = None
    hybrid_gnn_weight: float = 0.3
    hybrid_candidate_pool: int = 20


class DocumentEntityRetriever:
    def __init__(self, config: DocumentEntityRetrieverConfig) -> None:
        self.config = config
        self._tfidf_indexes = {}
        self.graph_dir = config.graph_dir.expanduser().resolve()
        self.papers = load_jsonl(self.graph_dir / "papers.jsonl")
        self.sections = load_jsonl(self.graph_dir / "sections.jsonl")
        annotate_section_role_context(self.sections)
        self.chunks = load_jsonl(self.graph_dir / "chunks.jsonl")
        self.figures = load_jsonl(self.graph_dir / "figures.jsonl")
        for figure in self.figures:
            if figure.get("image_path"):
                figure["image_path"] = str((self.graph_dir / figure["image_path"]).resolve())
        self.edges = load_jsonl(self.graph_dir / "edges.jsonl")
        self.paper_paper_edges = load_optional_jsonl(self.graph_dir / "paper_paper_edges.jsonl")
        self.paper_pair_summaries = load_optional_jsonl(self.graph_dir / "paper_pair_summaries.jsonl")
        self.entity_nodes = [
            node
            for node in load_jsonl(self.graph_dir / "nodes.jsonl")
            if str(node.get("node_type") or "") in ENTITY_NODE_TYPES
        ]

        self.paper_by_node_id = {paper["node_id"]: paper for paper in self.papers}
        self.section_by_node_id = {section["node_id"]: section for section in self.sections}
        self.chunk_by_node_id = {chunk["node_id"]: chunk for chunk in self.chunks}
        self.figure_by_node_id = {figure["node_id"]: figure for figure in self.figures}
        self.entity_by_node_id = {node["node_id"]: node for node in self.entity_nodes}
        self.paper_by_paper_id = {paper["paper_id"]: paper for paper in self.papers}
        self.paper_pair_summary_by_key: dict[tuple[str, str], dict[str, Any]] = {}

        self.paper_search_texts = [str(paper.get("search_text") or "") for paper in self.papers]
        self.chunk_search_texts = [
            str(chunk.get("search_text") or chunk.get("text") or "")
            for chunk in self.chunks
        ]
        self.entity_search_texts = [str(node.get("search_text") or "") for node in self.entity_nodes]
        self.figure_search_texts = [figure_query_text(figure) for figure in self.figures]
        self.figure_caption_texts = [figure_caption_text(figure) for figure in self.figures]
        self.figure_reference_texts = [figure_reference_label_text(figure) for figure in self.figures]

        self.sections_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.chunks_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.chunks_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.figures_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.figures_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.section_by_section_id: dict[str, dict[str, Any]] = {}
        self.section_lookup_text_by_node_id: dict[str, str] = {}
        self.section_compact_text_by_node_id: dict[str, str] = {}
        self.section_role_by_node_id: dict[str, str] = {}
        for section in self.sections:
            self.sections_by_paper[section["paper_id"]].append(section)
            self.section_by_section_id[str(section["section_id"])] = section
            section_node_id = str(section["node_id"])
            search_text = str(section.get("search_text") or "")
            self.section_lookup_text_by_node_id[section_node_id] = normalize_lookup_text(search_text)
            self.section_compact_text_by_node_id[section_node_id] = compact_lookup_text(search_text)
            self.section_role_by_node_id[section_node_id] = classify_query_section_role(section)
        for chunk in self.chunks:
            self.chunks_by_section[chunk["section_id"]].append(chunk)
            self.chunks_by_paper[str(chunk.get("paper_id") or "")].append(chunk)
        for figure in self.figures:
            self.figures_by_section[figure["section_id"]].append(figure)
            self.figures_by_paper[str(figure.get("paper_id") or "")].append(figure)

        self.paper_edges_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.section_edges_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.section_entity_edges_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.paper_node_ids_by_entity: dict[str, set[str]] = defaultdict(set)
        self.reference_entity_ids_by_paper: dict[str, list[str]] = defaultdict(list)
        self.cross_paper_edges_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges:
            edge_type = str(edge.get("edge_type") or "")
            if edge_type in PAPER_EDGE_TYPES and str(edge.get("target_id") or "") in self.entity_by_node_id:
                entity_node_id = str(edge["target_id"])
                paper_node_id = str(edge["source_id"])
                self.paper_edges_by_entity[entity_node_id].append(edge)
                self.paper_node_ids_by_entity[entity_node_id].add(paper_node_id)
                if edge_type == "paper_cites_reference":
                    self.reference_entity_ids_by_paper[paper_node_id].append(entity_node_id)
            if edge_type in SECTION_EDGE_TYPES and str(edge.get("target_id") or "") in self.entity_by_node_id:
                entity_node_id = str(edge["target_id"])
                self.section_edges_by_entity[entity_node_id].append(edge)
                self.section_entity_edges_by_section[str(edge["source_id"])].append(edge)
                section = self.section_by_node_id.get(str(edge["source_id"]))
                if section:
                    self.paper_node_ids_by_entity[entity_node_id].add(f"paper:{section['paper_id']}")

        for edge in self.paper_paper_edges:
            source_id = str(edge.get("source_id") or "")
            target_id = str(edge.get("target_id") or "")
            if source_id not in self.paper_by_node_id or target_id not in self.paper_by_node_id:
                continue
            self.cross_paper_edges_by_paper[source_id].append(edge)
            reverse_edge = dict(edge)
            reverse_edge["source_id"] = target_id
            reverse_edge["target_id"] = source_id
            reverse_edge["reverse_of"] = edge.get("edge_id")
            self.cross_paper_edges_by_paper[target_id].append(reverse_edge)

        for row in self.paper_pair_summaries:
            key = make_paper_pair_key(
                str(row.get("paper_a_node_id") or ""),
                str(row.get("paper_b_node_id") or ""),
            )
            if key:
                self.paper_pair_summary_by_key[key] = row

        self.entity_alias_lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.entity_alias_lengths: dict[str, int] = {}
        self._build_entity_alias_lookup()

        self.retrieval_backend = config.retrieval_backend
        if self.retrieval_backend == "auto":
            self.retrieval_backend = "hybrid" if config.gnn_index_dir is not None else "lexical"
        if self.retrieval_backend not in {"lexical", "gnn", "hybrid"}:
            raise ValueError("retrieval_backend must be auto, lexical, gnn, or hybrid")
        self.gnn = None
        self.hybrid = None
        if self.retrieval_backend in {"gnn", "hybrid"}:
            if config.gnn_index_dir is None:
                raise ValueError("gnn_index_dir is required for GNN retrieval")
            if not config.enable_cross_paper_edges:
                raise ValueError("GNN edge ablations require an index trained on the ablated graph; runtime edge disabling is unsupported")
            from .gnn_retrieval import GNNRetriever
            self.gnn = GNNRetriever(self)
        if self.retrieval_backend == "hybrid":
            from .hybrid_retrieval import HybridRetriever
            self.hybrid = HybridRetriever(self)

    def _cached_tfidf(self, query: str, docs: list[str], key) -> list[float]:
        try:
            from .prepared_tfidf import PreparedTfidf
        except ModuleNotFoundError as exc:
            if exc.name != "numpy":
                raise
            return tfidf_scores(query, docs)
        prepared = self._tfidf_indexes.get(key)
        if prepared is None or prepared.documents != tuple(docs):
            prepared = PreparedTfidf(docs)
            self._tfidf_indexes[key] = prepared
        return prepared.scores(query)

    def retrieve(
        self,
        query: str,
        top_k_papers: int | None = None,
        top_k_sections: int | None = None,
        top_k_chunks: int | None = None,
        top_k_figures: int | None = None,
        retrieval_mode: str = "hierarchical",
    ) -> dict[str, Any]:
        if self.hybrid is not None:
            return self.hybrid.retrieve(query, top_k_papers, top_k_sections, top_k_chunks, top_k_figures, retrieval_mode)
        if self.gnn is not None:
            return self.gnn.retrieve(query, top_k_papers, top_k_sections, top_k_chunks, top_k_figures, retrieval_mode)
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must not be empty")

        top_k_papers = max(1, int(top_k_papers or self.config.top_k_papers))
        top_k_sections = max(1, int(top_k_sections or self.config.top_k_sections))
        top_k_chunks = max(0, int(top_k_chunks or self.config.top_k_chunks))
        top_k_figures = max(0, int(top_k_figures or self.config.top_k_figures))
        retrieval_mode = normalize_retrieval_mode(retrieval_mode)

        if retrieval_mode == "flat_chunk":
            return self.retrieve_flat_chunk(
                query=query,
                top_k_papers=top_k_papers,
                top_k_sections=top_k_sections,
                top_k_chunks=top_k_chunks,
                top_k_figures=top_k_figures,
            )

        exhaustive_payload = self.retrieve_exhaustive(
            query=query,
            top_k_sections=top_k_sections,
            top_k_chunks=top_k_chunks,
            top_k_figures=top_k_figures,
            retrieval_mode=retrieval_mode,
        )
        if exhaustive_payload is not None:
            return exhaustive_payload

        exact_constraint_payload = self.retrieve_exact_constraint_query(
            query=query,
            top_k_papers=top_k_papers,
            top_k_sections=top_k_sections,
            top_k_chunks=top_k_chunks,
            top_k_figures=top_k_figures,
            retrieval_mode=retrieval_mode,
        )
        if exact_constraint_payload is not None:
            return exact_constraint_payload

        named_sources = named_source_paper_ids(query, self.papers)
        ranked_papers = self.rank_papers(query=query, top_k_papers=len(self.papers) if named_sources else top_k_papers)
        if named_sources:
            ranked_papers = [row for row in ranked_papers if row[0]["paper_id"] in named_sources][:top_k_papers]

        results: list[dict[str, Any]] = []
        for paper, total_score, direct_score, entity_support, cross_paper_support, figure_support in ranked_papers:
            results.append(
                self.build_paper_result(
                    paper=paper,
                    query=query,
                    top_k_sections=top_k_sections,
                    top_k_chunks=top_k_chunks,
                    top_k_figures=top_k_figures,
                    total_score=float(total_score),
                    direct_score=float(direct_score),
                    entity_support=entity_support,
                    cross_paper_support=cross_paper_support,
                    figure_support=figure_support,
                    retrieval_mode=retrieval_mode,
                )
            )

        payload = {
            "query": query,
            "graph_dir": str(self.graph_dir),
            "retrieval_mode": retrieval_mode,
            "result_kind": "papers",
            "result_count": len(results),
            "results": results,
        }
        if named_sources:
            payload.update(query_mode="named_source", selection_mode="named_source_constraint",
                           authoritative_candidate_ids=sorted(named_sources))
        if retrieval_mode == "paper_then_chunk":
            payload["selection_mode"] = "paper_rank_then_local_chunk"
        return payload

    def retrieve_flat_chunk(
        self,
        query: str,
        top_k_papers: int,
        top_k_sections: int,
        top_k_chunks: int,
        top_k_figures: int,
    ) -> dict[str, Any]:
        if self.hybrid is not None:
            return self.hybrid.retrieve(query, top_k_papers, top_k_sections, top_k_chunks, top_k_figures, "flat_chunk")
        if self.gnn is not None:
            return self.gnn.retrieve(query, top_k_papers, top_k_sections, top_k_chunks, top_k_figures, "flat_chunk")
        effective_top_k_chunks = max(1, int(top_k_chunks))
        final_chunk_budget = max(1, int(top_k_papers) * int(top_k_sections) * effective_top_k_chunks)
        candidate_pool_size = max(
            final_chunk_budget * 6,
            int(top_k_papers) * int(top_k_sections) * 4,
            int(top_k_papers) * 8,
            24,
        )
        chunk_hits = self.rank_chunks_globally(query=query, top_k=candidate_pool_size)

        paper_hits_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        section_hits_by_paper: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for hit in chunk_hits:
            chunk = hit["chunk"]
            paper_id = str(chunk.get("paper_id") or "")
            section_id = str(chunk.get("section_id") or "")
            if not paper_id or not section_id:
                continue
            paper_hits_by_paper[paper_id].append(hit)
            section_hits_by_paper[paper_id][section_id].append(hit)

        ranked_paper_ids = sorted(
            paper_hits_by_paper,
            key=lambda paper_id: (
                -self._score_flat_chunk_paper(paper_hits_by_paper.get(paper_id, [])),
                -max(
                    (float(hit.get("score") or 0.0) for hit in paper_hits_by_paper.get(paper_id, [])),
                    default=0.0,
                ),
                paper_id,
            ),
        )[:top_k_papers]

        results: list[dict[str, Any]] = []
        for paper_id in ranked_paper_ids:
            paper = self.paper_by_paper_id.get(paper_id)
            if not paper:
                continue
            results.append(
                self._build_flat_chunk_paper_result(
                    query=query,
                    paper=paper,
                    paper_hits=paper_hits_by_paper.get(paper_id, []),
                    section_hits=section_hits_by_paper.get(paper_id, {}),
                    top_k_sections=top_k_sections,
                    top_k_chunks=effective_top_k_chunks,
                    top_k_figures=top_k_figures,
                )
            )

        return {
            "query": query,
            "graph_dir": str(self.graph_dir),
            "retrieval_mode": "flat_chunk",
            "query_mode": "ranked",
            "selection_mode": "flat_chunk_global_tfidf",
            "result_kind": "papers",
            "result_count": len(results),
            "flat_chunk_pool_size": int(candidate_pool_size),
            "flat_chunk_budget": int(final_chunk_budget),
            "results": results,
        }

    def rank_chunks_globally(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.hybrid is not None:
            return self.hybrid.rank_chunks_globally(query, top_k)
        if self.gnn is not None:
            return [{"chunk": row, "score": score, "preview": preview_text(str(row.get("text") or ""), self.config.preview_chars)}
                    for row, score in self.gnn.index.rank(query, self.chunks, top_k)]
        if top_k <= 0 or not self.chunks:
            return []
        scores = self._cached_tfidf(query, self.chunk_search_texts, "chunks")
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda item: (-float(item[1]), int(item[0].get("order") or 0)),
        )[:top_k]
        results: list[dict[str, Any]] = []
        for chunk, score in ranked:
            results.append(
                {
                    "chunk": chunk,
                    "score": float(score),
                    "preview": preview_text(
                        str(chunk.get("text") or chunk.get("search_text") or ""),
                        self.config.preview_chars,
                    ),
                }
            )
        return results

    def _build_flat_chunk_paper_result(
        self,
        query: str,
        paper: dict[str, Any],
        paper_hits: list[dict[str, Any]],
        section_hits: dict[str, list[dict[str, Any]]],
        top_k_sections: int,
        top_k_chunks: int,
        top_k_figures: int,
    ) -> dict[str, Any]:
        paper_score = self._score_flat_chunk_paper(paper_hits)
        best_chunk_score = max((float(hit.get("score") or 0.0) for hit in paper_hits), default=0.0)
        ranked_section_ids = sorted(
            section_hits,
            key=lambda section_id: (
                -self._score_flat_chunk_section(section_hits.get(section_id, [])),
                int(
                    (self.section_by_section_id.get(section_id) or {}).get("order") or 0
                ),
            ),
        )[:top_k_sections]

        section_results = [
            self._build_flat_chunk_section_result(
                query=query,
                section_id=section_id,
                section_hits=section_hits.get(section_id, []),
                top_k_chunks=top_k_chunks,
                top_k_figures=top_k_figures,
            )
            for section_id in ranked_section_ids
            if self.section_by_section_id.get(section_id)
        ]

        best_hit = max(paper_hits, key=lambda item: float(item.get("score") or 0.0), default={})
        best_chunk = dict(best_hit.get("chunk") or {})
        best_section = self.section_by_section_id.get(str(best_chunk.get("section_id") or ""), {})
        best_section_title = str(best_section.get("section_title") or "")
        best_preview = str(best_hit.get("preview") or "").strip()
        why_matched = [
            {
                "kind": "flat_chunk_support",
                "score": round(float(paper_score), 6),
                "detail": (
                    f"flat chunk retrieval matched {len(paper_hits)} chunk hits "
                    f"across {len(section_hits)} section(s)"
                ),
            }
        ]
        if best_preview:
            detail = "best chunk hit"
            if best_section_title:
                detail += f" from section {best_section_title}"
            detail += f": {preview_text(best_preview, 180)}"
            why_matched.append(
                {
                    "kind": "flat_chunk_best_hit",
                    "score": round(float(best_chunk_score), 6),
                    "detail": detail,
                }
            )

        return {
            "paper_id": paper["paper_id"],
            "paper_title": paper.get("title") or "",
            "paper_author_text": paper.get("author_text") or "",
            "paper_institution_text": paper.get("institution_text") or "",
            "rank_score": round(float(paper_score), 6),
            "paper_direct_score": round(float(best_chunk_score), 6),
            "matched_entities": [],
            "cross_paper_support": [],
            "figure_support": [],
            "why_matched": why_matched,
            "top_sections": section_results,
        }

    def _build_flat_chunk_section_result(
        self,
        query: str,
        section_id: str,
        section_hits: list[dict[str, Any]],
        top_k_chunks: int,
        top_k_figures: int,
    ) -> dict[str, Any]:
        section = self.section_by_section_id.get(section_id, {})
        section_node_id = str(section.get("node_id") or "")
        ranked_hits = sorted(
            section_hits,
            key=lambda item: (
                -float(item.get("score") or 0.0),
                int((item.get("chunk") or {}).get("order") or 0),
            ),
        )

        chunk_results: list[dict[str, Any]] = []
        seen_chunk_node_ids: set[str] = set()
        for hit in ranked_hits:
            chunk = dict(hit.get("chunk") or {})
            node_id = str(chunk.get("node_id") or "")
            if not node_id or node_id in seen_chunk_node_ids:
                continue
            seen_chunk_node_ids.add(node_id)
            chunk_results.append(
                {
                    "node_id": node_id,
                    "rank_score": round(float(hit.get("score") or 0.0), 6),
                    "preview": str(hit.get("preview") or ""),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "record_kind": "chunk",
                }
            )
            if len(chunk_results) >= top_k_chunks:
                break

        if len(chunk_results) < top_k_chunks:
            local_chunk_rows = rank_local_records(
                query=query,
                rows=self.chunks_by_section.get(section_id, []),
                text_getter=lambda row: str(row.get("text") or ""),
                top_k=max(top_k_chunks * 2, top_k_chunks),
                preview_chars=self.config.preview_chars,
                record_kind="chunk",
            )
            for row in local_chunk_rows:
                node_id = str(row.get("node_id") or "")
                if not node_id or node_id in seen_chunk_node_ids:
                    continue
                seen_chunk_node_ids.add(node_id)
                chunk_results.append(row)
                if len(chunk_results) >= top_k_chunks:
                    break

        figure_results = rank_local_records(
            query=query,
            rows=self.figures_by_section.get(section_id, []),
            text_getter=lambda row: str(row.get("search_text") or ""),
            top_k=top_k_figures,
            preview_chars=self.config.preview_chars,
            record_kind="figure",
        )
        entity_edges = self.section_entity_edges_by_section.get(section_node_id, [])
        best_section_score = max((float(hit.get("score") or 0.0) for hit in ranked_hits), default=0.0)
        return {
            "section_id": section_id,
            "section_title": section.get("section_title") or "",
            "section_group": section.get("section_group") or "",
            "section_role": self.section_role_by_node_id.get(section_node_id, "other_content"),
            "rank_score": round(float(self._score_flat_chunk_section(section_hits)), 6),
            "section_direct_score": round(float(best_section_score), 6),
            "section_entity_overlap_score": 0.0,
            "section_query_alignment_score": 0.0,
            "section_visual_support_score": 0.0,
            "section_summary": preview_text(
                str(section.get("section_summary") or ""),
                self.config.preview_chars,
            ),
            "matched_entities": summarize_section_entities(entity_edges, self.entity_by_node_id),
            "top_chunks": chunk_results,
            "top_figures": figure_results,
        }

    def _score_flat_chunk_paper(self, paper_hits: list[dict[str, Any]]) -> float:
        top_scores = sorted(
            (float(hit.get("score") or 0.0) for hit in paper_hits),
            reverse=True,
        )[:4]
        weights = (1.0, 0.72, 0.55, 0.38)
        score = sum(weight * value for weight, value in zip(weights, top_scores))
        section_count = len(
            {
                str((hit.get("chunk") or {}).get("section_id") or "")
                for hit in paper_hits
                if str((hit.get("chunk") or {}).get("section_id") or "")
            }
        )
        if section_count > 1:
            score += 0.12 * min(section_count - 1, 3)
        return float(score)

    def _score_flat_chunk_section(self, section_hits: list[dict[str, Any]]) -> float:
        top_scores = sorted(
            (float(hit.get("score") or 0.0) for hit in section_hits),
            reverse=True,
        )[:3]
        weights = (1.0, 0.68, 0.5)
        score = sum(weight * value for weight, value in zip(weights, top_scores))
        if len(section_hits) > 1:
            score += 0.06 * min(len(section_hits) - 1, 3)
        return float(score)

    def retrieve_exhaustive(
        self,
        query: str,
        top_k_sections: int,
        top_k_chunks: int,
        top_k_figures: int,
        retrieval_mode: str = "hierarchical",
    ) -> dict[str, Any] | None:
        if not is_exhaustive_list_query(query):
            return None

        exact_entity_matches = self.find_exact_entity_matches(query)
        if not exact_entity_matches:
            return None

        declared_query_types = detect_query_entity_types(query)
        query_types = set(declared_query_types)
        query_types |= infer_query_types_from_exact_matches(exact_entity_matches)
        constraint_groups = self._resolve_exhaustive_constraint_groups(
            declared_query_types=declared_query_types,
            query_types=query_types,
            exact_entity_matches=exact_entity_matches,
        )
        if not constraint_groups:
            return None

        authoritative_paper_node_ids = set.intersection(
            *(set(group["paper_node_ids"]) for group in constraint_groups if group.get("paper_node_ids"))
        )
        if not authoritative_paper_node_ids:
            return None

        ranked_all = self.rank_papers(
            query=query,
            top_k_papers=max(len(self.papers), len(authoritative_paper_node_ids)),
        )
        ranked_lookup = {
            str(paper.get("node_id") or ""): (
                paper,
                float(total_score),
                float(direct_score),
                list(entity_support or []),
                list(cross_paper_support or []),
                list(figure_support or []),
            )
            for paper, total_score, direct_score, entity_support, cross_paper_support, figure_support in ranked_all
        }

        exact_entity_support_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scratch_scores: dict[str, float] = defaultdict(float)
        for group in constraint_groups:
            matched_text = str(group.get("display_name") or "")
            for entity_node_id in group.get("entity_node_ids") or []:
                entity = self.entity_by_node_id.get(str(entity_node_id))
                if not entity:
                    continue
                self._apply_entity_support(
                    entity=entity,
                    entity_score=3.2,
                    candidate_scores=scratch_scores,
                    entity_support_by_paper=exact_entity_support_by_paper,
                    match_source="exact",
                    matched_text=matched_text,
                )

        ordered_node_ids: list[str] = [
            str(paper.get("node_id") or "")
            for paper, _score, _direct, _entity, _cross, _figure in ranked_all
            if str(paper.get("node_id") or "") in authoritative_paper_node_ids
        ]
        for paper_node_id in sorted(authoritative_paper_node_ids):
            if paper_node_id not in ordered_node_ids:
                ordered_node_ids.append(paper_node_id)

        results: list[dict[str, Any]] = []
        authoritative_paper_ids: list[str] = []
        for paper_node_id in ordered_node_ids:
            ranked_row = ranked_lookup.get(paper_node_id)
            if ranked_row is not None:
                (
                    paper,
                    total_score,
                    direct_score,
                    entity_support,
                    cross_paper_support,
                    figure_support,
                ) = ranked_row
            else:
                paper = self.paper_by_node_id.get(paper_node_id)
                if not paper:
                    continue
                direct_score = self.score_paper_direct(query, paper_node_id)
                total_score = direct_score + (0.9 * len(exact_entity_support_by_paper.get(paper_node_id) or []))
                entity_support = list(exact_entity_support_by_paper.get(paper_node_id) or [])
                cross_paper_support = []
                figure_support = []

            paper_id = str(paper.get("paper_id") or "")
            if paper_id:
                authoritative_paper_ids.append(paper_id)
            results.append(
                self.build_paper_result(
                    paper=paper,
                    query=query,
                    top_k_sections=top_k_sections,
                    top_k_chunks=top_k_chunks,
                    top_k_figures=top_k_figures,
                    total_score=float(total_score),
                    direct_score=float(direct_score),
                    entity_support=list(entity_support or exact_entity_support_by_paper.get(paper_node_id) or []),
                    cross_paper_support=list(cross_paper_support or []),
                    figure_support=list(figure_support or []),
                    retrieval_mode=retrieval_mode,
                )
            )

        return {
            "query": query,
            "graph_dir": str(self.graph_dir),
            "result_kind": "papers",
            "result_count": len(results),
            "results": results,
            "query_mode": "exhaustive_list",
            "selection_mode": "exact_entity_set",
            "authoritative_candidate_ids": authoritative_paper_ids,
            "constraint_groups": [
                {
                    "entity_type": str(group.get("entity_type") or ""),
                    "display_name": str(group.get("display_name") or ""),
                    "matched_texts": list(group.get("matched_texts") or []),
                    "paper_count": len(group.get("paper_node_ids") or []),
                }
                for group in constraint_groups
            ],
        }

    def retrieve_exact_constraint_query(
        self,
        query: str,
        top_k_papers: int,
        top_k_sections: int,
        top_k_chunks: int,
        top_k_figures: int,
        retrieval_mode: str = "hierarchical",
    ) -> dict[str, Any] | None:
        if is_exhaustive_list_query(query) or extract_title_query_phrases_v2(query):
            return None
        if not has_constraint_query_intent(query):
            return None

        exact_entity_matches = self.find_exact_entity_matches(query)
        if not exact_entity_matches:
            return None

        declared_query_types = detect_query_entity_types(query)
        query_types = set(declared_query_types)
        query_types |= infer_query_types_from_exact_matches(exact_entity_matches)
        constraint_groups = self._resolve_exhaustive_constraint_groups(
            declared_query_types=declared_query_types,
            query_types=query_types,
            exact_entity_matches=exact_entity_matches,
        )
        if len(constraint_groups) < 2:
            return None

        authoritative_paper_node_ids = set.intersection(
            *(set(group["paper_node_ids"]) for group in constraint_groups if group.get("paper_node_ids"))
        )
        if not authoritative_paper_node_ids:
            return None

        max_candidate_count = max(6, top_k_papers * 4)
        if len(authoritative_paper_node_ids) > max_candidate_count:
            return None

        ranked_all = self.rank_papers(
            query=query,
            top_k_papers=max(len(authoritative_paper_node_ids), top_k_papers),
        )
        ranked_lookup = {
            str(paper.get("node_id") or ""): (
                paper,
                float(total_score),
                float(direct_score),
                list(entity_support or []),
                list(cross_paper_support or []),
                list(figure_support or []),
            )
            for paper, total_score, direct_score, entity_support, cross_paper_support, figure_support in ranked_all
        }

        ordered_node_ids: list[str] = [
            str(paper.get("node_id") or "")
            for paper, _score, _direct, _entity, _cross, _figure in ranked_all
            if str(paper.get("node_id") or "") in authoritative_paper_node_ids
        ]
        for paper_node_id in sorted(authoritative_paper_node_ids):
            if paper_node_id not in ordered_node_ids:
                ordered_node_ids.append(paper_node_id)

        results: list[dict[str, Any]] = []
        authoritative_paper_ids: list[str] = []
        for paper_node_id in ordered_node_ids:
            ranked_row = ranked_lookup.get(paper_node_id)
            if ranked_row is None:
                continue
            (
                paper,
                total_score,
                direct_score,
                entity_support,
                cross_paper_support,
                figure_support,
            ) = ranked_row
            paper_id = str(paper.get("paper_id") or "")
            if paper_id:
                authoritative_paper_ids.append(paper_id)
            results.append(
                self.build_paper_result(
                    paper=paper,
                    query=query,
                    top_k_sections=top_k_sections,
                    top_k_chunks=top_k_chunks,
                    top_k_figures=top_k_figures,
                    total_score=float(total_score),
                    direct_score=float(direct_score),
                    entity_support=list(entity_support or []),
                    cross_paper_support=list(cross_paper_support or []),
                    figure_support=list(figure_support or []),
                    retrieval_mode=retrieval_mode,
                )
            )

        if not results:
            return None

        return {
            "query": query,
            "graph_dir": str(self.graph_dir),
            "result_kind": "papers",
            "result_count": len(results),
            "results": results,
            "query_mode": "exact_constraint",
            "selection_mode": "constraint_intersection",
            "authoritative_candidate_ids": authoritative_paper_ids,
            "constraint_groups": [
                {
                    "entity_type": str(group.get("entity_type") or ""),
                    "display_name": str(group.get("display_name") or ""),
                    "matched_texts": list(group.get("matched_texts") or []),
                    "paper_count": len(group.get("paper_node_ids") or []),
                }
                for group in constraint_groups
            ],
        }

    def _resolve_exhaustive_constraint_groups(
        self,
        declared_query_types: set[str],
        query_types: set[str],
        exact_entity_matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        focus_types = query_types & (STRUCTURED_QUERY_FILTER_TYPES | SEMANTIC_QUERY_ENTITY_TYPES)
        if "reference" in declared_query_types:
            focus_types = {"reference"}
        if not focus_types:
            return []

        groups_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for match in exact_entity_matches:
            entity_type = str(match.get("entity_type") or "")
            if entity_type not in focus_types:
                continue
            entity_node_id = str(match.get("entity_node_id") or "")
            if not entity_node_id:
                continue
            compact_name = compact_lookup_text(
                str(match.get("matched_text") or match.get("canonical_name") or "")
            )
            if not compact_name:
                continue
            key = (entity_type, compact_name)
            group = groups_by_key.setdefault(
                key,
                {
                    "entity_type": entity_type,
                    "compact_name": compact_name,
                    "display_name": normalize_match_text(
                        str(match.get("matched_text") or match.get("canonical_name") or "")
                    ),
                    "entity_node_ids": set(),
                    "matched_texts": [],
                },
            )
            group["entity_node_ids"].add(entity_node_id)
            matched_text = normalize_match_text(str(match.get("matched_text") or ""))
            if matched_text and matched_text not in group["matched_texts"]:
                group["matched_texts"].append(matched_text)
            if not str(group.get("display_name") or ""):
                group["display_name"] = normalize_match_text(str(match.get("canonical_name") or ""))

        groups = list(groups_by_key.values())
        groups.sort(
            key=lambda item: (
                -len(str(item.get("compact_name") or "")),
                str(item.get("entity_type") or ""),
                str(item.get("display_name") or ""),
            )
        )

        pruned_groups: list[dict[str, Any]] = []
        for group in groups:
            compact_name = str(group.get("compact_name") or "")
            entity_type = str(group.get("entity_type") or "")
            if any(
                entity_type == str(kept.get("entity_type") or "")
                and compact_name
                and compact_name != str(kept.get("compact_name") or "")
                and compact_name in str(kept.get("compact_name") or "")
                for kept in pruned_groups
            ):
                continue
            paper_node_ids: set[str] = set()
            for entity_node_id in group.get("entity_node_ids") or set():
                paper_node_ids.update(self.paper_node_ids_by_entity.get(str(entity_node_id), set()))
            if entity_type == "reference" and paper_node_ids:
                filtered_paper_node_ids = self._filter_reference_group_paper_ids(group)
                if filtered_paper_node_ids:
                    paper_node_ids = filtered_paper_node_ids
            if not paper_node_ids:
                continue
            group["paper_node_ids"] = paper_node_ids
            pruned_groups.append(group)

        return pruned_groups

    def rank_papers(
        self,
        query: str,
        top_k_papers: int | None = None,
    ) -> list[
        tuple[
            dict[str, Any],
            float,
            float,
            list[dict[str, Any]],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]
    ]:
        if self.hybrid is not None:
            limit = max(1, int(self.config.top_k_papers if top_k_papers is None else top_k_papers))
            return self.hybrid.rank_papers(query, limit)
        if self.gnn is not None:
            limit = max(1, int(self.config.top_k_papers if top_k_papers is None else top_k_papers))
            return [(paper, score, self.gnn.index.score(query, paper["node_id"]),
                     self.gnn.entities(query, paper["node_id"]), [], [])
                    for paper, score in self.gnn.rank_papers(query, limit)]
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must not be empty")

        top_k_papers = max(1, int(top_k_papers or self.config.top_k_papers))
        paper_scores = self._cached_tfidf(query, self.paper_search_texts, "papers") if self.papers else []
        entity_scores = self._cached_tfidf(query, self.entity_search_texts, "entities") if self.entity_nodes else []
        query_types = detect_query_entity_types(query)
        visual_query = is_visual_query(query)
        title_query_phrases = extract_title_query_phrases_v2(query)
        exact_entity_matches = [] if title_query_phrases else self.find_exact_entity_matches(query)
        query_types |= infer_query_types_from_exact_matches(exact_entity_matches)

        candidate_scores: dict[str, float] = defaultdict(float)
        direct_paper_scores: dict[str, float] = {}
        figure_support_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        direct_weight = 0.92 if visual_query else (0.72 if len(exact_entity_matches) <= 1 else 0.24)
        for paper, score in zip(self.papers, paper_scores):
            paper_node_id = str(paper["node_id"])
            direct_paper_scores[paper_node_id] = float(score)
            candidate_scores[paper_node_id] += float(score) * direct_weight

        if title_query_phrases:
            self._apply_title_query_boosts(
                title_phrases=title_query_phrases,
                candidate_scores=candidate_scores,
            )
        else:
            self._apply_symbolic_query_boosts(
                symbolic_handles=extract_symbolic_query_handles(query),
                candidate_scores=candidate_scores,
            )

        top_entities = sorted(
            [
                (
                    entity,
                    adjust_entity_score_for_query(
                        entity_type=str(entity.get("node_type") or ""),
                        raw_score=float(score),
                        query_types=query_types,
                    ),
                    float(score),
                )
                for entity, score in zip(self.entity_nodes, entity_scores)
                if adjust_entity_score_for_query(
                    entity_type=str(entity.get("node_type") or ""),
                    raw_score=float(score),
                    query_types=query_types,
                ) >= self.config.min_score
            ],
            key=lambda item: (-item[1], -item[2], str(item[0].get("node_id") or "")),
        )[: self.config.top_k_entities]

        entity_support_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity, entity_score, _raw_score in top_entities:
            self._apply_entity_support(
                entity=entity,
                entity_score=float(entity_score),
                candidate_scores=candidate_scores,
                entity_support_by_paper=entity_support_by_paper,
                match_source="fuzzy",
            )

        if exact_entity_matches:
            exact_paper_sets: list[set[str]] = []
            exact_match_count_by_paper: dict[str, int] = defaultdict(int)
            for match in exact_entity_matches:
                entity = self.entity_by_node_id.get(str(match.get("entity_node_id") or ""))
                if not entity:
                    continue
                if visual_query:
                    exact_score = 2.1 if str(match.get("match_source") or "") == "quoted" else 1.6
                else:
                    exact_score = 3.6 if str(match.get("match_source") or "") == "quoted" else 2.8
                exact_score += float(match.get("type_bias") or 0.0)
                self._apply_entity_support(
                    entity=entity,
                    entity_score=exact_score,
                    candidate_scores=candidate_scores,
                    entity_support_by_paper=entity_support_by_paper,
                    match_source="exact",
                    matched_text=str(match.get("matched_text") or ""),
                )
                paper_node_ids = set(self.paper_node_ids_by_entity.get(str(entity["node_id"]), set()))
                if paper_node_ids:
                    exact_paper_sets.append(paper_node_ids)
                    for paper_node_id in paper_node_ids:
                        exact_match_count_by_paper[paper_node_id] += 1

            total_exact = len(exact_paper_sets)
            if total_exact > 0:
                max_exact_count = max(exact_match_count_by_paper.values(), default=0)
                full_intersection: set[str] = set()
                if total_exact >= 2:
                    full_intersection = set.intersection(*exact_paper_sets) if exact_paper_sets else set()
                count_bonus = 0.8 if visual_query else 1.5
                pair_bonus = 2.0 if visual_query else 5.0
                leader_bonus = 4.5 if visual_query else 10.0
                intersection_bonus = 12.0 if visual_query else 42.0
                intersection_scale = 3.5 if visual_query else 8.0
                penalty = 1.25 if visual_query else 4.0
                for paper_node_id, count in exact_match_count_by_paper.items():
                    candidate_scores[paper_node_id] += count_bonus * count
                    if total_exact >= 2:
                        candidate_scores[paper_node_id] += pair_bonus * count * count
                        if count == max_exact_count:
                            candidate_scores[paper_node_id] += leader_bonus + (1.4 * count)
                for paper_node_id in full_intersection:
                    candidate_scores[paper_node_id] += intersection_bonus + (intersection_scale * total_exact)
                if full_intersection:
                    for paper_node_id in exact_match_count_by_paper:
                        if paper_node_id not in full_intersection:
                            candidate_scores[paper_node_id] -= penalty

        if visual_query:
            self._apply_visual_figure_support(
                query=query,
                candidate_scores=candidate_scores,
                figure_support_by_paper=figure_support_by_paper,
                top_k_papers=top_k_papers,
            )

        self._apply_reference_node_fallback(
            query_types=query_types,
            exact_entity_matches=exact_entity_matches,
            candidate_scores=candidate_scores,
            entity_support_by_paper=entity_support_by_paper,
        )
        self._apply_reference_text_fallback(
            query_types=query_types,
            exact_entity_matches=exact_entity_matches,
            candidate_scores=candidate_scores,
            entity_support_by_paper=entity_support_by_paper,
        )
        self._apply_coverage_bonuses(
            candidate_scores=candidate_scores,
            entity_support_by_paper=entity_support_by_paper,
        )
        cross_paper_support_by_paper: dict[str, list[dict[str, Any]]] = {}
        if bool(self.config.enable_cross_paper_edges):
            cross_paper_support_by_paper = self._apply_cross_paper_support(
                query_types=query_types,
                candidate_scores=candidate_scores,
                direct_paper_scores=direct_paper_scores,
                entity_support_by_paper=entity_support_by_paper,
                top_k_papers=top_k_papers,
            )
        self._apply_structured_query_filter(
            query_types=query_types,
            exact_entity_matches=exact_entity_matches,
            candidate_scores=candidate_scores,
            entity_support_by_paper=entity_support_by_paper,
        )
        self._apply_semantic_query_boosts(
            query=query,
            query_types=query_types,
            exact_entity_matches=exact_entity_matches,
            candidate_scores=candidate_scores,
        )

        ranked_papers = sorted(
            [
                (
                    self.paper_by_node_id[paper_node_id],
                    score,
                    direct_paper_scores.get(paper_node_id, 0.0),
                    entity_support_by_paper.get(paper_node_id, []),
                    cross_paper_support_by_paper.get(paper_node_id, []),
                    figure_support_by_paper.get(paper_node_id, []),
                )
                for paper_node_id, score in candidate_scores.items()
                if paper_node_id in self.paper_by_node_id and score >= self.config.min_score
            ],
            key=lambda item: (-item[1], item[0]["paper_id"]),
        )[:top_k_papers]
        return ranked_papers

    def retrieve_within_paper(
        self,
        query: str,
        paper_id: str | None = None,
        paper_title: str | None = None,
        top_k_sections: int | None = None,
        top_k_chunks: int | None = None,
        top_k_figures: int | None = None,
    ) -> dict[str, Any]:
        if self.hybrid is not None:
            return self.hybrid.retrieve(query, top_k_sections=top_k_sections, top_k_chunks=top_k_chunks,
                top_k_figures=top_k_figures, paper=self.resolve_paper(paper_id=paper_id, paper_title=paper_title))
        if self.gnn is not None:
            return self.gnn.retrieve(query, top_k_sections=top_k_sections, top_k_chunks=top_k_chunks,
                                     top_k_figures=top_k_figures,
                                     paper=self.resolve_paper(paper_id=paper_id, paper_title=paper_title))
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must not be empty")

        top_k_sections = max(1, int(top_k_sections or self.config.top_k_sections))
        top_k_chunks = max(0, int(top_k_chunks or self.config.top_k_chunks))
        top_k_figures = max(0, int(top_k_figures or self.config.top_k_figures))

        paper = self.resolve_paper(paper_id=paper_id, paper_title=paper_title)
        paper_score = self.score_paper_direct(query, paper["node_id"])
        result = self.build_paper_result(
            paper=paper,
            query=query,
            top_k_sections=top_k_sections,
            top_k_chunks=top_k_chunks,
            top_k_figures=top_k_figures,
            total_score=paper_score,
            direct_score=paper_score,
            entity_support=[],
            cross_paper_support=[],
            figure_support=[],
        )
        return {
            "query": query,
            "graph_dir": str(self.graph_dir),
            "result_kind": "single_paper",
            "selected_paper_id": paper["paper_id"],
            "selected_paper_title": paper.get("title") or "",
            "result_count": 1,
            "results": [result],
        }

    def resolve_paper(
        self,
        paper_id: str | None = None,
        paper_title: str | None = None,
    ) -> dict[str, Any]:
        raw_paper_id = str(paper_id or "").strip()
        raw_paper_title = str(paper_title or "").strip()
        if raw_paper_id:
            if raw_paper_id in self.paper_by_paper_id:
                return self.paper_by_paper_id[raw_paper_id]
            if raw_paper_id in self.paper_by_node_id:
                return self.paper_by_node_id[raw_paper_id]
            raise ValueError(f"paper_id not found: {raw_paper_id}")
        if not raw_paper_title:
            raise ValueError("paper_id or paper_title must be provided for single-paper retrieval")

        normalized_query = normalize_lookup_text(raw_paper_title)
        exact_matches = [
            paper
            for paper in self.papers
            if normalize_lookup_text(str(paper.get("title") or "")) == normalized_query
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise ValueError(
                "paper_title matches multiple papers: "
                + "; ".join(str(paper.get("title") or "") for paper in exact_matches[:5])
            )

        partial_matches = [
            paper
            for paper in self.papers
            if normalized_query in normalize_lookup_text(str(paper.get("title") or ""))
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]
        if len(partial_matches) > 1:
            raise ValueError(
                "paper_title matches multiple papers: "
                + "; ".join(str(paper.get("title") or "") for paper in partial_matches[:5])
            )
        raise ValueError(f"paper_title not found: {raw_paper_title}")

    def score_paper_direct(self, query: str, paper_node_id: str) -> float:
        if self.hybrid is not None:
            # Hybrid scores use ranks because the two raw score scales differ.
            return next((score for paper, score, *_ in self.hybrid.rank_papers(query, len(self.papers))
                         if paper["node_id"] == paper_node_id), 0.0)
        if self.gnn is not None:
            return self.gnn.index.score(query, paper_node_id)
        if not self.papers:
            return 0.0
        paper_scores = self._cached_tfidf(query, self.paper_search_texts, "papers")
        for paper, score in zip(self.papers, paper_scores):
            if paper["node_id"] == paper_node_id:
                return float(score)
        return 0.0

    def build_paper_result(
        self,
        paper: dict[str, Any],
        query: str,
        top_k_sections: int,
        top_k_chunks: int,
        top_k_figures: int,
        total_score: float,
        direct_score: float,
        entity_support: list[dict[str, Any]],
        cross_paper_support: list[dict[str, Any]],
        figure_support: list[dict[str, Any]],
        retrieval_mode: str = "hierarchical",
    ) -> dict[str, Any]:
        query_types = detect_query_entity_types(query)
        matched_entities = sort_entity_support_for_query(
            rows=entity_support,
            query_types=query_types,
            top_k=self.config.top_k_entities,
        )
        cross_support_rows = summarize_cross_paper_support(
            rows=cross_paper_support,
            entity_by_node_id=self.entity_by_node_id,
            top_k=self.config.top_k_entities,
        )
        figure_support_rows = summarize_figure_support(figure_support, top_k=max(1, top_k_figures))
        effective_retrieval_mode = normalize_retrieval_mode(retrieval_mode)
        if effective_retrieval_mode == "paper_then_chunk":
            section_results = self.rank_sections_for_paper_without_routing(
                query=query,
                paper_id=paper["paper_id"],
                top_k_sections=top_k_sections,
                top_k_chunks=top_k_chunks,
                top_k_figures=top_k_figures,
                figure_support=figure_support_rows,
            )
        else:
            section_results = self.rank_sections_for_paper(
                query=query,
                paper_id=paper["paper_id"],
                top_k_sections=top_k_sections,
                top_k_chunks=top_k_chunks,
                top_k_figures=top_k_figures,
                entity_support=matched_entities,
                figure_support=figure_support_rows,
            )
        return {
            "paper_id": paper["paper_id"],
            "paper_title": paper.get("title") or "",
            "paper_author_text": paper.get("author_text") or "",
            "paper_institution_text": paper.get("institution_text") or "",
            "rank_score": round(float(total_score), 6),
            "paper_direct_score": round(float(direct_score), 6),
            "matched_entities": matched_entities,
            "cross_paper_support": cross_support_rows,
            "figure_support": figure_support_rows,
            "why_matched": build_paper_why_matched(
                direct_score=direct_score,
                matched_entities=matched_entities,
                cross_paper_support=cross_support_rows,
                figure_support=figure_support_rows,
            ),
            "top_sections": section_results,
        }

    def rank_sections_for_paper_without_routing(
        self,
        query: str,
        paper_id: str,
        top_k_sections: int,
        top_k_chunks: int,
        top_k_figures: int,
        figure_support: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if self.hybrid is not None:
            return self.hybrid.sections(query, paper_id, top_k_sections, top_k_chunks, top_k_figures,
                                        "paper_then_chunk", figure_support=figure_support)
        if self.gnn is not None:
            return self.gnn.sections(query, paper_id, top_k_sections, top_k_chunks, top_k_figures, "paper_then_chunk")
        sections = sorted(self.sections_by_paper.get(paper_id, []), key=lambda row: row["order"])
        if not sections:
            return []

        section_lookup = {
            str(section.get("section_id") or ""): section
            for section in sections
            if str(section.get("section_id") or "")
        }
        chunk_budget = max(0, int(top_k_sections)) * max(0, int(top_k_chunks))
        figure_budget = max(0, int(top_k_sections)) * max(0, int(top_k_figures))
        if chunk_budget <= 0 and figure_budget <= 0:
            return []

        preferred_figure_node_ids = {
            str(row.get("figure_node_id") or "")
            for row in (figure_support or [])
            if str(row.get("figure_node_id") or "")
        }
        chunk_hits = rank_local_records(
            query=query,
            rows=self.chunks_by_paper.get(paper_id, []),
            text_getter=lambda row: str(row.get("text") or ""),
            top_k=max(chunk_budget, 0),
            preview_chars=self.config.preview_chars,
            record_kind="chunk",
        )
        figure_hits = rank_local_records(
            query=query,
            rows=self.figures_by_paper.get(paper_id, []),
            text_getter=lambda row: str(row.get("search_text") or ""),
            top_k=max(figure_budget, 0),
            preview_chars=self.config.preview_chars,
            record_kind="figure",
            preferred_node_ids=preferred_figure_node_ids,
        )

        section_support: dict[str, dict[str, Any]] = {}
        for hit in chunk_hits:
            chunk = self.chunk_by_node_id.get(str(hit.get("node_id") or ""))
            section_id = str((chunk or {}).get("section_id") or "")
            if not section_id or section_id not in section_lookup:
                continue
            support = section_support.setdefault(section_id, {"chunk_hits": [], "figure_hits": []})
            support["chunk_hits"].append(hit)
        for hit in figure_hits:
            figure = self.figure_by_node_id.get(str(hit.get("node_id") or ""))
            section_id = str((figure or {}).get("section_id") or "")
            if not section_id or section_id not in section_lookup:
                continue
            support = section_support.setdefault(section_id, {"chunk_hits": [], "figure_hits": []})
            support["figure_hits"].append(hit)
        if not section_support:
            return []

        def section_rank_score(section_id: str) -> float:
            support = section_support.get(section_id) or {}
            chunk_scores = [float(row.get("rank_score") or 0.0) for row in support.get("chunk_hits") or []]
            figure_scores = [float(row.get("rank_score") or 0.0) for row in support.get("figure_hits") or []]
            scores = sorted(chunk_scores + figure_scores, reverse=True)
            if not scores:
                return 0.0
            best = scores[0]
            tail = sum(scores[1:3]) / max(1, len(scores[1:3])) if len(scores) > 1 else 0.0
            return best + (0.35 * tail)

        ranked_section_ids = sorted(
            section_support,
            key=lambda section_id: (
                -section_rank_score(section_id),
                int((section_lookup.get(section_id) or {}).get("order") or 0),
            ),
        )[: max(1, int(top_k_sections))]

        results: list[dict[str, Any]] = []
        for section_id in ranked_section_ids:
            section = section_lookup[section_id]
            section_node_id = str(section.get("node_id") or "")
            support = section_support.get(section_id) or {}
            ranked_chunk_hits = sorted(
                support.get("chunk_hits") or [],
                key=lambda item: (
                    -float(item.get("rank_score") or 0.0),
                    int((self.chunk_by_node_id.get(str(item.get("node_id") or "")) or {}).get("order") or 0),
                ),
            )
            ranked_figure_hits = sorted(
                support.get("figure_hits") or [],
                key=lambda item: (
                    -float(item.get("rank_score") or 0.0),
                    int((self.figure_by_node_id.get(str(item.get("node_id") or "")) or {}).get("order") or 0),
                ),
            )

            chunk_results: list[dict[str, Any]] = []
            seen_chunk_node_ids: set[str] = set()
            for row in ranked_chunk_hits:
                node_id = str(row.get("node_id") or "")
                if not node_id or node_id in seen_chunk_node_ids:
                    continue
                seen_chunk_node_ids.add(node_id)
                chunk_results.append(row)
                if len(chunk_results) >= max(0, int(top_k_chunks)):
                    break
            if len(chunk_results) < max(0, int(top_k_chunks)):
                local_chunk_rows = rank_local_records(
                    query=query,
                    rows=self.chunks_by_section.get(section_id, []),
                    text_getter=lambda row: str(row.get("text") or ""),
                    top_k=max(int(top_k_chunks), int(top_k_chunks) * 2),
                    preview_chars=self.config.preview_chars,
                    record_kind="chunk",
                )
                for row in local_chunk_rows:
                    node_id = str(row.get("node_id") or "")
                    if not node_id or node_id in seen_chunk_node_ids:
                        continue
                    seen_chunk_node_ids.add(node_id)
                    chunk_results.append(row)
                    if len(chunk_results) >= max(0, int(top_k_chunks)):
                        break

            figure_results: list[dict[str, Any]] = []
            seen_figure_node_ids: set[str] = set()
            for row in ranked_figure_hits:
                node_id = str(row.get("node_id") or "")
                if not node_id or node_id in seen_figure_node_ids:
                    continue
                seen_figure_node_ids.add(node_id)
                figure_results.append(row)
                if len(figure_results) >= max(0, int(top_k_figures)):
                    break
            if len(figure_results) < max(0, int(top_k_figures)):
                local_figure_rows = rank_local_records(
                    query=query,
                    rows=self.figures_by_section.get(section_id, []),
                    text_getter=lambda row: str(row.get("search_text") or ""),
                    top_k=max(int(top_k_figures), int(top_k_figures) * 2),
                    preview_chars=self.config.preview_chars,
                    record_kind="figure",
                    preferred_node_ids=preferred_figure_node_ids,
                )
                for row in local_figure_rows:
                    node_id = str(row.get("node_id") or "")
                    if not node_id or node_id in seen_figure_node_ids:
                        continue
                    seen_figure_node_ids.add(node_id)
                    figure_results.append(row)
                    if len(figure_results) >= max(0, int(top_k_figures)):
                        break

            entity_edges = self.section_entity_edges_by_section.get(section_node_id, [])
            best_support_score = 0.0
            if chunk_results:
                best_support_score = max(best_support_score, max(float(row.get("rank_score") or 0.0) for row in chunk_results))
            if figure_results:
                best_support_score = max(best_support_score, max(float(row.get("rank_score") or 0.0) for row in figure_results))
            results.append(
                {
                    "section_id": section_id,
                    "section_title": section.get("section_title") or "",
                    "section_group": section.get("section_group") or "",
                    "section_role": self.section_role_by_node_id.get(section_node_id, "other_content"),
                    "rank_score": round(float(section_rank_score(section_id)), 6),
                    "section_direct_score": round(float(best_support_score), 6),
                    "section_entity_overlap_score": 0.0,
                    "section_query_alignment_score": 0.0,
                    "section_visual_support_score": 0.0,
                    "section_summary": preview_text(str(section.get("section_summary") or ""), self.config.preview_chars),
                    "matched_entities": summarize_section_entities(entity_edges, self.entity_by_node_id),
                    "top_chunks": chunk_results,
                    "top_figures": figure_results,
                }
            )
        return results

    def rank_sections_for_paper(
        self,
        query: str,
        paper_id: str,
        top_k_sections: int,
        top_k_chunks: int,
        top_k_figures: int,
        entity_support: list[dict[str, Any]] | None = None,
        figure_support: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if self.hybrid is not None:
            return self.hybrid.sections(query, paper_id, top_k_sections, top_k_chunks, top_k_figures,
                                        "hierarchical", entity_support, figure_support)
        if self.gnn is not None:
            return self.gnn.sections(query, paper_id, top_k_sections, top_k_chunks, top_k_figures, "hierarchical")
        sections = sorted(self.sections_by_paper.get(paper_id, []), key=lambda row: row["order"])
        # A requested figure contributes its owner section without excluding
        # other sections that explain the same mechanism.
        requested_labels = figure_labels(query)
        matching_figures = {f["node_id"] for f in self.figures
            if f.get("paper_id") == paper_id and requested_labels & figure_labels(figure_caption_text(f))}
        if not sections:
            return []
        section_search_texts = [str(section.get("search_text") or "") for section in sections]
        section_scores = self._cached_tfidf(query, section_search_texts, ("sections", paper_id))
        # Weak fuzzy matches must not receive the fixed entity-overlap bonus.
        # Use the same confidence threshold as semantic query routing.
        matched_entities = [row for row in unique_entity_support(entity_support or [])
                            if row.get("match_source") == "exact"
                            or float(row.get("score") or 0.0) >= SECTION_ENTITY_MIN_SCORE]
        matched_entity_ids = {
            str(row.get("entity_node_id") or "")
            for row in matched_entities
            if str(row.get("entity_node_id") or "")
        }
        exact_entity_ids = {
            str(row.get("entity_node_id") or "")
            for row in matched_entities
            if str(row.get("match_source") or "") == "exact"
        }
        matched_section_ids = {
            str(row.get("via_section_id") or "")
            for row in matched_entities
            if str(row.get("via_section_id") or "")
        }
        query_types = detect_query_entity_types(query)
        section_focus = infer_section_query_focus(
            query=query,
            query_types=query_types,
            matched_entities=matched_entities,
        )
        figure_support_rows = unique_figure_support(figure_support or [])
        if matching_figures:
            figure_support_rows = [row for row in figure_support_rows if row.get("figure_node_id") in matching_figures]
        figure_support_by_section_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        preferred_figure_node_ids: set[str] = set()
        preferred_section_ids: set[str] = set()
        for row in figure_support_rows:
            section_id = str(row.get("section_id") or "")
            figure_node_id = str(row.get("figure_node_id") or "")
            if section_id:
                figure_support_by_section_id[section_id].append(row)
                preferred_section_ids.add(section_id)
            if figure_node_id:
                preferred_figure_node_ids.add(figure_node_id)

        ranked_sections = sorted(
            [
                build_ranked_section_row(
                    section=section,
                    direct_score=float(score),
                    overlap_score=self._score_section_entity_overlap(
                        section=section,
                        matched_entity_ids=matched_entity_ids,
                        exact_entity_ids=exact_entity_ids,
                        matched_section_ids=matched_section_ids,
                        query_types=query_types,
                    ),
                    alignment_score=score_section_query_alignment(
                        section=section,
                        section_focus=section_focus,
                    ),
                    visual_score=self._score_section_visual_support(
                        section=section,
                        figure_support_by_section_id=figure_support_by_section_id,
                    ),
                )
                for section, score in zip(sections, section_scores)
            ],
            key=lambda item: (-float(item["combined_score"]), int(item["section"].get("order") or 0)),
        )
        selected_sections = select_query_aware_sections(
            ranked_sections=ranked_sections,
            top_k_sections=top_k_sections,
            section_focus=section_focus,
            preferred_section_ids=preferred_section_ids,
        )

        results: list[dict[str, Any]] = []
        for row in selected_sections:
            section = row["section"]
            chunks = self.chunks_by_section.get(section["section_id"], [])
            figures = self.figures_by_section.get(section["section_id"], [])
            chunk_results = rank_local_records(
                query=query,
                rows=chunks,
                text_getter=lambda row: str(row.get("text") or ""),
                top_k=top_k_chunks,
                preview_chars=self.config.preview_chars,
                record_kind="chunk",
            )
            figure_results = rank_local_records(
                query=query,
                rows=figures,
                text_getter=lambda row: str(row.get("search_text") or ""),
                top_k=top_k_figures,
                preview_chars=self.config.preview_chars,
                record_kind="figure",
                preferred_node_ids=preferred_figure_node_ids,
            )
            entity_edges = self.section_entity_edges_by_section.get(section["node_id"], [])
            results.append(
                {
                    "section_id": section["section_id"],
                    "section_title": section.get("section_title") or "",
                    "section_group": section.get("section_group") or "",
                    "section_role": row["section_role"],
                    "rank_score": round(float(row["combined_score"]), 6),
                    "section_direct_score": round(float(row["direct_score"]), 6),
                    "section_entity_overlap_score": round(float(row["overlap_score"]), 6),
                    "section_query_alignment_score": round(float(row["alignment_score"]), 6),
                    "section_visual_support_score": round(float(row.get("visual_score") or 0.0), 6),
                    "section_summary": preview_text(str(section.get("section_summary") or ""), self.config.preview_chars),
                    "matched_entities": summarize_section_entities(entity_edges, self.entity_by_node_id),
                    "top_chunks": chunk_results,
                    "top_figures": figure_results,
                }
            )
        return results

    def find_exact_entity_matches(self, query: str) -> list[dict[str, Any]]:
        expected_types_global = detect_query_entity_types(query)
        matches_by_entity: dict[str, dict[str, Any]] = {}
        english_token_count = len(QUERY_EN_TOKEN_RE.findall(str(query or "")))
        ngram_max_n = min(24, max(10, english_token_count))

        for phrase, start, end in extract_quoted_spans(query):
            local_context = query[max(0, start - 12) : min(len(query), end + 12)]
            expected_types_local = detect_query_entity_types(local_context) or expected_types_global
            self._collect_exact_phrase_matches(
                phrase=phrase,
                expected_types=expected_types_local,
                match_source="quoted",
                matches_by_entity=matches_by_entity,
            )

        for phrase in build_query_english_ngrams(query, max_n=ngram_max_n):
            self._collect_exact_phrase_matches(
                phrase=phrase,
                expected_types=expected_types_global,
                match_source="ngram",
                matches_by_entity=matches_by_entity,
            )

        return sorted(
            matches_by_entity.values(),
            key=lambda item: (
                -float(item.get("match_score") or 0.0),
                -len(str(item.get("matched_text") or "")),
                str(item.get("entity_type") or ""),
                str(item.get("canonical_name") or ""),
            ),
        )

    def _collect_exact_phrase_matches(
        self,
        phrase: str,
        expected_types: set[str],
        match_source: str,
        matches_by_entity: dict[str, dict[str, Any]],
    ) -> None:
        seen_entity_ids: set[str] = set()
        for normalized_phrase in iter_lookup_variants(phrase):
            candidates = self.entity_alias_lookup.get(normalized_phrase, [])
            if not candidates:
                continue

            filtered_candidates = [
                candidate
                for candidate in candidates
                if not expected_types or str(candidate.get("entity_type") or "") in expected_types
            ]
            if not filtered_candidates:
                filtered_candidates = list(candidates)

            for candidate in filtered_candidates:
                entity_node_id = str(candidate["entity_node_id"])
                if entity_node_id in seen_entity_ids:
                    continue
                seen_entity_ids.add(entity_node_id)
                type_bias = 0.35 if str(candidate.get("entity_type") or "") in expected_types else 0.0
                match_score = 2.0 if match_source == "quoted" else 1.2
                match_score += type_bias
                current = matches_by_entity.get(entity_node_id)
                payload = {
                    "entity_node_id": entity_node_id,
                    "entity_type": candidate.get("entity_type"),
                    "canonical_name": candidate.get("canonical_name"),
                    "matched_text": normalize_match_text(phrase),
                    "match_source": match_source,
                    "type_bias": type_bias,
                    "match_score": match_score,
                }
                if current is None or (
                    float(payload["match_score"]) > float(current.get("match_score") or 0.0)
                ):
                    matches_by_entity[entity_node_id] = payload

    def _apply_entity_support(
        self,
        entity: dict[str, Any],
        entity_score: float,
        candidate_scores: dict[str, float],
        entity_support_by_paper: dict[str, list[dict[str, Any]]],
        match_source: str,
        matched_text: str | None = None,
    ) -> None:
        entity_node_id = str(entity["node_id"])
        for edge in self.paper_edges_by_entity.get(entity_node_id, []):
            paper_node_id = str(edge.get("source_id") or "")
            boost = entity_score * float(edge.get("confidence") or 1.0)
            candidate_scores[paper_node_id] += boost
            entity_support_by_paper[paper_node_id].append(
                make_entity_support_row(
                    entity=entity,
                    edge=edge,
                    score=entity_score,
                    match_source=match_source,
                    matched_text=matched_text,
                )
            )
        for edge in self.section_edges_by_entity.get(entity_node_id, []):
            section_node_id = str(edge.get("source_id") or "")
            section = self.section_by_node_id.get(section_node_id)
            if not section:
                continue
            paper_node_id = f"paper:{section['paper_id']}"
            boost = entity_score * float(edge.get("confidence") or 1.0) * 0.55
            candidate_scores[paper_node_id] += boost
            entity_support_by_paper[paper_node_id].append(
                make_entity_support_row(
                    entity=entity,
                    edge=edge,
                    score=entity_score,
                    match_source=match_source,
                    matched_text=matched_text,
                    via_section_id=str(section["section_id"]),
                )
            )

    def _apply_coverage_bonuses(
        self,
        candidate_scores: dict[str, float],
        entity_support_by_paper: dict[str, list[dict[str, Any]]],
    ) -> None:
        for paper_node_id, rows in entity_support_by_paper.items():
            unique_rows = unique_entity_support(rows)
            if not unique_rows:
                continue
            entity_ids = {
                str(row.get("entity_node_id") or "")
                for row in unique_rows
                if str(row.get("entity_node_id") or "")
            }
            entity_types = {
                str(row.get("entity_type") or "")
                for row in unique_rows
                if str(row.get("entity_type") or "")
            }
            exact_entity_ids = {
                str(row.get("entity_node_id") or "")
                for row in unique_rows
                if str(row.get("match_source") or "") == "exact"
            }
            section_ids = {
                str(row.get("via_section_id") or "")
                for row in unique_rows
                if str(row.get("via_section_id") or "")
            }

            bonus = 0.65 * len(entity_ids)
            bonus += 0.85 * len(entity_types)
            if len(entity_ids) >= 2:
                bonus += 1.1 * (len(entity_ids) - 1)
            if exact_entity_ids:
                bonus += 1.4 * len(exact_entity_ids)
                if len(exact_entity_ids) >= 2:
                    bonus += 2.2 * (len(exact_entity_ids) - 1)
            if section_ids:
                bonus += 0.25 * len(section_ids)
            candidate_scores[paper_node_id] += bonus

    def _apply_reference_node_fallback(
        self,
        query_types: set[str],
        exact_entity_matches: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        entity_support_by_paper: dict[str, list[dict[str, Any]]],
    ) -> None:
        if "reference" not in query_types:
            return

        reference_matches = [
            match
            for match in exact_entity_matches
            if str(match.get("entity_type") or "") == "reference"
        ]
        if not reference_matches:
            return

        target_records: list[dict[str, Any]] = []
        for match in reference_matches:
            entity = self.entity_by_node_id.get(str(match.get("entity_node_id") or ""))
            if not entity:
                continue
            target_variants = build_reference_match_variants(
                matched_text=str(match.get("matched_text") or ""),
                entity=entity,
            )
            if not target_variants:
                continue
            target_records.append(
                {
                    "entity_node_id": str(match.get("entity_node_id") or ""),
                    "target_variants": target_variants,
                }
            )
        if not target_records:
            return

        best_hits: dict[tuple[str, str], dict[str, Any]] = {}
        for paper_node_id, reference_entity_ids in self.reference_entity_ids_by_paper.items():
            deduped_reference_ids = dedupe_preserve_order(reference_entity_ids)
            for target in target_records:
                target_entity_node_id = str(target["entity_node_id"])
                if paper_node_id in self.paper_node_ids_by_entity.get(target_entity_node_id, set()):
                    continue
                best_hit: dict[str, Any] | None = None
                for candidate_entity_node_id in deduped_reference_ids:
                    candidate_entity = self.entity_by_node_id.get(str(candidate_entity_node_id))
                    if not candidate_entity:
                        continue
                    match_payload = score_reference_entity_match(
                        target_variants=target["target_variants"],
                        candidate_entity=candidate_entity,
                    )
                    if not match_payload:
                        continue
                    if best_hit is None or float(match_payload["score"]) > float(best_hit.get("score") or 0.0):
                        best_hit = {
                            "paper_node_id": paper_node_id,
                            "candidate_entity": candidate_entity,
                            "target_entity_node_id": target_entity_node_id,
                            "matched_text": str(match_payload.get("matched_text") or ""),
                            "score": float(match_payload["score"]),
                        }
                if best_hit:
                    best_hits[(paper_node_id, target_entity_node_id)] = best_hit

        for hit in best_hits.values():
            paper_node_id = str(hit["paper_node_id"])
            score = float(hit["score"])
            candidate_scores[paper_node_id] += score
            entity_support_by_paper[paper_node_id].append(
                make_entity_support_row(
                    entity=hit["candidate_entity"],
                    edge={"edge_type": "paper_cites_reference"},
                    score=score,
                    match_source="exact" if score >= 1.7 else "fuzzy",
                    matched_text=str(hit.get("matched_text") or ""),
                )
            )

    def _apply_reference_text_fallback(
        self,
        query_types: set[str],
        exact_entity_matches: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        entity_support_by_paper: dict[str, list[dict[str, Any]]],
    ) -> None:
        if "reference" not in query_types:
            return

        reference_matches = [
            match
            for match in exact_entity_matches
            if str(match.get("entity_type") or "") == "reference"
        ]
        if not reference_matches:
            return

        best_hits: dict[tuple[str, str], dict[str, Any]] = {}
        for match in reference_matches:
            entity_node_id = str(match.get("entity_node_id") or "")
            entity = self.entity_by_node_id.get(entity_node_id)
            if not entity:
                continue
            target_variants = build_reference_match_variants(
                matched_text=str(match.get("matched_text") or ""),
                entity=entity,
            )
            if not target_variants:
                continue

            for section in self.sections:
                section_node_id = str(section["node_id"])
                paper_node_id = f"paper:{section['paper_id']}"
                if paper_node_id in self.paper_node_ids_by_entity.get(entity_node_id, set()):
                    continue

                match_payload = score_reference_section_match(
                    section_lookup_text=self.section_lookup_text_by_node_id.get(section_node_id, ""),
                    section_compact_text=self.section_compact_text_by_node_id.get(section_node_id, ""),
                    section_role=self.section_role_by_node_id.get(section_node_id, ""),
                    target_variants=target_variants,
                )
                if not match_payload:
                    continue

                key = (paper_node_id, entity_node_id)
                current = best_hits.get(key)
                if current is None or float(match_payload["score"]) > float(current.get("score") or 0.0):
                    best_hits[key] = {
                        "paper_node_id": paper_node_id,
                        "section_id": str(section.get("section_id") or ""),
                        "entity": entity,
                        "matched_text": str(match_payload.get("matched_text") or ""),
                        "score": float(match_payload["score"]),
                    }

        for hit in best_hits.values():
            paper_node_id = str(hit["paper_node_id"])
            score = float(hit["score"])
            candidate_scores[paper_node_id] += score
            entity_support_by_paper[paper_node_id].append(
                make_entity_support_row(
                    entity=hit["entity"],
                    edge={"edge_type": "paper_cites_reference"},
                    score=score,
                    match_source="exact",
                    matched_text=str(hit.get("matched_text") or ""),
                    via_section_id=str(hit.get("section_id") or ""),
                )
            )

    def _filter_reference_group_paper_ids(self, group: dict[str, Any]) -> set[str]:
        target_variants: list[dict[str, str]] = []
        seen_targets: set[str] = set()

        display_name = normalize_match_text(str(group.get("display_name") or ""))
        matched_texts = [normalize_match_text(text) for text in (group.get("matched_texts") or [])]
        seed_text = next((text for text in [display_name, *matched_texts] if text), "")

        for entity_node_id in group.get("entity_node_ids") or set():
            entity = self.entity_by_node_id.get(str(entity_node_id))
            if not entity:
                continue
            for variant in build_reference_match_variants(seed_text, entity):
                compact_text = str(variant.get("compact_text") or "")
                if not compact_text or compact_text in seen_targets:
                    continue
                seen_targets.add(compact_text)
                target_variants.append(variant)

        if not target_variants:
            return set()

        best_hits: dict[str, float] = {}
        for entity_node_id in group.get("entity_node_ids") or set():
            for edge in self.paper_edges_by_entity.get(str(entity_node_id), []):
                if str(edge.get("edge_type") or "") != "paper_cites_reference":
                    continue
                paper_node_id = str(edge.get("source_id") or "")
                if not paper_node_id:
                    continue
                match_payload = score_reference_edge_match(target_variants, edge)
                if not match_payload:
                    continue
                match_score = float(match_payload.get("score") or 0.0)
                if match_score > float(best_hits.get(paper_node_id) or 0.0):
                    best_hits[paper_node_id] = match_score

        return {paper_node_id for paper_node_id, score in best_hits.items() if score >= 1.05}

    def _apply_cross_paper_support(
        self,
        query_types: set[str],
        candidate_scores: dict[str, float],
        direct_paper_scores: dict[str, float],
        entity_support_by_paper: dict[str, list[dict[str, Any]]],
        top_k_papers: int,
    ) -> dict[str, list[dict[str, Any]]]:
        if not self.cross_paper_edges_by_paper:
            return {}

        ranked_candidates = sorted(
            [
                (paper_node_id, float(score))
                for paper_node_id, score in candidate_scores.items()
                if paper_node_id in self.paper_by_node_id and float(score) >= self.config.min_score
            ],
            key=lambda item: (-item[1], item[0]),
        )[: max(18, top_k_papers * 6)]
        if not ranked_candidates:
            return {}

        max_anchor_score = max(float(score) for _, score in ranked_candidates) or 1.0
        cross_paper_support_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for anchor_paper_node_id, anchor_score in ranked_candidates:
            anchor_support_rows = unique_entity_support(
                entity_support_by_paper.get(anchor_paper_node_id, [])
            )
            anchor_entity_ids = {
                str(row.get("entity_node_id") or "")
                for row in anchor_support_rows
                if str(row.get("entity_node_id") or "")
            }
            anchor_entity_types = {
                str(row.get("entity_type") or "")
                for row in anchor_support_rows
                if str(row.get("entity_type") or "")
            }
            if not anchor_entity_ids and float(anchor_score) < (self.config.min_score * 2.0):
                continue

            for edge in self.cross_paper_edges_by_paper.get(anchor_paper_node_id, []):
                support_row = self._build_cross_paper_support_row(
                    edge=edge,
                    anchor_paper_node_id=anchor_paper_node_id,
                    anchor_score=float(anchor_score),
                    max_anchor_score=max_anchor_score,
                    query_types=query_types,
                    anchor_entity_ids=anchor_entity_ids,
                    anchor_entity_types=anchor_entity_types,
                    direct_paper_scores=direct_paper_scores,
                )
                if not support_row:
                    continue
                target_paper_node_id = str(support_row["target_paper_node_id"])
                candidate_scores[target_paper_node_id] += float(support_row["boost"])
                cross_paper_support_by_paper[target_paper_node_id].append(support_row)

        return cross_paper_support_by_paper

    def _apply_structured_query_filter(
        self,
        query_types: set[str],
        exact_entity_matches: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        entity_support_by_paper: dict[str, list[dict[str, Any]]],
    ) -> None:
        focus_types = query_types & STRUCTURED_QUERY_FILTER_TYPES
        if not focus_types or not exact_entity_matches:
            return

        exact_entity_ids_by_type: dict[str, set[str]] = defaultdict(set)
        for match in exact_entity_matches:
            entity_type = str(match.get("entity_type") or "")
            if entity_type in focus_types:
                entity_node_id = str(match.get("entity_node_id") or "")
                if entity_node_id:
                    exact_entity_ids_by_type[entity_type].add(entity_node_id)
        if not exact_entity_ids_by_type:
            return

        allowed_sets: list[set[str]] = []
        for entity_type, exact_entity_ids in exact_entity_ids_by_type.items():
            allowed_for_type = build_allowed_exact_support_set(
                entity_type=entity_type,
                exact_entity_ids=exact_entity_ids,
                entity_support_by_paper=entity_support_by_paper,
                paper_node_ids_by_entity=self.paper_node_ids_by_entity,
            )
            if allowed_for_type:
                allowed_sets.append(allowed_for_type)
        if not allowed_sets:
            return

        allowed_paper_node_ids = set.intersection(*allowed_sets) if len(allowed_sets) >= 2 else set(allowed_sets[0])
        if not allowed_paper_node_ids:
            return

        for paper_node_id in list(candidate_scores.keys()):
            if paper_node_id in allowed_paper_node_ids:
                candidate_scores[paper_node_id] += 0.35
                continue
            candidate_scores[paper_node_id] -= 24.0

    def _apply_semantic_query_boosts(
        self,
        query: str,
        query_types: set[str],
        exact_entity_matches: list[dict[str, Any]],
        candidate_scores: dict[str, float],
    ) -> None:
        semantic_query_types = query_types & SEMANTIC_QUERY_ENTITY_TYPES
        if not semantic_query_types:
            return

        concept_groups = build_semantic_exact_match_groups(
            exact_entity_matches=exact_entity_matches,
            semantic_query_types=semantic_query_types,
        )
        if not concept_groups:
            return

        paper_concepts: dict[str, set[str]] = defaultdict(set)
        for concept_key, matches in concept_groups.items():
            matched_paper_node_ids: set[str] = set()
            for match in matches:
                entity_node_id = str(match.get("entity_node_id") or "")
                if not entity_node_id:
                    continue
                matched_paper_node_ids.update(self.paper_node_ids_by_entity.get(entity_node_id, set()))
            for paper_node_id in matched_paper_node_ids:
                paper_concepts[paper_node_id].add(concept_key)

        if not paper_concepts:
            return

        total_concepts = len(concept_groups)
        max_coverage = max((len(values) for values in paper_concepts.values()), default=0)
        symbolic_handle_query = bool(extract_symbolic_query_handles(query))
        visual_or_symbolic_query = is_visual_query(query) or symbolic_handle_query
        conjunction_like = has_semantic_conjunction(query) and total_concepts >= 2
        full_coverage_papers = {
            paper_node_id
            for paper_node_id, covered_concepts in paper_concepts.items()
            if len(covered_concepts) == total_concepts
        }

        for paper_node_id, covered_concepts in paper_concepts.items():
            coverage = len(covered_concepts)
            bonus = 1.2 * coverage
            if coverage >= 2:
                bonus += 1.6 * (coverage - 1)
            if coverage == total_concepts:
                bonus += 4.8 + (1.4 * total_concepts)
            elif coverage == max_coverage and max_coverage >= 2:
                bonus += 2.0
            candidate_scores[paper_node_id] += bonus

        if visual_or_symbolic_query:
            return

        if conjunction_like and full_coverage_papers:
            for paper_node_id in list(candidate_scores.keys()):
                if paper_node_id in full_coverage_papers:
                    candidate_scores[paper_node_id] += 0.5
                    continue
                candidate_scores[paper_node_id] = min(
                    float(candidate_scores.get(paper_node_id) or 0.0),
                    self.config.min_score * 0.1,
                ) - 1.0
            return

        if total_concepts == 1:
            only_concept = next(iter(concept_groups.keys()))
            concept_papers = {
                paper_node_id
                for paper_node_id, covered_concepts in paper_concepts.items()
                if only_concept in covered_concepts
            }
            if concept_papers:
                for paper_node_id in list(candidate_scores.keys()):
                    if paper_node_id in concept_papers:
                        candidate_scores[paper_node_id] += 0.25
                        continue
                    candidate_scores[paper_node_id] -= 18.0

    def _build_cross_paper_support_row(
        self,
        edge: dict[str, Any],
        anchor_paper_node_id: str,
        anchor_score: float,
        max_anchor_score: float,
        query_types: set[str],
        anchor_entity_ids: set[str],
        anchor_entity_types: set[str],
        direct_paper_scores: dict[str, float],
    ) -> dict[str, Any] | None:
        target_paper_node_id = str(edge.get("target_id") or "")
        if target_paper_node_id not in self.paper_by_node_id or target_paper_node_id == anchor_paper_node_id:
            return None

        edge_type = str(edge.get("edge_type") or "")
        evidence_entity_ids = [
            str(entity_node_id)
            for entity_node_id in (edge.get("evidence_entity_ids") or [])
            if str(entity_node_id or "")
        ]
        evidence_entity_types = {
            str(entity_type)
            for entity_type in (edge.get("evidence_entity_types") or [])
            if str(entity_type or "")
        }
        relation_entity_types = set(CROSS_PAPER_EDGE_TO_ENTITY_TYPES.get(edge_type, set()))
        overlap_entity_ids = [entity_node_id for entity_node_id in evidence_entity_ids if entity_node_id in anchor_entity_ids]
        overlap_types = (evidence_entity_types & anchor_entity_types) | (relation_entity_types & query_types)

        if not overlap_entity_ids and not overlap_types:
            return None
        if float(direct_paper_scores.get(target_paper_node_id) or 0.0) < (self.config.min_score * 0.4) and not overlap_entity_ids:
            return None

        relation_weight = float(CROSS_PAPER_EDGE_TYPE_WEIGHTS.get(edge_type) or 0.85)
        confidence = float(edge.get("confidence") or 1.0)
        shared_count = float(edge.get("shared_count") or 0.0)
        shared_factor = 1.0 + (min(shared_count, 6.0) * 0.08)
        anchor_strength = min(1.35, float(anchor_score) / max(max_anchor_score, self.config.min_score))

        relevance = 0.0
        if overlap_entity_ids:
            relevance += 1.2 + (0.28 * len(overlap_entity_ids))
        if evidence_entity_types & anchor_entity_types:
            relevance += 0.55
        if relation_entity_types & query_types:
            relevance += 0.6
        if query_types and evidence_entity_types & query_types:
            relevance += 0.45
        if relevance <= 0.0:
            return None

        boost = min(4.0, relation_weight * shared_factor * confidence * anchor_strength * relevance * 0.72)
        if boost < 0.15:
            return None

        source_paper = self.paper_by_node_id.get(anchor_paper_node_id) or {}
        pair_summary = self.paper_pair_summary_by_key.get(
            make_paper_pair_key(anchor_paper_node_id, target_paper_node_id) or ("", "")
        ) or {}
        return {
            "target_paper_node_id": target_paper_node_id,
            "source_paper_id": source_paper.get("paper_id") or "",
            "source_paper_title": source_paper.get("title") or "",
            "edge_type": edge_type,
            "boost": round(float(boost), 6),
            "confidence": confidence,
            "shared_count": edge.get("shared_count"),
            "evidence_entity_ids": evidence_entity_ids,
            "evidence_entity_types": sorted(evidence_entity_types or relation_entity_types),
            "overlap_entity_ids": overlap_entity_ids,
            "relation_types": list(pair_summary.get("relation_types") or [edge_type]),
            "pair_total_weight": pair_summary.get("total_weight"),
        }

    def _score_section_entity_overlap(
        self,
        section: dict[str, Any],
        matched_entity_ids: set[str],
        exact_entity_ids: set[str],
        matched_section_ids: set[str],
        query_types: set[str],
    ) -> float:
        section_edges = self.section_entity_edges_by_section.get(str(section["node_id"]), [])
        if not section_edges:
            return 0.0

        section_entity_ids = {
            str(edge.get("target_id") or "")
            for edge in section_edges
            if str(edge.get("target_id") or "")
        }
        section_entity_types = {
            str((self.entity_by_node_id.get(str(edge.get("target_id") or "")) or {}).get("node_type") or "")
            for edge in section_edges
            if str((self.entity_by_node_id.get(str(edge.get("target_id") or "")) or {}).get("node_type") or "")
        }
        overlap_entity_ids = section_entity_ids & matched_entity_ids
        exact_overlap_ids = overlap_entity_ids & exact_entity_ids

        boost = 0.0
        if overlap_entity_ids:
            boost += 1.1 + (0.25 * len(overlap_entity_ids))
        if exact_overlap_ids:
            boost += 1.3 + (0.35 * len(exact_overlap_ids))
        if str(section.get("section_id") or "") in matched_section_ids:
            boost += 1.6
        if query_types and section_entity_types & query_types:
            boost += 0.25
        return boost

    def _score_section_visual_support(
        self,
        section: dict[str, Any],
        figure_support_by_section_id: dict[str, list[dict[str, Any]]],
    ) -> float:
        support_rows = figure_support_by_section_id.get(str(section.get("section_id") or ""), [])
        if not support_rows:
            return 0.0
        best_score = max(float(row.get("rank_score") or 0.0) for row in support_rows)
        bonus = 1.8 + (4.2 * best_score)
        if any(str(row.get("caption_preview") or "").strip() for row in support_rows):
            bonus += 0.25
        return bonus

    def _apply_visual_figure_support(
        self,
        query: str,
        candidate_scores: dict[str, float],
        figure_support_by_paper: dict[str, list[dict[str, Any]]],
        top_k_papers: int,
    ) -> None:
        ranked_figures = rank_figure_records(
            query=query,
            rows=self.figures,
            top_k=max(6, top_k_papers * 4),
            preview_chars=self.config.preview_chars,
            cached_scorer=self._cached_tfidf,
        )
        for rank, row in enumerate(ranked_figures):
            figure_node_id = str(row.get("node_id") or "")
            figure = self.figure_by_node_id.get(figure_node_id)
            if not figure:
                continue
            paper_node_id = str(figure.get("paper_node_id") or f"paper:{figure.get('paper_id') or ''}")
            if paper_node_id not in self.paper_by_node_id:
                continue
            rank_discount = max(0.45, 1.0 - (0.08 * rank))
            bonus = (72.0 * float(row.get("rank_score") or 0.0) * rank_discount)
            bonus += 10.0 * float(row.get("caption_score") or 0.0)
            bonus += 4.0 * float(row.get("label_score") or 0.0)
            candidate_scores[paper_node_id] += bonus
            figure_support_by_paper[paper_node_id].append(
                {
                    "figure_node_id": figure_node_id,
                    "figure_id": str(figure.get("figure_id") or ""),
                    "section_id": str(figure.get("section_id") or ""),
                    "section_node_id": str(figure.get("section_node_id") or ""),
                    "asset_kind": str(figure.get("asset_kind") or ""),
                    "rank_score": round(float(row.get("rank_score") or 0.0), 6),
                    "base_score": round(float(row.get("base_score") or 0.0), 6),
                    "caption_score": round(float(row.get("caption_score") or 0.0), 6),
                    "label_score": round(float(row.get("label_score") or 0.0), 6),
                    "page_start": figure.get("page_start"),
                    "image_path": str(figure.get("image_path") or ""),
                    "caption_preview": str(row.get("caption_preview") or ""),
                }
            )

    def _build_entity_alias_lookup(self) -> None:
        seen: set[tuple[str, str]] = set()
        for entity in self.entity_nodes:
            entity_node_id = str(entity["node_id"])
            canonical_name = str(entity.get("canonical_name") or entity.get("title") or "")
            for alias_text in iter_entity_alias_texts(entity):
                for normalized_alias in iter_lookup_variants(alias_text):
                    key = (normalized_alias, entity_node_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    self.entity_alias_lookup[normalized_alias].append(
                        {
                            "entity_node_id": entity_node_id,
                            "entity_type": str(entity.get("node_type") or ""),
                            "canonical_name": canonical_name,
                        }
                    )
                    self.entity_alias_lengths[normalized_alias] = max(
                        len(normalized_alias),
                        int(self.entity_alias_lengths.get(normalized_alias) or 0),
                    )

    def _apply_title_query_boosts(
        self,
        title_phrases: list[str],
        candidate_scores: dict[str, float],
    ) -> None:
        normalized_targets: list[tuple[str, str]] = []
        seen_targets: set[str] = set()
        for phrase in title_phrases:
            normalized_target = normalize_lookup_text(phrase)
            compact_target = compact_lookup_text(phrase)
            if not normalized_target or not compact_target or compact_target in seen_targets:
                continue
            seen_targets.add(compact_target)
            normalized_targets.append((normalized_target, compact_target))
        if not normalized_targets:
            return
        matched_paper_node_ids: set[str] = set()
        for paper in self.papers:
            paper_node_id = str(paper["node_id"])
            title_norm = normalize_lookup_text(str(paper.get("title") or ""))
            title_compact = compact_lookup_text(str(paper.get("title") or ""))
            if not title_norm:
                continue
            title_prefix = str(paper.get("title") or "").split(":", 1)[0]
            title_prefix_norm = normalize_lookup_text(title_prefix)
            title_prefix_compact = compact_lookup_text(title_prefix)
            bonus = 0.0
            for target_norm, target_compact in normalized_targets:
                if title_prefix_norm == target_norm or title_prefix_compact == target_compact:
                    bonus = max(bonus, 168.0)
                elif title_norm == target_norm or title_compact == target_compact:
                    bonus = max(bonus, 120.0)
                elif (
                    len(target_compact) >= 10
                    and (
                        target_norm in title_norm
                        or title_norm in target_norm
                        or target_compact in title_compact
                        or title_compact in target_compact
                    )
                ):
                    bonus = max(bonus, 52.0)
            if bonus > 0.0:
                matched_paper_node_ids.add(paper_node_id)
                candidate_scores[paper_node_id] += bonus

        if not matched_paper_node_ids:
            return

        isolation_penalty = 34.0 if len(normalized_targets) <= 2 else 24.0
        for paper in self.papers:
            paper_node_id = str(paper["node_id"])
            if paper_node_id in matched_paper_node_ids:
                candidate_scores[paper_node_id] += 6.0
                continue
            candidate_scores[paper_node_id] -= isolation_penalty

    def _apply_symbolic_query_boosts(
        self,
        symbolic_handles: list[str],
        candidate_scores: dict[str, float],
    ) -> None:
        normalized_handles = []
        seen_handles: set[str] = set()
        for handle in symbolic_handles:
            for variant in iter_lookup_variants(handle):
                compact_variant = compact_lookup_text(variant)
                if len(compact_variant) < 4 or compact_variant in seen_handles:
                    continue
                seen_handles.add(compact_variant)
                normalized_handles.append((variant, compact_variant))
        if not normalized_handles:
            return

        unique_prefix_matches: dict[str, list[str]] = defaultdict(list)
        for variant, compact_variant in normalized_handles:
            for paper in self.papers:
                paper_node_id = str(paper.get("node_id") or "")
                title_text = str(paper.get("title") or "")
                title_prefix = title_text.split(":", 1)[0]
                title_prefix_norm = normalize_lookup_text(title_prefix)
                title_prefix_compact = compact_lookup_text(title_prefix)
                if title_prefix_norm == variant or title_prefix_compact == compact_variant:
                    unique_prefix_matches[compact_variant].append(paper_node_id)

        for paper in self.papers:
            paper_node_id = str(paper.get("node_id") or "")
            title_text = str(paper.get("title") or "")
            title_norm = normalize_lookup_text(title_text)
            title_compact = compact_lookup_text(title_text)
            title_prefix = title_text.split(":", 1)[0]
            title_prefix_norm = normalize_lookup_text(title_prefix)
            title_prefix_compact = compact_lookup_text(title_prefix)
            figure_text = normalize_lookup_text(
                " ".join(
                    " ".join(
                        filter(
                            None,
                            [
                                figure_caption_text(figure),
                                figure_reference_label_text(figure),
                                str(figure.get("search_text") or ""),
                            ],
                        )
                    )
                    for figure in self.figures_by_paper.get(str(paper.get("paper_id") or ""), [])
                )
            )
            figure_compact = compact_lookup_text(figure_text)

            bonus = 0.0
            for variant, compact_variant in normalized_handles:
                if title_prefix_norm == variant or title_prefix_compact == compact_variant:
                    bonus = max(bonus, 180.0)
                elif title_norm == variant or title_compact == compact_variant:
                    bonus = max(bonus, 110.0)
                elif variant in title_norm or compact_variant in title_compact:
                    bonus = max(bonus, 96.0)
                elif variant in figure_text or compact_variant in figure_compact:
                    bonus = max(bonus, 48.0)
                if unique_prefix_matches.get(compact_variant) == [paper_node_id]:
                    bonus = max(bonus, 340.0)
            if bonus > 0.0:
                candidate_scores[paper_node_id] += bonus


def build_ranked_section_row(
    section: dict[str, Any],
    direct_score: float,
    overlap_score: float,
    alignment_score: float,
    visual_score: float = 0.0,
) -> dict[str, Any]:
    section_role = classify_query_section_role(section)
    return {
        "section": section,
        "section_role": section_role,
        "direct_score": float(direct_score),
        "overlap_score": float(overlap_score),
        "alignment_score": float(alignment_score),
        "visual_score": float(visual_score),
        "combined_score": float(direct_score)
        + float(overlap_score)
        + float(alignment_score)
        + float(visual_score),
    }


def infer_section_query_focus(
    query: str,
    query_types: set[str],
    matched_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    structured_types = set(query_types & STRUCTURED_QUERY_FILTER_TYPES)
    semantic_types = set(query_types & SEMANTIC_QUERY_ENTITY_TYPES)

    exact_entity_types = {
        str(row.get("entity_type") or "")
        for row in matched_entities
        if str(row.get("match_source") or "") == "exact" and str(row.get("entity_type") or "")
    }
    high_conf_entity_types = {
        str(row.get("entity_type") or "")
        for row in matched_entities
        if str(row.get("entity_type") or "")
        and float(row.get("score") or 0.0) >= SECTION_ENTITY_MIN_SCORE
    }

    if not semantic_types:
        semantic_types |= exact_entity_types & SEMANTIC_QUERY_ENTITY_TYPES
    if not semantic_types:
        semantic_types |= high_conf_entity_types & SEMANTIC_QUERY_ENTITY_TYPES

    if not structured_types:
        structured_types |= exact_entity_types & SECTION_QUERY_METADATA_TYPES

    reference_exact_only = (
        "reference" in exact_entity_types
        and not semantic_types
        and not (structured_types & SECTION_QUERY_METADATA_TYPES)
    )
    reference_query = "reference" in structured_types or reference_exact_only
    metadata_types = structured_types & SECTION_QUERY_METADATA_TYPES
    explanation_query = bool(re.search(r"\b(explain|how|why|compare|differ|differences?|mechanism|architecture|workflow)\b", query, re.I))
    semantic_query = bool(semantic_types) or (explanation_query and not metadata_types and not reference_query)
    metadata_query = bool(metadata_types) and not semantic_query and not reference_query
    mixed_query = bool(metadata_types) and semantic_query

    return {
        "structured_types": structured_types,
        "semantic_types": semantic_types,
        "metadata_types": metadata_types,
        "reference_query": reference_query,
        "semantic_query": semantic_query,
        "metadata_query": metadata_query,
        "mixed_query": mixed_query,
    }


def annotate_section_role_context(sections):
    """Keep related-work subsections from masquerading as proposed methods.

    MinerU can flatten heading levels; dotted heading numbers still identify
    an existing parent within the same paper. This only annotates query roles.
    """
    headings = {}
    for section in sections:
        match = re.match(r"^([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)\s+(.+)", str(section.get("section_title") or ""))
        if match:
            headings[(section["paper_id"], match[1])] = match[2]
    for section in sections:
        match = re.match(r"^([0-9]+(?:\.[0-9]+)*|[A-Z](?:\.[0-9]+)*)\s+", str(section.get("section_title") or ""))
        if not match:
            continue
        parts = match[1].split(".")
        titles = [headings.get((section["paper_id"], ".".join(parts[:end])), "") for end in range(1, len(parts) + 1)]
        if any(re.search(r"\b(related work|background|literature review)\b", title, re.I) for title in titles):
            section["query_role_context"] = "background"


def classify_query_section_role(section: dict[str, Any]) -> str:
    title = normalize_lookup_text(str(section.get("section_title") or ""))
    group = str(section.get("section_group") or "").strip().lower()

    if SECTION_TITLE_REFERENCE_RE.search(title):
        return "citation"
    if SECTION_TITLE_ADMIN_RE.search(title):
        return "admin"
    if section.get("query_role_context") == "background":
        return "other_content"
    if group == "front_matter":
        return "metadata"
    if group == "abstract":
        return "abstract"
    if group == "introduction":
        return "introduction"
    if group == "results" or SECTION_TITLE_RESULT_RE.search(title):
        return "results"
    if group == "methods" or SECTION_TITLE_METHOD_RE.search(title):
        return "methods"
    if group == "discussion":
        return "discussion"
    if group == "conclusion":
        return "conclusion"
    return "other_content"


def score_section_query_alignment(
    section: dict[str, Any],
    section_focus: dict[str, Any],
) -> float:
    role = classify_query_section_role(section)
    reference_query = bool(section_focus.get("reference_query"))
    semantic_query = bool(section_focus.get("semantic_query"))
    metadata_query = bool(section_focus.get("metadata_query"))
    mixed_query = bool(section_focus.get("mixed_query"))

    if mixed_query:
        return score_mixed_query_section_role(role=role, reference_query=reference_query)
    if semantic_query:
        return score_semantic_query_section_role(role=role, reference_query=reference_query)
    if reference_query:
        return score_reference_query_section_role(role=role)
    if metadata_query:
        return score_metadata_query_section_role(role=role)
    return score_default_query_section_role(role=role)


def score_metadata_query_section_role(role: str) -> float:
    if role == "metadata":
        return 1.65
    if role == "abstract":
        return 0.25
    if role == "introduction":
        return -0.05
    if role == "admin":
        return -1.05
    if role == "citation":
        return -0.75
    if role in CONTENT_SECTION_ROLES:
        return -0.45
    return -0.2


def score_reference_query_section_role(role: str) -> float:
    if role == "citation":
        return 1.75
    if role == "metadata":
        return -0.65
    if role == "admin":
        return -1.0
    if role == "abstract":
        return 0.12
    if role == "introduction":
        return 0.08
    return 0.0


def score_semantic_query_section_role(role: str, reference_query: bool) -> float:
    if role == "methods":
        return 1.15
    if role == "results":
        return 0.82
    if role == "abstract":
        return 0.52
    if role == "introduction":
        return 0.24
    if role == "discussion":
        return 0.08
    if role == "conclusion":
        return -0.04
    if role == "other_content":
        return 0.12
    if role == "citation":
        return -0.95 if reference_query else -1.3
    if role == "admin":
        return -1.45
    if role == "metadata":
        return -2.25
    return 0.0


def score_mixed_query_section_role(role: str, reference_query: bool) -> float:
    if role == "metadata":
        return 0.72
    if role == "methods":
        return 0.88
    if role == "results":
        return 0.62
    if role == "abstract":
        return 0.52
    if role == "introduction":
        return 0.18
    if role == "other_content":
        return 0.12
    if role == "discussion":
        return 0.04
    if role == "conclusion":
        return -0.02
    if role == "citation":
        return 0.7 if reference_query else -0.7
    if role == "admin":
        return -0.9
    return 0.0


def score_default_query_section_role(role: str) -> float:
    if role == "admin":
        return -0.45
    if role == "citation":
        return -0.18
    return 0.0


def select_query_aware_sections(
    ranked_sections: list[dict[str, Any]],
    top_k_sections: int,
    section_focus: dict[str, Any],
    preferred_section_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if top_k_sections <= 0 or not ranked_sections:
        return []
    if len(ranked_sections) <= top_k_sections:
        return ranked_sections

    selected: list[dict[str, Any]] = []
    used_section_ids: set[str] = set()

    def add_first_matching(roles: set[str]) -> None:
        for row in ranked_sections:
            section_id = str((row.get("section") or {}).get("section_id") or "")
            if not section_id or section_id in used_section_ids:
                continue
            if str(row.get("section_role") or "") not in roles:
                continue
            used_section_ids.add(section_id)
            selected.append(row)
            return

    if preferred_section_ids:
        for row in ranked_sections:
            if len(selected) >= top_k_sections:
                break
            section_id = str((row.get("section") or {}).get("section_id") or "")
            if not section_id or section_id in used_section_ids or section_id not in preferred_section_ids:
                continue
            used_section_ids.add(section_id)
            selected.append(row)

    if bool(section_focus.get("mixed_query")):
        add_first_matching({"metadata"})
        add_first_matching({"methods", "results", "abstract", "introduction", "other_content"})
        if bool(section_focus.get("reference_query")):
            add_first_matching({"citation"})
    elif bool(section_focus.get("metadata_query")):
        add_first_matching({"metadata"})
        add_first_matching({"abstract"})
    elif bool(section_focus.get("reference_query")) and not bool(section_focus.get("semantic_query")):
        add_first_matching({"citation"})
        add_first_matching({"abstract", "introduction"})
    elif bool(section_focus.get("semantic_query")):
        add_first_matching({"methods"})
        add_first_matching({"results"})
        add_first_matching({"abstract"})

    for row in ranked_sections:
        if len(selected) >= top_k_sections:
            break
        section_id = str((row.get("section") or {}).get("section_id") or "")
        if not section_id or section_id in used_section_ids:
            continue
        used_section_ids.add(section_id)
        selected.append(row)

    return sorted(
        selected[:top_k_sections],
        key=lambda item: (
            -float(item.get("combined_score") or 0.0),
            int((item.get("section") or {}).get("order") or 0),
        ),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def make_paper_pair_key(
    paper_a_node_id: str,
    paper_b_node_id: str,
) -> tuple[str, str] | None:
    left = str(paper_a_node_id or "").strip()
    right = str(paper_b_node_id or "").strip()
    if not left or not right:
        return None
    return tuple(sorted((left, right)))


def summarize_cross_paper_support(
    rows: list[dict[str, Any]],
    entity_by_node_id: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("source_paper_id") or ""),
            str(row.get("edge_type") or ""),
        )
        current = merged.get(key)
        if current is None:
            current = {
                "source_paper_id": str(row.get("source_paper_id") or ""),
                "source_paper_title": str(row.get("source_paper_title") or ""),
                "edge_type": str(row.get("edge_type") or ""),
                "boost": float(row.get("boost") or 0.0),
                "shared_count": row.get("shared_count"),
                "evidence_entity_ids": [],
                "evidence_entity_types": [],
                "relation_types": [],
                "pair_total_weight": row.get("pair_total_weight"),
            }
            merged[key] = current
        current["boost"] = max(float(current.get("boost") or 0.0), float(row.get("boost") or 0.0))
        current["shared_count"] = max(
            float(current.get("shared_count") or 0.0),
            float(row.get("shared_count") or 0.0),
        )
        current["pair_total_weight"] = max(
            float(current.get("pair_total_weight") or 0.0),
            float(row.get("pair_total_weight") or 0.0),
        )
        current["evidence_entity_ids"] = dedupe_preserve_order(
            list(current.get("evidence_entity_ids") or [])
            + list(row.get("overlap_entity_ids") or row.get("evidence_entity_ids") or [])
        )
        current["evidence_entity_types"] = dedupe_preserve_order(
            list(current.get("evidence_entity_types") or [])
            + list(row.get("evidence_entity_types") or [])
        )
        current["relation_types"] = dedupe_preserve_order(
            list(current.get("relation_types") or []) + list(row.get("relation_types") or [])
        )

    summarized: list[dict[str, Any]] = []
    for row in merged.values():
        evidence_names: list[str] = []
        for entity_node_id in list(row.get("evidence_entity_ids") or [])[:6]:
            entity = entity_by_node_id.get(str(entity_node_id))
            if not entity:
                continue
            evidence_name = str(entity.get("canonical_name") or entity.get("title") or "").strip()
            if not evidence_name and str(entity.get("node_type") or "") == "reference":
                evidence_name = best_reference_display_name(entity)
            if evidence_name:
                evidence_names.append(evidence_name)
        summarized.append(
            {
                "source_paper_id": row.get("source_paper_id") or "",
                "source_paper_title": row.get("source_paper_title") or "",
                "edge_type": row.get("edge_type") or "",
                "boost": round(float(row.get("boost") or 0.0), 6),
                "shared_count": int(float(row.get("shared_count") or 0.0)),
                "evidence_entity_types": list(row.get("evidence_entity_types") or []),
                "evidence_entity_names": evidence_names,
                "relation_types": list(row.get("relation_types") or []),
                "pair_total_weight": round(float(row.get("pair_total_weight") or 0.0), 6),
            }
        )
    return sorted(
        summarized,
        key=lambda item: (-float(item.get("boost") or 0.0), str(item.get("source_paper_id") or "")),
    )[:top_k]


def unique_figure_support(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        figure_node_id = str(row.get("figure_node_id") or row.get("node_id") or "")
        if not figure_node_id:
            continue
        current = unique.get(figure_node_id)
        if current is None or float(row.get("rank_score") or 0.0) > float(current.get("rank_score") or 0.0):
            unique[figure_node_id] = row
    return list(unique.values())


def summarize_figure_support(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return sorted(
        unique_figure_support(rows),
        key=lambda item: (
            -float(item.get("rank_score") or 0.0),
            -float(item.get("caption_score") or 0.0),
            str(item.get("figure_id") or item.get("figure_node_id") or ""),
        ),
    )[:top_k]


def is_visual_query(query: str) -> bool:
    text = str(query or "")
    # A graph as a data structure does not by itself request visual evidence.
    return bool(ENGLISH_VISUAL_QUERY_RE.search(text) or
                re.search(r"\b(show|display|draw)\b.{0,60}\bgraph\b", text, re.I))


def figure_query_text(figure: dict[str, Any]) -> str:
    return str(figure.get("search_text") or "").strip()


def figure_caption_text(figure: dict[str, Any]) -> str:
    return " ".join(str(item).strip() for item in (figure.get("caption") or []) if str(item).strip())


def figure_reference_label_text(figure: dict[str, Any]) -> str:
    return " ".join(str(item).strip() for item in (figure.get("reference_labels") or []) if str(item).strip())


def preferred_asset_kinds_for_query(query: str) -> set[str]:
    query_lower = str(query or "").lower()
    preferred: set[str] = set()
    for token, asset_kinds in VISUAL_ASSET_KIND_PREFERENCE.items():
        if token in query_lower:
            preferred.update(asset_kinds)
    return preferred


def figure_kind_bonus(query: str, figure: dict[str, Any]) -> float:
    preferred = preferred_asset_kinds_for_query(query)
    if not preferred:
        return 0.0
    asset_kind = str(figure.get("asset_kind") or "").lower()
    if asset_kind in preferred:
        return FIGURE_KIND_BIAS
    if asset_kind == "table" and "table" not in preferred:
        return -FIGURE_TABLE_PENALTY
    return -FIGURE_KIND_BIAS * 0.35


def named_source_paper_ids(query: str, papers: list[dict]) -> set[str]:
    """Recognize named sources in an explicit explanation or comparison request."""
    # A keyword search mentioning a paper may still seek related papers.
    if not re.search(r"\b(explain|describe|summari[sz]e|how|why|what|compare|contrast|differ(?:s|ences?)?|show)\b", query, re.I):
        return set()
    if is_exhaustive_list_query(query) or re.search(r"\b(cit(?:e[sd]?|ing|ations?)|references? to)\b", query, re.I):
        return set()
    aliases = defaultdict(set)
    for paper in papers:
        title = str(paper.get("title") or "").strip()
        if len(title) >= 8:
            aliases[title.casefold()].add(paper["paper_id"])
        if ":" in title:
            prefix = title.split(":", 1)[0].strip()
            if 3 <= len(prefix) <= 48 and len(prefix.split()) <= 4:
                aliases[prefix.casefold()].add(paper["paper_id"])
    matched = set()
    for alias, owners in aliases.items():
        if len(owners) == 1 and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", query, re.I):
            matched.update(owners)
    return matched


def figure_labels(text: str) -> set[tuple[str, str]]:
    return {("figure" if kind.startswith("fig") else kind, number)
            for kind, number in re.findall(r"\b(fig(?:ure)?|table)\.?\s*(\d+[a-z]?)\b", text.lower())}


def figure_caption_prefix_bonus(query: str, figure: dict[str, Any]) -> float:
    query_lower = str(query or "").lower()
    caption_lower = figure_caption_text(figure).lower()
    if not caption_lower:
        return 0.0
    bonus = 0.0
    requested = set(re.findall(r"\b(fig(?:ure)?|table)\.?\s*(\d+[a-z]?)\b", query_lower))
    actual = set(re.findall(r"\b(fig(?:ure)?|table)\.?\s*(\d+[a-z]?)\b", caption_lower))
    requested = {("figure" if kind.startswith("fig") else kind, number) for kind, number in requested}
    actual = {("figure" if kind.startswith("fig") else kind, number) for kind, number in actual}
    if requested & actual:
        bonus += FIGURE_REFERENCE_LABEL_WEIGHT
    if "figure" in query_lower and caption_lower.startswith("figure"):
        bonus += FIGURE_CAPTION_PREFIX_BONUS
    if "table" in query_lower and caption_lower.startswith("table"):
        bonus += FIGURE_CAPTION_PREFIX_BONUS
    if any(token in query_lower for token in ("t-sne", "tsne")) and any(
        token in caption_lower for token in ("t-sne", "tsne")
    ):
        bonus += FIGURE_CAPTION_PREFIX_BONUS
    return bonus


def informative_query_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in QUERY_EN_TOKEN_RE.findall(str(text or "").lower()):
        token = token.strip(".,:;!?()[]{}<>\"'")
        if len(token) < 4 or token in FIGURE_QUERY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def figure_query_overlap_bonus(query: str, figure: dict[str, Any]) -> float:
    query_tokens = informative_query_tokens(query)
    if not query_tokens:
        return 0.0
    haystack = " ".join(
        [
            figure_query_text(figure).lower(),
            figure_caption_text(figure).lower(),
            figure_reference_label_text(figure).lower(),
        ]
    )
    shared_tokens = [token for token in query_tokens if token in haystack]
    if not shared_tokens:
        return 0.0
    bonus = min(1.2, 0.12 * len(shared_tokens))
    if len(shared_tokens) >= 3:
        bonus += 0.28 + (0.04 * min(5, len(shared_tokens) - 3))
    return bonus


def rank_figure_records(
    query: str,
    rows: list[dict[str, Any]],
    top_k: int,
    preview_chars: int,
    preferred_node_ids: set[str] | None = None,
    cached_scorer=None,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not rows:
        return []
    search_texts = [figure_query_text(row) or " " for row in rows]
    caption_inputs = [figure_caption_text(row) for row in rows]
    label_inputs = [figure_reference_label_text(row) for row in rows]
    def score(texts, channel):
        return (cached_scorer(query, texts, ("figures", channel)) if cached_scorer is not None
                else tfidf_scores(query, texts))
    base_scores = score(search_texts, "text")
    caption_scores = score([text if text else " " for text in caption_inputs], "caption")
    label_scores = score([text if text else " " for text in label_inputs], "label")

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        base_score = float(base_scores[index])
        caption_score = float(caption_scores[index]) if caption_inputs[index].strip() else 0.0
        label_score = float(label_scores[index]) if label_inputs[index].strip() else 0.0
        combined_score = (
            base_score
            + (FIGURE_CAPTION_WEIGHT * caption_score)
            + (FIGURE_REFERENCE_LABEL_WEIGHT * label_score)
            + figure_kind_bonus(query, row)
            + figure_caption_prefix_bonus(query, row)
            + figure_query_overlap_bonus(query, row)
        )
        if preferred_node_ids and str(row.get("node_id") or "") in preferred_node_ids:
            combined_score += 1.25
        results.append(
            {
                "node_id": row["node_id"],
                "rank_score": round(float(combined_score), 6),
                "base_score": round(base_score, 6),
                "caption_score": round(caption_score, 6),
                "label_score": round(label_score, 6),
                "preview": preview_text(figure_query_text(row), preview_chars),
                "caption_preview": preview_text(caption_inputs[index], preview_chars),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "record_kind": "figure",
                "asset_kind": row.get("asset_kind"),
                "image_path": row.get("image_path"),
                "caption": row.get("caption") or [],
            }
        )
    return sorted(
        results,
        key=lambda item: (
            -float(item.get("rank_score") or 0.0),
            -float(item.get("caption_score") or 0.0),
            -float(item.get("base_score") or 0.0),
            str(item.get("node_id") or ""),
        ),
    )[:top_k]


def build_paper_why_matched(
    direct_score: float,
    matched_entities: list[dict[str, Any]],
    cross_paper_support: list[dict[str, Any]],
    figure_support: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if float(direct_score) > 0.0:
        reasons.append(
            {
                "kind": "direct_text",
                "score": round(float(direct_score), 6),
                "detail": "query matched paper-level title, abstract, author, institution, or keyword text",
            }
        )
    if matched_entities:
        entity_preview = ", ".join(
            f"{item['entity_type']}:{item.get('display_name') or item['canonical_name']}" for item in matched_entities[:4]
        )
        reasons.append(
            {
                "kind": "entity_support",
                "score": round(
                    sum(float(item.get("score") or 0.0) for item in matched_entities[:4]),
                    6,
                ),
                "detail": f"matched entity nodes: {entity_preview}",
            }
        )
    if cross_paper_support:
        best = cross_paper_support[0]
        evidence_preview = ", ".join(
            list(best.get("evidence_entity_names") or best.get("evidence_entity_types") or [])[:4]
        )
        detail = (
            f"connected from related paper '{best.get('source_paper_title') or best.get('source_paper_id')}' "
            f"via {best.get('edge_type') or 'cross_paper_edge'}"
        )
        if evidence_preview:
            detail += f" using shared evidence: {evidence_preview}"
        reasons.append(
            {
                "kind": "cross_paper_support",
                "score": round(float(best.get("boost") or 0.0), 6),
                "detail": detail,
            }
        )
    if figure_support:
        best_figure = figure_support[0]
        detail = (
            f"matched figure candidate {best_figure.get('figure_id') or best_figure.get('figure_node_id')}"
        )
        caption_preview = str(best_figure.get("caption_preview") or "").strip()
        if caption_preview:
            detail += f" with caption hint: {caption_preview}"
        reasons.append(
            {
                "kind": "figure_support",
                "score": round(float(best_figure.get("rank_score") or 0.0), 6),
                "detail": detail,
            }
        )
    return reasons


def rank_local_records(
    query: str,
    rows: list[dict[str, Any]],
    text_getter: Any,
    top_k: int,
    preview_chars: int,
    record_kind: str,
    preferred_node_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0 or not rows:
        return []
    if record_kind == "figure":
        return rank_figure_records(
            query=query,
            rows=rows,
            top_k=top_k,
            preview_chars=preview_chars,
            preferred_node_ids=preferred_node_ids,
        )
    search_texts = [str(text_getter(row) or "") for row in rows]
    scores = tfidf_scores(query, search_texts)
    ranked = sorted(
        zip(rows, scores),
        key=lambda item: (-float(item[1]), item[0].get("order", 0)),
    )[:top_k]
    results: list[dict[str, Any]] = []
    for row, score in ranked:
        payload = {
            "node_id": row["node_id"],
            "rank_score": round(float(score), 6),
            "preview": preview_text(str(text_getter(row) or ""), preview_chars),
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "record_kind": record_kind,
        }
        if record_kind == "figure":
            payload["asset_kind"] = row.get("asset_kind")
            payload["image_path"] = row.get("image_path")
            payload["caption"] = row.get("caption") or []
        results.append(payload)
    return results


def summarize_section_entities(
    edges: list[dict[str, Any]],
    entity_by_node_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        entity = entity_by_node_id.get(str(edge.get("target_id") or ""))
        if not entity:
            continue
        key = (entity["node_id"], edge["edge_type"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "entity_node_id": entity["node_id"],
                "entity_type": entity["node_type"],
                "canonical_name": entity.get("canonical_name") or entity.get("title") or "",
                "edge_type": edge["edge_type"],
                "confidence": edge.get("confidence"),
            }
        )
    return rows[:8]


def unique_entity_support(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("entity_node_id") or "") or (
            f"{str(row.get('entity_type') or '')}:{str(row.get('canonical_name') or '')}"
        )
        current = unique.get(key)
        if current is None or float(row.get("score") or 0.0) > float(current.get("score") or 0.0):
            unique[key] = row
    return list(unique.values())


def sort_entity_support_for_query(
    rows: list[dict[str, Any]],
    query_types: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    unique_rows = unique_entity_support(rows)
    structured_focus = query_types & STRUCTURED_QUERY_FILTER_TYPES

    return sorted(
        unique_rows,
        key=lambda item: (
            0 if str(item.get("match_source") or "") == "exact" else 1,
            0 if str(item.get("entity_type") or "") in structured_focus else 1,
            0 if str(item.get("entity_type") or "") in query_types else 1,
            -float(item.get("score") or 0.0),
            str(item.get("entity_type") or ""),
            str(item.get("display_name") or item.get("canonical_name") or ""),
        ),
    )[:top_k]


def adjust_entity_score_for_query(
    entity_type: str,
    raw_score: float,
    query_types: set[str],
) -> float:
    entity_type = str(entity_type or "")
    structured_focus = query_types & STRUCTURED_QUERY_FILTER_TYPES
    semantic_focus = query_types & SEMANTIC_QUERY_ENTITY_TYPES

    weight = 1.0
    if structured_focus:
        if entity_type in structured_focus:
            weight = 1.35
        elif entity_type in SEMANTIC_QUERY_ENTITY_TYPES:
            weight = 0.42
        elif entity_type == "reference" and "reference" not in structured_focus:
            weight = 0.22
        else:
            weight = 0.18
    elif semantic_focus:
        if entity_type in semantic_focus:
            weight = 1.35
        elif entity_type in SEMANTIC_QUERY_ENTITY_TYPES:
            weight = 0.86
        elif entity_type == "reference":
            weight = 0.2
        else:
            weight = 0.12
    elif entity_type == "reference":
        weight = 0.9

    return float(raw_score) * weight


def build_semantic_exact_match_groups(
    exact_entity_matches: list[dict[str, Any]],
    semantic_query_types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for match in exact_entity_matches:
        entity_type = str(match.get("entity_type") or "")
        if entity_type not in semantic_query_types:
            continue
        matched_text = normalize_lookup_text(str(match.get("matched_text") or ""))
        canonical_name = normalize_lookup_text(str(match.get("canonical_name") or ""))
        concept_key = matched_text or canonical_name
        if not concept_key:
            continue
        groups[concept_key].append(match)
    return groups


def has_semantic_conjunction(query: str) -> bool:
    lowered = str(query or "").casefold()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            " and ",
            " with ",
            " plus ",
        )
    )


def dedupe_preserve_order(values: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def build_allowed_exact_support_set(
    entity_type: str,
    exact_entity_ids: set[str],
    entity_support_by_paper: dict[str, list[dict[str, Any]]],
    paper_node_ids_by_entity: dict[str, set[str]],
) -> set[str]:
    allowed_for_type: set[str] = set()
    for paper_node_id, rows in entity_support_by_paper.items():
        for row in rows:
            if str(row.get("match_source") or "") != "exact":
                continue
            if str(row.get("entity_type") or "") != entity_type:
                continue
            if str(row.get("entity_node_id") or "") not in exact_entity_ids:
                continue
            allowed_for_type.add(str(paper_node_id))
            break
    if allowed_for_type:
        return allowed_for_type
    for entity_node_id in exact_entity_ids:
        allowed_for_type.update(paper_node_ids_by_entity.get(entity_node_id, set()))
    return allowed_for_type


def build_reference_match_variants(
    matched_text: str,
    entity: dict[str, Any],
) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    seen_compact: set[str] = set()
    raw_values = [
        matched_text,
        best_reference_display_name(entity),
        infer_reference_title(entity),
        *(entity.get("aliases") or []),
    ]
    for raw_value in raw_values:
        display_text = normalize_match_text(raw_value)
        compact_text = compact_lookup_text(display_text)
        if not compact_text or len(compact_text) < 18 or compact_text in seen_compact:
            continue
        seen_compact.add(compact_text)
        variants.append(
            {
                "display_text": display_text,
                "compact_text": compact_text,
            }
        )
        if len(variants) >= 6:
            break
    return variants


def score_reference_entity_match(
    target_variants: list[dict[str, str]],
    candidate_entity: dict[str, Any],
) -> dict[str, Any] | None:
    candidate_variants = build_reference_match_variants("", candidate_entity)
    if not candidate_variants:
        return None

    best_match: dict[str, Any] | None = None
    for target in target_variants:
        target_display = str(target.get("display_text") or "")
        target_compact = str(target.get("compact_text") or "")
        target_tokens = tokenize_lookup_words(target_display)
        if not target_compact or not target_tokens:
            continue
        for candidate in candidate_variants:
            candidate_display = str(candidate.get("display_text") or "")
            candidate_compact = str(candidate.get("compact_text") or "")
            candidate_tokens = tokenize_lookup_words(candidate_display)
            if not candidate_compact or not candidate_tokens:
                continue

            score = 0.0
            if target_compact == candidate_compact:
                score = 2.05
            elif target_compact in candidate_compact or candidate_compact in target_compact:
                score = 1.8
            else:
                shared_tokens = target_tokens & candidate_tokens
                recall = len(shared_tokens) / max(1, len(target_tokens))
                precision = len(shared_tokens) / max(1, len(candidate_tokens))
                if len(shared_tokens) >= 5 and recall >= 0.82:
                    score = 1.5 + (0.2 * min(precision, 1.0))
                elif len(shared_tokens) >= 4 and recall >= 0.7 and precision >= 0.55:
                    score = 1.1
            if score <= 0.0:
                continue
            match_payload = {
                "matched_text": target_display,
                "score": score,
            }
            if best_match is None or float(score) > float(best_match.get("score") or 0.0):
                best_match = match_payload
    return best_match


def score_reference_section_match(
    section_lookup_text: str,
    section_compact_text: str,
    section_role: str,
    target_variants: list[dict[str, str]],
) -> dict[str, Any] | None:
    if not target_variants:
        return None
    role = str(section_role or "")
    best_match: dict[str, Any] | None = None
    for variant in target_variants:
        compact_text = str(variant.get("compact_text") or "")
        display_text = str(variant.get("display_text") or "")
        if not compact_text or compact_text not in section_compact_text:
            continue
        score = 2.35 if role == "citation" else 1.15
        if len(compact_text) >= 48:
            score += 0.15
        match_payload = {
            "matched_text": display_text,
            "score": score,
        }
        if best_match is None or float(score) > float(best_match.get("score") or 0.0):
            best_match = match_payload
    if best_match:
        return best_match

    for variant in target_variants:
        display_text = str(variant.get("display_text") or "")
        normalized_variants = iter_lookup_variants(display_text)
        if not normalized_variants:
            continue
        if any(normalized_variant in section_lookup_text for normalized_variant in normalized_variants if normalized_variant):
            return {
                "matched_text": display_text,
                "score": 1.95 if role == "citation" else 0.95,
            }
    return None


def score_reference_edge_match(
    target_variants: list[dict[str, str]],
    edge: dict[str, Any],
) -> dict[str, Any] | None:
    edge_texts = dedupe_preserve_order(
        [
            normalize_match_text(text)
            for text in [
                *(edge.get("surface_forms") or []),
                *(edge.get("raw_citation_texts") or []),
            ]
            if normalize_match_text(text)
        ]
    )
    if not target_variants or not edge_texts:
        return None

    best_match: dict[str, Any] | None = None
    for variant in target_variants:
        display_text = str(variant.get("display_text") or "")
        compact_text = str(variant.get("compact_text") or "")
        target_tokens = tokenize_lookup_words(display_text)
        if not compact_text or not target_tokens:
            continue
        lookup_variants = iter_lookup_variants(display_text)
        for edge_text in edge_texts:
            edge_lookup_text = normalize_lookup_text(edge_text)
            edge_compact_text = compact_lookup_text(edge_text)
            edge_tokens = tokenize_lookup_words(edge_text)
            if not edge_lookup_text or not edge_compact_text or not edge_tokens:
                continue

            score = 0.0
            if compact_text in edge_compact_text:
                score = 2.6 if len(compact_text) >= 24 else 2.25
            elif any(
                normalized_variant and normalized_variant in edge_lookup_text
                for normalized_variant in lookup_variants
            ):
                score = 2.1
            else:
                shared_tokens = target_tokens & edge_tokens
                recall = len(shared_tokens) / max(1, len(target_tokens))
                if len(shared_tokens) >= 5 and recall >= 0.92:
                    score = 1.7
                elif len(shared_tokens) >= 5 and recall >= 0.8:
                    score = 1.3
                elif len(shared_tokens) >= 4 and recall >= 0.68:
                    score = 1.05
            if score <= 0.0:
                continue
            match_payload = {
                "matched_text": display_text,
                "score": score,
                "edge_text": edge_text,
            }
            if best_match is None or float(score) > float(best_match.get("score") or 0.0):
                best_match = match_payload
    return best_match


def normalize_lookup_text(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def compact_lookup_text(text: str) -> str:
    return LOOKUP_COMPACT_RE.sub("", normalize_lookup_text(text))


def tokenize_lookup_words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_lookup_text(text))
        if token and len(token) >= 2
    }


def iter_lookup_variants(text: str) -> list[str]:
    raw_text = str(text or "")
    candidates = [
        raw_text,
        raw_text.replace("-", " "),
        raw_text.replace("–", " ").replace("—", " "),
        raw_text.replace("/", " "),
        raw_text.replace("&", " and "),
    ]
    punctuation_light = re.sub(r"[\-–—/&]+", " ", raw_text)
    candidates.append(punctuation_light)

    variants: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_lookup_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        variants.append(normalized)
        compact = compact_lookup_text(normalized)
        if compact and compact not in seen:
            seen.add(compact)
            variants.append(compact)
    return variants


def normalize_match_text(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_retrieval_mode(mode: str | None) -> str:
    value = str(mode or "hierarchical").strip().lower()
    if value not in RETRIEVAL_MODES:
        raise ValueError(f"unsupported retrieval_mode: {mode}")
    return value


def is_exhaustive_list_query(text: str) -> bool:
    query_text = normalize_match_text(text)
    if not query_text:
        return False
    return any(pattern.search(query_text) for pattern in EXHAUSTIVE_LIST_QUERY_PATTERNS)


def has_constraint_query_intent(text: str) -> bool:
    lowered = normalize_lookup_text(str(text or ""))
    return bool(re.search(r"\b(find|show|locate|look up|lookup)\b", lowered))


def detect_query_entity_types(text: str) -> set[str]:
    query_text = str(text or "")
    lowered = query_text.casefold()
    matched_types: set[str] = set()
    for entity_type, hints in QUERY_TYPE_HINTS.items():
        if any(hint.casefold() in lowered for hint in hints):
            matched_types.add(entity_type)
    if any(token in lowered for token in (" citing ", " cited ", " bibliography", " works cited")):
        matched_types.add("reference")
    return matched_types


def infer_query_types_from_exact_matches(exact_entity_matches: list[dict[str, Any]]) -> set[str]:
    inferred: set[str] = set()
    for match in exact_entity_matches:
        entity_type = str(match.get("entity_type") or "")
        if entity_type:
            inferred.add(entity_type)
    return inferred


def extract_title_query_phrases(text: str) -> list[str]:
    raw_text = str(text or "")
    lowered = raw_text.casefold()
    if "title" not in lowered:
        return []
    quoted_phrases = [phrase for phrase, _, _ in extract_quoted_spans(raw_text)]
    if quoted_phrases:
        return quoted_phrases
    for pattern in TITLE_QUERY_PATTERNS:
        match = pattern.search(raw_text)
        if not match:
            continue
        phrase = normalize_match_text(match.group(1))
        if phrase:
            return [phrase]
    return []


def extract_title_query_phrases_v2(text: str) -> list[str]:
    raw_text = str(text or "")
    lowered = raw_text.casefold()
    phrases: list[str] = []
    seen_compact: set[str] = set()

    def add_phrase(raw_phrase: str) -> None:
        cleaned = strip_leading_request_prefix(normalize_match_text(raw_phrase).strip(" ,.;:!?()[]{}"))
        compact_text = compact_lookup_text(cleaned)
        if not looks_like_title_phrase(cleaned) or not compact_text or compact_text in seen_compact:
            return
        seen_compact.add(compact_text)
        phrases.append(cleaned)

    explicit_title_query = "title" in lowered
    if explicit_title_query:
        for phrase, _, _ in extract_quoted_spans(raw_text):
            add_phrase(phrase)
        for pattern in TITLE_QUERY_PATTERNS:
            match = pattern.search(raw_text)
            if not match:
                continue
            add_phrase(match.group(1))

    if TITLE_REFERENCE_CUE_RE.search(raw_text) and not is_exhaustive_list_query(raw_text):
        for cue_match in TITLE_REFERENCE_CUE_RE.finditer(raw_text):
            prefix_text = raw_text[: cue_match.start()]
            english_spans = [match.group(0) for match in TITLEISH_ENGLISH_SPAN_RE.finditer(prefix_text)]
            if not english_spans:
                continue
            for span in english_spans[-3:]:
                add_phrase(span)

    return phrases


def strip_leading_request_prefix(text: str) -> str:
    value = normalize_match_text(text)
    value = re.sub(
        r"^(?:find|find me|show me|show|look up|lookup|please find|please show|i want to see|i want|see)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return normalize_match_text(value)


def looks_like_title_phrase(text: str) -> bool:
    phrase = normalize_match_text(text)
    compact_phrase = compact_lookup_text(phrase)
    if len(compact_phrase) < 8:
        return False
    english_tokens = [token for token in QUERY_EN_TOKEN_RE.findall(phrase) if token]
    if len(english_tokens) >= 2:
        return True
    return any(marker in phrase for marker in (":", "-", "(", ")"))


def extract_quoted_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in QUERY_QUOTE_RE.finditer(str(text or "")):
        phrase = normalize_match_text(match.group(1))
        if phrase:
            spans.append((phrase, match.start(1), match.end(1)))
    return spans


def build_query_english_ngrams(text: str, max_n: int = 16) -> list[str]:
    tokens = [token for token in QUERY_EN_TOKEN_RE.findall(str(text or "")) if token]
    if not tokens:
        return []
    ngrams: list[str] = []
    seen: set[str] = set()
    max_n = max(1, int(max_n))
    for start in range(len(tokens)):
        upper = min(max_n, len(tokens) - start)
        for size in range(upper, 0, -1):
            phrase = " ".join(tokens[start : start + size]).strip()
            normalized_phrase = normalize_lookup_text(phrase)
            if not normalized_phrase or normalized_phrase in seen:
                continue
            seen.add(normalized_phrase)
            ngrams.append(phrase)
    return ngrams


def extract_symbolic_query_handles(text: str) -> list[str]:
    handles: list[str] = []
    seen: set[str] = set()
    for token in QUERY_EN_TOKEN_RE.findall(str(text or "")):
        compact_token = compact_lookup_text(token)
        if len(compact_token) < 4:
            continue
        has_internal_upper = any(char.isupper() for char in token[1:])
        is_all_caps = token.isupper()
        is_hyphenated_name = "-" in token and any(char.isupper() for char in token)
        if not (has_internal_upper or is_all_caps or is_hyphenated_name):
            continue
        normalized = normalize_lookup_text(token)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        handles.append(token)
    return handles


def iter_entity_alias_texts(entity: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for raw_value in [
        entity.get("canonical_name"),
        entity.get("title"),
        infer_reference_title(entity) if str(entity.get("node_type") or "") == "reference" else "",
        *(entity.get("aliases") or []),
    ]:
        alias_text = normalize_match_text(raw_value)
        normalized_alias = normalize_lookup_text(alias_text)
        if not normalized_alias or normalized_alias in seen:
            continue
        seen.add(normalized_alias)
        texts.append(alias_text)
    return texts


def make_entity_support_row(
    entity: dict[str, Any],
    edge: dict[str, Any],
    score: float,
    match_source: str,
    matched_text: str | None = None,
    via_section_id: str | None = None,
) -> dict[str, Any]:
    canonical_name = entity.get("canonical_name") or entity.get("title") or ""
    if not canonical_name and str(entity.get("node_type") or "") == "reference":
        canonical_name = best_reference_display_name(entity)
    display_name = canonical_name or (matched_text or "")
    row = {
        "entity_node_id": entity["node_id"],
        "entity_type": entity["node_type"],
        "canonical_name": canonical_name,
        "display_name": display_name,
        "score": round(float(score), 6),
        "edge_type": edge["edge_type"],
        "match_source": match_source,
    }
    if matched_text:
        row["matched_text"] = matched_text
    if via_section_id:
        row["via_section_id"] = via_section_id
    return row
