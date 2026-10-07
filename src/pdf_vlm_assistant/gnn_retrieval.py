"""GNN cosine ranking with the existing paper/section/evidence output schema."""
from __future__ import annotations

from collections import defaultdict

from .gnn_embeddings import EmbeddingIndex, node_text
from .image_retrieval import preview_text
from .gnn_paper_recall import GraphPaperRecall


class GNNRetriever:
    def __init__(self, owner):
        self.owner = owner
        self.config = owner.config
        self.index = EmbeddingIndex(self.config.gnn_index_dir, owner.graph_dir,
                                    self.config.gnn_device, self.config.gnn_encoder_model)
        self.paper_recall = GraphPaperRecall(
            [*owner.papers, *owner.sections, *owner.chunks, *owner.figures, *owner.entity_nodes],
            owner.edges, owner.papers, self.index.ids)
        self.last_recall_evidence = {}
        self.entity_edges = defaultdict(list)
        for edge in owner.edges:
            if edge["target_id"] in owner.entity_by_node_id:
                self.entity_edges[edge["source_id"]].append(edge)

    def rank_papers(self, query, limit, scores=None):
        from .document_entity_retrieval import detect_query_entity_types, infer_query_types_from_exact_matches
        if scores is None:
            scores = self.index.scores(query)
        query_types = detect_query_entity_types(query) or infer_query_types_from_exact_matches(self.owner.find_exact_entity_matches(query))
        ranked = self.paper_recall.rank(scores, limit, query_types)
        self.last_recall_evidence = {paper["node_id"]: evidence for paper, _, evidence in ranked}
        return [(paper, score) for paper, score, _ in ranked]

    def entities(self, query, source_id):
        by_id = {}
        for edge in self.entity_edges[source_id]:
            entity = self.owner.entity_by_node_id[edge["target_id"]]
            score = self.index.score(query, entity["node_id"])
            by_id.setdefault(entity["node_id"], {
                "entity_node_id": entity["node_id"], "entity_type": entity["node_type"],
                "canonical_name": entity.get("canonical_name") or entity.get("title") or entity.get("name") or "",
                "edge_type": edge["edge_type"], "confidence": edge.get("confidence"),
                "score": round(score, 6), "match_source": "gnn_embedding",
            })
        return sorted(by_id.values(), key=lambda row: (-row["score"], row["entity_node_id"]))[:max(0, self.config.top_k_entities)]

    def records(self, query, rows, limit, kind):
        results = []
        for row, score in self.index.rank(query, rows, limit):
            payload = {"node_id": row["node_id"], "rank_score": round(score, 6),
                       "preview": preview_text(node_text(row), self.config.preview_chars),
                       "page_start": row.get("page_start"), "page_end": row.get("page_end"), "record_kind": kind}
            if kind == "figure":
                payload.update(asset_kind=row.get("asset_kind"), image_path=row.get("image_path"), caption=row.get("caption") or [])
            results.append(payload)
        return results

    def sections(self, query, paper_node_id, section_k, chunk_k, figure_k, mode="hierarchical", allowed_chunks=None):
        paper = self.owner.paper_by_node_id.get(paper_node_id) or self.owner.paper_by_paper_id[paper_node_id]
        rows = self.owner.sections_by_paper.get(paper["paper_id"], [])
        ranked = self.index.rank(query, rows, len(rows))
        if mode in {"paper_then_chunk", "flat_chunk"}:
            def best_chunk(section):
                chunks = self.owner.chunks_by_section.get(section["section_id"], [])
                if allowed_chunks is not None:
                    chunks = [c for c in chunks if c["node_id"] in allowed_chunks]
                return max((self.index.score(query, c["node_id"]) for c in chunks), default=-float("inf"))
            ranked = [(row, best_chunk(row)) for row, _ in ranked]
            if mode == "flat_chunk":
                ranked = [pair for pair in ranked if pair[1] != -float("inf")]
            else:
                ranked = [(row, self.index.score(query, row["node_id"]) if score == -float("inf") else score)
                          for row, score in ranked]
            ranked.sort(key=lambda pair: (-pair[1], pair[0]["node_id"]))
        results = []
        for section, score in ranked[:section_k]:
            node_id = section["node_id"]
            chunks = self.owner.chunks_by_section.get(section["section_id"], [])
            if allowed_chunks is not None:
                chunks = [c for c in chunks if c["node_id"] in allowed_chunks]
            results.append({
                "section_id": section.get("section_id"), "section_title": section.get("section_title", ""),
                "section_group": section.get("section_group", ""), "rank_score": round(score, 6),
                "section_direct_score": round(self.index.score(query, node_id), 6),
                "section_summary": preview_text(section.get("section_summary") or section.get("section_text") or "", self.config.preview_chars),
                "matched_entities": self.entities(query, node_id),
                "top_chunks": self.records(query, chunks, chunk_k, "chunk"),
                "top_figures": self.records(query, self.owner.figures_by_section.get(section["section_id"], []), figure_k, "figure"),
            })
        return results

    def paper_result(self, query, paper, score, section_k, chunk_k, figure_k, mode="hierarchical", allowed_chunks=None):
        return {
            "paper_id": paper["paper_id"], "paper_title": paper.get("title", ""),
            "paper_author_text": paper.get("author_text", ""), "paper_institution_text": paper.get("institution_text", ""),
            "rank_score": round(score, 6), "paper_direct_score": round(self.index.score(query, paper["node_id"]), 6),
            "matched_entities": self.entities(query, paper["node_id"]), "cross_paper_support": [], "figure_support": [],
            "paper_recall_evidence": self.last_recall_evidence.get(paper["node_id"], []),
            "why_matched": [{"kind": "gnn_multi_node_recall" if self.last_recall_evidence else "gnn_embedding",
                             "detail": "Graph node cosine ranking; mapped paper ranks are fused with weighted RRF." if self.last_recall_evidence
                                       else "Query cosine similarity to trained graph node embeddings."}],
            "top_sections": self.sections(query, paper["node_id"], section_k, chunk_k, figure_k, mode, allowed_chunks),
        }

    def retrieve(self, query, top_k_papers=None, top_k_sections=None, top_k_chunks=None,
                 top_k_figures=None, retrieval_mode="hierarchical", paper=None):
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        if retrieval_mode not in {"hierarchical", "flat_chunk", "paper_then_chunk"}:
            raise ValueError(f"unsupported GNN retrieval mode: {retrieval_mode}")
        def limit(value, name, minimum):
            return max(minimum, int(getattr(self.config, name) if value is None else value))
        pk = limit(top_k_papers, "top_k_papers", 1)
        sk = limit(top_k_sections, "top_k_sections", 1)
        ck = limit(top_k_chunks, "top_k_chunks", 0)
        fk = limit(top_k_figures, "top_k_figures", 0)
        self.last_recall_evidence = {}
        allowed = None
        if paper is not None:
            ranked = [(paper, self.index.score(query, paper["node_id"]))]
        elif retrieval_mode == "flat_chunk":
            chunk_rows = self.index.rank(query, self.owner.chunks, pk * sk * max(ck, 1))
            allowed = {row["node_id"] for row, _ in chunk_rows}
            best = {}
            for row, score in chunk_rows:
                best.setdefault(row["paper_node_id"], score)
            ranked = [(self.owner.paper_by_node_id[node_id], score) for node_id, score in best.items()][:pk]
        else:
            ranked = self.rank_papers(query, pk)
        results = [self.paper_result(query, row, score, sk, ck, fk, retrieval_mode, allowed) for row, score in ranked]
        payload = {"query": query, "graph_dir": str(self.owner.graph_dir), "retrieval_backend": "gnn",
                   "gnn_index_dir": str(self.index.index_dir), "retrieval_mode": retrieval_mode,
                   "paper_recall": "weighted_rrf_multi_node" if self.last_recall_evidence else "node_cosine",
                   "recall_node_coverage": self.paper_recall.coverage(),
                   "result_kind": "single_paper" if paper is not None else "papers",
                   "result_count": len(results), "results": results}
        if paper is not None:
            payload.update(selected_paper_id=paper["paper_id"], selected_paper_title=paper.get("title", ""))
        return payload
