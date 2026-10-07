"""Weighted rank fusion of the complete lexical/rule and GNN retrievers."""
from __future__ import annotations

import copy
import math
from dataclasses import replace

from .image_retrieval import preview_text


RRF_K = 60


def validate_hybrid_options(weight, candidate_pool):
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("hybrid_gnn_weight must be finite and in [0, 1]")
    if isinstance(candidate_pool, bool) or not isinstance(candidate_pool, int) or candidate_pool < 1:
        raise ValueError("hybrid_candidate_pool must be a positive integer")


def fuse_ranked_rows(lexical, gnn, key, gnn_weight, limit, allowed=None):
    """Preserve each branch's ordering, deduplicate IDs, and retain provenance.

    The common (k+1) multiplier puts a rank-one contribution at its configured
    weight. It does not change RRF ordering. An absent branch contributes zero.
    """
    validate_hybrid_options(gnn_weight, 1)
    branches = {}
    for name, rows, weight in (("lexical", lexical, 1 - gnn_weight), ("gnn", gnn, gnn_weight)):
        entries = {}
        if weight > 0:
            for row in rows:
                node_id = row[key]
                if node_id not in entries:
                    rank = len(entries) + 1
                    entries[node_id] = (row, {"rank": rank, "raw_score": row.get("rank_score"),
                        "weight": weight, "contribution": weight * (RRF_K + 1) / (RRF_K + rank)})
        branches[name] = entries
    candidates = set(branches["lexical"]) | set(branches["gnn"])
    if allowed is not None:
        candidates &= set(allowed)
    records = []
    for node_id in candidates:
        sources = {name: entries[node_id][1] if node_id in entries else None for name, entries in branches.items()}
        score = sum(value["contribution"] for value in sources.values() if value)
        records.append((node_id, score, sources))
    records.sort(key=lambda item: (-item[1],
        (item[2]["lexical"] or {}).get("rank", math.inf),
        (item[2]["gnn"] or {}).get("rank", math.inf), str(item[0])))
    results = []
    for node_id, score, sources in records[:max(0, limit)]:
        source = branches["lexical"].get(node_id) or branches["gnn"][node_id]
        row = copy.deepcopy(source[0])
        row.update(rank_score=score, hybrid_sources=sources)
        results.append(row)
    return results


class HybridRetriever:
    def __init__(self, owner):
        self.owner, self.config, self.gnn = owner, owner.config, owner.gnn
        validate_hybrid_options(self.config.hybrid_gnn_weight, self.config.hybrid_candidate_pool)
        # Share read-only graph tables and lookup dictionaries, without loading
        # a second index/model or changing the calling retriever's backend.
        self.lexical = copy.copy(owner)
        self.lexical.config = replace(owner.config, retrieval_backend="lexical", gnn_index_dir=None)
        self.lexical.retrieval_backend = "lexical"
        self.lexical.gnn = self.lexical.hybrid = None

    @property
    def weight(self):
        return self.config.hybrid_gnn_weight

    def metadata(self):
        return {"lexical_weight": 1 - self.weight, "gnn_weight": self.weight,
                "fusion": "weighted_reciprocal_rank", "rrf_k": RRF_K,
                "candidate_pool": self.config.hybrid_candidate_pool,
                "scope": "papers, sections, chunks and figures; explicit lexical constraints preserved"}

    def _lexical_view(self, chunks, figures):
        view = copy.copy(self.lexical)
        # Older lexical APIs use `value or default`; explicit zero budgets must
        # also be zero in the view's defaults to disable evidence correctly.
        view.config = replace(view.config, top_k_chunks=chunks, top_k_figures=figures)
        return view

    def _fuse(self, lexical, gnn, key, limit, allowed=None):
        return fuse_ranked_rows(lexical, gnn, key, self.weight, limit, allowed)

    @staticmethod
    def _entities(lexical, gnn):
        rows = {}
        for source, items in (("lexical", lexical), ("gnn", gnn)):
            for item in items:
                key = item.get("entity_node_id") or (item.get("entity_type"), item.get("canonical_name"))
                if key not in rows:
                    rows[key] = dict(item, retrieval_sources=[])
                rows[key]["retrieval_sources"].append(source)
        return list(rows.values())

    def _sections(self, lexical, gnn, sk, ck, fk, query=None):
        left = {s["section_id"]: s for s in lexical}
        right = {s["section_id"]: s for s in gnn}
        rows = self._fuse(lexical, gnn, "section_id", sk)
        for row in rows:
            a, b = left.get(row["section_id"], {}), right.get(row["section_id"], {})
            row["matched_entities"] = self._entities(a.get("matched_entities", []), b.get("matched_entities", []))
            for name, limit in (("top_chunks", ck), ("top_figures", fk)):
                if query is not None and limit:
                    from .document_entity_retrieval import rank_local_records
                    kind = "chunk" if name == "top_chunks" else "figure"
                    table = self.owner.chunks_by_section if kind == "chunk" else self.owner.figures_by_section
                    candidates = table.get(row["section_id"], [])
                    left_local = rank_local_records(query=query, rows=candidates,
                        text_getter=lambda item: str(item.get("text" if kind == "chunk" else "search_text") or ""),
                        top_k=len(candidates), preview_chars=self.config.preview_chars, record_kind=kind)
                    right_local = self.gnn.records(query, candidates, len(candidates), kind)
                else:
                    left_local, right_local = a.get(name, []), b.get(name, [])
                row[name] = self._fuse(left_local, right_local, "node_id", limit)
        return rows

    def fuse_payloads(self, lexical, gnn, pk, sk, ck, fk):
        left = {p["paper_id"]: p for p in lexical["results"]}
        right = {p["paper_id"]: p for p in gnn["results"]}
        query_mode = lexical.get("query_mode")
        allowed = set(left) if query_mode in {"exhaustive_list", "exact_constraint", "named_source"} else None
        limit = len(left) if query_mode == "exhaustive_list" else pk
        rows = self._fuse(lexical["results"], gnn["results"], "paper_id", limit, allowed)
        for row in rows:
            a, b = left.get(row["paper_id"], {}), right.get(row["paper_id"], {})
            row["matched_entities"] = self._entities(a.get("matched_entities", []), b.get("matched_entities", []))
            row["paper_recall_evidence"] = copy.deepcopy(b.get("paper_recall_evidence", []))
            row["why_matched"] = [{"kind": "hybrid_rank_fusion", "detail":
                f"Lexical/rules {1-self.weight:.0%}, GNN {self.weight:.0%}; weighted rank fusion."}]
            for source, item in (("lexical", a), ("gnn", b)):
                row["why_matched"].extend(dict(reason, retrieval_source=source) for reason in item.get("why_matched", []))
            row["top_sections"] = self._sections(a.get("top_sections", []), b.get("top_sections", []), sk, ck, fk, lexical["query"])
        payload = {key: copy.deepcopy(value) for key, value in lexical.items() if key != "results"}
        payload.update(retrieval_backend="hybrid", hybrid=self.metadata(), results=rows, result_count=len(rows),
                       gnn_index_dir=str(self.gnn.index.index_dir), selection_mode="lexical_gnn_weighted_rank_fusion",
                       hybrid_constraint_mode=query_mode if allowed is not None else None,
                       hybrid_candidate_counts={"lexical": len(left), "gnn": len(right)})
        return payload

    def retrieve(self, query, top_k_papers=None, top_k_sections=None, top_k_chunks=None,
                 top_k_figures=None, retrieval_mode="hierarchical", paper=None):
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        if retrieval_mode not in {"hierarchical", "paper_then_chunk", "flat_chunk"}:
            raise ValueError(f"unsupported hybrid retrieval mode: {retrieval_mode}")
        def budget(value, name, minimum):
            return max(minimum, int(getattr(self.config, name) if value is None else value))
        pk = budget(top_k_papers, "top_k_papers", 1)
        sk = budget(top_k_sections, "top_k_sections", 1)
        ck = budget(top_k_chunks, "top_k_chunks", 0)
        fk = budget(top_k_figures, "top_k_figures", 0)

        def lexical_result(papers, sections, chunks, figures):
            view = self._lexical_view(chunks, figures)
            if paper is not None:
                return view.retrieve_within_paper(query, paper_id=paper["paper_id"],
                    top_k_sections=sections, top_k_chunks=chunks, top_k_figures=figures)
            return view.retrieve(query, papers, sections, chunks, figures, retrieval_mode)

        if self.weight in {0, 1}:
            # Endpoints are true branch ablations, without changes caused by
            # candidate overfetch or constraints from the disabled branch.
            result = (lexical_result(pk, sk, ck, fk) if self.weight == 0 else
                      self.gnn.retrieve(query, pk, sk, ck, fk, retrieval_mode, paper))
            result = dict(result, retrieval_backend="hybrid", hybrid=self.metadata(),
                          effective_backend="lexical" if self.weight == 0 else "gnn")
            return result
        pool = max(pk, self.config.hybrid_candidate_pool)
        # Overfetch local evidence before fusion, preserving its paper/section
        # ownership; a zero evidence budget never becomes a nonzero one.
        section_pool = max(sk, max((len(rows) for rows in self.owner.sections_by_paper.values()), default=sk))
        lexical = lexical_result(pool, section_pool, ck * 3, fk * 3)
        gnn = self.gnn.retrieve(query, pool, section_pool, ck * 3, fk * 3, retrieval_mode, paper)
        return self.fuse_payloads(lexical, gnn, pk, sk, ck, fk)

    def rank_papers(self, query, limit):
        result = self.retrieve(query, top_k_papers=limit, top_k_sections=1, top_k_chunks=0, top_k_figures=0)
        return [(self.owner.paper_by_paper_id[p["paper_id"]], p["rank_score"], p.get("paper_direct_score", 0),
                 p.get("matched_entities", []), p.get("cross_paper_support", []), p.get("figure_support", []))
                for p in result["results"][:max(0, limit)]]

    def sections(self, query, paper_id, sk, ck, fk, mode="hierarchical", entity_support=None, figure_support=None):
        if sk <= 0:
            return []
        paper = self.owner.paper_by_node_id.get(paper_id) or self.owner.paper_by_paper_id[paper_id]
        view = self._lexical_view(ck, fk)
        size = sk if self.weight in {0, 1} else sk * 3
        if self.weight < 1:
            if mode == "paper_then_chunk":
                left = view.rank_sections_for_paper_without_routing(query, paper["paper_id"], size, ck, fk, figure_support)
            else:
                left = view.rank_sections_for_paper(query, paper["paper_id"], size, ck, fk, entity_support, figure_support)
        else:
            left = []
        right = self.gnn.sections(query, paper["node_id"], size, ck, fk, mode) if self.weight > 0 else []
        if self.weight in {0, 1}:
            return left if self.weight == 0 else right
        return self._sections(left, right, sk, ck, fk)

    def rank_chunks_globally(self, query, top_k):
        if top_k <= 0:
            return []
        if self.weight == 0:
            return self.lexical.rank_chunks_globally(query, top_k)
        size = top_k if self.weight == 1 else max(top_k, self.config.hybrid_candidate_pool)
        left = self.lexical.rank_chunks_globally(query, size) if self.weight < 1 else []
        right = [{"chunk": row, "score": score, "preview": preview_text(str(row.get("text") or ""), self.config.preview_chars)}
                 for row, score in self.gnn.index.rank(query, self.owner.chunks, size)]
        for rows in (left, right):
            for row in rows:
                row.update(node_id=row["chunk"]["node_id"], rank_score=row["score"])
        fused = right if self.weight == 1 else self._fuse(left, right, "node_id", top_k)
        return [dict(row, score=row["rank_score"]) for row in fused[:top_k]]
