"""Map scored graph nodes to papers, then fuse ranks without lexical scores."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


CONTENT_TYPES = {"paper", "section", "chunk", "figure"}
CONTAINMENT = {"paper_has_section", "section_has_chunk", "section_has_figure", "contains"}


class GraphPaperRecall:
    def __init__(self, nodes, edges, papers, node_ids, *, candidate_limit=32, rrf_k=60):
        self.nodes = {n["node_id"]: n for n in nodes}
        self.papers = {p["node_id"]: p for p in papers}
        self.lookup = {node_id: i for i, node_id in enumerate(node_ids)}
        self.candidate_limit, self.rrf_k = candidate_limit, rrf_k
        paper_ids = {p["paper_id"]: p["node_id"] for p in papers}
        owners = {}
        for node_id, node in self.nodes.items():
            if node["node_type"] not in CONTENT_TYPES:
                continue
            paper = node_id if node_id in self.papers else (
                node.get("paper_node_id") or paper_ids.get(node.get("paper_id")))
            if paper in self.papers:
                owners[node_id] = paper
        # Only structural containment can establish ownership. Citation and
        # shared-entity paper-to-paper edges must not transfer ownership.
        pending = [e for e in edges if e["edge_type"] in CONTAINMENT
                   and self.nodes.get(e["target_id"], {}).get("node_type") in {"section", "chunk", "figure"}]
        while pending:
            remaining = []
            for edge in pending:
                source, target = edge["source_id"], edge["target_id"]
                if source in owners:
                    if target in owners and owners[target] != owners[source]:
                        raise ValueError(f"conflicting paper ownership: {target}")
                    owners[target] = owners[source]
                else:
                    remaining.append(edge)
            if len(remaining) == len(pending):
                break
            pending = remaining
        self.links = defaultdict(dict)
        for node_id, paper_id in owners.items():
            self.links[node_id][paper_id] = {"source_node_id": node_id, "edge_type": "ownership"}
        for edge in sorted(edges, key=lambda e: (e["source_id"], e["target_id"], e["edge_type"])):
            source, target = edge["source_id"], edge["target_id"]
            if source in owners and target in self.nodes and self.nodes[target]["node_type"] not in CONTENT_TYPES:
                if float(edge.get("confidence", 1)) > 0:
                    self.links[target].setdefault(owners[source], {
                        "source_node_id": source, "edge_type": edge["edge_type"]})
        self.entity_types = sorted({self.nodes[n]["node_type"] for n in self.links
                                    if self.nodes[n]["node_type"] not in CONTENT_TYPES})

    def rank(self, scores, limit, query_types=()):
        scores = np.asarray(scores)
        if scores.shape != (len(self.lookup),) or not np.isfinite(scores).all():
            raise ValueError("node scores must be finite and aligned with the embedding index")
        requested = sorted(set(query_types) & set(self.entity_types))
        active_entities = requested or self.entity_types
        # Query hints select entity TYPES only; matching individual names still
        # uses the node embeddings. These fixed weights do not use eval labels.
        weights = {"paper": 1.0, "section": 0.5 if requested else 1.0,
                   "chunk": 0.5 if requested else 1.0, "figure": 0.25 if requested else 0.5}
        weights.update({kind: (3.0 if requested else 1.0) / len(active_entities)
                        for kind in active_entities})
        best = defaultdict(dict)
        for node_id, papers in self.links.items():
            kind = self.nodes[node_id]["node_type"]
            if kind not in weights:
                continue
            score = float(scores[self.lookup[node_id]])
            if score <= 0 and kind != "paper":
                continue
            for paper_id, provenance in papers.items():
                old = best[kind].get(paper_id)
                if old is None or score > old[0] or (score == old[0] and node_id < old[1]):
                    best[kind][paper_id] = (score, node_id, provenance)
        totals, evidence = defaultdict(float), defaultdict(list)
        for kind, matches in best.items():
            ordered = sorted(matches.items(), key=lambda item: (-item[1][0], item[0]))
            rank, previous = 0, None
            for position, (paper_id, (score, node_id, provenance)) in enumerate(ordered, 1):
                # Equal scores share a rank. An entity connected to many papers
                # does not favour the first paper IDs or lose boundary ties.
                if previous is None or score != previous:
                    rank = position
                    previous = score
                if kind != "paper" and rank > self.candidate_limit:
                    break
                contribution = weights[kind] / (self.rrf_k + rank)
                totals[paper_id] += contribution
                evidence[paper_id].append({"node_id": node_id, "node_type": kind,
                    "cosine_score": score, "channel_rank": rank, "weight": weights[kind],
                    "rrf_contribution": contribution, **provenance})
        ranked = sorted(totals, key=lambda pid: (-totals[pid], -float(scores[self.lookup[pid]]), pid))
        return [(self.papers[pid], totals[pid], evidence[pid]) for pid in ranked[:max(0, limit)]]

    def coverage(self):
        counts = defaultdict(int)
        for node_id in self.links:
            counts[self.nodes[node_id]["node_type"]] += 1
        return dict(sorted(counts.items()))
