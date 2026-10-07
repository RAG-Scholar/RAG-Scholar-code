"""Version 2 node features: independent, complete paper metadata fields."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .gnn_embeddings import _text, node_text


FEATURE_VERSION = 2
PAPER_FIELDS = {
    "title": ("title",), "abstract": ("abstract",), "keywords": ("keywords",),
    "authors": ("authors", "author_text"),
    "institutions": ("institutions", "institution_text"), "venue": ("venue_text",),
}
METADATA_RELATIONS = {
    "paper_has_author": ("authors", "author"),
    "paper_has_institution": ("institutions", "institution"),
    "paper_published_in": ("venue", "venue"),
}


def paper_field_texts(nodes: list[dict], edges: list[dict]) -> dict[str, dict[str, str]]:
    """Use both recorded metadata and canonical names from existing graph links."""
    by_id = {n["node_id"]: n for n in nodes}
    linked = defaultdict(lambda: defaultdict(list))
    for edge in edges:
        relation = METADATA_RELATIONS.get(edge["edge_type"])
        target = by_id.get(edge["target_id"], {})
        if relation and target.get("node_type") == relation[1] and float(edge.get("confidence", 1)) > 0:
            linked[edge["source_id"]][relation[0]].append(
                target.get("canonical_name") or target.get("name") or node_text(target))
    result = {}
    for node in by_id.values():
        if node["node_type"] != "paper":
            continue
        fields = {}
        for field, keys in PAPER_FIELDS.items():
            values = [_text(node.get(key)).strip() for key in keys]
            values.extend(linked[node["node_id"]][field])
            value = "\n".join(dict.fromkeys(v for v in values if v))
            if value:
                fields[field] = value
        result[node["node_id"]] = fields or {"fallback": node_text(node)}
    return result


def normalized_mean(vectors: np.ndarray) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float32).mean(axis=0)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-8 else vectors[0].copy()


def encode_node_features(nodes, edges, encoder, progress=None):
    paper_fields = paper_field_texts(nodes, edges)
    texts, placements = [], []
    ordinary = []
    for i, node in enumerate(nodes):
        if node["node_type"] == "paper":
            fields = paper_fields[node["node_id"]]
        elif node["node_type"] in {"author", "institution", "venue"}:
            fields = {"entity": node_text(node)}
        else:
            ordinary.append(i)
            continue
        for field, text in fields.items():
            texts.append(text)
            placements.append((i, field))
    features = None
    if ordinary:
        encoded = encoder.encode([node_text(nodes[i]) for i in ordinary], progress)
        features = np.zeros((len(nodes), encoded.shape[1]), dtype=np.float32)
        features[ordinary] = encoded
    audit, grouped = {}, defaultdict(list)
    if texts:
        encoded, coverage = encoder.encode_complete(texts, progress)
        if features is None:
            features = np.zeros((len(nodes), encoded.shape[1]), dtype=np.float32)
        for (i, field), vector, detail in zip(placements, encoded, coverage):
            grouped[i].append(vector)
            if nodes[i]["node_type"] == "paper":
                audit.setdefault(nodes[i]["node_id"], {})[field] = detail
        for i, vectors in grouped.items():
            features[i] = normalized_mean(np.stack(vectors))
    return features, {"feature_version": FEATURE_VERSION,
                      "aggregation": "normalized window mean per field; normalized equal mean across nonempty fields",
                      "papers": audit}
