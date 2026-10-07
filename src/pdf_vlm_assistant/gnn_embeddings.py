"""Self-supervised, relation-aware GNN embeddings for an existing document graph.

The JSONL graph is read-only. Reverse edges exist only in the message-passing
view. Frozen text vectors define the shared node/query coordinate system.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _text(value) -> str:
    if isinstance(value, list):
        return " ".join(_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_text(item) for item in value.values())
    return str(value or "")


def node_text(node: dict) -> str:
    # Put the node's own content first: inherited paper metadata can otherwise
    # consume the transformer's token budget for every chunk in a paper.
    fields = {
        "paper": ("title", "abstract", "keywords", "author_text"),
        "section": ("section_title", "section_summary", "section_text"),
        "chunk": ("text",),
        "figure": ("caption", "ocr_text", "footnote"),
    }.get(node["node_type"], ("canonical_name", "title", "name", "aliases", "search_text"))
    parts = list(dict.fromkeys(_text(node.get(key)).strip() for key in fields))
    return "\n".join(part for part in parts if part) or _text(node.get("search_text")) or node["node_type"]


def graph_signature(nodes: list[dict], edges: list[dict], feature_version: int = 1) -> str:
    """Reject an index after graph text, topology, or edge confidence changes."""
    payload = {
        "nodes": sorted((n["node_id"], n["node_type"], node_text(n)) for n in nodes),
        "edges": sorted((e["source_id"], e["target_id"], e["edge_type"],
                         float(e.get("confidence", 1.0))) for e in edges),
    }
    if feature_version >= 2:
        from .gnn_features import PAPER_FIELDS
        keys = [key for fields in PAPER_FIELDS.values() for key in fields]
        payload["paper_metadata"] = sorted((n["node_id"], json.dumps(
            {key: n.get(key) for key in keys}, ensure_ascii=False, sort_keys=True))
            for n in nodes if n["node_type"] == "paper")
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass
class EncoderConfig:
    model_name: str = "maidalun1020/bce-embedding-base_v1"
    revision: str = "f542e557e78bd8c5feed08573f183d87bc3d5535"
    pooling: str = "cls"
    max_length: int = 256
    batch_size: int = 16
    local_files_only: bool = True


class TextEncoder:
    def __init__(self, config: EncoderConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        from transformers import AutoModel, AutoTokenizer
        if config.pooling != "cls" or config.max_length != 256:
            raise ValueError("Appendix C.1 uses CLS pooling with a 256-token limit")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision, local_files_only=config.local_files_only)
        self.model = AutoModel.from_pretrained(config.model_name, revision=config.revision, local_files_only=config.local_files_only).to(self.device).eval()
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != 768:
            raise ValueError("BCE node/query features must be 768-dimensional")

    def encode(self, texts: list[str], progress: Callable[[str], None] | None = None) -> np.ndarray:
        if not texts:
            raise ValueError("cannot encode an empty text list")
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                inputs = self.tokenizer(texts[start:start + self.config.batch_size], padding=True,
                                        truncation=True, max_length=self.config.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                vectors = hidden[:, 0]
                batches.append(F.normalize(vectors.float(), dim=-1).cpu().numpy())
                if progress and (start == 0 or (start // self.config.batch_size) % 20 == 0):
                    progress(f"encoded {min(start + self.config.batch_size, len(texts))}/{len(texts)} nodes")
        return np.concatenate(batches)


    def encode_complete(self, texts, progress=None):
        """Encode every token in independent windows, returning coverage per field."""
        from .gnn_features import normalized_mean
        budget = self.config.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        if budget < 1:
            raise ValueError("max_length must leave room for content and special tokens")
        pieces, ranges, coverage = [], [], []
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
            start = len(pieces)
            windows = [ids[i:i + budget] for i in range(0, len(ids), budget)] or [[]]
            for window in windows:
                pieces.append(self.tokenizer.prepare_for_model(window, add_special_tokens=True,
                              truncation=False, return_attention_mask=True, verbose=False))
            ranges.append((start, len(pieces)))
            coverage.append({"token_count": len(ids), "encoded_token_count": sum(map(len, windows)),
                             "window_count": len(windows), "truncated_tokens": 0})
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(pieces), self.config.batch_size):
                inputs = self.tokenizer.pad(pieces[start:start + self.config.batch_size],
                                            padding=True, return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                batch = hidden[:, 0]
                vectors.append(F.normalize(batch.float(), dim=-1).cpu().numpy())
                if progress and (start == 0 or start // self.config.batch_size % 20 == 0):
                    progress(f"encoded {min(start + self.config.batch_size, len(pieces))}/{len(pieces)} metadata windows")
        vectors = np.concatenate(vectors)
        return np.stack([normalized_mean(vectors[a:b]) for a, b in ranges]), coverage


@dataclass
class GraphTensors:
    node_types: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    weight: torch.Tensor
    type_names: list[str]
    relation_names: list[str]

    def to(self, device: str) -> "GraphTensors":
        return GraphTensors(*(getattr(self, name).to(device) for name in
                              ("node_types", "source", "target", "relation", "weight")),
                            self.type_names, self.relation_names)


def tensorize_graph(nodes: list[dict], edges: list[dict]) -> GraphTensors:
    lookup = {node["node_id"]: i for i, node in enumerate(nodes)}
    if len(lookup) != len(nodes) or not nodes:
        raise ValueError("graph must contain unique, nonempty node IDs")
    types = sorted({node["node_type"] for node in nodes})
    # Include endpoint types to distinguish heterogeneous relations with the
    # same edge label. Each direction has independent trainable parameters.
    records = []
    for edge in edges:
        if edge["source_id"] not in lookup or edge["target_id"] not in lookup:
            raise ValueError(f"edge refers to absent node: {edge.get('edge_id', edge)}")
        source, target = lookup[edge["source_id"]], lookup[edge["target_id"]]
        confidence = float(edge.get("confidence", 1.0))
        if not np.isfinite(confidence) or confidence < 0:
            raise ValueError("edge confidence must be finite and nonnegative")
        name = json.dumps([nodes[source]["node_type"], edge["edge_type"], nodes[target]["node_type"]])
        records.extend([(source, target, name, confidence), (target, source, "reverse:" + name, confidence)])
    names = sorted({row[2] for row in records})
    relation_lookup = {name: i for i, name in enumerate(names)}
    source = torch.tensor([r[0] for r in records], dtype=torch.long)
    target = torch.tensor([r[1] for r in records], dtype=torch.long)
    relation = torch.tensor([relation_lookup[r[2]] for r in records], dtype=torch.long)
    weight = torch.tensor([r[3] for r in records], dtype=torch.float32)
    # First normalize within each destination/relation, then average relations.
    # A prolific citation relation cannot drown out the section relation.
    if records:
        key = target * len(names) + relation
        _, group = torch.unique(key, return_inverse=True)
        sums = torch.zeros(int(group.max()) + 1).index_add_(0, group, weight)
        weight = weight / sums[group].clamp_min(1e-12)
        active = (sums > 0).float()
        unique_targets = torch.unique(key, sorted=True) // len(names)
        counts = torch.zeros(len(nodes)).index_add_(0, unique_targets, active)
        weight /= counts[target].clamp_min(1)
    return GraphTensors(torch.tensor([types.index(n["node_type"]) for n in nodes]), source, target,
                        relation, weight, types, names)


class RelationLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_count: int):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation_gate = nn.Embedding(max(1, relation_count), hidden_dim)
        nn.init.ones_(self.relation_gate.weight)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, graph: GraphTensors) -> torch.Tensor:
        transformed = self.neighbor_projection(hidden)
        messages = transformed[graph.source] * self.relation_gate(graph.relation) * graph.weight[:, None]
        neighbors = torch.zeros_like(hidden).index_add_(0, graph.target, messages)
        return self.norm(hidden + F.gelu(self.self_projection(hidden) + neighbors))


class DocumentGraphEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, type_count: int, relation_count: int,
                 layers: int = 2, residual_scale: float = 0.25):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.type_embedding = nn.Embedding(type_count, hidden_dim)
        self.layers = nn.ModuleList(RelationLayer(hidden_dim, relation_count) for _ in range(layers))
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        self.residual_scale = residual_scale

    def forward(self, features: torch.Tensor, graph: GraphTensors) -> torch.Tensor:
        hidden = self.input_projection(features) + self.type_embedding(graph.node_types)
        for layer in self.layers:
            hidden = layer(hidden, graph)
        delta = F.normalize(self.output_projection(hidden), dim=-1)
        return F.normalize(features + self.residual_scale * delta, dim=-1)


@dataclass
class TrainingConfig:
    epochs: int = 40
    hidden_dim: int = 128
    layers: int = 2
    learning_rate: float = 0.001
    mask_rate: float = 0.2
    contrastive_batch_size: int = 256
    temperature: float = 0.1
    contrastive_weight: float = 0.1
    anchor_weight: float = 1.0
    residual_scale: float = 0.25
    seed: int = 42
    device: str = "cpu"
    threads: int = 4


def train_embeddings(graph_dir: Path, output_dir: Path, encoder_config: EncoderConfig,
                     config: TrainingConfig, progress: Callable[[str], None] = print,
                     encoder_model: str | None = None) -> dict:
    if config.epochs < 1 or config.layers < 1 or config.hidden_dim < 1:
        raise ValueError("epochs, layers and hidden_dim must be positive")
    if not 0 < config.mask_rate < 1 or not 0 < config.residual_scale <= 1:
        raise ValueError("mask_rate must be in (0,1); residual_scale in (0,1]")
    if config.temperature <= 0 or config.learning_rate <= 0 or config.contrastive_batch_size < 2:
        raise ValueError("invalid learning rate, temperature or contrastive batch size")
    graph_dir, output_dir = Path(graph_dir).resolve(), Path(output_dir).resolve()
    if output_dir == graph_dir:
        raise ValueError("save the embedding index in a separate directory from the graph")
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    nodes, edges = read_jsonl(graph_dir / "nodes.jsonl"), read_jsonl(graph_dir / "edges.jsonl")
    if len(nodes) < 2:
        raise ValueError("self-supervised training requires at least two nodes")
    from .gnn_features import FEATURE_VERSION, encode_node_features
    old_index_path = output_dir / "index.json"
    if old_index_path.exists() and json.loads(old_index_path.read_text(encoding="utf-8")).get("feature_version", 1) != FEATURE_VERSION:
        raise ValueError("use a new output directory for v2 features; preserve the previous index")
    signature = graph_signature(nodes, edges, FEATURE_VERSION)
    graph = tensorize_graph(nodes, edges).to(config.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path, cache_meta_path = output_dir / "text_features.npy", output_dir / "features.json"
    # Ordered IDs are included because graph_signature deliberately ignores row order.
    feature_metadata = {"feature_version": FEATURE_VERSION, "graph_signature": signature, "node_ids": [n["node_id"] for n in nodes],
                        "encoder": asdict(encoder_config)}
    if cache_path.exists() and cache_meta_path.exists() and json.loads(cache_meta_path.read_text(encoding="utf-8")) == feature_metadata:
        features = np.load(cache_path, allow_pickle=False)
        progress(f"loaded cached features for {len(nodes)} nodes")
    else:
        encoder = TextEncoder(replace(encoder_config, model_name=encoder_model or encoder_config.model_name), config.device)
        features, audit = encode_node_features(nodes, edges, encoder, progress)
        (output_dir / "paper_feature_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        del encoder
        np.save(cache_path, features)
        cache_meta_path.write_text(json.dumps(feature_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if features.ndim != 2 or features.shape[0] != len(nodes) or not np.isfinite(features).all() or (np.linalg.norm(features, axis=1) < 1e-8).any():
        raise ValueError("invalid cached text features")
    x = torch.from_numpy(features).to(config.device)
    model_args = dict(feature_dim=x.shape[1], hidden_dim=config.hidden_dim, type_count=len(graph.type_names),
                      relation_count=len(graph.relation_names), layers=config.layers, residual_scale=config.residual_scale)
    torch.manual_seed(config.seed)
    model = DocumentGraphEncoder(**model_args).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(config.epochs):
        model.train()
        mask = torch.rand(len(nodes), device=config.device) < config.mask_rate
        if not mask.any():
            mask[torch.randint(len(nodes), (1,), device=config.device)] = True
        noisy = x.clone()
        noisy[mask] = 0
        predictions = model(noisy, graph)
        reconstruction = (1 - (predictions[mask] * x[mask]).sum(-1)).mean()
        visible = ~mask
        anchor = (1 - (predictions[visible] * x[visible]).sum(-1)).mean() if visible.any() else x.new_zeros(())
        chosen = torch.where(mask)[0]
        chosen = chosen[torch.randperm(len(chosen), device=config.device)[:config.contrastive_batch_size]]
        logits = predictions[chosen] @ x[chosen].T / config.temperature
        # Identical text belongs to the positive set, not the negative set.
        positives = (x[chosen] @ x[chosen].T) > 0.99999
        positives.fill_diagonal_(True)
        contrastive = (torch.logsumexp(logits, dim=1) -
                       torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)).mean()
        loss = reconstruction + config.anchor_weight * anchor + config.contrastive_weight * contrastive
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"epoch": epoch + 1, "loss": loss.item(), "masked_reconstruction": reconstruction.item(),
               "anchor_loss": anchor.item(), "contrastive_loss": contrastive.item()}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == config.epochs:
            progress(f"epoch {epoch + 1}/{config.epochs}: loss={loss.item():.4f}")
    model.eval()
    with torch.inference_mode():
        embeddings = model(x, graph).cpu().numpy()
    np.save(output_dir / "node_embeddings.npy", embeddings)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "model_args": model_args,
                "type_names": graph.type_names, "relation_names": graph.relation_names}, output_dir / "gnn_model.pt")
    metadata = {"schema_version": 2, "feature_version": FEATURE_VERSION, "graph_dir": "../graph", "graph_signature": signature,
                "node_ids": [node["node_id"] for node in nodes], "node_types": [node["node_type"] for node in nodes],
                "node_count": len(nodes), "edge_count": len(edges), "embedding_dim": int(embeddings.shape[1]),
                "encoder": asdict(encoder_config), "training": asdict(config),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "objective": "masked_node_text_reconstruction+contrastive_alignment+visible_text_anchor",
                "figure_features": "caption_and_ocr_text", "history": history,
                "mean_cosine_to_text": float((embeddings * features).sum(-1).mean())}
    (output_dir / "index.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


class EmbeddingIndex:
    def __init__(self, index_dir: Path, graph_dir: Path, device: str = "cpu", encoder_model: str | None = None):
        self.index_dir = Path(index_dir).resolve()
        self.metadata = json.loads((self.index_dir / "index.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != 2:
            raise ValueError("unsupported GNN index schema")
        nodes, edges = read_jsonl(Path(graph_dir) / "nodes.jsonl"), read_jsonl(Path(graph_dir) / "edges.jsonl")
        feature_version = self.metadata.get("feature_version", 1)
        if feature_version != 2:
            raise ValueError("unsupported node feature version")
        if graph_signature(nodes, edges, feature_version) != self.metadata["graph_signature"]:
            raise ValueError("GNN index does not match this graph; retrain/re-export after graph changes")
        self.ids = self.metadata["node_ids"]
        if len(self.ids) != len(set(self.ids)) or set(self.ids) != {n["node_id"] for n in nodes}:
            raise ValueError("GNN index node IDs do not match this graph")
        self.lookup = {node_id: i for i, node_id in enumerate(self.ids)}
        self.embeddings = np.load(self.index_dir / "node_embeddings.npy", allow_pickle=False)
        if self.embeddings.shape != (len(self.ids), self.metadata["embedding_dim"]) or not np.isfinite(self.embeddings).all():
            raise ValueError("invalid GNN embedding matrix")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        if (norms < 1e-8).any():
            raise ValueError("GNN index contains zero vectors")
        self.embeddings = self.embeddings / norms
        config = EncoderConfig(**self.metadata["encoder"])
        if encoder_model:
            config.model_name = encoder_model
        self.encoder_config, self.device = config, device
        self.encoder = None
        self.last_query, self.last_scores, self.last_vector = None, None, None

    def scores(self, query: str) -> np.ndarray:
        query = str(query).strip()
        if not query:
            raise ValueError("query must not be empty")
        if query != self.last_query:
            if self.encoder is None:
                self.encoder = TextEncoder(self.encoder_config, self.device)
            vector = self.encoder.encode([query])[0]
            if vector.shape[0] != self.embeddings.shape[1]:
                raise ValueError("query encoder dimension differs from the trained index")
            self.last_vector = vector
            self.last_scores = self.embeddings @ vector
            self.last_query = query
        return self.last_scores

    def query_vector(self, query: str) -> np.ndarray:
        self.scores(query)
        return self.last_vector

    def score(self, query: str, node_id: str) -> float:
        return float(self.scores(query)[self.lookup[node_id]])

    def rank(self, query: str, rows: list[dict], limit: int) -> list[tuple[dict, float]]:
        scores = self.scores(query)
        ranked = [(row, float(scores[self.lookup[row["node_id"]]])) for row in rows]
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]["node_id"]))[:max(0, limit)]
