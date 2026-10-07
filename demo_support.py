"""Validation and portable artifact writing for the three-paper workflow."""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

def relative_paths(value, base):
    """Rewrite local absolute paths under this package, without changing text."""
    if isinstance(value, dict):
        return {k: relative_paths(v, base) for k, v in value.items()}
    if isinstance(value, list):
        return [relative_paths(v, base) for v in value]
    if isinstance(value, str):
        root = str(ROOT)
        if value == root or value.startswith(root + os.sep):
            return Path(os.path.relpath(value, base)).as_posix()
    return value

def make_graph_portable(graph_dir):
    for path in graph_dir.glob("*.jsonl"):
        rows = [relative_paths(row, graph_dir) for row in read_jsonl(path)]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    path = graph_dir / "graph_stats.json"
    dump(path, relative_paths(json.loads(path.read_text(encoding="utf-8")), graph_dir))

def audit_graph(graph_dir):
    from pypdf import PdfReader
    nodes = read_jsonl(graph_dir / "nodes.jsonl")
    edges = read_jsonl(graph_dir / "edges.jsonl")
    cross = read_jsonl(graph_dir / "paper_paper_edges.jsonl")
    by_id = {row["node_id"]: row for row in nodes}
    counts = Counter(row["node_type"] for row in nodes)
    require(len(by_id) == len(nodes), "Duplicate node IDs")
    require(len({e["edge_id"] for e in edges}) == len(edges), "Duplicate edge IDs")
    require(all(e["source_id"] in by_id and e["target_id"] in by_id for e in edges), "Dangling edge")
    for kind in ["paper", "section", "chunk", "figure", "author", "method", "dataset", "reference"]:
        require(counts[kind] > 0, "Missing required entity type: " + kind)
    require(counts["paper"] == 3, "Expected exactly three papers")
    metadata = json.loads((ROOT / "data/papers.json").read_text(encoding="utf-8"))
    source_text = {}
    pdf_text = {}
    author_checks = []
    for record in metadata:
        pid = record["paper_id"]
        paper = by_id["paper:" + pid]
        items = json.loads((ROOT / record["segments"] / "paper_content_list.json").read_text(encoding="utf-8"))
        source_text[pid] = norm(" ".join(str(x.get("text", "")) for x in items))
        reader = PdfReader(ROOT / record["pdf"])
        page_texts = [page.extract_text() or "" for page in reader.pages]
        pdf_text[pid] = [norm(text) for text in page_texts]
        require(norm(record["title"]) in pdf_text[pid][0], "PDF title mismatch: " + pid)
        require(set(paper["authors"]) == set(record["authors"]), "Incorrect author segmentation: " + pid)
        for author in record["authors"]:
            require(norm(author) in pdf_text[pid][0], "Author absent from PDF first page: " + author)
        author_checks.append({"paper_id": pid, "author_count": len(record["authors"]), "pdf_page": 1})
        for key in ["chunk", "figure", "section"]:
            require(any(n["node_type"] == key and n.get("paper_id") == pid for n in nodes), pid + " missing " + key)
    keys = {(e["source_id"], e["target_id"], e["edge_type"]) for e in edges}
    shared_relations = {
        "paper_shares_author": "paper_has_author",
        "paper_shares_institution": "paper_has_institution",
        "paper_shares_method": "paper_uses_method",
        "paper_shares_dataset": "paper_uses_dataset",
        "paper_shares_model": "paper_uses_model",
        "paper_bibliographic_coupling": "paper_cites_reference",
    }
    audited = []
    for edge in cross:
        a, b = edge["source_id"], edge["target_id"]
        require(a != b, "Cross-paper self-loop")
        require(by_id[a]["node_type"] == by_id[b]["node_type"] == "paper", "Invalid cross-paper endpoint")
        require(edge["evidence_entity_ids"], "Missing bridge entity")
        evidence = []
        for eid in edge["evidence_entity_ids"]:
            require(eid in by_id, "Unknown bridge entity")
            entity = by_id[eid]
            kind = edge["edge_type"]
            if kind == "paper_cites_paper":
                require((a, eid, "paper_cites_reference") in keys, "Missing citation source")
                require((eid, b, "reference_resolved_to_paper") in keys, "Missing strong citation resolution")
            else:
                relation = shared_relations[kind]
                require((a, eid, relation) in keys and (b, eid, relation) in keys, "Unsupported shared-entity edge")
            name = entity.get("canonical_name") or entity.get("title") or entity.get("search_text", "").split("\n")[0]
            item = {"entity_id": eid, "name": name, "sources": []}
            endpoints = [a] if kind == "paper_cites_paper" else [a, b]
            for endpoint in endpoints:
                pid = by_id[endpoint]["paper_id"]
                base_edges = [e for e in edges if e["source_id"] == endpoint and e["target_id"] == eid]
                pages = [i + 1 for i, text in enumerate(pdf_text[pid]) if norm(name) in text]
                # Dataset aliases may contain a descriptive suffix absent from the PDF.
                if not pages and entity["node_type"] == "dataset":
                    pages = [i + 1 for i, text in enumerate(pdf_text[pid]) if norm(name.replace(" dataset", "")) in text]
                require(pages, "Bridge entity absent from source PDF: " + pid + " / " + name)
                snippets = []
                for base in base_edges:
                    for cid in base.get("evidence_chunk_ids", []):
                        chunk = by_id.get("chunk:" + cid)
                        require(chunk and chunk["paper_id"] == pid, "Invalid evidence chunk")
                        snippets.append({"chunk_id": chunk["node_id"], "page": chunk["page_start"] + 1 if chunk.get("page_start") is not None else None, "text": chunk.get("text", "")[:1200]})
                if entity["node_type"] == "method":
                    require(snippets, "Method relationship needs body evidence")
                    from pdf_vlm_assistant.document_entity_graph import method_is_comparison_only
                    require(any(not method_is_comparison_only(s["text"], name) for s in snippets), "Baseline-only method edge")
                item["sources"].append({"paper_id": pid, "pdf_pages": pages, "snippets": snippets})
            evidence.append(item)
        audited.append({"relation": edge["edge_type"], "source": a, "target": b, "evidence": evidence})
    author_edge = next(e for e in cross if e["edge_type"] == "paper_shares_author")
    require(set(author_edge["evidence_entity_ids"]) == {"author:qi_song", "author:qi_zhao", "author:wangqiu_zhou"}, "Wrong shared authors")
    method_edges = [e for e in cross if e["edge_type"] == "paper_shares_method"]
    require(len(method_edges) == 1 and {method_edges[0]["source_id"], method_edges[0]["target_id"]} == {"paper:g_retriever", "paper:sentgraph"}, "Unexpected shared-method relationship")
    require(("paper:rje", "paper:g_retriever", "paper_cites_paper") in keys, "Missing directional citation")
    require(("paper:g_retriever", "paper:rje", "paper_cites_paper") not in keys, "Spurious reverse citation")
    require(("paper:rje", "method:retrieval_augmented_generation", "paper_uses_method") not in keys, "Baseline mention promoted to method use")
    require(any(n["node_type"] == "chunk" and n.get("page_start") == 0 for n in nodes), "Page zero was discarded")
    for figure in (n for n in nodes if n["node_type"] == "figure"):
        path = graph_dir / figure["image_path"]
        require(path.is_file(), "Figure image missing")
        with Image.open(path) as image:
            require(image.width > 0 and image.height > 0, "Invalid figure image")
    return {"status": "passed", "node_count": len(nodes), "edge_count": len(edges), "node_types": dict(counts),
            "cross_paper_edge_count": len(cross), "author_pdf_checks": author_checks, "cross_paper_evidence": audited}

def audit_crop(payload):
    import numpy as np
    from scripts.select_pixel_crops_by_query import intersection_over_union
    image = Image.open(ROOT / payload["image_path"]).convert("RGB")
    rows = payload["selected_windows"]
    require(rows, "No selected crops")
    require(len(payload["top_scored_windows"]) > 1, "Missing candidate scores")
    require("Figure 3:" in " ".join(payload["figure_metadata"]["caption"]), "Wrong target figure")
    budget = payload["selection_budget"]
    require(payload.get("fallback_used") or sum(row["area_ratio"] for row in rows) <= budget["max_total_area_ratio"] + 1e-6, "Crop budget exceeded")
    require(len(rows) <= budget["keep_top_k"], "Too many crops")
    for i, row in enumerate(rows):
        x1, y1, x2, y2 = row["box"]
        require(0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height, "Crop bounds invalid")
        expected = image.crop((x1, y1, x2, y2))
        with Image.open(ROOT / row["crop_path"]) as saved:
            require(np.array_equal(np.asarray(expected), np.asarray(saved.convert("RGB"))), "Exported crop differs from specified pixels")
        for other in rows[:i]:
            require(intersection_over_union(row["box"], other["box"]) < budget["nms_iou_threshold"], "NMS overlap constraint violated")
        formula = payload["score_formula"]
        score = row["query_score"] - formula["text_penalty"] * row["text_redundancy"] - formula["area_penalty"] * row["area_ratio"]
        require(abs(score - row["retain_score"]) < 1e-5, "CLIP score formula mismatch")
    with Image.open(ROOT / payload["union_image"]) as merged:
        require(np.array_equal(np.asarray(image.crop(payload["union_box"])), np.asarray(merged.convert("RGB"))), "Union pixels differ")
    return {"status": "passed", "backend": "real CLIP", "selected_windows": len(rows),
            "summed_area_ratio": sum(row["area_ratio"] for row in rows), "budget": budget["max_total_area_ratio"],
            "pixel_equality": True, "figure_node_id": payload["figure_metadata"]["node_id"]}


def sanitize_runtime_log(text, extra_paths=()):
    """Keep diagnostics without recording the submitter's local paths."""
    replacements = [(ROOT, "<package-root>"), (Path.home(), "<home>"),
                    (Path(sys.prefix), "<python>"), (Path(sys.executable), "<python-executable>")]
    replacements += [(Path(p), "<model>") for p in extra_paths]
    variants = []
    for path, label in replacements:
        raw = str(path.resolve())
        for form in {raw, raw.replace(chr(92), "/"), raw.replace(chr(92), chr(92) * 2)}:
            variants.append((form, label))
    for form, label in sorted(variants, key=lambda item: -len(item[0])):
        text = text.replace(form, label)
    return text


def audit_entity_use(retriever, payloads):
    """Verify actual entity-channel contributions and counterfactual messages."""
    import torch
    import numpy as np
    from pdf_vlm_assistant.gnn_embeddings import DocumentGraphEncoder, tensorize_graph
    from pdf_vlm_assistant.gnn_paper_recall import CONTENT_TYPES, GraphPaperRecall
    nodes = read_jsonl(retriever.graph_dir / "nodes.jsonl")
    edges = retriever.edges
    entity_ids = {n["node_id"] for n in nodes if n["node_type"] not in CONTENT_TYPES}
    graph = tensorize_graph(nodes, edges)
    checkpoint = torch.load(retriever.gnn.index.index_dir / "gnn_model.pt", map_location="cpu", weights_only=True)
    model = DocumentGraphEncoder(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = torch.from_numpy(np.load(retriever.gnn.index.index_dir / "text_features.npy"))
    # Keep every original node/feature but remove entity links and their derived
    # cross-paper relations, as in the entity ablation definition (Appendix E.3).
    removed_edges = [e for e in edges if e["source_id"] not in entity_ids and e["target_id"] not in entity_ids
                     and not (set(e.get("evidence_entity_ids", [])) & entity_ids)]
    ablated_graph = tensorize_graph(nodes, removed_edges)
    # Align surviving relation IDs with this checkpoint; no retraining or relabeling.
    relation_ids = {name: i for i, name in enumerate(graph.relation_names)}
    mapping = torch.tensor([relation_ids[name] for name in ablated_graph.relation_names], dtype=torch.long)
    ablated_graph.relation = mapping[ablated_graph.relation]
    with torch.inference_mode():
        full = model(x, graph)
        without = model(x, ablated_graph)
    paper_rows = [i for i,n in enumerate(nodes) if n["node_type"] == "paper"]
    changes = {nodes[i]["paper_id"]: float(torch.linalg.vector_norm(full[i] - without[i])) for i in paper_rows}
    require(all(change > 1e-6 for change in changes.values()), "Entity messages do not affect paper representations")
    query_rows = []
    for case_id, payload in payloads.items():
        for paper in payload["results"]:
            evidence = [row for row in paper["paper_recall_evidence"] if row["node_type"] not in CONTENT_TYPES]
            require(evidence, "Entity recall channels unused: " + case_id)
            query_rows.append({"query_id": case_id, "paper_id": paper["paper_id"],
                               "entity_contribution": sum(e["rrf_contribution"] for e in evidence), "entities": evidence})
    for case_id, entity_id in [("shared_author", "author:qi_song"), ("shared_dataset", "dataset:webqsp_dataset")]:
        require(all(any(e["node_id"] == entity_id for e in row["entities"])
                    for row in query_rows if row["query_id"] == case_id), "Wrong entity-channel support")
    require(torch.allclose(full, torch.from_numpy(retriever.gnn.index.embeddings), atol=1e-6), "Stored GNN export differs")
    return {"status": "passed", "entity_node_count": len(entity_ids),
            "removed_message_source_edges": len(edges) - len(removed_edges),
            "paper_embedding_l2_change_without_entity_messages": changes,
            "entity_channel_support": query_rows,
            "interpretation": "Contribution check using the same checkpoint; not a retrained quality ablation."}


def audit_retained_evidence(case, bundle):
    """Check fixture concepts in source text only, excluding query and routing metadata."""
    def contains(text, term):
        return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE) is not None

    records = []
    for check in case.get("expected_evidence", []):
        snippets = []
        for paper in bundle.get("papers", []):
            if paper["paper_id"] != check["paper_id"]:
                continue
            for section in paper.get("sections", []):
                if section.get("section_summary"):
                    snippets.append((section["section_id"], section["section_summary"]))
                snippets.extend((c["node_id"], c["text"]) for c in section.get("chunks", []))
                snippets.extend((f["node_id"], f["caption"]) for f in section.get("figures", []))
        evidence = " ".join(text for _, text in snippets).casefold()
        missing = [options for options in check["all_of"] if not any(contains(evidence, term) for term in options)]
        support = [{"source_id": source, "text": text} for source, text in snippets
                   if any(contains(text, term) for options in check["all_of"] for term in options)]
        records.append({"paper_id": check["paper_id"], "description": check["description"],
                        "status": "passed" if not missing else "missing_evidence", "missing_concepts": missing,
                        "support": support})
    return {"status": "passed" if all(row["status"] == "passed" for row in records) else "missing_evidence",
            "scope": "Retained source-text concept coverage; generated-answer quality is not measured.", "checks": records}
