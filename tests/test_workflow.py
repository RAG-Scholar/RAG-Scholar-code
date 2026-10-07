"""Structural and semantic regressions; end-to-end CLIP is checked by run_demo.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from demo_support import audit_graph, make_graph_portable, read_jsonl, sanitize_runtime_log
from pdf_vlm_assistant.document_entity_graph import (
    DocumentEntityGraphConfig, build_document_entity_graph_from_segments,
    build_cross_paper_edges, method_is_comparison_only, contains_method_narrative_language,
)
from pdf_vlm_assistant.document_entity_retrieval import DocumentEntityRetriever, DocumentEntityRetrieverConfig
from scripts.select_pixel_crops_by_query import select_windows_budgeted

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix=".test-graph-", dir=ROOT)
        cls.graph = Path(cls.temp.name).resolve()
        cls.stats = build_document_entity_graph_from_segments(ROOT / "data/segments", cls.graph,
            DocumentEntityGraphConfig(enable_cross_paper_edges=True))
        make_graph_portable(cls.graph)
        cls.retriever = DocumentEntityRetriever(DocumentEntityRetrieverConfig(graph_dir=cls.graph))
        cls.edges = read_jsonl(cls.graph / "edges.jsonl")

    @classmethod
    def tearDownClass(cls):
        if cls.graph.parent != ROOT:
            raise RuntimeError("Unexpected temporary cleanup target")
        cls.temp.cleanup()

    def test_source_grounded_entities_and_every_cross_paper_edge(self):
        report = audit_graph(self.graph)
        self.assertEqual(report["cross_paper_edge_count"], 7)
        self.assertEqual(self.stats["skipped_paper_count"], 0)

    def test_constrained_retrieval_positive_and_negative_papers(self):
        cases = [
            ("Which papers are authored by Qi Song?", {"rje", "sentgraph"}),
            ("Which papers use retrieval-augmented generation?", {"g_retriever", "sentgraph"}),
            ("Which papers use the WebQSP dataset?", {"g_retriever", "rje"}),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                result = self.retriever.retrieve(query)
                self.assertEqual({p["paper_id"] for p in result["results"]}, expected)
                self.assertTrue(all(p["top_sections"] for p in result["results"]))

    def test_cross_edges_actually_contribute_to_retrieval(self):
        query = "RJE G-Retriever knowledge graph reasoning retrieval"
        enabled = self.retriever.retrieve(query)
        disabled = DocumentEntityRetriever(DocumentEntityRetrieverConfig(
            graph_dir=self.graph, enable_cross_paper_edges=False)).retrieve(query)
        self.assertTrue(any(p["cross_paper_support"] for p in enabled["results"]))
        self.assertFalse(any(p["cross_paper_support"] for p in disabled["results"]))

    def test_relative_paths_and_page_zero(self):
        figures = read_jsonl(self.graph / "figures.jsonl")
        self.assertTrue(all(not Path(f["image_path"]).is_absolute() for f in figures))
        self.assertTrue(all(Path(f["image_path"]).is_file() for f in self.retriever.figures))
        chunks = read_jsonl(self.graph / "chunks.jsonl")
        self.assertTrue(any(c["page_start"] == 0 for c in chunks))
        for paper in ["g_retriever", "rje", "sentgraph"]:
            self.assertTrue(any(c["page_start"] == 0 and c["paper_id"] == paper for c in chunks))

    def test_baseline_language_does_not_imply_method_use(self):
        self.assertTrue(method_is_comparison_only(
            "We compare with baseline methods, including Retrieval-Augmented Generation methods.",
            "retrieval-augmented generation"))
        self.assertFalse(method_is_comparison_only(
            "We propose a retrieval-augmented generation approach.", "retrieval-augmented generation"))
        self.assertTrue(contains_method_narrative_language("g-retriever outperforms traditional prompt tuning"))
        methods = read_jsonl(self.graph / "methods.jsonl")
        self.assertFalse(any("outperforms" in m["canonical_name"] for m in methods))

    def test_weak_reference_match_does_not_become_strong_citation(self):
        reference = next(e["target_id"] for e in self.edges
            if e["source_id"] == "paper:rje" and e["edge_type"] == "paper_cites_reference"
            and "g_retriever" in e["target_id"])
        weak = {"edge_type": "reference_weakly_resolved_to_paper", "source_id": reference,
                "target_id": "paper:g_retriever", "confidence": .65}
        citation = {"edge_type": "paper_cites_reference", "source_id": "paper:rje",
                    "target_id": reference, "confidence": .9}
        edges, _ = build_cross_paper_edges(papers=self.retriever.papers,
            author_edges=[], institution_edges=[], venue_edges=[], reference_edges=[citation],
            reference_resolution_edges=[weak], semantic_edges=[], config=DocumentEntityGraphConfig())
        self.assertFalse(any(e["edge_type"] == "paper_cites_paper" for e in edges))
        self.assertTrue(any(e["edge_type"] == "paper_weakly_cites_paper" for e in edges))

    def test_budget_cannot_be_exceeded_by_first_window_or_fallback(self):
        big = {"box": [0, 0, 100, 100], "area_ratio": 1., "efficiency_score": 10., "retain_score": 1.}
        small = {"box": [0, 0, 30, 30], "area_ratio": .09, "efficiency_score": 1., "retain_score": .4}
        result = select_windows_budgeted([big, small], 2, .5, .55)
        self.assertEqual([r["box"] for r in result], [small["box"]])
        self.assertEqual(select_windows_budgeted([big], 2, .5, .55), [])
        with self.assertRaises(ValueError):
            select_windows_budgeted([small], 0, .5, .55)

    def test_runtime_log_removes_machine_paths_but_preserves_paper_authors(self):
        model = ROOT / "model-cache"
        message = f"Image: {ROOT / 'data/example.png'}; model: {model}; home: {Path.home()}; Qi Song"
        cleaned = sanitize_runtime_log(message, [model])
        self.assertNotIn(str(ROOT), cleaned)
        self.assertNotIn(str(Path.home()), cleaned)
        self.assertIn("Qi Song", cleaned)
        self.assertIn("<model>", cleaned)
        self.assertIn("<package-root>", cleaned)

    def test_submission_contains_no_web_report_artifacts(self):
        self.assertEqual(list(ROOT.rglob("*.html")), [])
        self.assertEqual(list(ROOT.rglob("*.htm")), [])

    def test_budget_and_nms_both_apply(self):
        rows = [{"box": box, "area_ratio": .2, "efficiency_score": score, "retain_score": .3}
                for box, score in [([0, 0, 50, 40], 4), ([1, 0, 51, 40], 3),
                                   ([50, 40, 100, 80], 2), ([0, 50, 50, 90], 1)]]
        result = select_windows_budgeted(rows, 4, .5, .55)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(sum(r["area_ratio"] for r in result), .4)

if __name__ == "__main__":
    unittest.main(verbosity=2)
