"""Regression checks for source scope, floating assets, and retained source text."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from demo_support import audit_retained_evidence
from pdf_vlm_assistant.document_entity_graph import reassign_floating_figures
from pdf_vlm_assistant.document_entity_retrieval import (
    named_source_paper_ids, classify_query_section_role, infer_section_query_focus,
    DocumentEntityRetriever, DocumentEntityRetrieverConfig, is_visual_query, annotate_section_role_context)
from pdf_vlm_assistant.query_context_bundle import (
    QueryContextBundleConfig, build_query_context_bundle, shrink_bundle_once, format_page_range)


class RecallRegressionTests(unittest.TestCase):
    def test_evidence_audit_excludes_query_metadata_and_partial_words(self):
        case = {"expected_evidence": [{"paper_id": "p", "description": "Projection evidence",
                                      "all_of": [["projection"], ["align"]]}]}
        bundle = {"query": "projection align", "papers": [{"paper_id": "p", "why_selected": ["projection align"],
                   "sections": [{"section_id": "p:1", "section_entities": ["projection", "align"],
                   "chunks": [{"node_id": "chunk:p:1", "text": "A projection could be misaligned."}]}]}]}
        self.assertEqual(audit_retained_evidence(case, bundle)["status"], "missing_evidence")
        bundle["papers"][0]["sections"][0]["chunks"][0]["text"] = "A projection is used to align the graph token."
        self.assertEqual(audit_retained_evidence(case, bundle)["status"], "passed")

    def test_weak_generic_entity_does_not_displace_relevant_method_section(self):
        retriever = DocumentEntityRetriever.__new__(DocumentEntityRetriever)
        retriever.hybrid = retriever.gnn = None
        retriever.config = DocumentEntityRetrieverConfig()
        retriever.figures = []
        retriever.sections_by_paper = {"p": [
            {"node_id": "section:a", "section_id": "a", "section_title": "Subgraph Construction", "order": 0},
            {"node_id": "section:b", "section_id": "b", "section_title": "Other Methods", "order": 1}]}
        retriever._cached_tfidf = lambda *args: [.4, .1]
        retriever.chunks_by_section = retriever.figures_by_section = {}
        retriever.entity_by_node_id = {"task:retrieval": {"node_type": "task"}}
        retriever.section_entity_edges_by_section = {"section:b": [{"target_id": "task:retrieval"}]}
        weak = [{"entity_node_id": "task:retrieval", "entity_type": "task", "match_source": "fuzzy", "score": .16}]
        rows = retriever.rank_sections_for_paper("How is the subgraph constructed?", "p", 1, 0, 0, weak)
        self.assertEqual(rows[0]["section_id"], "a")

    def test_source_page_labels_are_one_based_and_keep_page_zero(self):
        self.assertEqual(format_page_range(0, 0), "p.1")
        self.assertEqual(format_page_range(5, 6), "p.6-7")
        self.assertEqual(format_page_range(None, None), "")

    def test_graph_data_structure_is_not_automatically_a_visual_request(self):
        self.assertFalse(is_visual_query("Explain how a graph encoder represents a graph."))
        self.assertTrue(is_visual_query("Explain Figure 3 and its graph encoder."))
        self.assertTrue(is_visual_query("Show the graph of retrieval results."))

    def test_related_work_role_follows_numbered_parent_within_its_paper(self):
        sections = [{"paper_id": "a", "section_title": "2 Related Work"},
                    {"paper_id": "a", "section_title": "2.1 Retrieval Methods"},
                    {"paper_id": "a", "section_title": "3.1 Graph Construction"},
                    {"paper_id": "b", "section_title": "2.1 Retrieval Methods"}]
        annotate_section_role_context(sections)
        self.assertEqual(classify_query_section_role(sections[1]), "other_content")
        self.assertEqual(classify_query_section_role(sections[2]), "methods")
        self.assertEqual(classify_query_section_role(sections[3]), "methods")
        self.assertEqual(classify_query_section_role({"section_title": "D.4 The Quality of Retrieval"}), "results")

    def test_named_sources_use_unique_metadata_aliases_and_word_boundaries(self):
        papers = [{"paper_id": "a", "title": "AlphaNet: A Graph Framework"},
                  {"paper_id": "b", "title": "BetaNet: A Retrieval Framework"}]
        self.assertEqual(named_source_paper_ids("Explain AlphaNet's encoder.", papers), {"a"})
        self.assertEqual(named_source_paper_ids("Compare AlphaNet and BetaNet.", papers), {"a", "b"})
        self.assertEqual(named_source_paper_ids("Explain MetaAlphaNet.", papers), set())
        papers.append({"paper_id": "c", "title": "AlphaNet: Another Framework"})
        self.assertEqual(named_source_paper_ids("Explain AlphaNet.", papers), set())

    def test_reference_and_exhaustive_queries_are_not_mistaken_for_source_scope(self):
        papers = [{"paper_id": "a", "title": "AlphaNet: A Graph Framework"}]
        self.assertEqual(named_source_paper_ids("Which papers use AlphaNet?", papers), set())
        self.assertEqual(named_source_paper_ids("Find papers citing AlphaNet.", papers), set())
        self.assertEqual(named_source_paper_ids("AlphaNet graph reasoning retrieval", papers), set())

    def test_method_subsections_and_result_headings_have_distinct_roles(self):
        self.assertEqual(classify_query_section_role({"section_title": "5.4 Answer Generation"}), "methods")
        self.assertEqual(classify_query_section_role({"section_title": "5.3 Subgraph Construction"}), "methods")
        self.assertEqual(classify_query_section_role({"section_title": "6 Retrieval Performance"}), "results")
        self.assertTrue(infer_section_query_focus("Explain how it works.", set(), [])['semantic_query'])

    def test_floating_figure_uses_nearby_explicit_discussion(self):
        figure = {"node_id": "figure:p:1", "figure_id": "p:1", "section_id": "old",
                  "section_node_id": "section:old", "page_start": 4,
                  "reference_labels": ["figure 2"], "caption": ["Figure 2: The framework."], "ocr_text": ""}
        chunks = [{"node_id": "chunk:p:1", "chunk_id": "p:1", "section_id": "new",
                   "page_start": 4, "order": 1, "text": "The framework is shown in Figure 2."}]
        sections = [{"section_id": "old", "node_id": "section:old", "section_title": "Datasets"},
                    {"section_id": "new", "node_id": "section:new", "section_title": "Method"}]
        reassign_floating_figures([figure], chunks, sections, "A Paper")
        self.assertEqual(figure["section_id"], "new")
        self.assertEqual(figure["section_assignment"]["evidence_chunk_id"], "chunk:p:1")
        self.assertIn("section Method", figure["search_text"])

    def test_distant_mentions_do_not_reassign_a_figure(self):
        figure = {"figure_id": "p:1", "section_id": "old", "page_start": 1,
                  "reference_labels": ["figure 2"]}
        chunks = [{"chunk_id": "p:1", "section_id": "new", "page_start": 8,
                   "order": 1, "text": "Figure 2 illustrates the method."}]
        reassign_floating_figures([figure], chunks, [], "A Paper")
        self.assertEqual(figure["section_id"], "old")

    def test_context_uses_full_source_text_instead_of_display_preview(self):
        source = "Background detail. " * 20 + "The projection aligns the graph token with the language model."
        section = {"section_id": "p:1", "section_title": "Generation", "rank_score": 1.,
                   "section_summary": "", "top_chunks": [{"node_id": "chunk:p:1", "preview": "Background...", "rank_score": 1.}],
                   "top_figures": []}
        payload = {"results": [{"paper_id": "p", "paper_title": "Example", "rank_score": 1., "top_sections": [section]}]}
        retriever = SimpleNamespace(retrieval_backend="hybrid", config=SimpleNamespace(gnn_index_dir=None),
            hybrid=None, chunk_by_node_id={"chunk:p:1": {"text": source}}, retrieve=lambda **kwargs: payload)
        bundle = build_query_context_bundle("Explain the projection.", QueryContextBundleConfig(max_total_tokens=800), retriever)
        self.assertIn("The projection aligns the graph token", bundle["llm_context_markdown"])
        self.assertEqual(bundle["papers"][0]["sections"][0]["chunks"][0]["text"], source)

    def test_trailing_chunk_pruning_preserves_primary_evidence_before_summaries(self):
        papers = [{"sections": [{"chunks": [{"text": "Primary evidence"}],
                                  "section_entities": ["a redundant entity"], "section_summary": "A duplicate summary"}]}]
        self.assertTrue(shrink_bundle_once(papers, "Explain the method."))
        self.assertEqual(papers[0]["sections"][0]["chunks"][0]["text"], "Primary evidence")
        self.assertEqual(papers[0]["sections"][0]["section_entities"], [])
        self.assertTrue(shrink_bundle_once(papers, "Explain the method."))
        self.assertEqual(papers[0]["sections"][0]["section_summary"], "")
        self.assertTrue(papers[0]["sections"][0]["chunks"])


if __name__ == "__main__":
    unittest.main()
