from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .citation_graph import (
    DOI_RE,
    build_citation_match_key,
    build_citation_nodes_for_paper,
    build_title_key,
    is_suspicious_doi,
    normalize_doi,
    normalize_pmid,
)
from .paper_graph import (
    DEFAULT_SECTION_NAME,
    IMAGE_ITEM_TYPES,
    TEXT_ITEM_TYPES,
    ChunkBuffer,
    append_text_piece,
    build_image_feature_text,
    build_node_search_text,
    classify_section_path,
    count_alpha_tokens,
    discover_paper_dirs,
    extract_caption_list,
    extract_footnote_list,
    extract_item_text,
    extract_paper_metadata,
    extract_reference_labels_from_item,
    has_institution_strong_hint,
    has_institution_weak_hint,
    is_low_signal_text,
    is_sentence_like_affiliation,
    is_section_heading,
    join_unique_nonempty,
    load_segment_items,
    looks_like_institution_line,
    merge_bboxes,
    normalize_bbox,
    normalize_text,
    page_range,
    page_span,
    resolve_image_path,
    split_long_text,
    split_paragraphs,
    strip_affiliation_leader,
    to_int,
    tokenize_for_similarity,
    unique_preserve_order,
    update_section_stack,
    write_jsonl,
)
from .reference_name_utils import (
    best_reference_display_name,
    build_reference_search_text,
    infer_reference_title,
)


PMID_RE = re.compile(r"\bPMID\s*[:：]?\s*(\d+)\b", re.IGNORECASE)
KEYWORDS_RE = re.compile(r"^\s*keywords?\s*[:：]\s*(.+)$", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
REF_LABEL_PATTERN_TEMPLATE = r"\b%s\b"
ABSTRACT_NOISE_PREFIXES = (
    "received ",
    "accepted ",
    "published ",
    "revised ",
    "* e-mail",
    "e-mail",
    "email",
    "correspondence",
    "keywords:",
    "keyword:",
    "per-review",
    "peer-review",
)
VENUE_HINTS = (
    "journal",
    "reports",
    "review",
    "medicine",
    "medical",
    "biology",
    "sciences",
    "informatics",
    "therapy",
    "oncology",
    "pharmacy",
    "neurology",
    "communications",
)
VENUE_STOP_PATTERNS = (
    re.compile(r"^\s*https?://", re.IGNORECASE),
    re.compile(r"\bdoi\s*:", re.IGNORECASE),
    re.compile(r"\bdoi\.org\b", re.IGNORECASE),
    re.compile(r"\bjournal\.[a-z0-9._-]+\.g\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:figure|fig\.?|table|chart)\s*\d+[a-z]?\b", re.IGNORECASE),
)
SECTION_HEADING_LIKE_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+")
PAPER_NODE_ORDER = [
    "paper",
    "section",
    "chunk",
    "figure",
    "author",
    "institution",
    "venue",
    "reference",
    "method",
    "dataset",
    "task",
    "metric",
    "model",
]
SHARED_ENTITY_CROSS_EDGE_SPECS = {
    "paper_has_author": ("paper_shares_author", 3.0),
    "paper_has_institution": ("paper_shares_institution", 1.4),
    "paper_published_in": ("paper_shares_venue", 1.0),
    "paper_uses_method": ("paper_shares_method", 2.2),
    "paper_uses_dataset": ("paper_shares_dataset", 2.4),
    "paper_uses_model": ("paper_shares_model", 1.8),
}
GENERIC_REFERENCE_BRIDGE_KEYS = {
    "doi_10_1016_j",
    "doi_10_1073_pnas",
    "doi_10_1371_journal",
}

SEMANTIC_EDGE_TYPE_MAP = {
    "method": ("section_uses_method", "paper_uses_method", "figure_supports_method"),
    "dataset": ("section_uses_dataset", "paper_uses_dataset", "figure_supports_dataset"),
    "task": ("section_addresses_task", "paper_addresses_task", "figure_supports_task"),
    "metric": ("section_reports_metric", "paper_reports_metric", "figure_supports_metric"),
    "model": ("section_uses_model", "paper_uses_model", "figure_supports_model"),
}

SEMANTIC_ALIAS_TABLE: dict[str, dict[str, list[str]]] = {
    "method": {
        "logistic regression": [
            "logistic regression",
            "binary logistic regression",
            "multivariable logistic regression",
            "multivariate logistic regression",
            "mixed-effects logistic regression",
            "mixed effects logistic regression",
        ],
        "linear regression": [
            "linear regression",
            "multivariable linear regression",
            "multivariate linear regression",
        ],
        "cox regression": [
            "cox regression",
            "cox proportional hazards",
            "cox proportional-hazards",
            "cox model",
        ],
        "survival analysis": [
            "survival analysis",
            "kaplan-meier",
            "kaplan meier",
        ],
        "principal component analysis": [
            "principal component analysis",
            "pca",
        ],
        "k-means clustering": [
            "k-means clustering",
            "k means clustering",
        ],
        "simulated annealing": [
            "simulated annealing",
        ],
        "quantum annealing": [
            "quantum annealing",
        ],
        "molecular docking": [
            "molecular docking",
        ],
        "meta-analysis": [
            "meta-analysis",
            "meta analysis",
        ],
        "network pharmacology": [
            "network pharmacology",
        ],
        "single-cell rna sequencing": [
            "single-cell rna sequencing",
            "single cell rna sequencing",
            "scrna-seq",
            "scRNA-seq",
        ],
        "whole-genome sequencing": [
            "whole-genome sequencing",
            "whole genome sequencing",
            "wgs",
        ],
        "machine learning": [
            "machine learning",
        ],
        "deep learning": [
            "deep learning",
        ],
        "polymerase chain reaction": [
            "polymerase chain reaction",
            "pcr",
        ],
        "elisa": [
            "enzyme-linked immunosorbent assay",
            "elisa",
        ],
    },
    "dataset": {
        "mimic-iv": ["mimic-iv", "mimic iv"],
        "mimic-iii": ["mimic-iii", "mimic iii"],
        "imagenet": ["imagenet"],
        "mnist": ["mnist"],
        "cifar-10": ["cifar-10", "cifar10"],
        "tcga": ["tcga", "the cancer genome atlas"],
        "seer": ["seer"],
        "uk biobank": ["uk biobank"],
    },
    "task": {
        "classification": ["classification"],
        "prediction": ["prediction", "risk prediction", "outcome prediction"],
        "detection": ["detection", "object detection", "lesion detection"],
        "segmentation": ["segmentation", "image segmentation", "lesion segmentation"],
        "diagnosis generation": ["diagnosis generation", "diagnosis"],
        "phenotype extraction": ["phenotype extraction", "information extraction", "extraction"],
        "text generation": ["text generation", "generation"],
        "surveillance": ["surveillance"],
        "retrieval": ["retrieval"],
        "forecasting": ["forecasting"],
        "protein folding": ["protein folding"],
    },
    "metric": {
        "accuracy": ["accuracy"],
        "precision": ["precision"],
        "recall": ["recall", "sensitivity"],
        "specificity": ["specificity"],
        "f1 score": ["f1 score", "f1-score", "f1"],
        "auc": ["auc", "auroc", "roc auc", "area under the curve"],
        "auprc": ["auprc", "area under the precision recall curve"],
        "dice coefficient": ["dice coefficient", "dice score", "dice"],
        "intersection over union": ["intersection over union", "iou"],
        "mean absolute error": ["mean absolute error", "mae"],
        "mean squared error": ["mean squared error", "mse"],
        "root mean squared error": ["root mean squared error", "rmse"],
        "odds ratio": ["odds ratio"],
        "hazard ratio": ["hazard ratio"],
    },
    "model": {
        "graph neural network": ["graph neural network", "gnn"],
        "transformer": ["transformer"],
        "clip": ["clip"],
        "bert": ["bert"],
        "SentenceBERT": ["sentencebert", "sentence-bert", "sentence bert", "sbert"],
        "gpt": ["gpt"],
        "large language model": ["large language model", "llm"],
        "u-net": ["u-net", "unet"],
        "resnet": ["resnet"],
        "cnn": ["convolutional neural network", "cnn"],
        "rnn": ["recurrent neural network", "rnn"],
        "lstm": ["long short-term memory", "lstm"],
        "gan": ["generative adversarial network", "gan"],
        "xgboost": ["xgboost"],
        "lightgbm": ["lightgbm"],
        "catboost": ["catboost"],
        "cancerllm": ["cancerllm"],
    },
}

TASK_DYNAMIC_RE = re.compile(
    r"\b((?:[A-Za-z0-9-]+\s+){0,3}(classification|prediction|detection|segmentation|diagnosis|extraction|generation|surveillance|retrieval|forecasting|folding))\b",
    re.IGNORECASE,
)
DATASET_DYNAMIC_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9-]{1,}(?:[- ][A-Z][A-Za-z0-9-]{1,}){0,4}\s+(dataset|corpus|cohort|benchmark|registry|biobank))\b"
)
MODEL_DYNAMIC_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9-]{2,}(?:LLM|BERT|GPT|Former|Net)|U-?Net|ResNet|DenseNet|InceptionV3|LLaMA|CancerLLM)\b"
)
AFFILIATION_INDEX_RE = re.compile(r"^\s*\d+\s*[-.)]?\s*")
COUNTRY_LIKE_VALUES = {
    "argentina",
    "australia",
    "austria",
    "belgium",
    "brazil",
    "brasil",
    "canada",
    "china",
    "france",
    "germany",
    "india",
    "israel",
    "italy",
    "japan",
    "korea",
    "mexico",
    "netherlands",
    "portugal",
    "spain",
    "sweden",
    "switzerland",
    "uk",
    "united kingdom",
    "united states",
    "usa",
}
GENERIC_DATASET_PREFIXES = {
    "a",
    "an",
    "our",
    "the",
    "their",
    "this",
    "these",
    "those",
}
GENERIC_DATASET_HEADS = {
    "dataset",
    "data",
    "training",
    "testing",
    "validation",
    "input",
    "output",
    "raw",
    "full",
    "current",
}
TASK_STRIP_PREFIXES = {
    "a",
    "an",
    "the",
    "our",
    "this",
    "that",
    "these",
    "those",
    "its",
    "their",
}
TASK_BANNED_TOKENS = {
    "actively",
    "adapts",
    "adjusts",
    "and",
    "approach",
    "approaches",
    "beyond",
    "capability",
    "crucial",
    "demonstrates",
    "demonstrating",
    "dynamically",
    "driven",
    "dynamics",
    "enables",
    "enabling",
    "environments",
    "for",
    "foundation",
    "framework",
    "from",
    "improvement",
    "improving",
    "in",
    "include",
    "includes",
    "including",
    "information",
    "involved",
    "latency",
    "mechanism",
    "more",
    "nature",
    "occurrence",
    "offering",
    "of",
    "on",
    "practice",
    "provides",
    "providing",
    "relationships",
    "shows",
    "significant",
    "stage",
    "temporary",
    "testbed",
    "to",
    "validation",
    "where",
    "while",
    "widely",
    "with",
    "work",
}
TASK_CORE_SUFFIXES = {
    "classification",
    "prediction",
    "detection",
    "segmentation",
    "diagnosis",
    "extraction",
    "generation",
    "surveillance",
    "retrieval",
    "forecasting",
    "folding",
}
TASK_GENERIC_MODIFIERS = {
    "accurate",
    "accurately",
    "accelerate",
    "accelerates",
    "accelerating",
    "efficient",
    "improved",
    "improving",
    "rapid",
    "reliable",
    "robust",
}
TASK_SENTENCE_TOKENS = {
    "about",
    "above",
    "achieve",
    "achieved",
    "achieving",
    "and",
    "during",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "the",
    "through",
    "to",
    "was",
    "were",
    "while",
    "with",
}
TASK_FOLDING_CONTEXT_TOKENS = {
    "ab",
    "coarse",
    "coarse-grained",
    "dna",
    "initio",
    "molecular",
    "peptide",
    "peptides",
    "protein",
    "proteins",
    "rna",
}
METHOD_STRIP_PREFIXES = {
    "a",
    "an",
    "al",
    "also",
    "are",
    "been",
    "being",
    "et",
    "has",
    "have",
    "had",
    "many",
    "we",
    "the",
    "our",
    "this",
    "that",
    "these",
    "those",
    "its",
    "their",
    "figure",
    "novel",
    "new",
    "paper",
    "papers",
    "table",
    "there",
    "then",
    "they",
    "theorem",
    "study",
    "studies",
    "proposed",
    "where",
}
METHOD_GENERIC_TOKENS = {
    "a",
    "an",
    "al",
    "also",
    "are",
    "been",
    "being",
    "called",
    "et",
    "we",
    "the",
    "our",
    "this",
    "that",
    "these",
    "those",
    "novel",
    "new",
    "proposed",
    "effective",
    "efficient",
    "robust",
    "advanced",
    "automatic",
    "automated",
    "apply",
    "applied",
    "applies",
    "applying",
    "develop",
    "developed",
    "developing",
    "develops",
    "designed",
    "design",
    "designing",
    "designs",
    "employ",
    "employed",
    "employing",
    "employs",
    "execute",
    "executed",
    "executes",
    "executing",
    "extended",
    "following",
    "general",
    "has",
    "have",
    "had",
    "integrated",
    "implement",
    "implemented",
    "implementing",
    "implements",
    "improved",
    "hybrid",
    "adaptive",
    "figure",
    "many",
    "paper",
    "papers",
    "perform",
    "performed",
    "performing",
    "performs",
    "present",
    "presented",
    "presents",
    "propose",
    "proposed",
    "proposes",
    "provide",
    "provided",
    "provides",
    "study",
    "studies",
    "table",
    "then",
    "there",
    "they",
    "theorem",
    "use",
    "used",
    "uses",
    "using",
    "utilize",
    "utilized",
    "utilizes",
    "utilizing",
    "where",
    "method",
    "approach",
    "framework",
    "technique",
    "strategy",
    "pipeline",
    "workflow",
    "protocol",
    "algorithm",
    "mechanism",
    "module",
    "scheme",
    "system",
}
METHOD_SENTENCE_TOKENS = {
    "about",
    "above",
    "across",
    "after",
    "and",
    "as",
    "at",
    "before",
    "between",
    "by",
    "called",
    "during",
    "figure",
    "for",
    "from",
    "has",
    "have",
    "had",
    "in",
    "into",
    "is",
    "it",
    "many",
    "of",
    "on",
    "that",
    "table",
    "than",
    "the",
    "there",
    "then",
    "they",
    "theorem",
    "through",
    "to",
    "under",
    "we",
    "using",
    "via",
    "was",
    "were",
    "while",
    "with",
    "where",
}
METHOD_SIGNAL_TOKENS = {
    "adaptation",
    "algorithm",
    "annealing",
    "attention",
    "autoencoder",
    "boosting",
    "clustering",
    "contrastive",
    "convolutional",
    "correlation",
    "distillation",
    "docking",
    "embedding",
    "extraction",
    "federated",
    "feature",
    "fine-tuning",
    "fusion",
    "graph",
    "hyperparameter",
    "join",
    "learning",
    "logical",
    "metaheuristic",
    "module",
    "multi-scale",
    "neural",
    "normalization",
    "optimization",
    "parsing",
    "pipeline",
    "pre-training",
    "pretraining",
    "pruning",
    "protocol",
    "random",
    "regression",
    "reinforcement",
    "representation",
    "sampling",
    "scheme",
    "screening",
    "selection",
    "self-supervised",
    "sequencing",
    "similarity",
    "sparse",
    "string",
    "strategy",
    "substring",
    "support",
    "technique",
    "temporal",
    "threshold",
    "transfer",
    "vector",
    "workflow",
}
METHOD_HEAD_TERMS = tuple(
    sorted(
        {
            "method",
            "approach",
            "framework",
            "technique",
            "strategy",
            "pipeline",
            "workflow",
            "protocol",
            "algorithm",
            "mechanism",
            "module",
            "scheme",
        },
        key=len,
        reverse=True,
    )
)
METHOD_TAIL_TERMS = tuple(
    sorted(
        {
            "annealing",
            "augmentation",
            "clustering",
            "distillation",
            "docking",
            "embedding",
            "extraction",
            "fine tuning",
            "fine-tuning",
            "fusion",
            "learning",
            "normalization",
            "optimization",
            "parsing",
            "pre training",
            "pre-training",
            "pretraining",
            "reasoning",
            "regression",
            "regularization",
            "sampling",
            "screening",
            "selection",
            "sequencing",
            "tuning",
        },
        key=len,
        reverse=True,
    )
)
METHOD_EXACT_SURFACES = tuple(
    sorted(
        {
            "artificial neural network",
            "autoencoder",
            "catboost",
            "contrastive learning",
            "convolutional neural network",
            "data normalization",
            "decision tree",
            "federated learning",
            "feature extraction",
            "feature selection",
            "few-shot learning",
            "gradient boosting",
            "graph neural network",
            "hyperparameter tuning",
            "knowledge distillation",
            "lightgbm",
            "linear scaling normalization",
            "long short-term memory",
            "long short term memory",
            "neural network",
            "parameter tuning",
            "prompt engineering",
            "random forest",
            "recurrent neural network",
            "reinforcement learning",
            "retrieval augmented generation",
            "retrieval-augmented generation",
            "self-supervised learning",
            "semi-supervised learning",
            "support vector machine",
            "transfer learning",
            "unsupervised learning",
            "variational autoencoder",
            "xgboost",
            "zero-shot learning",
        },
        key=len,
        reverse=True,
    )
)
METHOD_MODEL_BRIDGE_KEYS = {
    "catboost",
    "cnn",
    "gan",
    "graph_neural_network",
    "large_language_model",
    "lightgbm",
    "lstm",
    "rnn",
    "transformer",
    "u_net",
    "xgboost",
}
METHOD_DYNAMIC_EXACT_RE = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in METHOD_EXACT_SURFACES) + r")\b",
    re.IGNORECASE,
)
METHOD_DYNAMIC_TAIL_RE = re.compile(
    r"\b((?:[A-Za-z0-9][A-Za-z0-9+/.-]*\s+){0,5}(?:"
    + "|".join(re.escape(term) for term in METHOD_TAIL_TERMS)
    + r"))\b",
    re.IGNORECASE,
)
METHOD_DYNAMIC_HEAD_RE = re.compile(
    r"\b((?:[A-Za-z0-9][A-Za-z0-9+/.-]*\s+){0,5}(?:"
    + "|".join(re.escape(term) for term in METHOD_HEAD_TERMS)
    + r"))\b",
    re.IGNORECASE,
)
METHOD_CONTEXT_CUE_RE = re.compile(
    r"\b(?:"
    r"use|uses|used|using|"
    r"adopt|adopts|adopted|adopting|"
    r"apply|applies|applied|applying|"
    r"design|designs|designed|designing|"
    r"develop|develops|developed|developing|"
    r"employ|employs|employed|employing|"
    r"fine[- ]?tun(?:e|es|ed|ing)|"
    r"implement|implements|implemented|implementing|"
    r"integrat(?:e|es|ed|ing)|"
    r"introduc(?:e|es|ed|ing)|"
    r"optimiz(?:e|es|ed|ing)|"
    r"pre[- ]?train(?:ed|ing)?|"
    r"propos(?:e|es|ed|ing)|"
    r"train|trains|trained|training|"
    r"utiliz(?:e|es|ed|ing)|"
    r"based on"
    r")\b",
    re.IGNORECASE,
)
METHOD_CONTEXT_BOOSTS = {
    "front_matter": 0.05,
    "abstract": 0.05,
    "introduction": 0.02,
    "methods": 0.08,
    "results": 0.01,
    "discussion": 0.02,
    "conclusion": 0.03,
    "other": 0.01,
}
METHOD_DYNAMIC_ALLOWED_SECTION_GROUPS = {
    "abstract",
    "front_matter",
    "methods",
}
METHOD_DYNAMIC_ALLOWED_FIGURE_SECTION_GROUPS = {
    "abstract",
    "front_matter",
    "methods",
    "results",
}
METHOD_GENERIC_CANONICALS = {
    "artificial neural network",
    "cnn",
    "convolutional neural network",
    "data normalization",
    "deep learning",
    "feature extraction",
    "feature selection",
    "federated learning",
    "graph neural network",
    "hyperparameter tuning",
    "linear scaling normalization",
    "lstm",
    "machine learning",
    "parameter tuning",
    "random forest",
    "self-supervised learning",
    "semi-supervised learning",
    "support vector machine",
    "transfer learning",
    "unsupervised learning",
    "xgboost",
}
METHOD_SECTION_POSITIVE_HINT_RE = re.compile(
    r"\b(?:"
    r"method|methods|approach|approaches|algorithm|algorithms|"
    r"architecture|architectures|implementation|implementations|"
    r"training|preprocessing|pipeline|workflow|protocol|framework|"
    r"experimental|experiment|setup|design|optimization|tuning|model"
    r")\b",
    re.IGNORECASE,
)
METHOD_SECTION_NEGATIVE_HINT_RE = re.compile(
    r"\b(?:"
    r"related|background|literature|review|prior work|previous work|"
    r"references?|bibliograph"
    r")\b",
    re.IGNORECASE,
)
METHOD_CANONICAL_NARRATIVE_RE = re.compile(
    r"\b(?:"
    r"adopt(?:ed|ing|s)?|address(?:ing|es|ed)?|against|alongside|"
    r"assessment|author(?:s)?|because|broader|carries|carry|clinical|"
    r"collect(?:ed|ion)?|complex|conventional|discussion|"
    r"employ(?:ed|ing|s)?|enhanc(?:e|ed|ing|es)|evaluation|"
    r"explor(?:e|ed|ing)|focus|folds|if|including|indicates?|input|"
    r"introduc(?:e|ed|ing|es)|illustrat(?:e|ed|es|ing)|may|might|"
    r"involve(?:d|ing|s)?|often|or|output|paper|perform(?:ed|ing|s)?|present(?:ed|ing|s)?|"
    r"propos(?:e|ed|ing|es)|provid(?:e|ed|ing|es)|radiomic|"
    r"result(?:s)?|review|section|sharing|show(?:ed|ing|s)?|since|study|"
    r"supporting|table|technique|that|therefore|this|those|unlike|"
    r"used|using|utiliz(?:e|ed|ing|es)|validat(?:e|ed|ing|es)|views?|which|while|within"
    r")\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DocumentEntityGraphConfig:
    min_chars: int = 350
    target_chars: int = 700
    max_chars: int = 1100
    section_summary_chars: int = 420
    figure_nearby_chunk_k: int = 2
    enable_figure_entity_edges: bool = True
    semantic_min_confidence: float = 0.72
    reference_title_match_overlap: float = 0.92
    enable_cross_paper_edges: bool = False
    cross_paper_max_entity_group_size: int = 10
    cross_paper_max_reference_group_size: int = 16
    cross_paper_min_bibliographic_shared_refs: int = 2
    cross_paper_min_edge_weight: float = 0.55
    reference_navigation_min_confidence: float = 0.58
    reference_navigation_min_title_recall: float = 0.78
    reference_navigation_min_title_similarity: float = 0.62
    reference_navigation_min_title_tokens: int = 6
    reference_navigation_max_year_delta: int = 1
    reference_navigation_min_margin: float = 0.06


@dataclass(slots=True)
class SectionDraft:
    node: dict[str, Any]
    chunk_ids: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    page_indices: list[int] = field(default_factory=list)


@dataclass(slots=True)
class SemanticMention:
    entity_type: str
    canonical_name: str
    normalized_key: str
    surface_form: str
    confidence: float
    source: str


def build_document_entity_graph_from_segments(
    segments_dir: Path,
    output_dir: Path,
    config: DocumentEntityGraphConfig,
    max_papers: int | None = None,
) -> dict[str, Any]:
    segments_dir = segments_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_dirs = discover_paper_dirs(segments_dir)
    if max_papers is not None:
        paper_dirs = paper_dirs[:max_papers]

    papers: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    structure_edges: list[dict[str, Any]] = []
    paper_citations: list[dict[str, Any]] = []
    skipped_papers: list[dict[str, str]] = []

    for paper_dir in paper_dirs:
        try:
            record = build_document_structure_for_paper(paper_dir, config)
        except Exception as exc:  # pragma: no cover - defensive batch handling
            skipped_papers.append({"paper_id": paper_dir.name, "reason": repr(exc)})
            continue

        papers.append(record["paper"])
        sections.extend(record["sections"])
        chunks.extend(record["chunks"])
        figures.extend(record["figures"])
        structure_edges.extend(record["edges"])
        paper_citations.extend(record["citations"])

    if not papers:
        stats = {
            "segments_dir": str(segments_dir),
            "output_dir": str(output_dir),
            "paper_count": 0,
            "skipped_paper_count": len(skipped_papers),
            "skipped_papers": skipped_papers,
        }
        (output_dir / "graph_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return stats

    paper_by_id = {paper["paper_id"]: paper for paper in papers}
    section_by_id = {section["section_id"]: section for section in sections}
    chunks_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    figures_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_section[chunk["section_id"]].append(chunk)
    for figure in figures:
        figures_by_section[figure["section_id"]].append(figure)

    author_nodes, author_edges = build_author_entities(papers)
    institution_nodes, institution_edges = build_institution_entities(papers)
    venue_nodes, venue_edges = build_venue_entities(papers)
    reference_nodes, reference_edges = build_reference_entities(paper_citations)
    reference_resolution_edges = resolve_reference_to_paper_edges(reference_nodes, papers, config)

    semantic_result = build_semantic_entities_and_edges(
        papers=papers,
        sections=sections,
        figures=figures,
        chunks_by_section=chunks_by_section,
        figures_by_section=figures_by_section,
        config=config,
    )

    cross_paper_edges: list[dict[str, Any]] = []
    paper_pair_summaries: list[dict[str, Any]] = []
    if config.enable_cross_paper_edges:
        cross_paper_edges, paper_pair_summaries = build_cross_paper_edges(
            papers=papers,
            author_edges=author_edges,
            institution_edges=institution_edges,
            venue_edges=venue_edges,
            reference_edges=reference_edges,
            reference_resolution_edges=reference_resolution_edges,
            semantic_edges=semantic_result["edges"],
            config=config,
        )

    all_edges = (
        structure_edges
        + author_edges
        + institution_edges
        + venue_edges
        + reference_edges
        + reference_resolution_edges
        + semantic_result["edges"]
        + cross_paper_edges
    )
    assign_edge_ids(all_edges)

    all_nodes = (
        papers
        + sections
        + chunks
        + figures
        + author_nodes
        + institution_nodes
        + venue_nodes
        + reference_nodes
        + semantic_result["method_nodes"]
        + semantic_result["dataset_nodes"]
        + semantic_result["task_nodes"]
        + semantic_result["metric_nodes"]
        + semantic_result["model_nodes"]
    )

    write_outputs(
        output_dir=output_dir,
        papers=papers,
        sections=sections,
        chunks=chunks,
        figures=figures,
        author_nodes=author_nodes,
        institution_nodes=institution_nodes,
        venue_nodes=venue_nodes,
        reference_nodes=reference_nodes,
        semantic_result=semantic_result,
        all_nodes=all_nodes,
        all_edges=all_edges,
        cross_paper_edges=cross_paper_edges,
        paper_pair_summaries=paper_pair_summaries,
    )

    node_type_counts = dict(Counter(node["node_type"] for node in all_nodes))
    edge_type_counts = dict(Counter(edge["edge_type"] for edge in all_edges))
    stats = {
        "segments_dir": str(segments_dir),
        "output_dir": str(output_dir),
        "paper_count": len(papers),
        "section_count": len(sections),
        "chunk_count": len(chunks),
        "figure_count": len(figures),
        "author_count": len(author_nodes),
        "institution_count": len(institution_nodes),
        "venue_count": len(venue_nodes),
        "reference_count": len(reference_nodes),
        "method_count": len(semantic_result["method_nodes"]),
        "dataset_count": len(semantic_result["dataset_nodes"]),
        "task_count": len(semantic_result["task_nodes"]),
        "metric_count": len(semantic_result["metric_nodes"]),
        "model_count": len(semantic_result["model_nodes"]),
        "edge_count": len(all_edges),
        "cross_paper_edge_count": len(cross_paper_edges),
        "cross_paper_pair_count": len(paper_pair_summaries),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
        "cross_paper_edge_type_counts": dict(
            Counter(edge["edge_type"] for edge in cross_paper_edges)
        ),
        "skipped_paper_count": len(skipped_papers),
        "skipped_papers": skipped_papers,
        "files": {
            "papers": str(output_dir / "papers.jsonl"),
            "sections": str(output_dir / "sections.jsonl"),
            "chunks": str(output_dir / "chunks.jsonl"),
            "figures": str(output_dir / "figures.jsonl"),
            "authors": str(output_dir / "authors.jsonl"),
            "institutions": str(output_dir / "institutions.jsonl"),
            "venues": str(output_dir / "venues.jsonl"),
            "references": str(output_dir / "references.jsonl"),
            "methods": str(output_dir / "methods.jsonl"),
            "datasets": str(output_dir / "datasets.jsonl"),
            "tasks": str(output_dir / "tasks.jsonl"),
            "metrics": str(output_dir / "metrics.jsonl"),
            "models": str(output_dir / "models.jsonl"),
            "nodes": str(output_dir / "nodes.jsonl"),
            "edges": str(output_dir / "edges.jsonl"),
            "paper_paper_edges": str(output_dir / "paper_paper_edges.jsonl"),
            "paper_pair_summaries": str(output_dir / "paper_pair_summaries.jsonl"),
        },
    }
    (output_dir / "graph_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def build_document_structure_for_paper(
    paper_dir: Path,
    config: DocumentEntityGraphConfig,
) -> dict[str, Any]:
    items, parsed_root, content_source_path = load_segment_items(paper_dir)
    paper_id = paper_dir.name
    paper_metadata = extract_paper_metadata(items, paper_id)
    # Published source metadata corrects parser errors such as merged author names.
    source_metadata_path = paper_dir / "source_metadata.json"
    source_metadata = None
    if source_metadata_path.exists():
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        paper_metadata["paper_title"] = source_metadata["title"]
        paper_metadata["paper_authors"] = source_metadata["authors"]
        paper_metadata["paper_author_text"] = "; ".join(source_metadata["authors"])
        paper_metadata["metadata_sources"] = list(paper_metadata.get("metadata_sources", [])) + ["source_metadata"]
    paper_authors = repair_author_list(
        paper_metadata.get("paper_author_text") or "",
        paper_metadata.get("paper_authors") or [],
    )
    paper_institutions = repair_institution_list(
        paper_metadata.get("paper_institutions") or [],
    )
    paper_metadata["paper_authors"] = paper_authors
    paper_metadata["paper_institutions"] = paper_institutions
    paper_metadata["paper_institution_text"] = "; ".join(paper_institutions)
    title_index = find_title_index_local(items, paper_metadata["paper_title"])
    keywords = extract_keywords(items, title_index)
    venue_text = normalize_text(
        extract_venue_text(items, title_index, paper_metadata["paper_author_text"])
    )
    paper_doi = normalize_doi(extract_paper_doi(items) or "")
    paper_pmid = normalize_pmid(extract_paper_pmid(items) or "")
    paper_year = normalize_year_text(extract_paper_year(items, paper_id))

    if source_metadata:
        paper_year = str(source_metadata["year"])

    sections: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []

    section_drafts: dict[str, SectionDraft] = {}
    section_stack: list[str] = []
    buffer = ChunkBuffer()
    current_section_id: str | None = None
    current_section_path: list[str] = [DEFAULT_SECTION_NAME]

    chunk_counter = 0
    section_counter = 0
    figure_counter = 0

    def ensure_section(section_path: list[str]) -> dict[str, Any]:
        nonlocal current_section_id, current_section_path, section_counter
        normalized_path = list(section_path) or [DEFAULT_SECTION_NAME]
        if current_section_id and current_section_path == normalized_path:
            return section_drafts[current_section_id].node

        section_counter += 1
        section_id = f"{paper_id}:{section_counter:03d}"
        section_node = {
            "node_id": f"section:{section_id}",
            "node_type": "section",
            "section_id": section_id,
            "paper_id": paper_id,
            "paper_node_id": f"paper:{paper_id}",
            "section_title": normalized_path[-1],
            "section_level": len(normalized_path),
            "section_path": normalized_path,
            "section_group": classify_section_path(normalized_path),
            "order": section_counter,
            "page_start": None,
            "page_end": None,
            "section_text": "",
            "section_summary": "",
            "search_text": "",
        }
        section_drafts[section_id] = SectionDraft(node=section_node)
        sections.append(section_node)
        current_section_id = section_id
        current_section_path = normalized_path
        return section_node

    def flush_buffer() -> None:
        nonlocal chunk_counter
        if not buffer.paragraphs:
            return

        section_node = ensure_section(buffer.section_path)
        chunk_counter += 1
        page_start, page_end = page_range(buffer.page_indices)
        chunk_id = f"{paper_id}:{chunk_counter:04d}"
        chunk_text = buffer.text
        chunk_node = {
            "node_id": f"chunk:{chunk_id}",
            "node_type": "chunk",
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "paper_node_id": f"paper:{paper_id}",
            "section_id": section_node["section_id"],
            "section_node_id": section_node["node_id"],
            "order": chunk_counter,
            "page_start": page_start,
            "page_end": page_end,
            "page_span": page_span(page_start, page_end),
            "page_indices": [page for page in buffer.page_indices if page is not None],
            "bbox": merge_bboxes(buffer.bboxes),
            "text": chunk_text,
            "token_count": len(tokenize_for_similarity(chunk_text)),
            "source_item_indices": list(buffer.source_item_indices),
            "source_item_types": list(buffer.source_types),
            "search_text": build_node_search_text(
                join_unique_nonempty(
                    [
                        f"title {paper_metadata['paper_title']}",
                        f"authors {paper_metadata['paper_author_text']}" if paper_metadata["paper_author_text"] else "",
                        f"institutions {paper_metadata['paper_institution_text']}" if paper_metadata["paper_institution_text"] else "",
                        f"section {section_node['section_title']}",
                    ]
                ),
                chunk_text,
            ),
        }
        chunks.append(chunk_node)
        draft = section_drafts[section_node["section_id"]]
        draft.chunk_ids.append(chunk_node["chunk_id"])
        draft.texts.append(chunk_text)
        draft.page_indices.extend([page for page in buffer.page_indices if page is not None])
        buffer.clear()

    for item_index, item in enumerate(items):
        item_type = str(item.get("type") or "").strip().lower()
        page_idx = to_int(next((item[k] for k in ("page_idx", "page_no", "page") if item.get(k) is not None), None))
        bbox = normalize_bbox(item.get("bbox"))
        raw_text = extract_item_text(item)

        if raw_text and is_section_heading(item, raw_text):
            flush_buffer()
            update_section_stack(section_stack, raw_text, to_int(item.get("text_level")))
            current_section_id = None
            current_section_path = list(section_stack) or [DEFAULT_SECTION_NAME]
            continue

        if item_type in IMAGE_ITEM_TYPES or item.get("img_path"):
            section_node = ensure_section(list(section_stack) or [DEFAULT_SECTION_NAME])
            figure_counter += 1
            asset_kind = item_type if item_type in IMAGE_ITEM_TYPES else "image"
            image_path = resolve_image_path(str(item.get("img_path") or ""), parsed_root)
            caption = extract_caption_list(item)
            ocr_text = build_image_feature_text(item, max_chars=2000)
            figure_id = f"{paper_id}:{asset_kind}:{figure_counter:04d}"
            figure_node = {
                "node_id": f"figure:{figure_id}",
                "node_type": "figure",
                "figure_id": figure_id,
                "paper_id": paper_id,
                "paper_node_id": f"paper:{paper_id}",
                "section_id": section_node["section_id"],
                "section_node_id": section_node["node_id"],
                "asset_kind": asset_kind,
                "order": figure_counter,
                "caption": caption,
                "footnote": extract_footnote_list(item),
                "ocr_text": ocr_text,
                "image_path": str(image_path) if image_path else "",
                "page_start": page_idx,
                "page_end": page_idx,
                "page_span": page_span(page_idx, page_idx),
                "page_indices": [page_idx] if page_idx is not None else [],
                "bbox": bbox,
                "reference_labels": extract_reference_labels_from_item(item),
                "search_text": build_node_search_text(
                    join_unique_nonempty(
                        [
                            f"title {paper_metadata['paper_title']}",
                            f"section {section_node['section_title']}",
                        ]
                    ),
                    join_unique_nonempty(caption + [ocr_text]),
                ),
            }
            figures.append(figure_node)
            draft = section_drafts[section_node["section_id"]]
            draft.figure_ids.append(figure_node["figure_id"])
            if page_idx is not None:
                draft.page_indices.append(page_idx)
            continue

        if item_type not in TEXT_ITEM_TYPES:
            continue
        if is_reference_like_item(item, section_stack):
            continue
        if not raw_text:
            continue

        paragraphs = split_paragraphs(raw_text)
        for paragraph in paragraphs:
            paragraph = normalize_text(paragraph)
            if not paragraph or is_low_signal_text(paragraph):
                continue
            for piece in split_long_text(paragraph, config.max_chars):
                append_text_piece(
                    buffer=buffer,
                    section_stack=section_stack,
                    page_idx=page_idx,
                    item_index=item_index,
                    item_type=item_type,
                    bbox=bbox,
                    piece=piece,
                    config=_as_chunk_config(config),
                    flush_buffer=flush_buffer,
                )

    flush_buffer()

    reassign_floating_figures(figures, chunks, sections, paper_metadata["paper_title"])
    for draft in section_drafts.values():
        draft.figure_ids = []
        draft.page_indices = []
    for chunk in chunks:
        section_drafts[chunk["section_id"]].page_indices.extend(chunk["page_indices"])
    for figure in figures:
        draft = section_drafts[figure["section_id"]]
        draft.figure_ids.append(figure["figure_id"])
        draft.page_indices.extend(figure["page_indices"])

    for section_node in sections:
        draft = section_drafts[section_node["section_id"]]
        page_start, page_end = page_range(draft.page_indices)
        section_text = "\n\n".join(draft.texts).strip()
        section_summary = preview_text(section_text, config.section_summary_chars)
        section_node["page_start"] = page_start
        section_node["page_end"] = page_end
        section_node["section_text"] = section_text
        section_node["section_summary"] = section_summary
        section_node["search_text"] = build_node_search_text(
            join_unique_nonempty(
                [
                    f"title {paper_metadata['paper_title']}",
                    f"authors {paper_metadata['paper_author_text']}" if paper_metadata["paper_author_text"] else "",
                    f"institutions {paper_metadata['paper_institution_text']}" if paper_metadata["paper_institution_text"] else "",
                    f"section {section_node['section_title']}",
                ]
            ),
            join_unique_nonempty([section_summary, section_text]),
        )

    section_lookup = {section["section_id"]: section for section in sections}
    figure_lookup = {figure["figure_id"]: figure for figure in figures}
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

    abstract = extract_abstract_from_sections(sections)
    paper_profile_text = build_paper_profile_text_v1(
        title=paper_metadata["paper_title"],
        authors_text=paper_metadata["paper_author_text"],
        institution_text=paper_metadata["paper_institution_text"],
        venue_text=venue_text,
        abstract=abstract,
        keywords=keywords,
    )
    paper_node = {
        "node_id": f"paper:{paper_id}",
        "node_type": "paper",
        "paper_id": paper_id,
        "title": paper_metadata["paper_title"],
        "abstract": abstract,
        "year": paper_year,
        "doi": paper_doi,
        "pmid": paper_pmid,
        "venue_text": venue_text,
        "author_text": paper_metadata["paper_author_text"],
        "authors": paper_metadata["paper_authors"],
        "institution_text": paper_metadata["paper_institution_text"],
        "institutions": paper_metadata["paper_institutions"],
        "keywords": keywords,
        "metadata_sources": paper_metadata.get("metadata_sources") or ["mineru"],
        "paper_profile_text": paper_profile_text,
        "search_text": join_unique_nonempty(
            [
                paper_metadata["paper_title"],
                f"abstract {abstract}" if abstract else "",
                f"authors {paper_metadata['paper_author_text']}" if paper_metadata["paper_author_text"] else "",
                f"institutions {paper_metadata['paper_institution_text']}" if paper_metadata["paper_institution_text"] else "",
                f"venue {venue_text}" if venue_text else "",
                f"keywords {'; '.join(keywords)}" if keywords else "",
            ]
        ),
        "content_list_path": str(content_source_path) if content_source_path else "",
    }

    edges.extend(build_structure_edges(paper_node, sections, chunks, figures))
    edges.extend(
        build_figure_description_edges(
            figures=figures,
            chunks=chunks,
            section_lookup=section_lookup,
            chunk_lookup=chunk_lookup,
            config=config,
        )
    )

    citation_nodes = build_citation_nodes_for_paper(
        paper_dir=paper_dir,
        items=items,
        paper_metadata={
            "paper_title": paper_metadata["paper_title"],
            "paper_authors": paper_metadata["paper_authors"],
            "paper_author_text": paper_metadata["paper_author_text"],
            "paper_institutions": paper_metadata["paper_institutions"],
            "paper_institution_text": paper_metadata["paper_institution_text"],
            "paper_profile_text": paper_profile_text,
        },
    )
    for citation_node in citation_nodes:
        citations.append(
            {
                "paper_id": paper_id,
                "paper_node_id": paper_node["node_id"],
                "raw_citation_text": citation_node.get("citation_raw_text") or citation_node.get("text") or "",
                "page": citation_node.get("page_start"),
                "citation_title": citation_node.get("citation_title") or "",
                "citation_author_text": citation_node.get("citation_author_text") or "",
                "citation_year": citation_node.get("citation_year") or "",
                "citation_doi": citation_node.get("citation_doi") or "",
                "citation_pmid": citation_node.get("citation_pmid") or "",
                "citation_venue": citation_node.get("citation_venue") or "",
                "citation_title_key": citation_node.get("citation_title_key") or "",
                "citation_match_key": citation_node.get("citation_match_key") or "",
            }
        )

    return {
        "paper": paper_node,
        "sections": sections,
        "chunks": chunks,
        "figures": figures,
        "edges": edges,
        "citations": citations,
    }


def reassign_floating_figures(figures, chunks, sections, paper_title):
    """Resolve a floating asset to its nearest explicit discussion, within one page."""
    section_lookup = {s["section_id"]: s for s in sections}
    for figure in figures:
        references = explicit_reference_chunk_ids(figure, chunks)
        if not references or any(c["chunk_id"] in references and c["section_id"] == figure["section_id"] for c in chunks):
            continue
        page = figure.get("page_start")
        if page is None:
            continue
        nearby = [c for c in chunks if c["chunk_id"] in references and c.get("page_start") is not None
                  and abs(c["page_start"] - page) <= 1]
        if not nearby:
            continue
        owner = min(nearby, key=lambda c: (abs(c["page_start"] - page), c["order"]))
        section = section_lookup[owner["section_id"]]
        figure["section_assignment"] = {"method": "nearby_explicit_reference",
            "previous_section_id": figure["section_id"], "evidence_chunk_id": owner["node_id"]}
        figure["section_id"], figure["section_node_id"] = section["section_id"], section["node_id"]
        figure["search_text"] = build_node_search_text(
            f"title {paper_title}\nsection {section['section_title']}",
            join_unique_nonempty(list(figure.get("caption") or []) + [figure.get("ocr_text") or ""]))


def build_structure_edges(
    paper_node: dict[str, Any],
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    sections_by_paper = sorted(sections, key=lambda row: row["order"])
    chunks_by_paper = sorted(chunks, key=lambda row: row["order"])

    for section in sections_by_paper:
        edges.append(
            new_edge(
                edge_type="paper_has_section",
                source_id=paper_node["node_id"],
                target_id=section["node_id"],
                confidence=1.0,
                source="structure",
            )
        )
    for left, right in zip(sections_by_paper, sections_by_paper[1:]):
        edges.append(new_edge("section_next", left["node_id"], right["node_id"], 1.0, "structure"))

    for left, right in zip(chunks_by_paper, chunks_by_paper[1:]):
        edges.append(new_edge("chunk_next", left["node_id"], right["node_id"], 1.0, "structure"))

    for chunk in chunks:
        edges.append(
            new_edge(
                edge_type="section_has_chunk",
                source_id=chunk["section_node_id"],
                target_id=chunk["node_id"],
                confidence=1.0,
                source="structure",
            )
        )
    for figure in figures:
        edges.append(
            new_edge(
                edge_type="section_has_figure",
                source_id=figure["section_node_id"],
                target_id=figure["node_id"],
                confidence=1.0,
                source="structure",
            )
        )
    return edges


def build_figure_description_edges(
    figures: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    section_lookup: dict[str, dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    config: DocumentEntityGraphConfig,
) -> list[dict[str, Any]]:
    chunks_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_section[chunk["section_id"]].append(chunk)

    edges: list[dict[str, Any]] = []
    for figure in figures:
        section_chunks = chunks_by_section.get(figure["section_id"], [])
        descriptive_chunks = find_descriptive_chunks(figure, section_chunks, config.figure_nearby_chunk_k)
        for rank, chunk in enumerate(descriptive_chunks, start=1):
            confidence = 0.96 if chunk["chunk_id"] in explicit_reference_chunk_ids(figure, section_chunks) else max(0.72, 0.90 - ((rank - 1) * 0.08))
            edges.append(
                new_edge(
                    edge_type="figure_described_by_chunk",
                    source_id=figure["node_id"],
                    target_id=chunk["node_id"],
                    confidence=confidence,
                    source="reference_label" if confidence >= 0.95 else "nearby_text",
                    evidence_chunk_ids=[chunk["chunk_id"]],
                    evidence_figure_ids=[figure["figure_id"]],
                )
            )
    return edges


def explicit_reference_chunk_ids(figure: dict[str, Any], chunks: list[dict[str, Any]]) -> set[str]:
    labels = [normalize_text(label).lower() for label in (figure.get("reference_labels") or []) if normalize_text(label)]
    if not labels:
        return set()
    matches: set[str] = set()
    for chunk in chunks:
        text = normalize_text(chunk.get("text") or "").lower()
        if any(re.search(REF_LABEL_PATTERN_TEMPLATE % re.escape(label), text) for label in labels):
            matches.add(chunk["chunk_id"])
    return matches


def find_descriptive_chunks(
    figure: dict[str, Any],
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    explicit_ids = explicit_reference_chunk_ids(figure, chunks)
    explicit_chunks = [chunk for chunk in chunks if chunk["chunk_id"] in explicit_ids]
    if explicit_chunks:
        return explicit_chunks[:top_k]

    figure_page = figure.get("page_start")
    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: (
            abs((chunk.get("page_start") or 0) - (figure_page or 0)),
            abs(int(chunk.get("order") or 0) - int(figure.get("order") or 0)),
        ),
    )
    return sorted_chunks[:top_k]


def build_author_entities(papers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for paper in papers:
        for author_order, author in enumerate(paper.get("authors") or [], start=1):
            key = normalize_author_key(author)
            if not key:
                continue
            canonical_name = choose_author_canonical_name(author)
            node = registry.setdefault(
                key,
                {
                    "node_id": f"author:{key}",
                    "node_type": "author",
                    "author_id": key,
                    "canonical_name": canonical_name,
                    "aliases": set(),
                    "normalized_key": key,
                    "orcid": None,
                    "search_text": "",
                },
            )
            node["aliases"].add(normalize_text(author))
            edges.append(
                new_edge(
                    edge_type="paper_has_author",
                    source_id=paper["node_id"],
                    target_id=node["node_id"],
                    confidence=1.0,
                    source="metadata",
                    surface_forms=[author],
                    author_order=author_order,
                )
            )

    nodes = finalize_alias_nodes(registry, alias_field="aliases")
    return nodes, edges


def build_institution_entities(papers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for paper in papers:
        for institution in paper.get("institutions") or []:
            canonical_name = canonicalize_institution_name(institution)
            key = normalize_entity_key(canonical_name)
            if not key:
                continue
            node = registry.setdefault(
                key,
                {
                    "node_id": f"institution:{key}",
                    "node_type": "institution",
                    "institution_id": key,
                    "canonical_name": canonical_name,
                    "aliases": set(),
                    "normalized_key": key,
                    "country": extract_country_from_affiliation(institution),
                    "search_text": "",
                },
            )
            if not node.get("country"):
                node["country"] = extract_country_from_affiliation(institution)
            node["aliases"].add(normalize_text(institution))
            edges.append(
                new_edge(
                    edge_type="paper_has_institution",
                    source_id=paper["node_id"],
                    target_id=node["node_id"],
                    confidence=0.98,
                    source="metadata",
                    surface_forms=[institution],
                    raw_affiliation_text=institution,
                )
            )

    nodes = finalize_alias_nodes(registry, alias_field="aliases")
    return nodes, edges


def build_venue_entities(papers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for paper in papers:
        venue_text = normalize_text(paper.get("venue_text") or "")
        if not venue_text or not is_valid_venue_text(venue_text):
            continue
        key = normalize_entity_key(venue_text)
        if not key:
            continue
        node = registry.setdefault(
            key,
            {
                "node_id": f"venue:{key}",
                "node_type": "venue",
                "venue_id": key,
                "canonical_name": venue_text,
                "aliases": set(),
                "normalized_key": key,
                "issn": None,
                "venue_type": None,
                "search_text": "",
            },
        )
        node["aliases"].add(venue_text)
        edges.append(
            new_edge(
                edge_type="paper_published_in",
                source_id=paper["node_id"],
                target_id=node["node_id"],
                confidence=0.92,
                source="metadata",
                surface_forms=[venue_text],
            )
        )

    nodes = finalize_alias_nodes(registry, alias_field="aliases")
    return nodes, edges


def build_reference_entities(paper_citations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry: dict[str, dict[str, Any]] = {}
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}

    for citation in paper_citations:
        reference_key = citation.get("citation_match_key") or build_citation_match_key(
            doi=citation.get("citation_doi") or "",
            lead_author=first_author_key(citation.get("citation_author_text") or ""),
            year=citation.get("citation_year") or "",
            title_key=build_title_key(citation.get("citation_title") or citation.get("raw_citation_text") or ""),
        )
        if not reference_key:
            continue

        registry_node = registry.setdefault(
            reference_key,
            {
                "node_id": f"reference:{normalize_entity_key(reference_key)}",
                "node_type": "reference",
                "reference_id": reference_key,
                "title": citation.get("citation_title") or "",
                "authors_text": citation.get("citation_author_text") or "",
                "year": citation.get("citation_year") or "",
                "venue_text": citation.get("citation_venue") or "",
                "doi": citation.get("citation_doi") or "",
                "pmid": citation.get("citation_pmid") or "",
                "normalized_signature": reference_key,
                "raw_variants": set(),
                "aliases": set(),
                "search_text": "",
            },
        )
        raw_text = normalize_text(citation.get("raw_citation_text") or "")
        if raw_text:
            registry_node["raw_variants"].add(raw_text)
        if citation.get("citation_title"):
            registry_node["aliases"].add(normalize_text(citation["citation_title"]))

        edge_key = (citation["paper_node_id"], registry_node["node_id"])
        edge = edge_map.setdefault(
            edge_key,
            new_edge(
                edge_type="paper_cites_reference",
                source_id=citation["paper_node_id"],
                target_id=registry_node["node_id"],
                confidence=0.94 if citation.get("citation_doi") else 0.84,
                source="bibliography",
                surface_forms=[citation.get("citation_title") or citation.get("raw_citation_text") or ""],
                mention_count=0,
                raw_citation_texts=[],
                page=citation.get("page"),
            ),
        )
        edge["mention_count"] = int(edge.get("mention_count") or 0) + 1
        if raw_text:
            edge.setdefault("raw_citation_texts", [])
            if raw_text not in edge["raw_citation_texts"]:
                edge["raw_citation_texts"].append(raw_text)

    provisional_reference_nodes = []
    for node in registry.values():
        raw_variants = sorted(node.pop("raw_variants"))
        aliases = sorted(node.pop("aliases"))
        hydrated = {
            **node,
            "raw_variants": raw_variants,
            "aliases": aliases,
        }
        inferred_title = best_reference_display_name(hydrated) or infer_reference_title(hydrated)
        if inferred_title:
            hydrated["title"] = inferred_title
        if hydrated.get("title") and normalize_text(hydrated["title"]) not in {
            normalize_text(value) for value in aliases
        }:
            aliases = [normalize_text(hydrated["title"]), *aliases]
        hydrated["aliases"] = unique_preserve_order([normalize_text(value) for value in aliases if normalize_text(value)])
        hydrated["search_text"] = build_reference_search_text(hydrated)
        provisional_reference_nodes.append(hydrated)

    reference_nodes, reference_id_map = consolidate_reference_nodes(provisional_reference_nodes)
    edges = merge_reference_citation_edges(list(edge_map.values()), reference_id_map)
    return reference_nodes, edges


def consolidate_reference_nodes(
    reference_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if len(reference_nodes) < 2:
        node_map = {str(node.get("node_id") or ""): str(node.get("node_id") or "") for node in reference_nodes}
        return sorted(reference_nodes, key=lambda row: str(row.get("node_id") or "")), node_map

    reference_records = {
        str(node["node_id"]): build_reference_merge_record(node)
        for node in reference_nodes
    }
    parent = {node_id: node_id for node_id in reference_records}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            return
        left_score = reference_records[left_root]["anchor_score"]
        right_score = reference_records[right_root]["anchor_score"]
        if right_score > left_score or (
            right_score == left_score and right_root < left_root
        ):
            parent[left_root] = right_root
        else:
            parent[right_root] = left_root

    for field_name in ("doi", "pmid"):
        grouped_ids: dict[str, list[str]] = defaultdict(list)
        for node_id, record in reference_records.items():
            value = str(record.get(field_name) or "")
            if value:
                grouped_ids[value].append(node_id)
        for grouped_node_ids in grouped_ids.values():
            if len(grouped_node_ids) < 2:
                continue
            anchor = grouped_node_ids[0]
            for node_id in grouped_node_ids[1:]:
                union(anchor, node_id)

    title_key_groups: dict[str, list[str]] = defaultdict(list)
    for node_id, record in reference_records.items():
        for title_key in record.get("candidate_title_keys") or []:
            title_key_groups[str(title_key)].append(node_id)
    for title_key, grouped_node_ids in title_key_groups.items():
        unique_ids = sorted(set(grouped_node_ids))
        if len(unique_ids) < 2 or len(unique_ids) > 64:
            continue
        for left_index in range(len(unique_ids)):
            left_id = unique_ids[left_index]
            left_record = reference_records[left_id]
            for right_id in unique_ids[left_index + 1 :]:
                right_record = reference_records[right_id]
                if should_merge_reference_records(left_record, right_record, shared_title_key=title_key):
                    union(left_id, right_id)

    clusters: dict[str, list[str]] = defaultdict(list)
    for node_id in reference_records:
        clusters[find(node_id)].append(node_id)

    merged_nodes: list[dict[str, Any]] = []
    reference_id_map: dict[str, str] = {}
    for cluster_node_ids in clusters.values():
        member_records = [reference_records[node_id] for node_id in sorted(cluster_node_ids)]
        anchor_record = max(
            member_records,
            key=lambda row: (
                float(row.get("anchor_score") or 0.0),
                len(str((row.get("node") or {}).get("title") or "")),
                str((row.get("node") or {}).get("node_id") or ""),
            ),
        )
        merged_node = merge_reference_cluster_nodes(member_records, anchor_record)
        merged_nodes.append(merged_node)
        for record in member_records:
            reference_id_map[str(record["node"]["node_id"])] = str(merged_node["node_id"])

    merged_nodes.sort(key=lambda row: str(row.get("node_id") or ""))
    return merged_nodes, reference_id_map


def build_reference_merge_record(reference: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(reference.get("title") or best_reference_display_name(reference) or "")
    if not title:
        title = infer_reference_title(reference)
    title_tokens = set(tokenize_for_similarity(title))
    candidate_title_keys = collect_reference_candidate_title_keys(reference, title=title)
    anchor_score = (
        (8.0 if normalize_doi(reference.get("doi") or "") else 0.0)
        + (6.0 if normalize_pmid(reference.get("pmid") or "") else 0.0)
        + min(4.0, float(len(title_tokens)))
        + min(2.0, float(len(reference.get("aliases") or [])) * 0.25)
        + min(2.0, float(len(reference.get("raw_variants") or [])) * 0.15)
    )
    return {
        "node": reference,
        "doi": normalize_doi(reference.get("doi") or ""),
        "pmid": normalize_pmid(reference.get("pmid") or ""),
        "year": normalize_year_text(reference.get("year")),
        "title": title,
        "title_match_text": normalize_resolution_text(title),
        "title_tokens": title_tokens,
        "first_author_key": first_author_key(reference.get("authors_text") or ""),
        "candidate_title_keys": candidate_title_keys,
        "anchor_score": anchor_score,
    }


def collect_reference_candidate_title_keys(reference: dict[str, Any], *, title: str = "") -> list[str]:
    values = [title or normalize_text(reference.get("title") or "")]
    values.extend(normalize_text(value) for value in (reference.get("aliases") or []) if normalize_text(value))
    if not any(values):
        values.extend(
            normalize_text(value)
            for value in (reference.get("raw_variants") or [])
            if normalize_text(value)
        )
    keys: list[str] = []
    for value in values:
        title_key = build_title_key(value)
        if is_informative_reference_title_key(title_key):
            keys.append(title_key)
    return unique_preserve_order(keys)


def is_informative_reference_title_key(title_key: str) -> bool:
    tokens = [token for token in str(title_key or "").split("-") if token]
    return len(tokens) >= 4


def should_merge_reference_records(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    shared_title_key: str,
) -> bool:
    if left.get("doi") and left.get("doi") == right.get("doi"):
        return True
    if left.get("pmid") and left.get("pmid") == right.get("pmid"):
        return True
    if not shared_title_key:
        return False

    shared_token_count = len([token for token in shared_title_key.split("-") if token])
    left_title = str(left.get("title_match_text") or "")
    right_title = str(right.get("title_match_text") or "")
    title_exact = bool(left_title and right_title and left_title == right_title)
    title_similarity = token_f1_overlap(
        set(left.get("title_tokens") or []),
        set(right.get("title_tokens") or []),
    )
    title_recall = token_set_recall(
        set(left.get("title_tokens") or []),
        set(right.get("title_tokens") or []),
    )
    reverse_recall = token_set_recall(
        set(right.get("title_tokens") or []),
        set(left.get("title_tokens") or []),
    )
    first_author_match = bool(
        left.get("first_author_key")
        and right.get("first_author_key")
        and left.get("first_author_key") == right.get("first_author_key")
    )
    year_delta = compare_year_text(str(left.get("year") or ""), str(right.get("year") or ""))
    year_close = year_delta is None or abs(int(year_delta)) <= 1

    if title_exact and (year_close or first_author_match or shared_token_count >= 8):
        return True
    if shared_token_count >= 8 and year_close and (first_author_match or title_similarity >= 0.94):
        return True
    if shared_token_count >= 10 and max(title_recall, reverse_recall) >= 0.95:
        return True
    if shared_token_count >= 6 and year_close and first_author_match and title_similarity >= 0.88:
        return True
    return False


def merge_reference_cluster_nodes(
    member_records: list[dict[str, Any]],
    anchor_record: dict[str, Any],
) -> dict[str, Any]:
    anchor_node = dict(anchor_record["node"])
    merged_aliases: list[str] = []
    merged_raw_variants: list[str] = []
    merged_titles: list[str] = []
    merged_reference_ids: list[str] = []

    best_doi = ""
    best_pmid = ""
    best_year = ""
    best_authors = ""
    best_venue = ""
    best_signature = ""

    for record in member_records:
        node = record["node"]
        merged_reference_ids.append(str(node.get("node_id") or ""))
        merged_titles.append(normalize_text(record.get("title") or node.get("title") or ""))
        merged_aliases.extend(normalize_text(value) for value in (node.get("aliases") or []) if normalize_text(value))
        merged_raw_variants.extend(normalize_text(value) for value in (node.get("raw_variants") or []) if normalize_text(value))
        if not best_doi and normalize_doi(node.get("doi") or ""):
            best_doi = normalize_doi(node.get("doi") or "")
        if not best_pmid and normalize_pmid(node.get("pmid") or ""):
            best_pmid = normalize_pmid(node.get("pmid") or "")
        if not best_year and normalize_year_text(node.get("year")):
            best_year = normalize_year_text(node.get("year"))
        if not best_authors and normalize_text(node.get("authors_text") or ""):
            best_authors = normalize_text(node.get("authors_text") or "")
        if not best_venue and normalize_text(node.get("venue_text") or ""):
            best_venue = normalize_text(node.get("venue_text") or "")
        if not best_signature and normalize_text(node.get("normalized_signature") or ""):
            best_signature = normalize_text(node.get("normalized_signature") or "")

    anchor_node["doi"] = best_doi or normalize_doi(anchor_node.get("doi") or "")
    anchor_node["pmid"] = best_pmid or normalize_pmid(anchor_node.get("pmid") or "")
    anchor_node["year"] = best_year or normalize_year_text(anchor_node.get("year"))
    anchor_node["authors_text"] = best_authors or normalize_text(anchor_node.get("authors_text") or "")
    anchor_node["venue_text"] = best_venue or normalize_text(anchor_node.get("venue_text") or "")
    anchor_node["normalized_signature"] = best_signature or normalize_text(anchor_node.get("normalized_signature") or "")
    anchor_node["raw_variants"] = unique_preserve_order([value for value in merged_raw_variants if value])
    anchor_node["aliases"] = unique_preserve_order([value for value in merged_aliases if value])
    anchor_node["merged_reference_node_ids"] = sorted(set(merged_reference_ids))
    anchor_node["merged_reference_count"] = len(anchor_node["merged_reference_node_ids"])

    title_record = {
        **anchor_node,
        "aliases": unique_preserve_order([*merged_titles, *(anchor_node.get("aliases") or [])]),
        "raw_variants": anchor_node.get("raw_variants") or [],
    }
    merged_title = best_reference_display_name(title_record) or infer_reference_title(title_record)
    if merged_title:
        anchor_node["title"] = merged_title
    if anchor_node.get("title") and normalize_text(anchor_node["title"]) not in {
        normalize_text(value) for value in (anchor_node.get("aliases") or [])
    }:
        anchor_node["aliases"] = [normalize_text(anchor_node["title"]), *(anchor_node.get("aliases") or [])]
    anchor_node["aliases"] = unique_preserve_order(
        [normalize_text(value) for value in (anchor_node.get("aliases") or []) if normalize_text(value)]
    )
    anchor_node["search_text"] = build_reference_search_text(anchor_node)
    return anchor_node


def merge_reference_citation_edges(
    citation_edges: list[dict[str, Any]],
    reference_id_map: dict[str, str],
) -> list[dict[str, Any]]:
    merged_edges: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in citation_edges:
        source_id = str(edge.get("source_id") or "")
        target_id = reference_id_map.get(str(edge.get("target_id") or ""), str(edge.get("target_id") or ""))
        key = (source_id, target_id)
        merged = merged_edges.setdefault(
            key,
            {
                **edge,
                "target_id": target_id,
                "surface_forms": set(),
                "raw_citation_texts": [],
                "mention_count": 0,
            },
        )
        merged["confidence"] = max(float(merged.get("confidence") or 0.0), float(edge.get("confidence") or 0.0))
        merged["mention_count"] = int(merged.get("mention_count") or 0) + int(edge.get("mention_count") or 1)
        for value in edge.get("surface_forms") or []:
            normalized_value = normalize_text(value)
            if normalized_value:
                merged["surface_forms"].add(normalized_value)
        for value in edge.get("raw_citation_texts") or []:
            normalized_value = normalize_text(value)
            if normalized_value and normalized_value not in merged["raw_citation_texts"]:
                merged["raw_citation_texts"].append(normalized_value)
        if not merged.get("page") and edge.get("page") is not None:
            merged["page"] = edge.get("page")

    rows: list[dict[str, Any]] = []
    for edge in merged_edges.values():
        if isinstance(edge.get("surface_forms"), set):
            edge["surface_forms"] = sorted(edge["surface_forms"])
        rows.append(edge)
    rows.sort(key=lambda row: (str(row.get("source_id") or ""), str(row.get("target_id") or "")))
    return rows


def resolve_reference_to_paper_edges(
    reference_nodes: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    config: DocumentEntityGraphConfig,
) -> list[dict[str, Any]]:
    paper_records = [build_paper_resolution_record(paper) for paper in papers]
    paper_by_doi = {
        record["doi"]: record
        for record in paper_records
        if record["doi"] and not bool(record.get("suspicious_doi"))
    }
    paper_by_pmid = {
        record["pmid"]: record
        for record in paper_records
        if record["pmid"]
    }
    edges: list[dict[str, Any]] = []
    for reference in reference_nodes:
        reference_record = build_reference_resolution_record(reference)
        ranked_candidates = rank_reference_paper_candidates(
            reference_record=reference_record,
            paper_records=paper_records,
            config=config,
        )
        matched = match_reference_to_paper(
            reference_record=reference_record,
            paper_records=paper_records,
            paper_by_doi=paper_by_doi,
            paper_by_pmid=paper_by_pmid,
            config=config,
            ranked_candidates=ranked_candidates,
        )
        if not matched:
            weak_match = select_weak_reference_navigation_match(
                reference_record=reference_record,
                ranked_candidates=ranked_candidates,
                config=config,
            )
            if not weak_match:
                continue
            matched_paper, match_info = weak_match
            if matched_paper["paper_id"] == reference.get("reference_id"):
                continue
            edges.append(
                new_edge(
                    edge_type="reference_weakly_resolved_to_paper",
                    source_id=reference["node_id"],
                    target_id=matched_paper["node_id"],
                    confidence=match_info["confidence"],
                    source="reference_navigation",
                    match_method=match_info["match_method"],
                    resolver_score=round(match_info["score"], 4),
                    title_similarity=round(match_info["title_similarity"], 4),
                    title_token_recall=round(match_info["title_token_recall"], 4),
                    year_match=match_info["year_match"],
                    year_delta=match_info["year_delta"],
                    first_author_match=match_info["first_author_match"],
                    pmid_match=match_info["pmid_match"],
                    doi_prefix_match=match_info["doi_prefix_match"],
                    matched_on_raw_title=match_info["matched_on_raw_title"],
                )
            )
            continue
        matched_paper, match_info = matched
        if matched_paper["paper_id"] == reference.get("reference_id"):
            continue
        edges.append(
            new_edge(
                edge_type="reference_resolved_to_paper",
                source_id=reference["node_id"],
                target_id=matched_paper["node_id"],
                confidence=match_info["confidence"],
                source="reference_resolution",
                match_method=match_info["match_method"],
                resolver_score=round(match_info["score"], 4),
                title_similarity=round(match_info["title_similarity"], 4),
                title_token_recall=round(match_info["title_token_recall"], 4),
                year_match=match_info["year_match"],
                year_delta=match_info["year_delta"],
                first_author_match=match_info["first_author_match"],
                pmid_match=match_info["pmid_match"],
                doi_prefix_match=match_info["doi_prefix_match"],
                matched_on_raw_title=match_info["matched_on_raw_title"],
            )
        )
    return edges


def build_paper_resolution_record(paper: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(paper.get("title") or "")
    title_tokens = set(tokenize_for_similarity(title))
    authors = paper.get("authors") or []
    first_author_source = authors[0] if authors else paper.get("author_text") or ""
    doi = normalize_doi(paper.get("doi") or "")
    pmid = normalize_pmid(paper.get("pmid") or "")
    year = normalize_year_text(paper.get("year"))
    return {
        "paper": paper,
        "node_id": paper["node_id"],
        "paper_id": paper["paper_id"],
        "doi": doi,
        "suspicious_doi": bool(doi and is_suspicious_doi(doi)),
        "pmid": pmid,
        "year": year,
        "title": title,
        "title_key": build_title_key(title),
        "title_match_text": normalize_resolution_text(title),
        "title_tokens": title_tokens,
        "first_author_key": first_author_key(first_author_source),
    }


def build_reference_resolution_record(reference: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(reference.get("title") or "")
    raw_variants = [
        normalize_text(value)
        for value in reference.get("raw_variants") or []
        if normalize_text(value)
    ]
    aliases = [
        normalize_text(value)
        for value in reference.get("aliases") or []
        if normalize_text(value)
    ]
    reference_text = join_unique_nonempty([title, *aliases[:2], *raw_variants[:3]])
    doi = normalize_doi(reference.get("doi") or "")
    pmid = normalize_pmid(reference.get("pmid") or "")
    return {
        "reference": reference,
        "doi": doi,
        "pmid": pmid,
        "year": normalize_year_text(reference.get("year")),
        "title": title,
        "title_key": build_title_key(title),
        "title_match_text": normalize_resolution_text(title),
        "title_tokens": set(tokenize_for_similarity(title)),
        "reference_text": reference_text,
        "reference_text_match": normalize_resolution_text(reference_text),
        "reference_text_tokens": set(tokenize_for_similarity(reference_text)),
        "first_author_key": first_author_key(reference.get("authors_text") or ""),
        "suspicious_doi": bool(doi and is_suspicious_doi(doi)),
    }


def match_reference_to_paper(
    reference_record: dict[str, Any],
    paper_records: list[dict[str, Any]],
    paper_by_doi: dict[str, dict[str, Any]],
    paper_by_pmid: dict[str, dict[str, Any]],
    config: DocumentEntityGraphConfig,
    ranked_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    doi = reference_record["doi"]
    if doi and doi in paper_by_doi and not reference_record["suspicious_doi"]:
        matched_paper = paper_by_doi[doi]
        return matched_paper, build_exact_reference_match_info(
            match_method="doi_exact",
            score=1.0,
            pmid_match=False,
            doi_prefix_match=False,
            matched_on_raw_title=False,
        )

    pmid = reference_record["pmid"]
    if pmid and pmid in paper_by_pmid:
        matched_paper = paper_by_pmid[pmid]
        return matched_paper, build_exact_reference_match_info(
            match_method="pmid_exact",
            score=0.995,
            pmid_match=True,
            doi_prefix_match=False,
            matched_on_raw_title=False,
        )

    scored_candidates = ranked_candidates
    if scored_candidates is None:
        scored_candidates = rank_reference_paper_candidates(
            reference_record=reference_record,
            paper_records=paper_records,
            config=config,
        )
    if not scored_candidates:
        return None
    best_score, best_paper, best_match = scored_candidates[0]
    second_score = scored_candidates[1][0] if len(scored_candidates) > 1 else 0.0
    if best_score < 0.89:
        return None
    if (
        second_score
        and best_score < 0.97
        and (best_score - second_score) < 0.03
        and best_match["match_method"] not in {"raw_title_author_year", "raw_title_plus_metadata"}
    ):
        return None
    return best_paper, best_match


def rank_reference_paper_candidates(
    reference_record: dict[str, Any],
    paper_records: list[dict[str, Any]],
    config: DocumentEntityGraphConfig,
) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    scored_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for paper_record in paper_records:
        match_info = score_reference_paper_match(reference_record, paper_record, config)
        if match_info["score"] <= 0.0:
            continue
        scored_candidates.append((match_info["score"], paper_record, match_info))
    scored_candidates.sort(
        key=lambda item: (
            -item[0],
            -float(item[2]["title_token_recall"]),
            -float(item[2]["title_similarity"]),
            item[1]["paper_id"],
        )
    )
    return scored_candidates


def select_weak_reference_navigation_match(
    reference_record: dict[str, Any],
    ranked_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]],
    config: DocumentEntityGraphConfig,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not ranked_candidates:
        return None

    best_score, best_paper, best_match = ranked_candidates[0]
    second_score = ranked_candidates[1][0] if len(ranked_candidates) > 1 else 0.0
    title_token_count = len(best_paper.get("title_tokens") or [])
    year_delta = best_match.get("year_delta")
    year_ok = year_delta is None or abs(int(year_delta)) <= int(config.reference_navigation_max_year_delta)
    has_strong_title_signal = bool(
        best_match["matched_on_raw_title"]
        or best_match["title_similarity"] >= float(config.reference_navigation_min_title_similarity)
        or (
            best_match["title_token_recall"] >= 0.92
            and title_token_count >= max(6, int(config.reference_navigation_min_title_tokens))
        )
        or (
            best_match["title_token_recall"] >= 0.80
            and title_token_count >= max(9, int(config.reference_navigation_min_title_tokens))
        )
    )
    has_anchor_signal = bool(
        best_match["matched_on_raw_title"]
        or best_match["first_author_match"]
        or best_match["doi_prefix_match"]
        or best_match.get("title_key_exact")
    )
    margin_ok = (
        second_score <= 0.0
        or (best_score - second_score) >= float(config.reference_navigation_min_margin)
        or best_match["matched_on_raw_title"]
    )
    if best_score >= 0.89:
        return None
    if best_score < float(config.reference_navigation_min_confidence):
        return None
    if best_match["title_token_recall"] < float(config.reference_navigation_min_title_recall):
        return None
    if title_token_count < int(config.reference_navigation_min_title_tokens):
        return None
    if not year_ok:
        return None
    if not has_strong_title_signal:
        return None
    if (
        not has_anchor_signal
        and float(best_match.get("title_similarity") or 0.0) < 0.88
    ):
        return None
    if not margin_ok:
        return None

    weak_info = dict(best_match)
    weak_info["match_method"] = determine_weak_reference_navigation_method(best_match, title_token_count)
    weak_info["confidence"] = max(0.54, min(0.84, 0.88 * float(best_match["score"])))
    return best_paper, weak_info


def determine_weak_reference_navigation_method(match_info: dict[str, Any], title_token_count: int) -> str:
    if match_info.get("matched_on_raw_title"):
        return "weak_raw_title_navigation"
    if float(match_info.get("title_similarity") or 0.0) >= 0.72:
        return "weak_fuzzy_title_navigation"
    if (
        float(match_info.get("title_token_recall") or 0.0) >= 0.90
        and title_token_count >= 9
    ):
        return "weak_long_title_overlap_navigation"
    return "weak_reference_navigation"


def build_exact_reference_match_info(
    match_method: str,
    score: float,
    pmid_match: bool,
    doi_prefix_match: bool,
    matched_on_raw_title: bool,
) -> dict[str, Any]:
    return {
        "match_method": match_method,
        "score": score,
        "confidence": score,
        "title_similarity": 1.0,
        "title_token_recall": 1.0,
        "title_key_exact": True,
        "year_match": True,
        "year_delta": 0,
        "first_author_match": False,
        "pmid_match": pmid_match,
        "doi_prefix_match": doi_prefix_match,
        "matched_on_raw_title": matched_on_raw_title,
    }


def score_reference_paper_match(
    reference_record: dict[str, Any],
    paper_record: dict[str, Any],
    config: DocumentEntityGraphConfig,
) -> dict[str, Any]:
    title_similarity = token_f1_overlap(reference_record["title_tokens"], paper_record["title_tokens"])
    comparison_tokens = reference_record["title_tokens"] or reference_record["reference_text_tokens"]
    title_token_recall = token_set_recall(comparison_tokens, paper_record["title_tokens"])
    matched_on_raw_title = bool(
        paper_record["title_match_text"]
        and (
            len(paper_record["title_tokens"]) >= 4
            or len(paper_record["title_match_text"]) >= 24
        )
        and (
            paper_record["title_match_text"] in reference_record["reference_text_match"]
            or (
                reference_record["title_match_text"]
                and paper_record["title_match_text"] in reference_record["title_match_text"]
            )
        )
    )
    title_key_exact = bool(
        reference_record["title_key"]
        and paper_record["title_key"]
        and reference_record["title_key"] == paper_record["title_key"]
    )
    first_author_match = bool(
        reference_record["first_author_key"]
        and paper_record["first_author_key"]
        and reference_record["first_author_key"] == paper_record["first_author_key"]
    )
    year_delta = compare_year_text(reference_record["year"], paper_record["year"])
    year_match = year_delta == 0 if year_delta is not None else False
    year_close = year_delta is not None and abs(year_delta) <= 1
    long_title_match = (
        title_key_exact
        and len(reference_record["title_tokens"]) >= 8
        and len(paper_record["title_tokens"]) >= 8
    )
    doi_prefix_match = bool(
        reference_record["doi"]
        and reference_record["suspicious_doi"]
        and paper_record["doi"]
        and paper_record["doi"].startswith(reference_record["doi"])
    )

    if (
        year_delta is not None
        and abs(year_delta) > 1
        and not matched_on_raw_title
        and not doi_prefix_match
        and not long_title_match
    ):
        return build_reference_match_info(
            match_method="none",
            score=0.0,
            title_similarity=title_similarity,
            title_token_recall=title_token_recall,
            title_key_exact=title_key_exact,
            year_match=year_match,
            year_delta=year_delta,
            first_author_match=first_author_match,
            pmid_match=False,
            doi_prefix_match=doi_prefix_match,
            matched_on_raw_title=matched_on_raw_title,
        )

    score = 0.0
    match_method = "none"
    required_overlap = max(float(config.reference_title_match_overlap or 0.0), 0.92)

    if matched_on_raw_title and first_author_match and (year_match or year_close):
        score = 0.97 if year_match else 0.95
        match_method = "raw_title_author_year"
    elif matched_on_raw_title and (year_match or year_close or first_author_match or doi_prefix_match):
        score = 0.94 if year_match else 0.93
        match_method = "raw_title_plus_metadata"
    elif title_key_exact and first_author_match and (year_match or year_close):
        score = 0.93 if year_match else 0.91
        match_method = "title_key_author_year"
    elif long_title_match and year_close:
        score = 0.90 if year_match else 0.89
        match_method = "title_key_close_year"
    elif title_key_exact and (year_match or doi_prefix_match):
        score = 0.91 if year_match else 0.89
        match_method = "title_key_year"
    elif title_token_recall >= required_overlap and first_author_match and (year_match or year_close):
        score = 0.90 if year_match else 0.89
        match_method = "fuzzy_title_author_year"
    elif title_token_recall >= 0.97 and (year_match or doi_prefix_match):
        score = 0.89
        match_method = "fuzzy_title_year"
    else:
        score = max(
            score,
            0.72 * title_token_recall + 0.10 * float(first_author_match) + 0.08 * float(year_match),
        )

    return build_reference_match_info(
        match_method=match_method,
        score=min(score, 0.999),
        title_similarity=title_similarity,
        title_token_recall=title_token_recall,
        title_key_exact=title_key_exact,
        year_match=year_match,
        year_delta=year_delta,
        first_author_match=first_author_match,
        pmid_match=False,
        doi_prefix_match=doi_prefix_match,
        matched_on_raw_title=matched_on_raw_title,
    )


def build_reference_match_info(
    match_method: str,
    score: float,
    title_similarity: float,
    title_token_recall: float,
    title_key_exact: bool,
    year_match: bool,
    year_delta: int | None,
    first_author_match: bool,
    pmid_match: bool,
    doi_prefix_match: bool,
    matched_on_raw_title: bool,
) -> dict[str, Any]:
    return {
        "match_method": match_method,
        "score": float(score),
        "confidence": float(score),
        "title_similarity": float(title_similarity),
        "title_token_recall": float(title_token_recall),
        "title_key_exact": bool(title_key_exact),
        "year_match": bool(year_match),
        "year_delta": year_delta,
        "first_author_match": bool(first_author_match),
        "pmid_match": bool(pmid_match),
        "doi_prefix_match": bool(doi_prefix_match),
        "matched_on_raw_title": bool(matched_on_raw_title),
    }


def normalize_resolution_text(text: str) -> str:
    normalized = strip_accents(normalize_text(text)).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_year_text(value: Any) -> str:
    text = normalize_text(str(value or ""))
    match = YEAR_RE.search(text)
    if not match:
        return ""
    return match.group(0)


def compare_year_text(left: str, right: str) -> int | None:
    if not left or not right:
        return None
    try:
        return int(left) - int(right)
    except ValueError:
        return None


def token_set_recall(observed_tokens: set[str], expected_tokens: set[str]) -> float:
    if not observed_tokens or not expected_tokens:
        return 0.0
    overlap = len(observed_tokens & expected_tokens)
    return overlap / max(len(expected_tokens), 1)


def token_f1_overlap(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return (2.0 * precision * recall) / denominator


def build_semantic_entities_and_edges(
    papers: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    chunks_by_section: dict[str, list[dict[str, Any]]],
    figures_by_section: dict[str, list[dict[str, Any]]],
    config: DocumentEntityGraphConfig,
) -> dict[str, Any]:
    entity_registries: dict[str, dict[str, dict[str, Any]]] = {
        "method": {},
        "dataset": {},
        "task": {},
        "metric": {},
        "model": {},
    }
    section_edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    figure_edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    sections_by_id = {section["section_id"]: section for section in sections}
    paper_by_id = {paper["paper_id"]: paper for paper in papers}
    sections_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunks_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    figures_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for section in sections:
        sections_by_paper[str(section["paper_id"])].append(section)
    for section_id, section_chunks in chunks_by_section.items():
        parent_section = sections_by_id.get(section_id)
        if not parent_section:
            continue
        chunks_by_paper[str(parent_section["paper_id"])].extend(section_chunks)
    for section_id, section_figures in figures_by_section.items():
        parent_section = sections_by_id.get(section_id)
        if not parent_section:
            continue
        figures_by_paper[str(parent_section["paper_id"])].extend(section_figures)

    for section in sections:
        section_text = join_unique_nonempty(
            [
                section.get("section_title") or "",
                section.get("section_summary") or "",
                section.get("section_text") or "",
            ]
        )
        mentions = extract_semantic_mentions(
            section_text,
            section_group=str(section.get("section_group") or ""),
            source_kind="section",
        )
        for mention in mentions:
            if mention.confidence < config.semantic_min_confidence:
                continue
            if mention.entity_type == "method" and method_is_comparison_only(section_text, mention.surface_form):
                continue
            if mention.entity_type == "method" and not should_keep_method_mention(
                mention,
                section_group=str(section.get("section_group") or ""),
                section_title=str(section.get("section_title") or ""),
                source_kind="section",
            ):
                continue
            entity_node = get_or_create_semantic_entity(entity_registries[mention.entity_type], mention)
            evidence_chunk_ids = find_surface_chunk_ids(
                chunks_by_section.get(section["section_id"], []),
                mention.surface_form,
                fallback_k=2,
            )
            evidence_figure_ids = find_surface_figure_ids(
                figures_by_section.get(section["section_id"], []),
                mention.surface_form,
            )
            section_edge_type = SEMANTIC_EDGE_TYPE_MAP[mention.entity_type][0]
            merge_edge(
                section_edge_map,
                edge_type=section_edge_type,
                source_id=section["node_id"],
                target_id=entity_node["node_id"],
                confidence=mention.confidence,
                source=mention.source,
                surface_form=mention.surface_form,
                evidence_section_id=section["section_id"],
                evidence_chunk_ids=evidence_chunk_ids,
                evidence_figure_ids=evidence_figure_ids,
            )

    if config.enable_figure_entity_edges:
        for figure in figures:
            figure_text = join_unique_nonempty(
                list(figure.get("caption") or []) + [figure.get("ocr_text") or ""]
            )
            mentions = extract_semantic_mentions(
                figure_text,
                section_group=str(sections_by_id.get(figure["section_id"], {}).get("section_group") or ""),
                source_kind="figure",
            )
            for mention in mentions:
                if mention.confidence < max(config.semantic_min_confidence, 0.80):
                    continue
                parent_section = sections_by_id.get(figure["section_id"], {})
                if mention.entity_type == "method" and not should_keep_method_mention(
                    mention,
                    section_group=str(parent_section.get("section_group") or ""),
                    section_title=str(parent_section.get("section_title") or ""),
                    source_kind="figure",
                ):
                    continue
                entity_node = get_or_create_semantic_entity(entity_registries[mention.entity_type], mention)
                figure_edge_type = SEMANTIC_EDGE_TYPE_MAP[mention.entity_type][2]
                merge_edge(
                    figure_edge_map,
                    edge_type=figure_edge_type,
                    source_id=figure["node_id"],
                    target_id=entity_node["node_id"],
                    confidence=mention.confidence,
                    source="caption",
                    surface_form=mention.surface_form,
                    evidence_section_id=figure["section_id"],
                    evidence_figure_ids=[figure["figure_id"]],
                )

    paper_edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for paper in papers:
        paper_text = join_unique_nonempty(
            [
                paper.get("title") or "",
                paper.get("abstract") or "",
                f"keywords {'; '.join(paper.get('keywords') or [])}" if paper.get("keywords") else "",
            ]
        )
        mentions = extract_semantic_mentions(
            paper_text,
            section_group="abstract",
            source_kind="paper_profile",
        )
        for mention in mentions:
            if mention.confidence < max(config.semantic_min_confidence, 0.78):
                continue
            if mention.entity_type == "method" and not should_keep_method_mention(
                mention,
                section_group="abstract",
                section_title=str(paper.get("title") or ""),
                source_kind="paper_profile",
            ):
                continue
            entity_node = get_or_create_semantic_entity(entity_registries[mention.entity_type], mention)
            evidence_section_ids = find_surface_section_ids(
                sections_by_paper.get(paper["paper_id"], []),
                mention.surface_form,
            )
            evidence_chunk_ids = find_surface_chunk_ids(
                chunks_by_paper.get(paper["paper_id"], []),
                mention.surface_form,
                fallback_k=3,
            )
            evidence_figure_ids = find_surface_figure_ids(
                figures_by_paper.get(paper["paper_id"], []),
                mention.surface_form,
            )
            paper_edge_type = SEMANTIC_EDGE_TYPE_MAP[mention.entity_type][1]
            merge_edge(
                paper_edge_map,
                edge_type=paper_edge_type,
                source_id=paper["node_id"],
                target_id=entity_node["node_id"],
                confidence=float(mention.confidence),
                source="paper_profile",
                surface_form=mention.surface_form,
                evidence_section_id=evidence_section_ids[0] if evidence_section_ids else None,
                evidence_chunk_ids=evidence_chunk_ids,
                evidence_figure_ids=evidence_figure_ids,
            )

    for edge in list(section_edge_map.values()):
        section_id = source_section_id_from_edge(edge)
        if not section_id:
            continue
        paper_id = sections_by_id[section_id]["paper_id"]
        paper_edge_type = promote_section_edge_type(edge["edge_type"])
        merge_edge(
            paper_edge_map,
            edge_type=paper_edge_type,
            source_id=paper_by_id[paper_id]["node_id"],
            target_id=edge["target_id"],
            confidence=float(edge.get("confidence") or 0.0),
            source="section_text",
            surface_form=None,
            evidence_section_id=section_id,
            evidence_chunk_ids=edge.get("evidence_chunk_ids") or [],
            evidence_figure_ids=edge.get("evidence_figure_ids") or [],
            mention_increment=int(edge.get("mention_count") or 1),
            surface_forms=edge.get("surface_forms") or [],
        )

    for edge in list(figure_edge_map.values()):
        section_id = source_section_id_from_edge(edge)
        if not section_id:
            continue
        paper_id = sections_by_id[section_id]["paper_id"]
        figure_edge_type = edge["edge_type"]
        paper_edge_type = figure_edge_type.replace("figure_supports_", "paper_uses_")
        if "task" in figure_edge_type:
            paper_edge_type = "paper_addresses_task"
        elif "metric" in figure_edge_type:
            paper_edge_type = "paper_reports_metric"
        merge_edge(
            paper_edge_map,
            edge_type=paper_edge_type,
            source_id=paper_by_id[paper_id]["node_id"],
            target_id=edge["target_id"],
            confidence=float(edge.get("confidence") or 0.0),
            source="caption",
            surface_form=None,
            evidence_section_id=section_id,
            evidence_chunk_ids=edge.get("evidence_chunk_ids") or [],
            evidence_figure_ids=edge.get("evidence_figure_ids") or [],
            mention_increment=int(edge.get("mention_count") or 1),
            surface_forms=edge.get("surface_forms") or [],
        )

    method_nodes = finalize_alias_nodes(entity_registries["method"], alias_field="aliases")
    dataset_nodes = finalize_alias_nodes(entity_registries["dataset"], alias_field="aliases")
    task_nodes = finalize_alias_nodes(entity_registries["task"], alias_field="aliases")
    metric_nodes = finalize_alias_nodes(entity_registries["metric"], alias_field="aliases")
    model_nodes = finalize_alias_nodes(entity_registries["model"], alias_field="aliases")

    edges = list(section_edge_map.values()) + list(figure_edge_map.values()) + list(paper_edge_map.values())
    return {
        "method_nodes": method_nodes,
        "dataset_nodes": dataset_nodes,
        "task_nodes": task_nodes,
        "metric_nodes": metric_nodes,
        "model_nodes": model_nodes,
        "edges": edges,
    }


def build_cross_paper_edges(
    papers: list[dict[str, Any]],
    author_edges: list[dict[str, Any]],
    institution_edges: list[dict[str, Any]],
    venue_edges: list[dict[str, Any]],
    reference_edges: list[dict[str, Any]],
    reference_resolution_edges: list[dict[str, Any]],
    semantic_edges: list[dict[str, Any]],
    config: DocumentEntityGraphConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paper_count = max(1, len(papers))
    paper_lookup = {paper["node_id"]: paper for paper in papers}
    cross_edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    grouped_entity_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in author_edges + institution_edges + venue_edges + semantic_edges:
        grouped_entity_edges[str(edge["edge_type"])].append(edge)

    for source_edge_type, (cross_edge_type, base_weight) in SHARED_ENTITY_CROSS_EDGE_SPECS.items():
        paper_level_edges = [
            edge for edge in grouped_entity_edges.get(source_edge_type, [])
            if str(edge.get("source_id") or "").startswith("paper:")
        ]
        entity_to_papers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in paper_level_edges:
            entity_to_papers[str(edge["target_id"])].append(edge)
        for entity_id, edges_for_entity in entity_to_papers.items():
            # Model families do not prove that two papers used the same model.
            if source_edge_type == "paper_uses_model" and entity_id in {
                "model:large_language_model", "model:gpt", "model:llama", "model:bert"
            }:
                continue
            unique_paper_edges = unique_edges_by_source(edges_for_entity)
            entity_group_size = len(unique_paper_edges)
            if entity_group_size < 2 or entity_group_size > config.cross_paper_max_entity_group_size:
                continue
            rarity = inverse_document_frequency(paper_count, entity_group_size)
            if rarity <= 0:
                continue
            for left_index in range(entity_group_size):
                left_edge = unique_paper_edges[left_index]
                for right_index in range(left_index + 1, entity_group_size):
                    right_edge = unique_paper_edges[right_index]
                    paper_a = str(left_edge["source_id"])
                    paper_b = str(right_edge["source_id"])
                    if paper_a == paper_b:
                        continue
                    weight = base_weight * rarity * harmonic_mean_confidence(
                        left_edge, right_edge
                    )
                    merge_cross_paper_edge(
                        edge_map=cross_edge_map,
                        edge_type=cross_edge_type,
                        source_id=min(paper_a, paper_b),
                        target_id=max(paper_a, paper_b),
                        weight=weight,
                        confidence=max(
                            float(left_edge.get("confidence") or 0.0),
                            float(right_edge.get("confidence") or 0.0),
                        ),
                        source="shared_entity",
                        evidence_entity_id=entity_id,
                        evidence_entity_type=entity_id.split(":", 1)[0],
                    )

    paper_ids_by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in reference_edges:
        if str(edge.get("edge_type") or "") != "paper_cites_reference":
            continue
        paper_ids_by_reference[str(edge["target_id"])].append(edge)

    references_by_paper_resolution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    weak_references_by_paper_resolution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in reference_resolution_edges:
        edge_type = str(edge.get("edge_type") or "")
        if edge_type == "reference_resolved_to_paper":
            references_by_paper_resolution[str(edge["source_id"])].append(edge)
        elif edge_type == "reference_weakly_resolved_to_paper":
            weak_references_by_paper_resolution[str(edge["source_id"])].append(edge)

    for reference_id, citation_edges in paper_ids_by_reference.items():
        if not is_informative_reference_bridge_id(reference_id):
            continue
        unique_citation_edges = unique_edges_by_source(citation_edges)
        citing_paper_count = len(unique_citation_edges)
        if (
            citing_paper_count < config.cross_paper_min_bibliographic_shared_refs
            or citing_paper_count > config.cross_paper_max_reference_group_size
        ):
            continue
        rarity = inverse_document_frequency(paper_count, citing_paper_count)
        if rarity <= 0:
            continue
        for left_index in range(citing_paper_count):
            left_edge = unique_citation_edges[left_index]
            for right_index in range(left_index + 1, citing_paper_count):
                right_edge = unique_citation_edges[right_index]
                paper_a = str(left_edge["source_id"])
                paper_b = str(right_edge["source_id"])
                if paper_a == paper_b:
                    continue
                weight = 1.6 * rarity * harmonic_mean_confidence(left_edge, right_edge)
                merge_cross_paper_edge(
                    edge_map=cross_edge_map,
                    edge_type="paper_bibliographic_coupling",
                    source_id=min(paper_a, paper_b),
                    target_id=max(paper_a, paper_b),
                    weight=weight,
                    confidence=max(
                        float(left_edge.get("confidence") or 0.0),
                        float(right_edge.get("confidence") or 0.0),
                    ),
                    source="shared_reference",
                    evidence_entity_id=reference_id,
                    evidence_entity_type="reference",
                )

    resolved_papers_by_reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in reference_resolution_edges:
        if edge["edge_type"] == "reference_resolved_to_paper":
            resolved_papers_by_reference[str(edge["source_id"])].append(edge)

    for reference_id, citation_edges in paper_ids_by_reference.items():
        if not is_informative_reference_bridge_id(reference_id):
            continue
        resolved_targets = resolved_papers_by_reference.get(reference_id, [])
        if not resolved_targets:
            continue
        for citation_edge in citation_edges:
            citing_paper_id = str(citation_edge["source_id"])
            for resolved_edge in resolved_targets:
                cited_paper_id = str(resolved_edge["target_id"])
                if citing_paper_id == cited_paper_id:
                    continue
                weight = 3.2 * harmonic_mean_confidence(citation_edge, resolved_edge)
                merge_cross_paper_edge(
                    edge_map=cross_edge_map,
                    edge_type="paper_cites_paper",
                    source_id=citing_paper_id,
                    target_id=cited_paper_id,
                    weight=weight,
                    confidence=min(
                        float(citation_edge.get("confidence") or 0.0),
                        float(resolved_edge.get("confidence") or 0.0),
                    ),
                    source="resolved_reference",
                    evidence_entity_id=reference_id,
                    evidence_entity_type="reference",
                )

    for reference_id, citation_edges in paper_ids_by_reference.items():
        if not is_informative_reference_bridge_id(reference_id):
            continue
        weak_resolved_targets = weak_references_by_paper_resolution.get(reference_id, [])
        if not weak_resolved_targets:
            continue
        for citation_edge in citation_edges:
            citing_paper_id = str(citation_edge["source_id"])
            for weak_edge in weak_resolved_targets:
                cited_paper_id = str(weak_edge["target_id"])
                if citing_paper_id == cited_paper_id:
                    continue
                weight = 1.7 * harmonic_mean_confidence(citation_edge, weak_edge)
                merge_cross_paper_edge(
                    edge_map=cross_edge_map,
                    edge_type="paper_weakly_cites_paper",
                    source_id=citing_paper_id,
                    target_id=cited_paper_id,
                    weight=weight,
                    confidence=min(
                        float(citation_edge.get("confidence") or 0.0),
                        float(weak_edge.get("confidence") or 0.0),
                    ),
                    source="weak_reference_navigation",
                    evidence_entity_id=reference_id,
                    evidence_entity_type="reference",
                )

    cross_edges = [
        edge
        for edge in cross_edge_map.values()
        if float(edge.get("weight") or 0.0) >= config.cross_paper_min_edge_weight
        and edge["source_id"] in paper_lookup
        and edge["target_id"] in paper_lookup
    ]
    cross_edges.sort(
        key=lambda edge: (
            edge["edge_type"],
            edge["source_id"],
            edge["target_id"],
        )
    )
    pair_summaries = build_paper_pair_summaries(cross_edges, paper_lookup)
    return cross_edges, pair_summaries


def unique_edges_by_source(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for edge in edges:
        source_id = str(edge.get("source_id") or "")
        current = unique.get(source_id)
        if current is None or float(edge.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            unique[source_id] = edge
    return sorted(unique.values(), key=lambda edge: str(edge.get("source_id") or ""))


def is_informative_reference_bridge_id(reference_node_id: str) -> bool:
    if not reference_node_id.startswith("reference:"):
        return True
    key = reference_node_id.split("reference:", 1)[1]
    if key in GENERIC_REFERENCE_BRIDGE_KEYS:
        return False
    if not key.startswith("doi_"):
        return True

    doi_tail = key[4:]
    if len(doi_tail) < 16:
        return False
    parts = [part for part in doi_tail.split("_") if part]
    if len(parts) < 4:
        return False
    if len(parts) >= 6:
        return True

    alpha_like_parts = [part for part in parts[2:] if re.search(r"[a-z]", part)]
    if len(alpha_like_parts) >= 2 and len(parts[-1]) >= 5:
        return True
    return False


def inverse_document_frequency(total_papers: int, group_size: int) -> float:
    if total_papers <= 0 or group_size <= 0:
        return 0.0
    return math.log1p(total_papers / float(group_size))


def harmonic_mean_confidence(left_edge: dict[str, Any], right_edge: dict[str, Any]) -> float:
    left_conf = max(0.0, float(left_edge.get("confidence") or 0.0))
    right_conf = max(0.0, float(right_edge.get("confidence") or 0.0))
    return (2.0 * left_conf * right_conf) / (left_conf + right_conf) if left_conf + right_conf > 0 else 0.0


def merge_cross_paper_edge(
    edge_map: dict[tuple[str, str, str], dict[str, Any]],
    edge_type: str,
    source_id: str,
    target_id: str,
    weight: float,
    confidence: float,
    source: str,
    evidence_entity_id: str,
    evidence_entity_type: str,
) -> None:
    key = (edge_type, source_id, target_id)
    edge = edge_map.setdefault(
        key,
        {
            "edge_type": edge_type,
            "source_id": source_id,
            "target_id": target_id,
            "confidence": confidence,
            "weight": 0.0,
            "source": source,
            "evidence_entity_ids": set(),
            "evidence_entity_types": set(),
            "shared_count": 0,
            "cross_paper": True,
        },
    )
    edge["weight"] = float(edge.get("weight") or 0.0) + float(weight)
    edge["confidence"] = max(float(edge.get("confidence") or 0.0), float(confidence))
    edge["source"] = source
    edge["evidence_entity_ids"].add(evidence_entity_id)
    edge["evidence_entity_types"].add(evidence_entity_type)
    edge["shared_count"] = int(edge.get("shared_count") or 0) + 1


def build_paper_pair_summaries(
    cross_edges: list[dict[str, Any]],
    paper_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pair_map: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in cross_edges:
        left = str(edge["source_id"])
        right = str(edge["target_id"])
        pair_key = tuple(sorted((left, right)))
        pair = pair_map.setdefault(
            pair_key,
            {
                "paper_a_node_id": pair_key[0],
                "paper_b_node_id": pair_key[1],
                "paper_a_id": paper_lookup[pair_key[0]]["paper_id"],
                "paper_b_id": paper_lookup[pair_key[1]]["paper_id"],
                "paper_a_title": paper_lookup[pair_key[0]].get("title") or "",
                "paper_b_title": paper_lookup[pair_key[1]].get("title") or "",
                "relation_types": set(),
                "total_weight": 0.0,
                "edge_count": 0,
                "shared_count": 0,
                "evidence_entity_ids": set(),
                "evidence_entity_types": set(),
                "directed_citation_count": 0,
                "weak_navigation_count": 0,
            },
        )
        pair["relation_types"].add(edge["edge_type"])
        pair["total_weight"] = float(pair.get("total_weight") or 0.0) + float(edge.get("weight") or 0.0)
        pair["edge_count"] = int(pair.get("edge_count") or 0) + 1
        pair["shared_count"] = int(pair.get("shared_count") or 0) + int(edge.get("shared_count") or 0)
        for value in edge.get("evidence_entity_ids") or []:
            pair["evidence_entity_ids"].add(value)
        for value in edge.get("evidence_entity_types") or []:
            pair["evidence_entity_types"].add(value)
        if edge["edge_type"] == "paper_cites_paper":
            pair["directed_citation_count"] = int(pair.get("directed_citation_count") or 0) + 1
        if edge["edge_type"] == "paper_weakly_cites_paper":
            pair["weak_navigation_count"] = int(pair.get("weak_navigation_count") or 0) + 1

    summaries: list[dict[str, Any]] = []
    for pair in pair_map.values():
        pair["relation_types"] = sorted(pair["relation_types"])
        pair["evidence_entity_ids"] = sorted(pair["evidence_entity_ids"])
        pair["evidence_entity_types"] = sorted(pair["evidence_entity_types"])
        summaries.append(pair)
    summaries.sort(
        key=lambda row: (
            -float(row.get("total_weight") or 0.0),
            -int(row.get("edge_count") or 0),
            row["paper_a_id"],
            row["paper_b_id"],
        )
    )
    return summaries


def get_or_create_semantic_entity(
    registry: dict[str, dict[str, Any]],
    mention: SemanticMention,
) -> dict[str, Any]:
    node = registry.setdefault(
        mention.normalized_key,
        {
            "node_id": f"{mention.entity_type}:{mention.normalized_key}",
            "node_type": mention.entity_type,
            "entity_id": f"{mention.entity_type}:{mention.normalized_key}",
            "canonical_name": mention.canonical_name,
            "aliases": set(),
            "normalized_key": mention.normalized_key,
            "description": "",
            "ontology_source": "rule_v1",
            "search_text": "",
        },
    )
    node["aliases"].add(mention.surface_form)
    return node


def extract_semantic_mentions(
    text: str,
    *,
    section_group: str | None = None,
    source_kind: str = "section",
) -> list[SemanticMention]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    mentions: list[SemanticMention] = []
    seen: set[tuple[str, str, str]] = set()
    occupied_alias_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def add_mention(
        entity_type: str,
        canonical_name: str,
        surface: str,
        confidence: float,
        source: str,
    ) -> None:
        surface = normalize_text(surface)
        if not surface:
            return
        key = normalize_entity_key(canonical_name)
        sig = (entity_type, key, surface.lower())
        if sig in seen:
            return
        seen.add(sig)
        mentions.append(
            SemanticMention(
                entity_type=entity_type,
                canonical_name=canonical_name,
                normalized_key=key,
                surface_form=surface,
                confidence=confidence,
                source=source,
            )
        )

    alias_rows: list[tuple[str, str, str]] = []
    for entity_type, canonical_map in SEMANTIC_ALIAS_TABLE.items():
        for canonical_name, aliases in canonical_map.items():
            for alias in unique_preserve_order([canonical_name, *(aliases or [])]):
                if normalize_text(alias):
                    alias_rows.append((entity_type, canonical_name, normalize_text(alias)))
    alias_rows.sort(key=lambda item: (item[0], -len(item[2]), item[2].lower()))

    for entity_type, canonical_name, alias in alias_rows:
        pattern = re.compile(r"\b%s\b" % re.escape(alias), re.IGNORECASE)
        for match in pattern.finditer(normalized):
            start, end = match.start(), match.end()
            if any(start >= span_start and end <= span_end for span_start, span_end in occupied_alias_spans[entity_type]):
                continue
            occupied_alias_spans[entity_type].append((start, end))
            surface = normalize_text(match.group(0))
            add_mention(
                entity_type=entity_type,
                canonical_name=canonical_name,
                surface=surface,
                confidence=0.94,
                source="rule_extract",
            )
            if entity_type == "model":
                context_window = extract_match_window(normalized, start, end)
                bridge_confidence = maybe_promote_model_surface_to_method(
                    canonical_name=canonical_name,
                    surface=surface,
                    context_window=context_window,
                    section_group=section_group,
                    source_kind=source_kind,
                )
                if bridge_confidence is not None:
                    method_canonical = canonicalize_method_surface(canonical_name) or canonical_name
                    add_mention(
                        entity_type="method",
                        canonical_name=canonicalize_dynamic_entity_name("method", method_canonical),
                        surface=surface,
                        confidence=bridge_confidence,
                        source="model_bridge",
                    )

    for match in METHOD_DYNAMIC_EXACT_RE.finditer(normalized):
        surface = normalize_text(match.group(1))
        cleaned_surface = canonicalize_method_surface(surface)
        if not is_valid_dynamic_method_surface(cleaned_surface, strict_tail_prefix=False):
            continue
        canonical = canonicalize_dynamic_entity_name("method", cleaned_surface)
        add_mention(
            entity_type="method",
            canonical_name=canonical,
            surface=cleaned_surface,
            confidence=min(0.90, 0.84 + method_context_bonus(section_group, source_kind)),
            source="rule_method_exact",
        )

    for match in METHOD_DYNAMIC_TAIL_RE.finditer(normalized):
        surface = normalize_text(match.group(1))
        cleaned_surface = canonicalize_method_surface(surface)
        if not is_valid_dynamic_method_surface(cleaned_surface):
            continue
        canonical = canonicalize_dynamic_entity_name("method", cleaned_surface)
        add_mention(
            entity_type="method",
            canonical_name=canonical,
            surface=cleaned_surface,
            confidence=min(0.88, 0.80 + method_context_bonus(section_group, source_kind)),
            source="rule_method_tail",
        )

    for match in METHOD_DYNAMIC_HEAD_RE.finditer(normalized):
        surface = normalize_text(match.group(1))
        cleaned_surface = canonicalize_method_surface(surface)
        cleaned_surface = maybe_trim_method_head_surface(cleaned_surface)
        if not is_valid_dynamic_method_surface(cleaned_surface):
            continue
        context_window = extract_match_window(normalized, match.start(1), match.end(1))
        if not METHOD_CONTEXT_CUE_RE.search(context_window) and str(section_group or "") != "methods":
            continue
        canonical = canonicalize_dynamic_entity_name("method", cleaned_surface)
        add_mention(
            entity_type="method",
            canonical_name=canonical,
            surface=cleaned_surface,
            confidence=min(0.86, 0.76 + method_context_bonus(section_group, source_kind)),
            source="rule_method_head",
        )

    for match in DATASET_DYNAMIC_RE.finditer(normalized):
        surface = normalize_text(match.group(1))
        if not is_valid_dynamic_dataset_surface(surface):
            continue
        canonical = canonicalize_dynamic_entity_name("dataset", surface)
        add_mention("dataset", canonical, surface, 0.82, "rule_extract")

    for match in MODEL_DYNAMIC_RE.finditer(normalized):
        surface = normalize_text(match.group(1))
        canonical = canonicalize_dynamic_entity_name("model", surface)
        add_mention("model", canonical, surface, 0.86, "rule_extract")
        context_window = extract_match_window(normalized, match.start(1), match.end(1))
        bridge_confidence = maybe_promote_model_surface_to_method(
            canonical_name=canonical,
            surface=surface,
            context_window=context_window,
            section_group=section_group,
            source_kind=source_kind,
        )
        if bridge_confidence is not None:
            method_canonical = canonicalize_method_surface(canonical) or canonical
            add_mention(
                "method",
                canonicalize_dynamic_entity_name("method", method_canonical),
                surface,
                bridge_confidence,
                "model_bridge",
            )

    for match in TASK_DYNAMIC_RE.finditer(normalized):
        surface = normalize_text(match.group(1))
        canonical_task = canonicalize_task_surface(surface)
        if not canonical_task:
            continue
        surface = canonical_task
        canonical = canonicalize_dynamic_entity_name("task", surface)
        add_mention("task", canonical, surface, 0.78, "rule_extract")

    return sorted(
        mentions,
        key=lambda mention: (-mention.confidence, mention.entity_type, mention.canonical_name),
    )


def merge_edge(
    edge_map: dict[tuple[str, str, str], dict[str, Any]],
    edge_type: str,
    source_id: str,
    target_id: str,
    confidence: float,
    source: str,
    surface_form: str | None,
    evidence_section_id: str | None = None,
    evidence_chunk_ids: list[str] | None = None,
    evidence_figure_ids: list[str] | None = None,
    mention_increment: int = 1,
    surface_forms: list[str] | None = None,
) -> None:
    key = (edge_type, source_id, target_id)
    edge = edge_map.setdefault(
        key,
        {
            "edge_type": edge_type,
            "source_id": source_id,
            "target_id": target_id,
            "confidence": confidence,
            "source": source,
            "surface_forms": set(),
            "mention_count": 0,
            "evidence_section_ids": set(),
            "evidence_chunk_ids": set(),
            "evidence_figure_ids": set(),
        },
    )
    edge["confidence"] = max(float(edge.get("confidence") or 0.0), confidence)
    edge["source"] = source
    if surface_form:
        edge["surface_forms"].add(normalize_text(surface_form))
    for item in surface_forms or []:
        if normalize_text(item):
            edge["surface_forms"].add(normalize_text(item))
    edge["mention_count"] = int(edge.get("mention_count") or 0) + mention_increment
    if evidence_section_id:
        edge["evidence_section_ids"].add(evidence_section_id)
    for chunk_id in evidence_chunk_ids or []:
        edge["evidence_chunk_ids"].add(chunk_id)
    for figure_id in evidence_figure_ids or []:
        edge["evidence_figure_ids"].add(figure_id)


def new_edge(
    edge_type: str,
    source_id: str,
    target_id: str,
    confidence: float,
    source: str,
    **extra: Any,
) -> dict[str, Any]:
    edge = {
        "edge_id": "",
        "edge_type": edge_type,
        "source_id": source_id,
        "target_id": target_id,
        "confidence": float(confidence),
        "source": source,
    }
    edge.update(extra)
    return edge


def assign_edge_ids(edges: list[dict[str, Any]]) -> None:
    for edge_index, edge in enumerate(edges, start=1):
        edge["edge_id"] = f"edge:{edge_index:07d}"
        if isinstance(edge.get("surface_forms"), set):
            edge["surface_forms"] = sorted(edge["surface_forms"])
        if isinstance(edge.get("evidence_section_ids"), set):
            edge["evidence_section_ids"] = sorted(edge["evidence_section_ids"])
        if isinstance(edge.get("evidence_chunk_ids"), set):
            edge["evidence_chunk_ids"] = sorted(edge["evidence_chunk_ids"])
        if isinstance(edge.get("evidence_figure_ids"), set):
            edge["evidence_figure_ids"] = sorted(edge["evidence_figure_ids"])
        if isinstance(edge.get("evidence_entity_ids"), set):
            edge["evidence_entity_ids"] = sorted(edge["evidence_entity_ids"])
        if isinstance(edge.get("evidence_entity_types"), set):
            edge["evidence_entity_types"] = sorted(edge["evidence_entity_types"])


def finalize_alias_nodes(
    registry: dict[str, dict[str, Any]],
    alias_field: str,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for key, node in registry.items():
        aliases = sorted(normalize_text(value) for value in node.pop(alias_field) if normalize_text(value))
        node["aliases"] = aliases
        node["search_text"] = join_unique_nonempty(
            [node.get("canonical_name") or ""] + aliases
        )
        nodes.append(node)
    nodes.sort(key=lambda row: row["node_id"])
    return nodes


def write_outputs(
    output_dir: Path,
    papers: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    author_nodes: list[dict[str, Any]],
    institution_nodes: list[dict[str, Any]],
    venue_nodes: list[dict[str, Any]],
    reference_nodes: list[dict[str, Any]],
    semantic_result: dict[str, Any],
    all_nodes: list[dict[str, Any]],
    all_edges: list[dict[str, Any]],
    cross_paper_edges: list[dict[str, Any]],
    paper_pair_summaries: list[dict[str, Any]],
) -> None:
    write_jsonl(output_dir / "papers.jsonl", papers)
    write_jsonl(output_dir / "sections.jsonl", sections)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "figures.jsonl", figures)
    write_jsonl(output_dir / "authors.jsonl", author_nodes)
    write_jsonl(output_dir / "institutions.jsonl", institution_nodes)
    write_jsonl(output_dir / "venues.jsonl", venue_nodes)
    write_jsonl(output_dir / "references.jsonl", reference_nodes)
    write_jsonl(output_dir / "methods.jsonl", semantic_result["method_nodes"])
    write_jsonl(output_dir / "datasets.jsonl", semantic_result["dataset_nodes"])
    write_jsonl(output_dir / "tasks.jsonl", semantic_result["task_nodes"])
    write_jsonl(output_dir / "metrics.jsonl", semantic_result["metric_nodes"])
    write_jsonl(output_dir / "models.jsonl", semantic_result["model_nodes"])
    ordered_nodes = sorted(all_nodes, key=lambda node: (PAPER_NODE_ORDER.index(node["node_type"]), node["node_id"]))
    write_jsonl(output_dir / "nodes.jsonl", ordered_nodes)
    write_jsonl(output_dir / "edges.jsonl", all_edges)
    write_jsonl(output_dir / "paper_paper_edges.jsonl", cross_paper_edges)
    write_jsonl(output_dir / "paper_pair_summaries.jsonl", paper_pair_summaries)


def promote_section_edge_type(edge_type: str) -> str:
    if edge_type.startswith("section_uses_method"):
        return "paper_uses_method"
    if edge_type.startswith("section_uses_dataset"):
        return "paper_uses_dataset"
    if edge_type.startswith("section_addresses_task"):
        return "paper_addresses_task"
    if edge_type.startswith("section_reports_metric"):
        return "paper_reports_metric"
    if edge_type.startswith("section_uses_model"):
        return "paper_uses_model"
    return edge_type


def source_section_id_from_edge(edge: dict[str, Any]) -> str | None:
    evidence_section_ids = edge.get("evidence_section_ids") or []
    if evidence_section_ids:
        if isinstance(evidence_section_ids, set):
            return str(sorted(evidence_section_ids)[0])
        return str(evidence_section_ids[0])
    source_id = str(edge.get("source_id") or "")
    if source_id.startswith("section:"):
        return source_id.split("section:", 1)[1]
    return None


def find_surface_section_ids(sections: list[dict[str, Any]], surface_form: str, fallback_k: int = 2) -> list[str]:
    normalized_surface = normalize_text(surface_form).lower()
    if not sections:
        return []
    if normalized_surface:
        matches = []
        for section in sections:
            section_text = join_unique_nonempty(
                [
                    section.get("section_title") or "",
                    section.get("section_summary") or "",
                    section.get("section_text") or "",
                ]
            ).lower()
            if normalized_surface in section_text:
                matches.append(str(section["section_id"]))
        if matches:
            return matches[:fallback_k]
    preferred = [
        str(section["section_id"])
        for section in sections
        if str(section.get("section_group") or "") in {"abstract", "front_matter", "methods"}
    ]
    if preferred:
        return preferred[:fallback_k]
    return [str(section["section_id"]) for section in sections[:fallback_k]]


def find_surface_chunk_ids(chunks: list[dict[str, Any]], surface_form: str, fallback_k: int = 2) -> list[str]:
    normalized_surface = normalize_text(surface_form).lower()
    if not normalized_surface:
        return [chunk["chunk_id"] for chunk in chunks[:fallback_k]]
    matches = [
        chunk["chunk_id"]
        for chunk in chunks
        if normalized_surface in normalize_text(chunk.get("text") or "").lower()
    ]
    if matches:
        return matches[:fallback_k]
    return [chunk["chunk_id"] for chunk in chunks[:fallback_k]]


def find_surface_figure_ids(figures: list[dict[str, Any]], surface_form: str) -> list[str]:
    normalized_surface = normalize_text(surface_form).lower()
    if not normalized_surface:
        return []
    matches: list[str] = []
    for figure in figures:
        text = join_unique_nonempty(list(figure.get("caption") or []) + [figure.get("ocr_text") or ""]).lower()
        if normalized_surface in text:
            matches.append(figure["figure_id"])
    return matches[:3]


def extract_keywords(items: list[dict[str, Any]], title_index: int) -> list[str]:
    for index, item in enumerate(items):
        if title_index >= 0 and index <= title_index:
            continue
        text = normalize_text(extract_item_text(item))
        if not text:
            continue
        match = KEYWORDS_RE.match(text)
        if not match:
            continue
        keywords = [
            normalize_text(part)
            for part in re.split(r"[;,|]", match.group(1))
            if normalize_text(part)
        ]
        return unique_preserve_order(keywords)
    return []


def extract_venue_text(items: list[dict[str, Any]], title_index: int, authors_text: str) -> str:
    lines: list[str] = []
    for index, item in enumerate(items):
        if title_index >= 0 and index <= title_index:
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type not in TEXT_ITEM_TYPES:
            continue
        text_level = to_int(item.get("text_level"))
        if text_level is not None and text_level >= 2:
            continue
        text = normalize_text(extract_item_text(item))
        if not text or len(text) > 120:
            continue
        if text == normalize_text(authors_text):
            continue
        if looks_like_institution_line(text):
            continue
        if KEYWORDS_RE.match(text):
            continue
        if SECTION_HEADING_LIKE_RE.match(text):
            continue
        if not is_valid_venue_text(text):
            continue
        lowered = text.lower()
        if any(lowered.startswith(prefix) for prefix in ("abstract", "background", "introduction", "keywords")):
            break
        lines.append(text)
        if len(lines) >= 12:
            break
    for line in lines:
        lowered = line.lower()
        if (
            any(hint in lowered for hint in VENUE_HINTS)
            and count_alpha_tokens(line) <= 14
            and is_valid_venue_text(line)
        ):
            return line
    return ""


def extract_paper_doi(items: list[dict[str, Any]]) -> str | None:
    for item in items[:40]:
        text = normalize_text(extract_item_text(item))
        if not text:
            continue
        match = DOI_RE.search(text)
        if match:
            return match.group(0).rstrip(".,;)").lower()
    return None


def extract_paper_pmid(items: list[dict[str, Any]]) -> str | None:
    for item in items[:40]:
        text = normalize_text(extract_item_text(item))
        if not text:
            continue
        match = PMID_RE.search(text)
        if match:
            return match.group(1)
    return None


def extract_paper_year(items: list[dict[str, Any]], paper_id: str) -> int | None:
    prefix_match = re.match(r"^(19|20)\d{2}", paper_id)
    if prefix_match:
        return int(prefix_match.group(0))
    for item in items[:40]:
        text = normalize_text(extract_item_text(item))
        if not text:
            continue
        match = YEAR_RE.search(text)
        if match:
            return int(match.group(0))
    return None


def extract_abstract_from_sections(sections: list[dict[str, Any]]) -> str:
    for section in sections:
        title = normalize_text(section.get("section_title") or "").lower()
        if "abstract" in title:
            return sanitize_abstract_text(section.get("section_text") or "")
    return ""


def build_paper_profile_text_v1(
    title: str,
    authors_text: str,
    institution_text: str,
    venue_text: str,
    abstract: str,
    keywords: list[str],
) -> str:
    return join_unique_nonempty(
        [
            title,
            f"abstract {abstract}" if abstract else "",
            f"authors {authors_text}" if authors_text else "",
            f"institutions {institution_text}" if institution_text else "",
            f"venue {venue_text}" if venue_text else "",
            f"keywords {'; '.join(keywords)}" if keywords else "",
        ]
    )


def is_reference_like_item(item: dict[str, Any], section_stack: list[str]) -> bool:
    sub_type = str(item.get("sub_type") or "").strip().lower()
    if sub_type in {"ref_text", "reference"}:
        return True
    if not section_stack:
        return False
    section_title = normalize_text(section_stack[-1]).lower()
    return "reference" in section_title or "bibliograph" in section_title


def preview_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", normalize_text(text))
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def sanitize_abstract_text(text: str) -> str:
    lines = [normalize_text(line) for line in str(text).splitlines() if normalize_text(line)]
    kept = []
    for line in lines:
        lowered = line.lower()
        if any(lowered.startswith(prefix) for prefix in ABSTRACT_NOISE_PREFIXES):
            continue
        if "e-mail" in lowered or "email" in lowered:
            continue
        kept.append(line)
    return normalize_text("\n\n".join(kept))


def repair_author_list(authors_text: str, fallback_authors: list[str]) -> list[str]:
    fallback = [normalize_text(author) for author in fallback_authors if normalize_text(author)]
    normalized_text = normalize_text(authors_text)
    if not normalized_text:
        return fallback

    looks_suspicious = (
        not fallback
        or any("," in author for author in fallback)
        or (len(fallback) <= 1 and normalized_text.count(",") >= 2)
    )
    if not looks_suspicious:
        return fallback

    working = normalized_text.replace("\\", " ")
    working = re.sub(r"[†‡§¶*]+", " ", working)
    working = re.sub(r"[^\w\s,\-.'&À-ÖØ-öø-ÿ]", " ", working)
    working = re.sub(r"\s+and\s+", "|", working, flags=re.IGNORECASE)
    working = re.sub(r"\s*&\s*", "|", working)
    working = re.sub(r"\b\d+(?:\s*,\s*\d+)*\b", " ", working)
    if working.count(",") >= 2:
        candidates = [
            normalize_text(part)
            for part in re.split(r",|\|", working)
            if normalize_text(part)
        ]
    else:
        working = re.sub(
            r",\s*(?=[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ.\-]+(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ.\-]+){1,4}\b)",
            "|",
            working,
        )
        candidates = [normalize_text(part) for part in working.split("|") if normalize_text(part)]

    authors: list[str] = []
    for candidate in candidates:
        cleaned = re.sub(r"[\d*†‡§¶]+", " ", candidate)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
        if count_alpha_tokens(cleaned) < 2:
            continue
        authors.append(cleaned)
    return unique_preserve_order(authors) or fallback


def repair_institution_list(institutions: list[str]) -> list[str]:
    parts: list[str] = []
    for institution in institutions:
        normalized = normalize_text(strip_affiliation_leader(institution))
        if not normalized:
            continue
        for semi_part in re.split(r"\s*;\s*", normalized):
            semi_part = normalize_text(strip_affiliation_leader(semi_part))
            if not semi_part:
                continue
            lowered = semi_part.lower()
            if "@" in semi_part or "e-mail" in lowered or "email" in lowered:
                continue
            split_parts = re.split(r",\s*(?=\d+\s+)", semi_part)
            for split_part in split_parts:
                split_part = normalize_text(strip_affiliation_leader(split_part))
                if (
                    split_part
                    and is_valid_institution_candidate(split_part)
                    and not is_address_only_affiliation(split_part)
                ):
                    parts.append(split_part)
    return unique_preserve_order(parts)


def normalize_author_key(text: str) -> str:
    return normalize_entity_key(normalize_person_name(text))


def choose_author_canonical_name(text: str) -> str:
    return normalize_text(re.sub(r"\s+", " ", strip_accents(text)))


def normalize_person_name(text: str) -> str:
    normalized = strip_accents(normalize_text(text)).lower()
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def canonicalize_institution_name(text: str) -> str:
    normalized = normalize_text(strip_affiliation_leader(text))
    if not normalized:
        return ""
    if not is_valid_institution_candidate(normalized):
        return ""
    parts = [
        strip_affiliation_index(normalize_text(part))
        for part in re.split(r"[;,]", normalized)
        if normalize_text(part)
    ]
    parts = [
        part
        for part in parts
        if part and is_valid_institution_candidate(part) and not is_address_only_affiliation(part)
    ]
    def part_score(part: str) -> int:
        lowered = part.lower()
        score = 0
        if any(token in lowered for token in ("university", "universidad", "univ.", "polytechnic")):
            score += 6
        if any(token in lowered for token in ("institute", "institut", "academy", "hospital", "cnrs")):
            score += 5
        if any(token in lowered for token in ("centre", "center", "laboratory", "laboratorio", "laboratoire")):
            score += 4
        if any(token in lowered for token in ("school", "faculty", "department", "research", "clinic")):
            score += 2
        if is_sentence_like_affiliation(part):
            score -= 8
        if part[:1].islower():
            score -= 2
        score -= max(0, len(re.findall(r"[A-Za-z0-9-]+", part)) - 8)
        return score

    scored_parts = [(part_score(part), index, part) for index, part in enumerate(parts)]
    scored_parts.sort(key=lambda item: (-item[0], item[1]))
    if scored_parts and scored_parts[0][0] > 0:
        return scored_parts[0][2]
    for part in parts:
        lowered = part.lower()
        if any(
            token in lowered
            for token in ("research", "faculty", "department", "clinic")
        ) and not is_probable_country_name(part) and is_valid_institution_candidate(part):
            return part
    fallback = parts[0] if parts else strip_affiliation_index(normalized)
    if is_probable_country_name(fallback) or not is_valid_institution_candidate(fallback):
        return ""
    return fallback


def normalize_entity_key(text: str) -> str:
    normalized = strip_accents(normalize_text(text)).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def compact_entity_alias_text(text: str) -> str:
    normalized = strip_accents(normalize_text(text)).lower()
    return re.sub(r"[^a-z0-9]+", "", normalized)


def match_semantic_alias_canonical(entity_type: str, surface: str) -> str:
    normalized_surface = normalize_text(surface)
    if not normalized_surface:
        return ""
    normalized_lookup = strip_accents(normalized_surface).lower()
    compact_lookup = compact_entity_alias_text(normalized_surface)
    for canonical_name, aliases in SEMANTIC_ALIAS_TABLE.get(entity_type, {}).items():
        candidates = [canonical_name, *(aliases or [])]
        for alias in candidates:
            alias_text = normalize_text(alias)
            if not alias_text:
                continue
            alias_lookup = strip_accents(alias_text).lower()
            alias_compact = compact_entity_alias_text(alias_text)
            if normalized_lookup == alias_lookup or (compact_lookup and compact_lookup == alias_compact):
                return canonical_name
    return ""


def strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def first_author_key(author_text: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z]+", strip_accents(author_text).lower())]
    if not tokens:
        return ""
    if len(tokens[0]) == 1 and len(tokens) > 1:
        return tokens[-1]
    return tokens[0]


def canonicalize_dynamic_entity_name(entity_type: str, surface: str) -> str:
    cleaned = normalize_text(surface).strip(" .,:;")
    aliased = match_semantic_alias_canonical(entity_type, cleaned)
    if aliased:
        return aliased
    if entity_type in {"method", "task", "metric"}:
        return cleaned.lower()
    return cleaned


def strip_affiliation_index(text: str) -> str:
    stripped = AFFILIATION_INDEX_RE.sub("", text or "")
    return normalize_text(strip_affiliation_leader(stripped))


def is_probable_country_name(text: str) -> bool:
    lowered = strip_accents(normalize_text(text)).lower().strip(" .,:;")
    return lowered in COUNTRY_LIKE_VALUES


def extract_country_from_affiliation(text: str) -> str | None:
    parts = [normalize_text(part) for part in re.split(r"[;,]", normalize_text(text)) if normalize_text(part)]
    for part in reversed(parts):
        lowered = strip_accents(part).lower().strip(" .,:;")
        if lowered in COUNTRY_LIKE_VALUES:
            return part.strip(" .,:;")
    return None


def is_address_only_affiliation(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    stripped = strip_affiliation_index(normalized)
    lowered = strip_accents(stripped).lower()
    if is_probable_country_name(stripped):
        return True
    if re.search(r"\b\d{4,}\b", stripped) and not any(
        token in lowered
        for token in (
            "university",
            "universidad",
            "univ.",
            "college",
            "school",
            "institute",
            "hospital",
            "academy",
            "centre",
            "center",
            "polytechnic",
            "cnrs",
            "laboratory",
            "laboratorio",
            "laboratoire",
            "research",
        )
    ):
        return True
    if re.search(r"\b(?:street|st\.|road|rd\.|avenue|ave\.?|blvd|boulevard|drive|dr\.?)\b", lowered):
        return True
    return False


def is_valid_institution_candidate(text: str) -> bool:
    normalized = normalize_text(strip_affiliation_leader(text))
    if not normalized:
        return False
    if count_alpha_tokens(normalized) < 2:
        return False
    if len(normalized) > 180:
        return False
    tokens = re.findall(r"[A-Za-z0-9-]+", normalized)
    if len(tokens) > 18:
        return False
    if is_probable_country_name(normalized):
        return False
    strong_hint = has_institution_strong_hint(normalized)
    weak_hint = has_institution_weak_hint(normalized)
    if not strong_hint and not weak_hint:
        return False
    if is_sentence_like_affiliation(normalized) and not strong_hint:
        return False
    lowered = normalized.lower()
    if re.search(r"\b(?:study|review|results|patients|performance|treatment|diagnosis)\b", lowered) and not strong_hint:
        return False
    if (
        not strong_hint
        and weak_hint
        and "," not in normalized
        and "(" not in normalized
        and len(tokens) > 10
    ):
        return False
    return True


def is_valid_venue_text(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if count_alpha_tokens(normalized) < 2:
        return False
    if any(pattern.search(normalized) for pattern in VENUE_STOP_PATTERNS):
        return False
    return True


def is_valid_dynamic_dataset_surface(surface: str) -> bool:
    normalized = normalize_text(surface)
    lowered = normalized.lower().strip(" .,:;")
    if not normalized:
        return False
    if lowered in {
        "dataset",
        "the dataset",
        "this dataset",
        "our dataset",
        "their dataset",
        "a dataset",
        "an dataset",
    }:
        return False
    tokens = [token for token in re.findall(r"[A-Za-z0-9-]+", lowered) if token]
    if len(tokens) < 2:
        return False
    if tokens[0] in GENERIC_DATASET_PREFIXES:
        return False
    head_tokens = tokens[:-1]
    if head_tokens and all(token in GENERIC_DATASET_HEADS for token in head_tokens):
        return False
    return True


def canonicalize_task_surface(surface: str) -> str:
    cleaned = normalize_text(surface).strip(" .,:;")
    if not cleaned:
        return ""
    tokens = [token for token in re.findall(r"[A-Za-z0-9-]+", cleaned)]
    if not tokens:
        return ""
    while tokens and tokens[0].lower() in TASK_STRIP_PREFIXES:
        tokens.pop(0)
    if not tokens:
        return ""
    lowered_tokens = [token.lower() for token in tokens]
    if len(lowered_tokens) < 2:
        return ""
    if re.fullmatch(r"(?:19|20)\d{2}|\d+[a-z]*", lowered_tokens[0]):
        return ""
    if any(re.fullmatch(r"(?:19|20)\d{2}|\d+[a-z]*", token) for token in lowered_tokens[:-1]):
        return ""
    if any(token in TASK_BANNED_TOKENS for token in lowered_tokens[:-1]):
        return ""
    if lowered_tokens[-1] not in TASK_CORE_SUFFIXES:
        return ""
    if lowered_tokens[0] in TASK_BANNED_TOKENS or lowered_tokens[0] in TASK_SENTENCE_TOKENS:
        return ""
    if any(token in TASK_SENTENCE_TOKENS for token in lowered_tokens[:-1]):
        return ""
    if all(token in TASK_GENERIC_MODIFIERS for token in lowered_tokens[:-1]):
        return ""
    if len(tokens[0]) == 1:
        return ""
    if len(tokens) > 5:
        return ""
    if lowered_tokens[-1] == "folding" and not any(
        token in TASK_FOLDING_CONTEXT_TOKENS for token in lowered_tokens[:-1]
    ):
        return ""
    return normalize_text(" ".join(tokens))


def canonicalize_method_surface(surface: str) -> str:
    cleaned = normalize_text(surface).strip(" .,:;")
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(?:\d+\.)+(?=\S)", "", cleaned)
    tokens = [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+/.-]*", cleaned)]
    if not tokens:
        return ""
    while tokens and (
        tokens[0].lower().strip(" .,:;()[]{}") in METHOD_STRIP_PREFIXES
        or re.fullmatch(r"(?:\d+\.)+\d*|\d+[a-z]*", tokens[0].lower().strip(" .,:;()[]{}"))
    ):
        tokens.pop(0)
    while len(tokens) > 2 and (
        tokens[0].lower().strip(" .,:;()[]{}") in METHOD_GENERIC_TOKENS
        or tokens[0].lower().strip(" .,:;()[]{}") in METHOD_SENTENCE_TOKENS
        or re.fullmatch(r"(?:\d+\.)+\d*|\d+[a-z]*", tokens[0].lower().strip(" .,:;()[]{}"))
    ):
        tokens.pop(0)
    if not tokens:
        return ""
    return normalize_text(" ".join(tokens))


def has_informative_method_token(tokens: list[str]) -> bool:
    for token in tokens:
        lowered = token.lower().strip(" .,:;()[]{}")
        if lowered in METHOD_GENERIC_TOKENS or lowered in METHOD_SENTENCE_TOKENS:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}|\d+[a-z]*", lowered):
            continue
        if lowered in METHOD_SIGNAL_TOKENS or "-" in token or "/" in token:
            return True
        if token.isupper() and 1 < len(token) <= 10:
            return True
    return False


def is_valid_dynamic_method_surface(surface: str, *, strict_tail_prefix: bool = True) -> bool:
    canonical = canonicalize_method_surface(surface)
    if not canonical:
        return False
    tokens = [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+/.-]*", canonical)]
    if not tokens:
        return False
    lowered_tokens = [token.lower().strip(" .,:;()[]{}") for token in tokens]
    if len(tokens) == 1:
        return lowered_tokens[0] in {
            "ann",
            "catboost",
            "cnn",
            "gan",
            "gnn",
            "gru",
            "lightgbm",
            "lstm",
            "mlp",
            "rnn",
            "svm",
            "xgboost",
        }
    if len(tokens) > 8:
        return False
    if any(token in METHOD_SENTENCE_TOKENS for token in lowered_tokens[:-1]):
        return False
    if all(token in METHOD_GENERIC_TOKENS for token in lowered_tokens):
        return False
    if strict_tail_prefix:
        for tail_term in METHOD_TAIL_TERMS:
            tail_tokens = [token for token in tail_term.split() if token]
            if lowered_tokens[-len(tail_tokens) :] == tail_tokens:
                prefix_tokens = tokens[: len(tokens) - len(tail_tokens)]
                if not prefix_tokens:
                    return False
                if not has_informative_method_token(prefix_tokens):
                    return False
                break
    if lowered_tokens[-1] in METHOD_HEAD_TERMS and not has_informative_method_token(tokens[:-1]):
        return False
    return has_informative_method_token(tokens)


def maybe_trim_method_head_surface(surface: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+/.-]*", surface)]
    if len(tokens) < 3:
        return surface
    if tokens[-1].lower().strip(" .,:;()[]{}") not in METHOD_HEAD_TERMS:
        return surface
    trimmed = normalize_text(" ".join(tokens[:-1]))
    trimmed_lower = trimmed.lower()
    matches_exact_surface = trimmed_lower in {term.lower() for term in METHOD_EXACT_SURFACES}
    ends_with_method_tail = any(
        trimmed_lower == tail or trimmed_lower.endswith(f" {tail}") for tail in METHOD_TAIL_TERMS
    )
    if (matches_exact_surface or ends_with_method_tail) and is_valid_dynamic_method_surface(
        trimmed,
        strict_tail_prefix=False,
    ):
        return trimmed
    return surface


def method_context_bonus(section_group: str | None, source_kind: str) -> float:
    bonus = float(METHOD_CONTEXT_BOOSTS.get(str(section_group or ""), 0.0))
    if source_kind == "figure":
        bonus += 0.01
    return bonus


def extract_match_window(text: str, start: int, end: int, radius: int = 80) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def maybe_promote_model_surface_to_method(
    canonical_name: str,
    surface: str,
    context_window: str,
    section_group: str | None,
    source_kind: str,
) -> float | None:
    bridge_key = normalize_entity_key(canonical_name)
    if bridge_key not in METHOD_MODEL_BRIDGE_KEYS:
        return None
    base = None
    if str(section_group or "") == "methods":
        base = 0.82
    elif METHOD_CONTEXT_CUE_RE.search(context_window):
        base = 0.76
    if base is None:
        return None
    if not is_valid_dynamic_method_surface(surface):
        return None
    return min(0.90, base + method_context_bonus(section_group, source_kind))


def is_generic_method_canonical(canonical_name: str) -> bool:
    return normalize_text(canonical_name).lower() in METHOD_GENERIC_CANONICALS


def has_method_positive_section_hint(section_title: str) -> bool:
    return bool(METHOD_SECTION_POSITIVE_HINT_RE.search(normalize_text(section_title)))


def has_method_negative_section_hint(section_title: str) -> bool:
    return bool(METHOD_SECTION_NEGATIVE_HINT_RE.search(normalize_text(section_title)))


def section_allows_dynamic_method_mentions(
    section_group: str | None,
    section_title: str,
    source_kind: str,
) -> bool:
    group = str(section_group or "")
    if source_kind == "figure":
        if group in METHOD_DYNAMIC_ALLOWED_FIGURE_SECTION_GROUPS:
            return True
    elif group in METHOD_DYNAMIC_ALLOWED_SECTION_GROUPS:
        return True
    if has_method_negative_section_hint(section_title):
        return False
    return has_method_positive_section_hint(section_title)


def contains_method_narrative_language(canonical_name: str) -> bool:
    normalized = normalize_text(canonical_name).lower()
    if not normalized:
        return False
    if normalized in METHOD_GENERIC_CANONICALS:
        return False
    return bool(METHOD_CANONICAL_NARRATIVE_RE.search(normalized) or re.search(r"\b(?:outperforms?|achieves?|improves?)\b", normalized))


def should_keep_method_mention(
    mention: SemanticMention,
    *,
    section_group: str | None,
    section_title: str,
    source_kind: str,
) -> bool:
    canonical = normalize_text(mention.canonical_name).lower()
    if not canonical:
        return False
    section_group = str(section_group or "")
    dynamic_sources = {
        "model_bridge",
        "rule_method_exact",
        "rule_method_head",
        "rule_method_tail",
    }
    dynamic_allowed = section_allows_dynamic_method_mentions(
        section_group=section_group,
        section_title=section_title,
        source_kind=source_kind,
    )
    negative_section = has_method_negative_section_hint(section_title)

    if mention.source in {"rule_method_head", "rule_method_tail", "model_bridge"} and not dynamic_allowed:
        return False
    if negative_section and mention.source in dynamic_sources:
        return False
    if contains_method_narrative_language(canonical):
        return False
    if section_group in {"conclusion", "discussion", "introduction", "other"}:
        if mention.source in dynamic_sources and not dynamic_allowed:
            return False
        if mention.source == "rule_extract" and is_generic_method_canonical(canonical) and not dynamic_allowed:
            return False
    return True


def find_title_index_local(items: list[dict[str, Any]], paper_title: str) -> int:
    normalized_title = normalize_text(paper_title)
    for index, item in enumerate(items):
        text = normalize_text(extract_item_text(item))
        if text and text == normalized_title:
            return index
    return -1


@dataclass(slots=True)
class _ChunkConfigAdapter:
    min_chars: int
    target_chars: int
    max_chars: int


def _as_chunk_config(config: DocumentEntityGraphConfig) -> _ChunkConfigAdapter:
    return _ChunkConfigAdapter(
        min_chars=config.min_chars,
        target_chars=config.target_chars,
        max_chars=config.max_chars,
    )


def method_is_comparison_only(text: str, surface: str) -> bool:
    """A baseline-only mention is insufficient evidence for a uses-method edge."""
    sentences = re.split(r"(?<=[.!?])\s+|\n", text.lower())
    mentions = [s for s in sentences if surface.lower() in s]
    comparison = re.compile(r"\b(?:baselines?|outperforms?|compared (?:to|with)|compare (?:against|with))\b")
    return bool(mentions) and all(comparison.search(s) for s in mentions)
