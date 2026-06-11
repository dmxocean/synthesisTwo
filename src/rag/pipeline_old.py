# -*- coding: utf-8 -*-
"""
Legacy RAG pipeline implementation for the Radio Barcelona archive

This module provides the older version of retrieval logic for the GUIRAD collection
"""

from __future__ import annotations

import os
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Routes
PATH_FILE = os.path.abspath(__file__)
PATH_RAG = os.path.dirname(PATH_FILE)
PATH_SRC = os.path.dirname(PATH_RAG)
PATH_ROOT = os.path.dirname(PATH_SRC)

from llama_index.core import Document, PromptTemplate, Settings, StorageContext, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter as QdrantFilter, FieldCondition, MatchValue

from src.core.config import COLLECTION_REAL

DEFAULT_COLLECTION = COLLECTION_REAL
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_LLM_MODEL = "mistral:7b-instruct"
DEFAULT_TOP_K = 8
DEFAULT_SIMILARITY_CUTOFF = 0.45
DEFAULT_RRF_K = 60
DEFAULT_BM25_TOP_K = 50
DEFAULT_ROUTER_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_SCROLL_BATCH_SIZE = 256

SYSTEM_PROMPT = """You are RADAR, a knowledgeable research assistant for the Radio Barcelona historical archive (GUIRAD collection, 1924-1953).
Answer in clear, natural language like an expert historian explaining findings to a colleague.
Always ground your answers in the retrieved evidence. When it matters, mention which document and page the information comes from.
Be direct about uncertainty - if the evidence is weak or incomplete, say so plainly in plain language.
Do not invent facts. If the retrieved context does not contain what was asked, explain what you did find and what is missing.
"""

QA_TEMPLATE = PromptTemplate(
    """Context information is below.
---------------------
{context_str}
---------------------
You are answering a researcher question about the Radio Barcelona historical archive.
Question: {query_str}

Instructions:
- Answer as a knowledgeable historian would, in natural flowing sentences.
- Use only the context above - do not invent facts.
- When citing evidence, mention the document and page naturally (e.g. "In document guiradbcn_a1932m05, page 3...").
- Mention forensic flags (confidence, verification status, alerts) only when directly relevant to the question.
- If evidence is conflicting or weak, say so naturally ("The evidence is ambiguous..." / "Only one record mentions...").
- If the context does not contain what was asked, say so directly ("The retrieved records don't mention...").
"""
)

ROUTER_PLANNER_TEMPLATE = PromptTemplate(
    """You are a retrieval planner for an archival RAG system.
Question: {query_str}

Allowed retrieval_mode values:
- vector
- hybrid
- metadata_lookup
- provenance_lookup

Allowed filter fields and valid values/operators:
- verification_status: eq, values in ["verified", "uncertain"]
- human_review_required: eq, values in [true, false]
- unit_type: eq, values in ["page", "region"]
- source_type: eq, string
- language: contains, string
- document_id: eq, string
- page_id: eq, string

Return ONLY valid JSON with this schema:
{{
  "retrieval_mode": "vector|hybrid|metadata_lookup|provenance_lookup",
  "filters": [
    {{"field": "verification_status", "operator": "eq", "value": "verified"}}
  ],
  "confidence": 0.0,
  "rationale": "brief explanation"
}}

Rules:
- Use hybrid when the question likely needs lexical/OCR/exact-match support.
- Use metadata_lookup for primarily field/filter-driven requests.
- Use provenance_lookup when the user explicitly asks for record_id/document_id/page_id/provenance/evidence trail.
- Use vector when the question is semantic, broad, or underspecified.
- Do not invent filters unless the question supports them.
- Confidence must be between 0 and 1.
- If uncertain, prefer vector and fewer filters.
- No markdown fences.
"""
)

JSON_QA_TEMPLATE = PromptTemplate(
    """Context information is below.
---------------------
{context_str}
---------------------
Question: {query_str}

Return ONLY valid JSON with this schema:
{{
  "answer": "natural-language answer in flowing sentences",
  "confidence": "high|medium|low",
  "needs_human_review": true,
  "applied_filters": [{{"field": "verification_status", "operator": "eq", "value": "verified"}}],
  "evidence_summary": [
    {{
      "record_id": "...",
      "document_id": "...",
      "page_id": "...",
      "verification_status": "...",
      "forensic_confidence_score": 0.0,
      "why_relevant": "..."
    }}
  ],
  "uncertainties": ["..."],
  "insufficient_evidence": false
}}

Rules:
- Answer as a knowledgeable historian, in natural flowing sentences - not a bullet list or formal report.
- Use only the context above; do not invent facts.
- Cite document_id/page_id naturally within the answer text when they strengthen the evidence.
- Mention forensic flags only when they are directly relevant to the question.
- If evidence is conflicting or weak, express this naturally in the answer and set confidence to low.
- If no relevant evidence is in the context, set insufficient_evidence to true and say so naturally in the answer.
- Do not output Markdown fences.
"""
)


def safe_parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


@dataclass
class PipelineConfig:
    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_path: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    collection_name: str = DEFAULT_COLLECTION
    embed_model: str = DEFAULT_EMBED_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    top_k: int = DEFAULT_TOP_K
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF
    ollama_request_timeout: float = 120.0
    qdrant_check_compatibility: bool = False
    rrf_k: int = DEFAULT_RRF_K
    bm25_top_k: int = DEFAULT_BM25_TOP_K
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    router_confidence_threshold: float = DEFAULT_ROUTER_CONFIDENCE_THRESHOLD
    scroll_batch_size: int = DEFAULT_SCROLL_BATCH_SIZE


@dataclass
class QueryPlan:
    retrieval_mode: str
    metadata_filters: Optional[MetadataFilters]
    rationale: str
    inferred_constraints: List[Dict[str, Any]]
    route_source: str = "rules"
    confidence: Optional[float] = None
    fallback_applied: bool = False
    raw_planner_output: Optional[Dict[str, Any]] = None


class RadioArchiveMetadataReader:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def load_data(self) -> List[Document]:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        suffix = os.path.splitext(self.input_path)[1].lower()
        if suffix == ".jsonl":
            records = self._read_jsonl(self.input_path)
        elif suffix == ".json":
            records = self._read_json(self.input_path)
        else:
            raise ValueError("Input must be .jsonl or .json")
        return [self._record_to_document(r) for r in records]

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _read_json(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be a list of records")

    def _record_to_document(self, record: Dict[str, Any]) -> Document:
        retrieval_text = self._build_retrieval_text(record)
        metadata = self._extract_metadata(record)
        return Document(text=retrieval_text, metadata=metadata)

    @staticmethod
    def _build_retrieval_text(record: Dict[str, Any]) -> str:
        title = record.get("title", "")
        retrieval_text = record.get("retrieval_text", "")
        summary = (record.get("summary") or {}).get("short_description", "")
        ocr_text = (record.get("ocr") or {}).get("full_text", "")
        keywords = (record.get("summary") or {}).get("keywords", [])
        entities = [e.get("text", "") for e in record.get("entities", []) if e.get("text")]
        mark_types = [m.get("mark_type", "") for m in record.get("visual_marks", []) if m.get("mark_type")]
        alerts = (record.get("forensic_flags") or {}).get("alerts", [])
        topics = (record.get("ai_metadata") or {}).get("detected_topics", [])
        linked_spans = [s.get("text", "") for s in record.get("linked_text_spans", []) if s.get("text")]
        provenance = record.get("provenance") or {}
        parts = [
            f"Title: {title}",
            f"Retrieval text: {retrieval_text}",
            f"Summary: {summary}",
            f"OCR text: {ocr_text}",
            f"Source image path: {provenance.get('source_image_path', '')}",
        ]
        if keywords:
            parts.append("Summary keywords: " + ", ".join(keywords))
        if topics:
            parts.append("Detected topics: " + ", ".join(topics))
        if entities:
            parts.append("Entities: " + ", ".join(entities))
        if mark_types:
            parts.append("Visual marks: " + ", ".join(mark_types))
        if linked_spans:
            parts.append("Linked text spans: " + " | ".join(linked_spans))
        if alerts:
            parts.append("Forensic alerts: " + ", ".join(alerts))
        return "\n".join(p for p in parts if p.strip())

    @staticmethod
    def _extract_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        forensic = record.get("forensic_flags") or {}
        archival = record.get("archival_metadata") or {}
        physical = record.get("physical_metadata") or {}
        ocr = record.get("ocr") or {}
        provenance = record.get("provenance") or {}
        metadata = {
            "record_id": record.get("record_id"),
            "document_id": record.get("document_id"),
            "page_id": record.get("page_id"),
            "region_id": record.get("region_id"),
            "unit_type": record.get("unit_type"),
            "source_type": record.get("source_type"),
            "collection_id": record.get("collection_id"),
            "collection_name": record.get("collection_name"),
            "title": record.get("title"),
            "date_created": record.get("date_created"),
            "language": ",".join(record.get("language", [])) if isinstance(record.get("language"), list) else record.get("language"),
            "creator": ",".join(record.get("creator", [])) if isinstance(record.get("creator"), list) else record.get("creator"),
            "contributors": ",".join(record.get("contributors", [])) if isinstance(record.get("contributors"), list) else record.get("contributors"),
            "repository": archival.get("repository"),
            "institution": archival.get("institution"),
            "fonds": archival.get("fonds"),
            "series": archival.get("series"),
            "call_number": archival.get("call_number"),
            "file_name": physical.get("file_name"),
            "source_image_path": provenance.get("source_image_path"),
            "ocr_confidence_score": ocr.get("confidence_score"),
            "forensic_confidence_score": forensic.get("confidence_score"),
            "verification_status": forensic.get("verification_status"),
            "uncertainty_state": forensic.get("uncertainty_state"),
            "human_review_required": "true" if forensic.get("human_review_required") is True else "false" if forensic.get("human_review_required") is False else None,
            "alerts": ",".join(forensic.get("alerts", [])),
            "entity_texts": ",".join([e.get("text", "") for e in record.get("entities", []) if e.get("text")]),
            "mark_types": ",".join([m.get("mark_type", "") for m in record.get("visual_marks", []) if m.get("mark_type")]),
            "topic_texts": ",".join((record.get("ai_metadata") or {}).get("detected_topics", [])),
        }
        return metadata


class BM25Retriever(BaseRetriever):
    def __init__(self, nodes: List[TextNode], top_k: int = DEFAULT_TOP_K, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B):
        super().__init__()
        self.nodes = nodes
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.tokenized_nodes = [self._tokenize(node.text) for node in nodes]
        self.doc_freqs = self._build_doc_freqs(self.tokenized_nodes)
        self.avg_doc_len = sum(len(toks) for toks in self.tokenized_nodes) / max(len(self.tokenized_nodes), 1)
        self.doc_lens = [len(toks) for toks in self.tokenized_nodes]
        self.N = len(self.tokenized_nodes)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _build_doc_freqs(tokenized_docs: List[List[str]]) -> Dict[str, int]:
        doc_freqs: Dict[str, int] = defaultdict(int)
        for toks in tokenized_docs:
            for token in set(toks):
                doc_freqs[token] += 1
        return dict(doc_freqs)

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens: List[str], doc_index: int) -> float:
        tf = Counter(self.tokenized_nodes[doc_index])
        doc_len = self.doc_lens[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            numerator = tf[term] * (self.k1 + 1)
            denominator = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-9))
            score += self._idf(term) * (numerator / denominator)
        return score

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_tokens = self._tokenize(query_bundle.query_str)
        scored: List[Tuple[float, TextNode]] = []
        for idx, node in enumerate(self.nodes):
            score = self._score(query_tokens, idx)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [NodeWithScore(node=node, score=score) for score, node in scored[: self.top_k]]


class RRFFusionRetriever(BaseRetriever):
    def __init__(self, vector_retriever: BaseRetriever, bm25_retriever: BaseRetriever, top_k: int = DEFAULT_TOP_K, rrf_k: int = DEFAULT_RRF_K):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        fused_scores: Dict[str, float] = defaultdict(float)
        node_lookup: Dict[str, TextNode] = {}

        for rank, item in enumerate(vector_nodes, start=1):
            key = item.node.node_id or item.node.hash
            fused_scores[key] += 1.0 / (self.rrf_k + rank)
            node_lookup[key] = item.node

        for rank, item in enumerate(bm25_nodes, start=1):
            key = item.node.node_id or item.node.hash
            fused_scores[key] += 1.0 / (self.rrf_k + rank)
            node_lookup[key] = item.node

        fused = [NodeWithScore(node=node_lookup[key], score=score) for key, score in fused_scores.items()]
        fused.sort(key=lambda item: item.score or 0.0, reverse=True)
        return fused[: self.top_k]


class ProvenanceFormatter:
    @staticmethod
    def format_source_line(index: int, node_with_score: NodeWithScore) -> str:
        meta = node_with_score.node.metadata
        score = node_with_score.score if node_with_score.score is not None else 0.0
        return (
            f"[{index}] score={score:.4f} "
            f"record_id={meta.get('record_id')} document_id={meta.get('document_id')} "
            f"page_id={meta.get('page_id')} verification_status={meta.get('verification_status')} "
            f"human_review_required={meta.get('human_review_required')} "
            f"forensic_confidence_score={meta.get('forensic_confidence_score')} alerts={meta.get('alerts')}"
        )

    @staticmethod
    def format_json_sources(nodes: List[NodeWithScore], limit: int = 5) -> List[Dict[str, Any]]:
        rows = []
        for item in nodes[:limit]:
            meta = item.node.metadata
            rows.append(
                {
                    "score": item.score,
                    "record_id": meta.get("record_id"),
                    "document_id": meta.get("document_id"),
                    "page_id": meta.get("page_id"),
                    "verification_status": meta.get("verification_status"),
                    "human_review_required": meta.get("human_review_required"),
                    "forensic_confidence_score": meta.get("forensic_confidence_score"),
                    "alerts": meta.get("alerts"),
                    "source_image_path": meta.get("source_image_path"),
                }
            )
        return rows


class QueryClassifier:
    ALLOWED_FILTERS: Dict[str, Dict[str, Any]] = {
        "verification_status": {"operators": {"eq"}, "values": {"verified", "uncertain"}},
        "human_review_required": {"operators": {"eq"}, "values": {"true", "false"}},
        "unit_type": {"operators": {"eq"}, "values": {"page", "region"}},
        "source_type": {"operators": {"eq"}, "values": None},
        "language": {"operators": {"contains"}, "values": None},
        "document_id": {"operators": {"eq"}, "values": None},
        "page_id": {"operators": {"eq"}, "values": None},
    }

    def __init__(self, llm: Ollama, confidence_threshold: float = DEFAULT_ROUTER_CONFIDENCE_THRESHOLD):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    def classify(self, question: str, explicit_filters: Optional[MetadataFilters] = None) -> QueryPlan:
        rule_plan = self._classify_rules(question, explicit_filters=explicit_filters)
        if rule_plan.retrieval_mode != "vector" or rule_plan.inferred_constraints:
            return rule_plan
        planner_raw = self._run_llm_planner(question)
        return self._build_plan_from_planner(planner_raw, explicit_filters=explicit_filters)

    def _has_content_terms(self, text: str) -> bool:
        content_patterns = [
            r"\babout\b",
            r"\bmention(?:s|ed|ing)?\b",
            r"\bcontains?\b",
            r"\bcontaining\b",
            r"\bwith\b",
            r"\brelated to\b",
            r"\btopic\b",
            r"\btopics\b",
            r"\bkeyword\b",
            r"\bkeywords\b",
            r"\bocr\b",
            r"\btranscription\b",
            r"\bguerra\b",
            r"\bfrente\b",
            r"\bbolet[ií]n\b",
            r"\bcensorship\b",
            r"\bintervenci[oó]n\b",
            r"\brep[úu]blica\b",
            r"\bbarrio\b",
            r"\bf[úu]tbol\b",
            r"\bm[úu]sica\b",
        ]
        return any(re.search(pattern, text) for pattern in content_patterns)

    def _classify_rules(self, question: str, explicit_filters: Optional[MetadataFilters] = None) -> QueryPlan:
        text = question.lower()
        inferred: List[MetadataFilter] = []
        inferred_constraints: List[Dict[str, Any]] = []
        retrieval_mode = "vector"
        rationale = "Default semantic retrieval"

        exact_terms = ["exact", "literal", "verbatim", "wording", "phrase", "contains", "mention", "keyword", "transcription", "ocr", "spelling"]
        provenance_terms = ["provenance", "evidence trail", "sources", "record_id", "document_id", "page_id", "which records support"]
        metadata_terms = ["filter", "verified", "uncertain", "human review", "review required", "page", "region", "document_id", "page_id"]
        listing_terms = ["show", "list", "find", "give", "which", "records", "pages"]

        has_metadata_terms = any(term in text for term in metadata_terms)
        has_listing_terms = any(term in text for term in listing_terms)
        has_content_terms = self._has_content_terms(text)

        if any(term in text for term in exact_terms):
            retrieval_mode = "hybrid"
            rationale = "Question appears lexical or OCR-sensitive, so hybrid retrieval is preferred by the rule-based router"
        elif any(term in text for term in provenance_terms):
            retrieval_mode = "provenance_lookup"
            rationale = "Question explicitly asks for provenance or source identifiers"
        elif has_metadata_terms and has_listing_terms and not has_content_terms:
            retrieval_mode = "metadata_lookup"
            rationale = "Question is primarily metadata/filter-driven, so deterministic lookup is preferred"
        elif has_metadata_terms and has_content_terms:
            retrieval_mode = "hybrid"
            rationale = "Question mixes metadata constraints with topical content terms, so hybrid retrieval is preferred"

        if "verified" in text:
            inferred.append(MetadataFilter(key="verification_status", value="verified", operator=FilterOperator.EQ))
            inferred_constraints.append({"field": "verification_status", "operator": "eq", "value": "verified", "source": "rules"})
        if any(term in text for term in ["uncertain", "doubtful", "unverified"]):
            inferred.append(MetadataFilter(key="verification_status", value="uncertain", operator=FilterOperator.EQ))
            inferred_constraints.append({"field": "verification_status", "operator": "eq", "value": "uncertain", "source": "rules"})
        if any(term in text for term in ["human review", "review required"]):
            inferred.append(MetadataFilter(key="human_review_required", value="true", operator=FilterOperator.EQ))
            inferred_constraints.append({"field": "human_review_required", "operator": "eq", "value": "true", "source": "rules"})
        if "region" in text:
            inferred.append(MetadataFilter(key="unit_type", value="region", operator=FilterOperator.EQ))
            inferred_constraints.append({"field": "unit_type", "operator": "eq", "value": "region", "source": "rules"})
        elif "page" in text:
            inferred.append(MetadataFilter(key="unit_type", value="page", operator=FilterOperator.EQ))
            inferred_constraints.append({"field": "unit_type", "operator": "eq", "value": "page", "source": "rules"})

        merged_filters = merge_metadata_filters(explicit_filters, inferred)
        return QueryPlan(
            retrieval_mode=retrieval_mode,
            metadata_filters=merged_filters,
            rationale=rationale,
            inferred_constraints=inferred_constraints,
            route_source="rules",
            confidence=1.0 if (retrieval_mode != "vector" or inferred_constraints) else None,
        )

    def _run_llm_planner(self, question: str) -> Dict[str, Any]:
        prompt = ROUTER_PLANNER_TEMPLATE.format(query_str=question)
        raw = self.llm.complete(prompt)
        return safe_parse_json_response(str(raw))

    def _build_plan_from_planner(self, planner_raw: Dict[str, Any], explicit_filters: Optional[MetadataFilters] = None) -> QueryPlan:
        retrieval_mode = planner_raw.get("retrieval_mode", "vector")
        if retrieval_mode not in {"vector", "hybrid", "metadata_lookup", "provenance_lookup"}:
            retrieval_mode = "vector"

        confidence = planner_raw.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        validated_filters, inferred_constraints = self._validate_planner_filters(planner_raw.get("filters", []))
        rationale = planner_raw.get("rationale", "LLM planner route")

        fallback_applied = False
        if confidence < self.confidence_threshold:
            retrieval_mode = "vector"
            validated_filters = []
            inferred_constraints = []
            rationale = f"LLM planner confidence below threshold ({confidence:.2f} < {self.confidence_threshold:.2f}); falling back to baseline vector retrieval without inferred filters"
            fallback_applied = True

        merged_filters = merge_metadata_filters(explicit_filters, validated_filters)
        return QueryPlan(
            retrieval_mode=retrieval_mode,
            metadata_filters=merged_filters,
            rationale=rationale,
            inferred_constraints=inferred_constraints,
            route_source="llm_planner" if not fallback_applied else "fallback",
            confidence=confidence,
            fallback_applied=fallback_applied,
            raw_planner_output=planner_raw,
        )

    def _validate_planner_filters(self, raw_filters: Any) -> Tuple[List[MetadataFilter], List[Dict[str, Any]]]:
        if not isinstance(raw_filters, list):
            return [], []
        filters: List[MetadataFilter] = []
        constraints: List[Dict[str, Any]] = []
        for item in raw_filters:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            operator = item.get("operator")
            value = item.get("value")
            spec = self.ALLOWED_FILTERS.get(field)
            if not spec or operator not in spec["operators"]:
                continue
            allowed_values = spec["values"]
            if field == "human_review_required":
                if isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered not in {"true", "false"}:
                        continue
                    value = lowered
                else:
                    continue
            if allowed_values is not None and value not in allowed_values:
                continue
            if field in {"source_type", "language", "document_id", "page_id"} and not isinstance(value, str):
                continue
            filters.append(MetadataFilter(key=field, value=value, operator=self._to_filter_operator(operator)))
            constraints.append({"field": field, "operator": operator, "value": value, "source": "llm_planner"})
        return filters, constraints

    @staticmethod
    def _to_filter_operator(operator: str) -> FilterOperator:
        return {"eq": FilterOperator.EQ, "contains": FilterOperator.CONTAINS}[operator]


class RadioBarcelonaRAG:
    def __init__(self, config: PipelineConfig):
        self.config = config
        if config.qdrant_path:
            self.qdrant_client = QdrantClient(path=config.qdrant_path)
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=config.collection_name,
                index_doc_id=False,  # Payload indexes have no effect in embedded mode
            )
        else:
            self.qdrant_client = QdrantClient(url=config.qdrant_url, check_compatibility=config.qdrant_check_compatibility)
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=config.collection_name,
            )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.embed_model = HuggingFaceEmbedding(model_name=config.embed_model)
        self.llm = Ollama(model=config.llm_model, base_url=config.ollama_url, request_timeout=config.ollama_request_timeout)
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        self.query_classifier = QueryClassifier(self.llm, confidence_threshold=config.router_confidence_threshold)
        self._bm25_cache: Dict[str, List[TextNode]] = {}

    def build_index(self, documents: List[Document]) -> VectorStoreIndex:
        return VectorStoreIndex.from_documents(documents, storage_context=self.storage_context, show_progress=True)

    def load_index(self) -> VectorStoreIndex:
        return VectorStoreIndex.from_vector_store(vector_store=self.vector_store, embed_model=self.embed_model)

    def _metadata_match(self, node: TextNode, metadata_filters: Optional[MetadataFilters]) -> bool:
        if not metadata_filters:
            return True
        for flt in metadata_filters.filters:
            value = node.metadata.get(flt.key)
            if flt.operator == FilterOperator.EQ and value != flt.value:
                return False
            if flt.operator == FilterOperator.CONTAINS:
                value_str = "" if value is None else str(value)
                if str(flt.value) not in value_str:
                    return False
        return True

    def _filters_cache_key(self, metadata_filters: Optional[MetadataFilters]) -> str:
        if not metadata_filters:
            return "__all__"
        items = []
        for flt in metadata_filters.filters:
            items.append((flt.key, str(flt.operator), json.dumps(flt.value, sort_keys=True, ensure_ascii=False)))
        items.sort()
        return json.dumps(items, ensure_ascii=False)

    def _metadata_filters_to_qdrant_filter(self, metadata_filters: Optional[MetadataFilters]) -> Optional[QdrantFilter]:
        if not metadata_filters:
            return None
        must_conditions = []
        for flt in metadata_filters.filters:
            if flt.operator != FilterOperator.EQ:
                continue
            must_conditions.append(FieldCondition(key=f"metadata.{flt.key}", match=MatchValue(value=flt.value)))
        if not must_conditions:
            return None
        return QdrantFilter(must=must_conditions)

    def _point_to_text_node(self, point: Any) -> Optional[TextNode]:
        payload = getattr(point, "payload", None) or {}
        metadata = payload.get("metadata") or {}
        text = payload.get("text") or payload.get("document") or payload.get("page_content") or ""
        if not text:
            return None
        node_id = str(getattr(point, "id", None) or metadata.get("record_id") or "")
        return TextNode(text=text, metadata=metadata, id_=node_id or None)

    def _scroll_all_candidate_nodes(self, metadata_filters: Optional[MetadataFilters] = None) -> List[TextNode]:
        cache_key = self._filters_cache_key(metadata_filters)
        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key]

        qdrant_filter = self._metadata_filters_to_qdrant_filter(metadata_filters)
        nodes: List[TextNode] = []
        offset = None
        while True:
            points, offset = self.qdrant_client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=qdrant_filter,
                limit=self.config.scroll_batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                node = self._point_to_text_node(point)
                if node is None:
                    continue
                if not self._metadata_match(node, metadata_filters):
                    continue
                nodes.append(node)
            if offset is None:
                break

        self._bm25_cache[cache_key] = nodes
        return nodes

    def make_retriever(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
        metadata_filters: Optional[MetadataFilters] = None,
        retrieval_mode: str = "vector",
    ) -> BaseRetriever:
        index = self.load_index()
        final_top_k = similarity_top_k or self.config.top_k
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=final_top_k,
            filters=metadata_filters,
        )

        if retrieval_mode in {"vector", "metadata_lookup", "provenance_lookup"}:
            return vector_retriever
        bm25_nodes = self._scroll_all_candidate_nodes(metadata_filters=metadata_filters)
        bm25_retriever = BM25Retriever(
            nodes=bm25_nodes,
            top_k=max(final_top_k, self.config.bm25_top_k),
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        if retrieval_mode == "bm25":          # lexical-only (used by the retrieval ablation)
            return bm25_retriever
        return RRFFusionRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=final_top_k,
            rrf_k=self.config.rrf_k,
        )

    def make_query_engine(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
        metadata_filters: Optional[MetadataFilters] = None,
        retrieval_mode: str = "vector",
        json_mode: bool = False,
    ):
        retriever = self.make_retriever(
            question=question,
            similarity_top_k=similarity_top_k,
            metadata_filters=metadata_filters,
            retrieval_mode=retrieval_mode,
        )
        postprocessors = []
        if retrieval_mode == "vector":
            postprocessors.append(SimilarityPostprocessor(similarity_cutoff=self.config.similarity_cutoff))
        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            text_qa_template=JSON_QA_TEMPLATE if json_mode else QA_TEMPLATE,
            system_prompt=SYSTEM_PROMPT,
        )

    def retrieve_only(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
        metadata_filters: Optional[MetadataFilters] = None,
        retrieval_mode: str = "vector",
    ) -> List[NodeWithScore]:
        if retrieval_mode == "metadata_lookup":
            return self.retrieve_metadata_only(metadata_filters=metadata_filters, limit=similarity_top_k or self.config.top_k)
        retriever = self.make_retriever(
            question=question,
            similarity_top_k=similarity_top_k,
            metadata_filters=metadata_filters,
            retrieval_mode=retrieval_mode,
        )
        nodes = retriever.retrieve(question)
        if retrieval_mode == "vector":
            cutoff = self.config.similarity_cutoff
            nodes = [n for n in nodes if (n.score or 0.0) >= cutoff]
        return nodes

    def retrieve_metadata_only(self, metadata_filters: Optional[MetadataFilters], limit: int) -> List[NodeWithScore]:
        nodes = self._scroll_all_candidate_nodes(metadata_filters=metadata_filters)
        rows = [NodeWithScore(node=node, score=1.0) for node in nodes[:limit]]
        return rows

    def query_with_router(self, question: str, explicit_filters: Optional[MetadataFilters] = None, json_mode: bool = False) -> Dict[str, Any]:
        plan = self.query_classifier.classify(question, explicit_filters=explicit_filters)

        if plan.retrieval_mode == "metadata_lookup":
            source_nodes = self.retrieve_metadata_only(plan.metadata_filters, self.config.top_k)
            response_text = build_metadata_lookup_response(question, source_nodes)
            response = response_text if not json_mode else json.dumps(
                {
                    "answer": response_text,
                    "confidence": "high" if source_nodes else "low",
                    "needs_human_review": any(str(n.node.metadata.get("human_review_required")).lower() == "true" for n in source_nodes),
                    "applied_filters": plan.inferred_constraints,
                    "evidence_summary": ProvenanceFormatter.format_json_sources(source_nodes),
                    "uncertainties": [] if source_nodes else ["No records matched the metadata filters."],
                    "insufficient_evidence": not bool(source_nodes),
                },
                ensure_ascii=False,
            )
            return {"plan": plan, "response": response, "source_nodes": source_nodes}

        if plan.retrieval_mode == "provenance_lookup":
            source_nodes = self.retrieve_only(question, metadata_filters=plan.metadata_filters, retrieval_mode="hybrid")
            response_text = build_provenance_response(source_nodes)
            response = response_text if not json_mode else json.dumps(
                {
                    "answer": response_text,
                    "confidence": "high" if source_nodes else "low",
                    "needs_human_review": any(str(n.node.metadata.get("human_review_required")).lower() == "true" for n in source_nodes),
                    "applied_filters": plan.inferred_constraints,
                    "evidence_summary": ProvenanceFormatter.format_json_sources(source_nodes),
                    "uncertainties": [] if source_nodes else ["No provenance-bearing results were retrieved."],
                    "insufficient_evidence": not bool(source_nodes),
                },
                ensure_ascii=False,
            )
            return {"plan": plan, "response": response, "source_nodes": source_nodes}

        query_engine = self.make_query_engine(
            question=question,
            metadata_filters=plan.metadata_filters,
            retrieval_mode=plan.retrieval_mode,
            json_mode=json_mode,
        )
        response = query_engine.query(question)
        source_nodes = getattr(response, "source_nodes", [])
        return {"plan": plan, "response": response, "source_nodes": source_nodes}


def build_metadata_lookup_response(question: str, nodes: List[NodeWithScore]) -> str:
    if not nodes:
        return f"No records matched the requested metadata constraints for: {question}"
    lines = [f"Found {len(nodes)} matching records for: {question}"]
    for item in nodes:
        meta = item.node.metadata
        lines.append(
            f"- record_id={meta.get('record_id')}, document_id={meta.get('document_id')}, page_id={meta.get('page_id')}, "
            f"verification_status={meta.get('verification_status')}, human_review_required={meta.get('human_review_required')}"
        )
    return "\n".join(lines)


def build_provenance_response(nodes: List[NodeWithScore]) -> str:
    if not nodes:
        return "No provenance-bearing results were retrieved"
    lines = ["Provenance summary:"]
    for idx, item in enumerate(nodes, start=1):
        lines.append(ProvenanceFormatter.format_source_line(idx, item))
    return "\n".join(lines)


def merge_metadata_filters(explicit_filters: Optional[MetadataFilters], inferred_filters: List[MetadataFilter]) -> Optional[MetadataFilters]:
    combined: List[MetadataFilter] = []
    if explicit_filters:
        combined.extend(explicit_filters.filters)
    combined.extend(inferred_filters)
    if not combined:
        return None
    return MetadataFilters(filters=combined, condition=FilterCondition.AND)
