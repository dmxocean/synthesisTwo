# -*- coding: utf-8 -*-
"""
Radio Barcelona routed RAG pipeline executable workflow

This script provides a command-line interface for the Radio Barcelona RAG system. It supports building the index from metadata, querying with LLM synthesis, retrieving nodes without synthesis, and performing retrieval evaluation against a gold set. The pipeline integrates vector and lexical search with a routing layer to optimize retrieval based on query intent
"""

import os
import re
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition, FilterOperator

from src.core import metrics as RM
from src.core.config import get_artifact_dir, PATH_STORAGE_QDRANT, OLLAMA_HOST

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_KS = [1, 3, 5, 10]
ABLATION_MODES = ["vector", "bm25", "hybrid"]

from src.rag.pipeline import (
    RadioBarcelonaRAG, PipelineConfig, QueryPlan, RadioArchiveMetadataReader, ProvenanceFormatter,
    DEFAULT_QDRANT_URL, DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL, DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_CUTOFF, DEFAULT_RRF_K, DEFAULT_BM25_TOP_K, DEFAULT_BM25_K1, DEFAULT_BM25_B,
    DEFAULT_ROUTER_CONFIDENCE_THRESHOLD, DEFAULT_SCROLL_BATCH_SIZE,
)


def build_filters_from_args(args: argparse.Namespace) -> Optional[MetadataFilters]:
    """
    Construct metadata filters from command-line arguments
    """
    filters: List[MetadataFilter] = []
    if getattr(args, "verification_status", None):
        filters.append(MetadataFilter(key="verification_status", value=args.verification_status, operator=FilterOperator.EQ))
    if getattr(args, "human_review_required", None) is not None:
        hr_value = "true" if args.human_review_required else "false"
        filters.append(MetadataFilter(key="human_review_required", value=hr_value, operator=FilterOperator.EQ))
    if getattr(args, "source_type", None):
        filters.append(MetadataFilter(key="source_type", value=args.source_type, operator=FilterOperator.EQ))
    if getattr(args, "unit_type", None):
        filters.append(MetadataFilter(key="unit_type", value=args.unit_type, operator=FilterOperator.EQ))
    if getattr(args, "language", None):
        filters.append(MetadataFilter(key="language", value=args.language, operator=FilterOperator.CONTAINS))
    if getattr(args, "document_id", None):
        filters.append(MetadataFilter(key="document_id", value=args.document_id, operator=FilterOperator.EQ))
    if getattr(args, "page_id", None):
        filters.append(MetadataFilter(key="page_id", value=args.page_id, operator=FilterOperator.EQ))
    if not filters:
        return None
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def load_eval_questions(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation questions from a JSON file
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Evaluation file must be a JSON list")
    return data


def safe_parse_json_response(text: str) -> Dict[str, Any]:
    """
    Attempt to parse JSON from a string that might contain additional text
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def cmd_build(rag: RadioBarcelonaRAG, args: argparse.Namespace) -> None:
    """
    Load and index documents into the vector store
    """
    reader = RadioArchiveMetadataReader(args.input)
    documents = reader.load_data()
    rag.build_index(documents)
    print(f"Indexed {len(documents)} documents into collection '{rag.config.collection_name}'")


def cmd_query(rag: RadioBarcelonaRAG, args: argparse.Namespace) -> None:
    """
    Execute a RAG query and print the result with provenance
    """
    explicit_filters = build_filters_from_args(args)
    result = rag.query_with_router(args.question, explicit_filters=explicit_filters, json_mode=args.json_output)
    response = result["response"]
    plan: QueryPlan = result["plan"]
    source_nodes = result["source_nodes"]

    print("\nQUERY PLAN\n")  # Log query execution strategy
    print(json.dumps({
        "retrieval_mode": plan.retrieval_mode,
        "rationale": plan.rationale,
        "inferred_constraints": plan.inferred_constraints,
        "route_source": plan.route_source,
        "confidence": plan.confidence,
        "fallback_applied": plan.fallback_applied,
    }, indent=2, ensure_ascii=False))

    print("\nRESPONSE\n")  # Log generated answer
    if args.json_output:
        payload = safe_parse_json_response(str(response))
        payload["provenance"] = ProvenanceFormatter.format_json_sources(source_nodes)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(str(response))

    print("\nSOURCES\n")  # Log retrieval evidence
    for i, sn in enumerate(source_nodes, start=1):
        print(ProvenanceFormatter.format_source_line(i, sn))


def cmd_retrieve(rag: RadioBarcelonaRAG, args: argparse.Namespace) -> None:
    """
    Retrieve nodes for a query without generating a response
    """
    explicit_filters = build_filters_from_args(args)
    plan = rag.query_classifier.classify(args.question, explicit_filters=explicit_filters)
    nodes = rag.retrieve_only(args.question, metadata_filters=plan.metadata_filters, retrieval_mode=plan.retrieval_mode)

    print("\nQUERY PLAN\n")  # Log classified intent
    print(json.dumps({
        "retrieval_mode": plan.retrieval_mode,
        "rationale": plan.rationale,
        "inferred_constraints": plan.inferred_constraints,
        "route_source": plan.route_source,
        "confidence": plan.confidence,
        "fallback_applied": plan.fallback_applied,
    }, indent=2, ensure_ascii=False))

    print("\nRETRIEVED NODES\n")  # Log retrieved text segments
    for i, node in enumerate(nodes, start=1):
        print(ProvenanceFormatter.format_source_line(i, node))
        print(f" text={node.node.text[:500]}\n")


def _retrieved_ids(rag: RadioBarcelonaRAG, question: str, retrieval_mode: str,
                   metadata_filters: Optional[MetadataFilters] = None) -> List[str]:
    """
    Helper to get record IDs for a query
    """
    nodes = rag.retrieve_only(question, retrieval_mode=retrieval_mode, metadata_filters=metadata_filters)
    return [n.node.metadata.get("record_id") for n in nodes]


def _question_metrics(retrieved: List[str], expected: List[str], ks: List[int]) -> Dict[str, Any]:
    """
    Calculate hit-rate, recall, nDCG and MRR for a single question
    """
    return {
        "hit_rate": {str(k): RM.hit_at_k(retrieved, expected, k) for k in ks},
        "recall":   {str(k): RM.recall_at_k(retrieved, expected, k) for k in ks},
        "ndcg":     {str(k): RM.ndcg_at_k(retrieved, expected, k) for k in ks},
        "mrr":      RM.mean_reciprocal_rank(retrieved, expected),
    }


def _mean_metrics(per_question: List[Dict[str, Any]], ks: List[int]) -> Dict[str, Any]:
    """
    Aggregate per-question metrics into a mean summary
    """
    n = max(1, len(per_question))
    out = {"mrr": sum(q["mrr"] for q in per_question) / n}
    for name in ("hit_rate", "recall", "ndcg"):
        out[name] = {str(k): sum(q[name][str(k)] for q in per_question) / n for k in ks}
    return out


def cmd_eval(rag: RadioBarcelonaRAG, args: argparse.Namespace) -> None:
    """
    Run retrieval evaluation and ablation studies
    """
    eval_rows = load_eval_questions(args.questions)
    ks = [k for k in EVAL_KS if k <= args.top_k] or [args.top_k]

    router_q: List[Dict[str, Any]] = []
    ablation_q: Dict[str, List[Dict[str, Any]]] = {m: [] for m in ABLATION_MODES}
    mode_counts: Dict[str, int] = defaultdict(int)
    route_hits: Dict[str, List[float]] = defaultdict(list)
    per_question_log: List[Dict[str, Any]] = []

    for row in eval_rows:
        question = row["question"]
        expected = list(row.get("expected_record_ids", []))
        plan = rag.query_classifier.classify(question)
        mode_counts[plan.retrieval_mode] += 1

        retrieved = _retrieved_ids(rag, question, plan.retrieval_mode, plan.metadata_filters)
        qm = _question_metrics(retrieved, expected, ks)
        router_q.append(qm)
        route_hits[plan.retrieval_mode].append(qm["hit_rate"][str(max(ks))])

        if not args.no_ablation:
            for mode in ABLATION_MODES:
                ablation_q[mode].append(_question_metrics(
                    _retrieved_ids(rag, question, mode), expected, ks))

        per_question_log.append({
            "question": question, "retrieval_mode": plan.retrieval_mode,
            "expected_record_ids": expected, "retrieved_record_ids": retrieved,
            "hit": bool(qm["hit_rate"][str(max(ks))]),
        })
        print(json.dumps(per_question_log[-1], ensure_ascii=False))

    out = {
        "stage": "rag",
        "n_questions": len(eval_rows),
        "ks": ks,
        "router": {
            **_mean_metrics(router_q, ks),
            "route_distribution": dict(mode_counts),
            "route_hit_rate": {m: (sum(v) / len(v) if v else 0.0) for m, v in route_hits.items()},
        },
        "ablation": ({m: _mean_metrics(ablation_q[m], ks) for m in ABLATION_MODES}
                     if not args.no_ablation else {}),
        "per_question": per_question_log,
    }
    out_path = os.path.join(get_artifact_dir("rag", "qwen_pipeline", "metrics"), "rag.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nEVAL SUMMARY")  # Log evaluation results
    print(json.dumps({"router": out["router"], "ablation": out["ablation"]}, indent=2, ensure_ascii=False))
    print(f"Wrote metrics to {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the RAG pipeline
    """
    parser = argparse.ArgumentParser(description="Radio Barcelona final routed RAG pipeline")
    parser.add_argument("--qdrant-url", default=None,
                        help="Qdrant server URL; when set, switches to server mode and overrides the embedded --qdrant-path")
    parser.add_argument("--qdrant-path", default=PATH_STORAGE_QDRANT, help="Path to local Qdrant storage (embedded mode; the default)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--similarity-cutoff", type=float, default=DEFAULT_SIMILARITY_CUTOFF)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--bm25-k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--bm25-b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument("--router-confidence-threshold", type=float, default=DEFAULT_ROUTER_CONFIDENCE_THRESHOLD)
    parser.add_argument("--scroll-batch-size", type=int, default=DEFAULT_SCROLL_BATCH_SIZE)

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build", help="Load JSONL/JSON metadata and index it in Qdrant")
    p_build.add_argument("--input", required=True, help="Path to JSONL or JSON metadata file")

    p_query = subparsers.add_parser("query", help="Run a routed RAG query")
    p_query.add_argument("--question", required=True)
    p_query.add_argument("--json-output", action="store_true", help="Return structured JSON answer")
    p_query.add_argument("--verification-status", choices=["verified", "uncertain"])
    p_query.add_argument("--human-review-required", type=lambda x: x.lower() == "true", default=None)
    p_query.add_argument("--source-type")
    p_query.add_argument("--unit-type", choices=["page", "region"])
    p_query.add_argument("--language")
    p_query.add_argument("--document-id")
    p_query.add_argument("--page-id")

    p_retrieve = subparsers.add_parser("retrieve", help="Retrieve nodes only, without LLM synthesis")
    p_retrieve.add_argument("--question", required=True)
    p_retrieve.add_argument("--verification-status", choices=["verified", "uncertain"])
    p_retrieve.add_argument("--human-review-required", type=lambda x: x.lower() == "true", default=None)
    p_retrieve.add_argument("--source-type")
    p_retrieve.add_argument("--unit-type", choices=["page", "region"])
    p_retrieve.add_argument("--language")
    p_retrieve.add_argument("--document-id")
    p_retrieve.add_argument("--page-id")

    p_eval = subparsers.add_parser("eval", help="Run retrieval evaluation (hit@k/recall@k/MRR/nDCG + ablation)")
    p_eval.add_argument("--questions", required=True, help="Path to eval JSON with questions and expected IDs")
    p_eval.add_argument("--no-ablation", action="store_true",
                        help="skip the vector/bm25/hybrid ablation (faster; router metrics only)")

    return parser.parse_args()


def main() -> None:
    """
    Entry point for the Radio Barcelona RAG CLI
    """
    args = parse_args()
    # Embedded local storage by default; an explicit --qdrant-url switches to server mode
    use_server = args.qdrant_url is not None
    config = PipelineConfig(
        qdrant_url=args.qdrant_url or DEFAULT_QDRANT_URL,
        qdrant_path=None if use_server else args.qdrant_path,
        ollama_url=OLLAMA_HOST,
        collection_name=args.collection,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        similarity_cutoff=args.similarity_cutoff,
        rrf_k=args.rrf_k,
        bm25_top_k=args.bm25_top_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        router_confidence_threshold=args.router_confidence_threshold,
        scroll_batch_size=args.scroll_batch_size,
    )

    rag = RadioBarcelonaRAG(config)
    try:
        if args.command == "build":
            cmd_build(rag, args)
        elif args.command == "query":
            cmd_query(rag, args)
        elif args.command == "retrieve":
            cmd_retrieve(rag, args)
        elif args.command == "eval":
            cmd_eval(rag, args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    finally:
        rag.qdrant_client.close()


if __name__ == "__main__":
    main()
