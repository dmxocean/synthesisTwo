# -*- coding: utf-8 -*-
"""
FastAPI application for the research viewer + chatbot (component)

Exposes the read API the frontend consumes: the document catalog, per-page detail
(separated layer text, marks, confidence, derived-image urls), the raw PDF, and
the precomputed derived images

The viewer endpoints read only the indexed stores (records + derived images), so
they need no GPU and no Qwen/RAG runtime

The chat endpoint is wired to the RAG library to provide archival synthesis

The launcher (scripts/serving/serve_api.py) runs this with uvicorn
"""

import os
import re
import glob
import json
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag.pipeline import RadioBarcelonaRAG, PipelineConfig, ProvenanceFormatter
from app.backend import store_reader

# Routes
PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_DIR_RAW = os.path.join(PATH_ROOT, "data", "raw")
PATH_DIR_DERIVED = os.path.join(PATH_ROOT, "outputs", "derived")
PATH_STORAGE_QDRANT = os.environ.get("QDRANT_PATH", os.path.join(PATH_ROOT, "storage", "qdrant"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COLLECTION_REAL = "radio_barcelona_real"


class ChatRequest(BaseModel):
    question: str


def create_app() -> FastAPI:
    """Builds and returns the FastAPI app"""
    app = FastAPI(title="RADAR research viewer", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # The Next.js dev server
        allow_methods=["*"], allow_headers=["*"],
    )

    # Initialize RAG system at startup
    rag_config = PipelineConfig(
        qdrant_path=PATH_STORAGE_QDRANT,
        collection_name=COLLECTION_REAL,
        ollama_url=OLLAMA_HOST,
    )
    # Note: initialization might take a moment (loading embedding model)
    rag_system = RadioBarcelonaRAG(rag_config)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/documents")
    def documents():
        """The document catalog with page ids"""
        return [d.model_dump() for d in store_reader.list_documents()]

    @app.get("/api/documents/{doc_id}/pages/{page_id}")
    def page_detail(doc_id: str, page_id: str):
        """Per-page detail: separated layer text, marks, confidence, derived-image urls"""
        page = store_reader.get_page(doc_id, page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"page {doc_id}/{page_id} not found")
        return page.model_dump()

    @app.get("/api/pdf/{doc_id}")
    def pdf(doc_id: str):
        """Streams the original PDF for a document (searched under data/raw)"""
        matches = glob.glob(os.path.join(PATH_DIR_RAW, "**", f"{doc_id}.pdf"), recursive=True)
        if not matches:
            raise HTTPException(status_code=404, detail=f"pdf for {doc_id} not found")
        return FileResponse(matches[0], media_type="application/pdf")

    @app.get("/api/page-image/{doc_id}/{page_id}")
    def page_image(doc_id: str, page_id: str):
        """
        Renders the clean original page to PNG (the base image for the layer composite)

        Cached to outputs/derived/{doc}/{page}_raw.png on first request, then served from disk
        The page index is parsed from the trailing _pNNNN of the page_id
        """
        cache_path = os.path.join(PATH_DIR_DERIVED, doc_id, f"{page_id}_raw.png")
        if os.path.exists(cache_path):
            return FileResponse(cache_path, media_type="image/png")

        m = re.search(r"_p(\d+)$", page_id)
        if not m:
            raise HTTPException(status_code=400, detail=f"cannot parse page index from {page_id}")
        page_index = int(m.group(1))

        matches = glob.glob(os.path.join(PATH_DIR_RAW, "**", f"{doc_id}.pdf"), recursive=True)
        if not matches:
            raise HTTPException(status_code=404, detail=f"pdf for {doc_id} not found")

        from PIL import Image
        from src.preprocessing.pdf import DocumentProcessor
        page_np = DocumentProcessor.pdf_page_to_image(matches[0], page_index=page_index, dpi=150)
        if page_np is None:
            raise HTTPException(status_code=404, detail=f"failed to render {doc_id} page {page_index}")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        Image.fromarray(page_np).save(cache_path)
        return FileResponse(cache_path, media_type="image/png")

    @app.get("/api/heatmap/{doc_id}/{page_id}")
    def heatmap_clean(doc_id: str, page_id: str):
        """
        Clean uncertainty heatmap rendered directly from .conf.npy - no colorbar, no title

        Applies the viridis colormap as a pure lookup table (no matplotlib axes/figure)
        uncertainty = 1 - max_prob: dark-blue pixels are confident, yellow pixels are uncertain
        Cached to outputs/derived/{doc}/{page}_heatmap.png on first request
        """
        cache_path = os.path.join(PATH_DIR_DERIVED, doc_id, f"{page_id}_heatmap.png")
        if os.path.exists(cache_path):
            return FileResponse(cache_path, media_type="image/png")

        npy_path = os.path.join(PATH_DIR_DERIVED, doc_id, f"{page_id}.conf.npy")
        if not os.path.exists(npy_path):
            raise HTTPException(status_code=404, detail="confidence map not found (run build_index first)")

        conf = np.load(npy_path).astype(np.float32)
        uncertainty = np.clip(1.0 - conf, 0.0, 1.0)

        from matplotlib.cm import viridis  # Colormap only - no figure/axes
        rgba = (viridis(uncertainty) * 255).astype(np.uint8)

        from PIL import Image
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(cache_path)
        return FileResponse(cache_path, media_type="image/png")

    @app.get("/api/derived/{doc_id}/{filename}")
    def derived(doc_id: str, filename: str):
        """Serves a precomputed derived image (layer mask, heatmap, overlay)"""
        if os.path.sep in filename or "/" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="invalid filename")
        path = os.path.join(PATH_DIR_DERIVED, doc_id, filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="derived image not found")
        return FileResponse(path)

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        """Archival RAG chat wired to src/rag"""
        try:
            result = rag_system.query_with_router(req.question, json_mode=True)
            
            # The router returns a LlamaIndex response object or a string in result["response"]
            # With json_mode=True, it should be a JSON-formatted string
            response_obj = result["response"]
            if hasattr(response_obj, "response"):
                 response_text = response_obj.response
            else:
                 response_text = str(response_obj)

            # Parse the synthesis answer
            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if LLM didn't return pure JSON
                payload = {"answer": response_text, "insufficient_evidence": False}

            # Inject forensic provenance
            payload["provenance"] = ProvenanceFormatter.format_json_sources(result["source_nodes"])
            payload["plan"] = {
                "mode": result["plan"].retrieval_mode,
                "rationale": result["plan"].rationale
            }
            
            return payload
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
