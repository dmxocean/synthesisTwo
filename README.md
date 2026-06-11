# RADAR

RADAR turns the GUIRAD archive (Radio Barcelona broadcast scripts, 1924-1953) into structured, searchable, research-grade historical knowledge. Raw PDFs are ingested, segmented into semantic layers, transcribed region by region, classified for noise marks, and indexed into a retrieval system accessible through a web application.

The pipeline runs sequentially: PDF acquisition and preprocessing, synthetic training data generation, segmentation model training (SegFormer MiT-B3 and U-Net), VLM-based transcription (Qwen3-VL-8B), noise classification (ResNet-18), record building, vector indexing (Qdrant with BAAI/bge-m3 embeddings), and a FastAPI/Next.js frontend for browsing and querying the archive.


## Installation

Requirements: Conda, CUDA 12.4, roughly 20 GB VRAM for the full VLM, Docker for external services.

### Python environment

```bash
conda create -n RADAR python=3.10 -y
conda activate RADAR
```

### PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### System dependency for PDF rasterisation

```bash
conda install -c conda-forge poppler
# or: sudo apt-get install poppler-utils
```

### Package and dependencies

```bash
pip install -e ".[segmentation,detection,synthetic,rag,serving]"
```

Install only the extras relevant to the stages you intend to run:

| Extra | Covers |
|---|---|
| `segmentation` | SegFormer and U-Net training and inference |
| `detection` | Qwen3-VL transcription and noise ResNet-18 |
| `synthetic` | Synthetic data factory |
| `rag` | Retrieval pipeline and vector store |
| `serving` | FastAPI backend |
| `dev` | Testing and linting |

The editable install registers `src/` as a package so `from src.*` imports work everywhere when running from the repository root.

### Hugging Face access

```bash
hf auth login   # paste your token; accept the Qwen3-VL licence at huggingface.co
```

Models are downloaded on first use:

| Model | Size | Used by |
|---|---|---|
| Qwen/Qwen3-VL-8B-Instruct | ~16 GB | Transcription and noise description |
| BAAI/bge-m3 | ~2 GB | RAG embeddings (1024-dim) |

On machines with less than 16 GB VRAM, substitute `Qwen/Qwen3-VL-2B-Instruct` or install flash-attn for memory savings: `pip install flash-attn --no-build-isolation`.

### External services

For the RAG and serving stages only:

```bash
docker run -d -p 6333:6333 qdrant/qdrant       # vector store
docker run -d -p 27017:27017 mongo             # document catalog
ollama pull mistral:7b-instruct                # answer LLM (~4 GB)
```

Or start all at once:

```bash
docker compose up -d
```

### Verify the install

```bash
python -c "from transformers import Qwen3VLForConditionalGeneration; print('VLM import OK')"
python -c "from src.core.config import DATA_ROOT; print('data root:', DATA_ROOT)"
python -m pytest -q
```


## Running the application

All commands run from the repository root with the RADAR environment active. Run the backend and frontend in separate terminals.

```bash
uvicorn app.backend.app:app --reload
```

```bash
cd app/frontend && npm install && npm run dev
```

The frontend opens at `http://localhost:3000`. The left panel shows the document catalog with search; the right panel provides a layered viewer (handwritten, printed, and noise segmentation layers with toggles), a heatmap view of model uncertainty, and a clean transcription tab. The chatbot queries the vector index through the RAG pipeline.


## Pipeline stages

| Stage | Script directory | Output |
|---|---|---|
| 0 Install | | environment ready |
| 1 Acquire | scripts/scraping/ | data/raw PDFs |
| 2 Cluster | scripts/preprocessing/ | manifest.json, epoch PNGs |
| 3 Annotations | scripts/annotation/ | COCO JSON, region crops |
| 4 Assets | scripts/synthetic/ | templates, IAM data, noise crops |
| 5 Synthesise | scripts/synthetic/ | data/synthetic/factory/ |
| 6 Train | scripts/segmentation/, scripts/detection/ | outputs/*/weights/best.pt |
| 7 Evaluate | scripts/segmentation/, scripts/detection/ | outputs/*/metrics/*.json |
| 8 Index | scripts/indexing/ | data/index/records/ |
| 9 RAG | scripts/rag/ | storage/qdrant/, rag.json |
| 10 Serve | app/ | FastAPI backend and Next.js frontend |


## Models

| Model | Architecture | Task |
|---|---|---|
| Segmentation | SegFormer MiT-B3 (44.6M) and U-Net ResNet-50 | 3-channel segmentation: handwritten, printed, noise |
| Noise classifier | ResNet-18 (timm, 224x224) | 5-class noise typing: stamps, circles, crosses, lines, marks |
| Transcription | Qwen3-VL-8B-Instruct (bfloat16) | Per-region OCR and HTR |
| Embeddings | BAAI/bge-m3 | 1024-dim dense vectors for retrieval |
| Answer LLM | mistral:7b-instruct via Ollama | RAG-grounded question answering |

Segmentation outputs 3 binary channels (NS=0, HW=1, PR=2) at 768x768. Training data combines the synthetic factory tiles and the DELINE8K dataset, split 70/15/15 by source page using a deterministic hash.


## Project structure

```
RADAR/
 app/
   backend/          FastAPI server (app/backend/app.py)
   frontend/         Next.js 14 (React / TypeScript)
 data/
   interim/          Layout annotations and sample tiles by epoch cluster
   synthetic/        Factory tiles (5000) and DELINE8K tiles (8000)
 docs/               Stage runbooks, schema docs, architecture notes
 notebooks/          Exploratory analysis by stage
 outputs/            Model weights, metric JSONs, training logs, predictions
 scripts/            CLI entry points, one subdirectory per stage
   preprocessing/
   synthetic/
   annotation/
   segmentation/
   detection/
   indexing/
   metadata/
   rag/
   scraping/
   tools/
 src/                Importable library, no executables
   core/             Config, splits, metrics, confidence, geometry
   preprocessing/    PDF ingestion and tiling
   synthetic/        Synthetic data factory (tile renderer, noise, text primitives)
   segmentation/     SegFormer and U-Net, loss functions, data loaders
   detection/        Qwen3-VL transcription, noise classifier, record builder
   indexing/         Record serialization and store population
   rag/              Retrieval pipeline (vector, hybrid, BM25, RRF, LLM router)
   metadata/         Metadata extraction and enrichment
   scraping/         GUIRAD archive scraping utilities
 storage/            Qdrant embedded vector store (excluded from git)
 tests/
 bentham/            Bentham handwriting corpus utilities
 information/        Archive reference material
 docker-compose.yml
 environment.yml     Full conda environment spec (alternative to pyproject extras)
 pyproject.toml
```
## Contributors

Gerzon Diaz Marcani  
Héctor Salguero Martinez  
David Piera Jimenez  
Victor Brao Ruiz  
David Sanllehí Vico  