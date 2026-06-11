# RADAR

A self-contained application that serves the indexed GUIRAD archive as a **document
viewer** + **chatbot**

```
app/
├── backend/    FastAPI server (Python)
└── frontend/   Next.js viewer (TypeScript + Tailwind)
```

## Backend (Python)

Reads the indexed stores (`data/index/records` + `data/derived`) and serves them

No GPU and no Qwen/RAG runtime are needed for the viewer endpoints

```bash
conda activate Synthesis
pip install -e ".[serving]"
python -m app.backend.main --reload  # http://127.0.0.1:8000
```

Endpoints:  
`/api/documents`  
`/api/documents/{doc}/pages/{page}`  
`/api/pdf/{doc}`  
`/api/derived/{doc}/{file}`  
`/api/chat` (wired to the RAG pipeline next)

## Frontend (Next.js)

Renders the precomputed layer images with a layer toggle, the separated
transcriptions and marks colour-coded by confidence, and a chatbot pane

```bash
cd app/frontend
npm install
npm run dev  # http://localhost:3000
```

`next.config.mjs` proxies `/api/*` to the backend (override with `BACKEND_URL`)
