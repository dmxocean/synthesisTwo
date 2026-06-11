# Radio Barcelona RAG pipeline: terminal usage and evaluation queries

This document explains how to run the Radio Barcelona RAG pipeline from the terminal and includes a starter evaluation query table derived from the synthetic metadata collection.

## Files and pipeline variants

The project currently has three relevant pipeline scripts:

- `llm_rag_pipeline.py`: baseline vector-only prototype with build, query, retrieve, and eval commands.
- `llm_rag_pipeline_v2.py`: small improvements from the baseline implementation (structured JSON answers, hybrid retrieval      scaffolding, a lightweight query classifier/router, a separate provenance formatter)
- `llm_rag_pipeline_v3.py`: adds bm25 to replace the simple keyword retriever, and RRF replaces the simple weighted score-fusion between vector and hybrid retrieval.
- `llm_rag_pipeline_v4.py`: extended version with ROUTED RETRIEVAL, hybrid BM25 + vector retrieval via RRF fusion, structured JSON answers, and provenance formatting.
- `llm_rag_pipeline_v5.py`: extension that is not tested yet, pero pinta bien

The baseline script supports `build`, `query`, `retrieve`, and `eval` commands from the CLI.[cite:963]
The improved script supports the same commands and adds `--json-output`, `--similarity-cutoff`, `--rrf-k`, and `--bm25-candidate-pool` options.[cite:1067]

## Prerequisites

Install the Python dependencies used by the baseline script:[cite:963]

```bash
pip install llama-index llama-index-vector-stores-qdrant llama-index-embeddings-huggingface \
  llama-index-llms-ollama qdrant-client fastembed
```

If you want BM25-related experimentation, install the optional lexical dependency too.[cite:963]

```bash
pip install rank-bm25
```

Start the required local services before running the pipeline:

- Start Ollama locally, for example with `ollama serve`.[cite:963]
- Pull a local model such as `mistral:7b-instruct` with `ollama pull mistral:7b-instruct`.[cite:963]
- Start Qdrant locally, for example with Docker: `docker run -p 6333:6333 qdrant/qdrant`.[cite:963]

## Build the index

Both scripts load a `.json` or `.jsonl` metadata file and index it into the Qdrant collection `radio_barcelona_metadata` by default.[cite:963][cite:1067]

Use the baseline script:

```bash
python llm_rag_pipeline.py build --input synthetic_radio_barcelona_metadata_v2.json
```

Use the improved routed hybrid script:

```bash
python llm_rag_pipeline_bm25_rrf.py build --input synthetic_radio_barcelona_metadata_v2.json
```

You can override the defaults with global flags such as `--qdrant-url`, `--collection`, `--embed-model`, `--llm-model`, and `--top-k` in both scripts.[cite:963][cite:1067]

Example:

```bash
python llm_rag_pipeline_bm25_rrf.py \
  --qdrant-url http://localhost:6333 \
  --collection radio_barcelona_metadata \
  --embed-model BAAI/bge-m3 \
  --llm-model mistral:7b-instruct \
  --top-k 8 \
  build --input synthetic_radio_barcelona_metadata_v2.json
```

## Run retrieval and question answering

### Baseline pipeline

The baseline script supports semantic querying with optional metadata filters such as `--verification-status`, `--human-review-required`, `--source-type`, `--unit-type`, `--language`, `--document-id`, and `--page-id`.[cite:963]

Basic answer generation:

```bash
python llm_rag_pipeline.py query \
  --question "find pages mentioning Barcelona with uncertain annotations"
```

Retrieve nodes only, without LLM synthesis:

```bash
python llm_rag_pipeline.py retrieve \
  --question "find pages mentioning Barcelona with uncertain annotations"
```

Example with explicit filters:

```bash
python llm_rag_pipeline.py query \
  --question "find pages mentioning Barcelona with uncertain annotations" \
  --verification-status uncertain \
  --unit-type page
```

### Improved routed hybrid pipeline

The improved script uses a classifier to choose `vector` or `hybrid` retrieval and can also return structured JSON answers with `--json-output`.[cite:1067]

Basic routed query:

```bash
python llm_rag_pipeline_bm25_rrf.py query \
  --question "exact mention of autorizado para emisión"
```

Structured JSON answer:

```bash
python llm_rag_pipeline_bm25_rrf.py query \
  --question "find uncertain pages mentioning Barcelona" \
  --json-output
```

Retrieve only, while still printing the inferred query plan:

```bash
python llm_rag_pipeline_bm25_rrf.py retrieve \
  --question "exact mention of evítese toda improvisación"
```

You can also tune retrieval behavior with `--similarity-cutoff`, `--rrf-k`, and `--bm25-candidate-pool`.[cite:1067]

Example:

```bash
python llm_rag_pipeline_bm25_rrf.py \
  --top-k 10 \
  --similarity-cutoff 0.45 \
  --rrf-k 60 \
  --bm25-candidate-pool 5000 \
  query --question "exact phrase leer únicamente la versión autorizada"
```

## Run evaluation

Both scripts provide an `eval` command that accepts a JSON file containing questions and expected record IDs.[cite:963][cite:1067]
The improved script also reports how many queries were routed to `vector` vs `hybrid` retrieval.[cite:1067]

Baseline evaluation:

```bash
python llm_rag_pipeline.py eval --questions output/eval_queries_radio_barcelona.json
```

Improved routed hybrid evaluation:

```bash
python llm_rag_pipeline_bm25_rrf.py eval --questions output/eval_queries_radio_barcelona.json
```

## Starter evaluation query table

The following evaluation set was derived from the synthetic metadata file and maps each question to expected retrieval behavior and gold records from the collection.[cite:1093]

| Query | Expected mode | Expected filters | Expected records |
|---|---|---|---|
| `uncertain region records about guerra and frente` | `vector` | `verification_status=uncertain; unit_type=region` | `rec_1937_001_p005` |
| `verified page records about música or selección instrumental` | `vector` | `verification_status=verified; unit_type=page` | `rec_1941_002_p003`, `rec_1936_010_p005` |
| `exact mention of autorizado para emisión` | `hybrid` | none | `rec_1941_002_p003`, `rec_1938_003_p004`, `rec_1943_011_p004`, `rec_1940_006_p003`, `rec_1938_062_p005` |
| `page records that require human review and mention República` | `vector` | `human_review_required=true; unit_type=page` | `rec_1936_005_p004` |
| `records about barrio Barcelona with low confidence and human review needed` | `vector` | `human_review_required=true` | `rec_1934_004_p001` |
| `exact phrase leer únicamente la versión autorizada` | `hybrid` | none | `rec_1939_013_p006`, `rec_1938_062_p005` |
| `verified region records about República or canciones populares` | `vector` | `verification_status=verified; unit_type=region` | `rec_1934_014_p005` |
| `uncertain page records with politically sensitive content` | `vector` | `verification_status=uncertain; unit_type=page` | `rec_1936_005_p004`, `rec_1939_013_p006` |
| `exact mention of evítese toda improvisación` | `hybrid` | none | `rec_1938_003_p004`, `rec_1943_007_p003`, `rec_1936_010_p005` |
| `verified page records about emisión de continuidad and Barcelona` | `vector` | `verification_status=verified; unit_type=page` | `rec_1943_007_p003`, `rec_1941_073_p002` |
| `records mentioning Gràcia with uncertain forensic status` | `vector` | `verification_status=uncertain` | `rec_1953_012_p004`, `rec_1939_013_p006` |
| `exact mention of orden revisado` | `hybrid` | none | `rec_1953_012_p004`, `rec_1934_014_p005`, `rec_1944_061_p003` |

## Suggested workflow

A practical terminal workflow is:

IN ONE TERMINAL:
1.1 conda activate llm_rag
1.2 cd qdrant
1.3 ./qdrant

IN ANOTHER TERMINAL
2.1 conda activate llm_rag
2.2 ollama serve

IN A THIRD TERMINAL:
3.1 conda activate llm_rag
3.2 run the pipeline
    - python llm_rag_pipeline_v4.py  --collection radio_barcelona_metadata_v4  build --input metadata/metadata/synthetic_radio_barcelona_metadata_v2.jsonl
    - python llm_rag_pipeline_v4.py  --collection radio_barcelona_metadata_v3  query --question "exact mention of orden revisado"

1. Start Qdrant and Ollama.[cite:963]
2. Build the index from the metadata JSON or JSONL file.[cite:963][cite:1067]
3. Run `retrieve` first to inspect raw candidates and provenance lines.[cite:963][cite:1067]
4. Run `query` for synthesized answers, and add `--json-output` on the improved script when you want structured output.[cite:1067]
5. Run `eval` against the generated evaluation JSON to measure hit rate and MRR.[cite:963][cite:1067]


## Router notes and route cheat sheet

### Is the router too rule-based?

Yes, the current router is partly rule-based. In the current routed scripts, the classifier checks for surface cues such as `exact`, `mention`, `contains`, `keyword`, `transcription`, `ocr`, `literal`, and `spelling` to switch from default vector retrieval to hybrid retrieval, and it also infers filters from words like `verified`, `uncertain`, `human review`, `region`, and `page`.[cite:1067][cite:1025]

That concern is valid: users can express the same intent in many different ways, so a pure keyword-trigger router will never cover all phrasings. A good design is to keep the rules, but treat them as a high-precision first layer rather than the whole intelligence of the system.[cite:1067][cite:1025]

A practical way to think about it is:

- Rules are useful for obvious cases, especially archival phrase-search queries such as `exact mention of ...`, field-driven queries such as `verified pages`, and OCR-sensitive requests.[cite:1067][cite:1025]
- The fallback should still be robust semantic retrieval, so if no route is confidently triggered, the query goes through vector retrieval and the LLM can still answer from semantically relevant evidence.[cite:1067][cite:1025]
- Later, the rule-based classifier can be replaced or complemented with an LLM-based query classifier or a small learned classifier, but the rule-based version is still a very reasonable intermediate architecture for prototyping.[cite:1067][cite:1025]

So the short answer is: yes, it is rule-based today, but that is not necessarily a problem if you treat the router as a routing aid plus fallback system, not as the only source of intelligence.[cite:1067][cite:1025]

### Should metadata retrieval bypass the LLM?

Mostly yes for the retrieval step, but not necessarily for the final response. In your pipeline, metadata filters are already first-class inputs, and direct field matching is more precise than asking an LLM to infer field values from unstructured context.[cite:1067][cite:1025]

The recommended pattern is:

1. Use direct lookup or filtered retrieval to get the right records.
2. Then optionally pass those retrieved instances to the LLM so it can generate a more natural-language answer, summarize the results, and explain uncertainty in a researcher-friendly way.[cite:1067][cite:1025]

That means `bypass the LLM` should be understood as `do not use the LLM as the retrieval mechanism when exact structured lookup is better`, not `never use the LLM afterward`. For your use case, a very good architecture is retrieval by deterministic filters first, then LLM synthesis second.[cite:1067][cite:1025]

### Proposed mental model

A simple mental model for the routed system is:

- If the user wants **exact wording**, use phrase-oriented retrieval first.
- If the user wants **specific words**, use keyword/BM25-style retrieval.
- If the user wants **meaning or topic**, use vector retrieval.
- If the user wants **specific metadata fields**, use direct filter lookup.
- After retrieval, optionally let the LLM write the answer in natural language using only the retrieved evidence.[cite:1067][cite:1025]

### Route cheat sheet

| Route | What it is for | Typical user intent | Example queries |
|---|---|---|---|
| `phrase_exact` | Literal text search | The user wants the exact wording or phrase | `exact mention of orden revisado`; `find the exact phrase autorizado para emisión`; `contains "leer únicamente la versión autorizada"`; `literal text servicio de censura`; `show pages where "Radio Barcelona" appears exactly` |
| `keyword_lexical` | Word-based search without requiring exact phrase order | The user cares about specific terms, OCR tokens, or spelling-sensitive matches | `pages mentioning censura and Barcelona`; `keyword orden revisado`; `OCR mention of pasodobles`; `records with República and revisión`; `find pages containing improvisación` |
| `semantic_vector` | Meaning-based retrieval | The user wants documents about a topic, theme, or concept | `documents about censorship practices in broadcasting`; `records related to Francoist administrative intervention`; `pages discussing daily life in Gràcia`; `documents about music programming control`; `records about local commerce and neighborhood life` |
| `metadata_lookup` | Direct field matching and structured filtering | The user is really asking for field values or exact structured subsets | `verified pages from 1946`; `records with human review required`; `document_id doc_1946_064`; `page_id p006`; `uncertain region records in Catalan` |
| `provenance_lookup` | Traceability and evidence reporting | The user wants source identifiers and verification info | `show document_id and page_id for records about pasodobles`; `which records support the answer about censorship?`; `give me provenance for uncertain pages`; `show record_id and verification status for these matches`; `what sources back this answer?` |
| `hybrid` | Mixed lexical + semantic need | The user wants specific wording, but approximate semantic recall may still help | `OCR mention of orden revisado in uncertain pages`; `exact or near mention of intervención`; `records about censorship that mention Radio Barcelona`; `find pages with autorizado para emisión even if OCR is noisy`; `look for pages about República with literal mention of Barcelona` |

### Very simple rule of thumb

Use this quick test when thinking about routing:

- `these exact words` -> `phrase_exact`
- `these words somewhere in the record` -> `keyword_lexical`
- `this idea or topic` -> `semantic_vector`
- `this field or filter` -> `metadata_lookup`
- `show me the evidence trail` -> `provenance_lookup`
- `a mix of wording and meaning` -> `hybrid`[cite:1067][cite:1025]

### Recommended answer strategy by route

| Route | Retrieval | LLM used afterward? |
|---|---|---|
| `phrase_exact` | Exact substring or phrase search over text-bearing fields | Optional, mainly for summarizing matches |
| `keyword_lexical` | BM25 or lexical retrieval | Usually yes, to summarize evidence |
| `semantic_vector` | Vector retrieval | Yes, this is the standard RAG flow |
| `metadata_lookup` | Direct metadata filter lookup | Yes, when you want a natural-language synthesis of the filtered records |
| `provenance_lookup` | Provenance-first evidence listing | Optional, often a short explanatory summary is enough |
| `hybrid` | BM25 + vector fusion | Yes, if the retrieved evidence is good |

This keeps the retrieval step precise while still letting the LLM produce a natural answer after the right records are found.[cite:1067][cite:1025]
