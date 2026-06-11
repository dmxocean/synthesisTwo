# Synthetic Data Generation Suite

This suite provides the full pipeline for building multi-layer synthetic training documents from historical GUIRAD PDFs: it covers asset preparation, layout template management, document assembly, and dataset export

## 1. Pipeline Execution Sequence

Scripts are executed from the project root in the following logical order:

### Phase A: Asset Preparation

Infrastructure for creating the visual primitives used during synthesis: this phase is typically executed once to populate the asset libraries

1. **`src/synthetic/iam/build.py`**: extracts alpha-masked sentences and words from raw IAM files into `data/iam/library/`

   **Prerequisites:** the IAM Handwriting Database requires a free registration at `https://fki.tic.heia-fr.ch/databases/iam-handwriting-database`. After registering, download `lines.tgz` (scanned line images) and `ascii.tgz` (text metadata) and extract both into `data/iam/raw/`:

   ```bash
   tar -xzf lines.tgz -C data/iam/raw/
   tar -xzf ascii.tgz -C data/iam/raw/
   ```

   The expected structure before running `build.py`:
   ```
   data/iam/raw/
       ascii/
           lines.txt
           words.txt
       lines/
           {writer_id}/{form_id}/{line_id}.png
   ```

   If the library has already been built on another machine, copying `data/iam/library/` directly to the server avoids rebuilding from scratch.
2. **`src/synthetic/paper/harvester.py`**: removes ink from raw PDFs to recover A4 paper textures into `data/paper/`

### Phase B: Layout Sampling

Selects representative pages from the raw PDF archive for manual annotation: this phase is executed once or whenever the layout variety needs expansion

1. **`src/synthetic/layouts/sampler.py`**: DBSCAN-based proportional sampler: exports A4 PNGs to `data/interim/layouts/samples/` with a manifest

After exporting, images are uploaded to external labeling platforms for manual annotation: the resulting COCO JSON export is used to populate `data/layouts/templates/`

### Phase C: Synthesis

The final generation phase where all primitives are combined: this phase is executed repeatedly to generate the required dataset volume

1. **`src/synthetic/verify/main.py`**: smoke test that exercises every pipeline component and confirms output file integrity: run this after any structural change
   ```bash
   python -m src.synthetic.verify.main
   ```
2. **`src/synthetic/main.py`**: master entry point: produces RGB composites, 3-channel semantic masks, and COCO annotation JSON for every generated document
   ```bash
   python src/synthetic/main.py --count 100 --epoch all
   ```

## 2. Core Modules

### `src/synthetic/core/`
*   **`assembler.py`**: composites handwriting, printed text, and noise layers onto authentic paper textures at native asset scale

### `src/synthetic/generators/`
*   **`handwriting.py`**: renders handwriting blocks by tiling IAM sentence and word assets at native scanned size - ground truth text comes from IAM paired labels
*   **`printed.py`**: renders printed text blocks using a historical TTF font at a fixed line height: text is generated internally using Romance-language syllable patterns
*   **`noise.py`**: pastes noise artifacts (stamps, crossouts, lines, marks, crosses) at native pixel size using position-based type weighting

### `src/synthetic/providers/`
*   **`assets.py`**: indexes the IAM sentence and word libraries into memory at startup for zero-latency lookups during synthesis

### `src/synthetic/layouts/`
*   **`blueprint.py`**: loads layout templates from `data/layouts/templates/{epoch}/` and applies rubber-sheet jitter to every region

### `src/synthetic/export/`
*   **`format.py`**: serializes each synthetic document to `data/synthetic/images/`, `data/synthetic/masks/`, and `data/synthetic/annotations/`

### `src/utils/`
*   **`textgen.py`**: embedded Spanish word list used by `PrintedGenerator` to produce natural-looking fill text
*   **`document.py`**: PDF-to-image conversion and A4 standardization utilities
*   **`gpu.py`**: device autodetection for CPU, CUDA, or MPS

## 3. Utility Scripts

**IAM style browser**: renders one PNG per writer per letter group at true synthesis scale
```bash
python src/synthetic/iam/preview.py
```

## 4. Usage

**Verify the pipeline before generating:**
```bash
python -m src.synthetic.verify.main
```

**Verify polygon refinement and asset alpha cleaning:**
```bash
python -m src.synthetic.verify.polygons --count 3 --modes all --prewarm
```

**Generate documents:**
```bash
python src/synthetic/main.py --count 100 --epoch all
```

**Arguments:**

| Argument | Values | Default | Description |
|---|---|---|---|
| `--count` | int | all templates × variants | document cap |
| `--epoch` | `all`, `monarchy`, `republic`, `war`, `francoist` | `all` | target historical epochs |
| `--variants` | int | 20 | variants per template |
| `--inject` | int | 5 | max donor regions injected per document |

## 5. Output Structure

```
data/synthetic/
    images/          RGB A4 composite documents (2480x3508 at 300 DPI)
    masks/           3-channel semantic masks (R: handwriting, G: printed, B: noise)
    annotations/     per-document COCO-compatible JSON with word-level bounding boxes
```

Output files are named using the convention `synth_{mode}_{epoch}_{index:04d}.png`
