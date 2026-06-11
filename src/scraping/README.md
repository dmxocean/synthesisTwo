# Scraper

Advanced dataset architect for the autonomous acquisition and metadata indexing of the [GUIRAD](https://ddd.uab.cat/search?cc=guirad&ln=es) historical archive: this module handles high-speed batch discovery and resilient asset retrieval

## Chronological Coverage

The project segments the archive into four primary historical periods to ensure representative sampling and epoch-aware analysis:

*   **Monarchy & Dictatorship (1924:1930)**: earliest records of Radio Barcelona scripts
*   **Second Spanish Republic (1931:1935)**: significant growth in the document volume
*   **Spanish Civil War (1936:1938)**: period of intense archival variance and shifting nomenclature
*   **Francoist Regime (1939:1953)**: post-war records with standardized administrative formats

## Logic & Architecture

The module utilizes a path-agnostic design, resolving its internal directory structure dynamically: it leverages the UAB repository's **MARCXML** interface to perform bulk discovery (100 records per request), significantly reducing network overhead

### Components

`guirad.py`:
primary coordinator: it implements the discovery pipeline, manages the incremental state of the metadata index, and generates comprehensive status reports

`getfiles.py`:
asset acquisition engine: it provides robust streaming downloads with automatic retries and validates local filesystem integrity to prevent redundant transfers

## Usage

The launchers are designed to be executable from any system path

**Automated Sync (1924:1945)**:
iterates through the primary research years and executes the synchronized discovery and download process

**Windows (PowerShell):**
```powershell
./download_guirad.ps1
```

**Linux/macOS (Bash):**
```bash
./download_guirad.sh
```

**Direct Execution (`guirad.py`)**

All commands are run from the project root

**Full Synchronized Map (Default Mode):**
discover all records and download missing PDF assets: this mode ensures that every record identified in the MARCXML metadata has a corresponding local PDF file
```bash
python src/scraper/guirad.py
```

**Global Metadata Indexing (`--info` Mode):**
execute a non-downloading scan across all years to update `guirad.json` and generate `report.txt`: this mode is used for auditing the archive status without initiating data transfers
```bash
python src/scraper/guirad.py --info
```

**Targeted Year Query:**
restrict the pipeline to a specific chronological year
```bash
python src/scraper/guirad.py --year 1932
```

## Dataset Artifacts

*   `data/analysis/guirad.json`: global metadata registry (titles, dates, languages, and remote links)
*   `data/analysis/report.txt`: statistical audit distinguishing between **Identified** and **Acquired** assets across all epochs
*   `data/raw/`: hierarchical storage of original PDF documents organized by year

## Environment

Requires `requests` and `beautifulsoup4`
