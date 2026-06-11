# -*- coding: utf-8 -*-
"""
Centralized environment detection and filesystem path configuration

This module resolves the root data directory from environment variables or local paths and defines global constants for the project. It handles path resolution for raw assets, synthetic data, model weights, and storage directories. All paths are anchored to the repository root via absolute resolution to ensure consistent behavior across different execution environments
"""

import os
import sys
from typing import Optional

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_PATH, "data")
STORAGE_DIR = os.path.join(BASE_PATH, "storage")

def _load_dotenv(path: str) -> None:
    """
    Load key-value pairs from a dot-env file into the environment
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass

def _auto_detect_server_data_root(project_name: Optional[str] = None) -> Optional[str]:
    """
    Locate the server-mounted data root for the current user
    """
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    base = project_name or os.path.basename(BASE_PATH)
    if not user:
        return None
    candidate = os.path.join("/datos", user, base, "data")
    if os.path.exists(candidate) and os.path.isdir(candidate):
        return candidate
    return None

_load_dotenv(os.path.join(BASE_PATH, ".env"))

DATA_ROOT = os.path.join(BASE_PATH, "data")

if __name__ != "__main__":
    if "pytest" not in sys.modules and not os.environ.get("RADAR_QUIET"):
        print(f"[*] Synthesis Data Root: {DATA_ROOT}")  # Report resolved path on import

def is_server_mode() -> bool:
    """
    Determine if the application is running in a server-mounted environment
    """
    try:
        return os.path.abspath(DATA_ROOT) != os.path.abspath(os.path.join(BASE_PATH, "data"))
    except Exception:
        return False

PATH_DIR_ANALYSIS = os.path.join(DATA_ROOT, "analysis")
PATH_DIR_RAW = os.path.join(DATA_ROOT, "raw")
PATH_DIR_PAPER = os.path.join(DATA_ROOT, "paper")
PATH_DIR_IAM_LIB = os.path.join(DATA_ROOT, "iam", "library")
PATH_DIR_IAM_SAMPLES = os.path.join(DATA_ROOT, "iam", "samples")
PATH_DIR_NOISE = os.path.join(DATA_ROOT, "assets", "manual", "noise")
PATH_DIR_MANUAL_HW = os.path.join(DATA_ROOT, "assets", "manual", "handwritten")
PATH_DIR_TEMPLATES = os.path.join(DATA_ROOT, "layouts", "templates")
PATH_DIR_LAYOUTS_SAMPLES = os.path.join(DATA_ROOT, "interim", "layouts", "samples")
PATH_DIR_LAYOUTS_ANNOTATIONS = os.path.join(DATA_ROOT, "interim", "layouts", "annotations")
PATH_DIR_OUTPUT = os.path.join(DATA_ROOT, "synthetic", "factory")
PATH_FILE_FONT = os.path.join(DATA_ROOT, "assets", "fonts", "SpecialElite-Regular.ttf")
PATH_FILE_JSON = os.path.join(PATH_DIR_ANALYSIS, "guirad.json")
PATH_FILE_REPORT = os.path.join(PATH_DIR_ANALYSIS, "report.txt")
PATH_DIR_OUTPUTS = os.path.join(BASE_PATH, "outputs")
PATH_DIR_DERIVED = os.path.join(PATH_DIR_OUTPUTS, "derived")
PATH_DIR_INDEX = os.path.join(PATH_DIR_OUTPUTS, "index")
PATH_DIR_RECORDS = os.path.join(PATH_DIR_INDEX, "records")
PATH_DIR_RAG_METADATA = os.path.join(PATH_DIR_OUTPUTS, "rag", "metadata")
PATH_FILE_RAG_CORPUS = os.path.join(PATH_DIR_RAG_METADATA, "records.jsonl")
COLLECTION_REAL = "radio_barcelona_real"

DEFAULT_QDRANT_PATH = os.path.join(BASE_PATH, "storage", "qdrant")
PATH_STORAGE_QDRANT = os.environ.get("QDRANT_PATH", DEFAULT_QDRANT_PATH)
if not os.path.isabs(PATH_STORAGE_QDRANT):
    PATH_STORAGE_QDRANT = os.path.join(BASE_PATH, PATH_STORAGE_QDRANT)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "RADAR")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "records")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

EPOCHS = ("monarchy", "republic", "war", "francoist")
DICT_EPOCHS = {
    "monarchy": range(1924, 1931),
    "republic": range(1931, 1936),
    "war": range(1936, 1939),
    "francoist": range(1939, 1954),
}

SEG_THRESHOLDS = [0.1, 0.1, 0.1]

def get_artifact_dir(phase: str, model_name: str, artifact_type: str) -> str:
    """
    Retrieve the standardized directory path for model artifacts

    Args:
        phase (str): Pipeline stage such as segmentation or detection
        model_name (str): Identifier for the specific model architecture
        artifact_type (str): Category of artifact like weights or metrics
    Returns:
        str: Absolute path to the resolved and created directory
    """
    if artifact_type not in ["weights", "logs", "metrics", "predictions", "figures"]:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    path = os.path.join(PATH_DIR_OUTPUTS, phase, model_name, artifact_type)
    os.makedirs(path, exist_ok=True)
    return path

PATH_FILE_SPLIT = os.path.join(PATH_DIR_OUTPUT, "split.json")
