# -*- coding: utf-8 -*-
"""
Synthetic metadata generation for the Radio Barcelona archive executable

This script generates a synthetic corpus of archival records simulating the Radio Barcelona historical collection. It produces rich metadata including OCR text, visual marks, forensic flags, and entities across different historical periods. The output is used to test retrieval and indexing pipelines without requiring real sensitive data
"""

import os
import json
import random
from datetime import datetime, timezone

# Route block
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_JSONL = os.path.join(BASE_PATH, "metadata", "synthetic_radio_barcelona_metadata_v2.jsonl")
OUTPUT_JSON = os.path.join(BASE_PATH, "metadata", "synthetic_radio_barcelona_metadata_v2.json")

SEED = 42
random.seed(SEED)

N_RECORDS = 80
EXPORT_JSONL = True
EXPORT_JSON = True

COLLECTION_ID = "uab_radio_scripts"
COLLECTION_NAME = "UAB Radio Scripts Archive"
INSTITUTION = "Universitat Autònoma de Barcelona"
REPOSITORY = "UAB Library"

PLACES = ["Barcelona", "Sabadell", "Gràcia", "Mataró", "Terrassa", "Catalunya", "Hospitalet"]
PEOPLE = [
    "locutor no identificado",
    "redactor no identificado",
    "sección cultural",
    "sección deportiva",
    "archivo interno",
    "sección femenina",
    "programa infantil",
]

PROGRAM_TYPES_BY_PERIOD = {
    "1934_1935": [
        ("sports_bulletin", 0.28),
        ("music_program", 0.24),
        ("local_news", 0.28),
        ("cultural_program", 0.20),
    ],
    "1936_1938": [
        ("war_bulletin", 0.34),
        ("emergency_news", 0.24),
        ("political_announcement", 0.20),
        ("cultural_program", 0.10),
        ("sports_bulletin", 0.06),
        ("music_program", 0.06),
    ],
    "1939_1945": [
        ("official_bulletin", 0.26),
        ("continuity_script", 0.22),
        ("administrative_notice", 0.18),
        ("music_program", 0.16),
        ("sports_bulletin", 0.10),
        ("local_news", 0.08),
    ],
    "1946_1953": [
        ("women_program", 0.14),
        ("children_program", 0.14),
        ("interview", 0.18),
        ("advertising", 0.14),
        ("sports_bulletin", 0.16),
        ("music_program", 0.12),
        ("local_news", 0.12),
    ],
}

TOPICS_BY_TYPE = {
    "sports_bulletin": ["fútbol", "marcador", "boletín deportivo", "Sabadell", "Barcelona"],
    "music_program": ["música", "pasodobles", "concierto", "selección instrumental"],
    "local_news": ["vida cotidiana", "comercio", "Barcelona", "barrio"],
    "cultural_program": ["teatro", "canciones populares", "cultura", "República"],
    "war_bulletin": ["frente", "boletín", "guerra", "versión autorizada"],
    "emergency_news": ["situación militar", "Barcelona", "orden", "comunicado"],
    "political_announcement": ["República", "comunicado", "nota autorizada", "emisión"],
    "official_bulletin": ["radiodifusión", "intervención", "emisoras privadas", "saludo oficial"],
    "continuity_script": ["emisión de continuidad", "locutor", "saludo oficial", "Barcelona"],
    "administrative_notice": ["autorizado", "revisión", "emisión", "sello"],
    "women_program": ["hogar", "recetas", "consejos", "programa femenino"],
    "children_program": ["canción", "concurso", "adivinanzas", "programa infantil"],
    "interview": ["entrevista", "Gràcia", "comercio", "vida cotidiana"],
    "advertising": ["cuña publicitaria", "música ligera", "anuncio", "locutor"],
}

TITLE_MAP = {
    "sports_bulletin": "Guion deportivo con resultados de fútbol",
    "music_program": "Programa musical con revisión previa",
    "local_news": "Guion de actualidad local",
    "cultural_program": "Programa cultural con referencias sociales",
    "war_bulletin": "Parte de guerra and anuncio de lectura restringida",
    "emergency_news": "Guion informativo sobre disturbios and situación política",
    "political_announcement": "Comunicado político para emisión controlada",
    "official_bulletin": "Boletín con referencias a radiodifusión e intervención",
    "continuity_script": "Guion de continuidad con saludo oficial",
    "administrative_notice": "Región con sello and texto de autorización",
    "women_program": "Programa femenino and consejos del hogar",
    "children_program": "Guion infantil con canción and concurso",
    "interview": "Guion de entrevista local",
    "advertising": "Guion publicitario and música ligera",
}

BASE_MARKS = [
    {"mark_type": "censorship_stamp", "subtype": "probable", "description": "Rectangular stamp near top margin"},
    {"mark_type": "crossed_out_text", "subtype": "handwritten", "description": "Diagonal pen strokes over one sentence"},
    {"mark_type": "handwritten_note", "subtype": "editorial_correction", "description": "Editorial correction in right margin"},
    {"mark_type": "administrative_stamp", "subtype": "reviewed", "description": "Rectangular review stamp"},
    {"mark_type": "archive_stamp", "subtype": "later_processing", "description": "Later archival stamp"},
]

REQUIRED_TOP = [
    "schema_version", "record_id", "document_id", "unit_type", "source_type",
    "retrieval_text", "forensic_flags", "provenance"
]

REQUIRED_FORENSIC = ["confidence_score", "verification_status", "uncertainty_state"]
REQUIRED_PROVENANCE = ["source_image_path"]


def weighted_choice(weighted_items):
    """
    Select an item from a list based on associated weights
    """
    items = [x[0] for x in weighted_items]
    weights = [x[1] for x in weighted_items]
    return random.choices(items, weights=weights, k=1)[0]


def choose_period(year):
    """
    Identify the historical period for a given year
    """
    if year <= 1935:
        return "1934_1935"
    if year <= 1938:
        return "1936_1938"
    if year <= 1945:
        return "1939_1945"
    return "1946_1953"


def random_date(year):
    """
    Generate a random ISO date string for a given year
    """
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def make_bbox():
    """
    Generate a random bounding box within standard page dimensions
    """
    x1 = random.randint(80, 1800)
    y1 = random.randint(80, 2800)
    x2 = x1 + random.randint(180, 900)
    y2 = y1 + random.randint(80, 500)
    return [x1, y1, min(x2, 2400), min(y2, 3400)]


def gen_entities(program_type, year):
    """
    Generate synthetic named entities based on program type and year
    """
    entities = []
    if random.random() < 0.72:
        entities.append({
            "text": "Radio Barcelona",
            "type": "organization",
            "normalized": "Radio Barcelona",
            "confidence_score": round(random.uniform(0.93, 0.99), 2)
        })
    if random.random() < 0.62:
        place = random.choice(PLACES)
        entities.append({
            "text": place,
            "type": "place",
            "normalized": place,
            "confidence_score": round(random.uniform(0.9, 0.99), 2)
        })
    if 1936 <= year <= 1938 and random.random() < 0.42:
        entities.append({
            "text": "República",
            "type": "political_entity",
            "normalized": "Segunda República Española",
            "confidence_score": round(random.uniform(0.7, 0.88), 2)
        })
    return entities


def gen_mark_text(year):
    """
    Generate typical stamp text for a given historical period
    """
    if year >= 1939:
        return ["DIRECCIÓN GENERAL DE RADIODIFUSIÓN", "BARCELONA", "INTERVENCIÓN DE EMISORAS PRIVADAS"]
    return ["REVISADO", "BARCELONA"]


def gen_visual_marks(year, program_type, degraded=False):
    """
    Generate synthetic visual marks based on historical context
    """
    marks = []
    period = choose_period(year)

    if period == "1936_1938":
        n = random.choice([1, 2, 2, 3])
    elif period == "1939_1945":
        n = random.choice([1, 2, 2, 3])
    else:
        n = random.choice([0, 1, 1, 2, 2])

    candidate_pool = BASE_MARKS.copy()

    if period == "1936_1938":
        candidate_pool += [{
            "mark_type": "censorship_stamp",
            "subtype": "probable",
            "description": "Rectangular censorship mark"
        }]
    if period == "1939_1945":
        candidate_pool += [{
            "mark_type": "stamp",
            "subtype": "pasted",
            "description": "Circular red official stamp",
            "shape": "circular",
            "colour": "red"
        }]

    for i in range(n):
        base = random.choice(candidate_pool).copy()
        low, high = (0.35, 0.82) if degraded else (0.55, 0.98)
        mark = {
            "mark_id": f"m{i+1:03d}",
            "mark_type": base["mark_type"],
            "subtype": base["subtype"],
            "description": base["description"],
            "bbox": make_bbox(),
            "confidence_score": round(random.uniform(low, high), 2),
            "verification_status": random.choices(
                ["verified", "uncertain"],
                weights=[0.72, 0.28],
                k=1
            )[0]
        }
        if "shape" in base:
            mark["shape"] = base["shape"]
        if "colour" in base:
            mark["colour"] = base["colour"]
            mark["text"] = gen_mark_text(year)
        marks.append(mark)

    if year >= 1939 and random.random() < 0.28:
        marks.append({
            "mark_id": f"m{len(marks)+1:03d}",
            "mark_type": "stamp",
            "subtype": "pasted",
            "shape": "circular",
            "colour": "red",
            "description": "Red circular official stamp with intervention text",
            "text": [
                "DIRECCIÓN GENERAL DE RADIODIFUSIÓN",
                "INTERVENCIÓN DE EMISORAS PRIVADAS"
            ],
            "bbox": make_bbox(),
            "confidence_score": round(random.uniform(0.9, 0.99), 2),
            "verification_status": "verified"
        })

    return marks


def gen_ocr_text(program_type, topics):
    """
    Generate synthetic OCR text using templates and topics
    """
    templates = {
        "sports_bulletin": "Resultados de fútbol de la jornada. {a}. {b}. Revisar marcador antes de emisión.",
        "music_program": "Programa musical. {a}. {b}. El locutor anunciará cada pieza sin comentarios adicionales.",
        "local_news": "Boletín local. {a}. {b}. Información para los oyentes de Barcelona.",
        "cultural_program": "Programa cultural. {a}. {b}. Corregir orden del segundo bloque.",
        "war_bulletin": "Parte informativo. {a}. {b}. Leer únicamente la versión autorizada para antena.",
        "emergency_news": "Boletín extraordinario. {a}. {b}. Mantener tono de calma y orden.",
        "political_announcement": "Comunicado. {a}. {b}. El locutor leerá la nota autorizada.",
        "official_bulletin": "Boletín de servicio. {a}. {b}. Léase con entera exactitud.",
        "continuity_script": "Emisión de continuidad. {a}. {b}. Evítese toda improvisación.",
        "administrative_notice": "Autorizado para emisión. {a}. {b}. Orden revisado.",
        "women_program": "Programa femenino. {a}. {b}. Duración aproximada: doce minutos.",
        "children_program": "Programa infantil. {a}. {b}. Concurso y despedida del locutor.",
        "interview": "Entrevista local. {a}. {b}. Orden de preguntas al margen.",
        "advertising": "Cuña publicitaria. {a}. {b}. Entrada del locutor y cierre final."
    }
    return templates[program_type].format(a=topics, b=topics if len(topics) > 1 else topics)[1]


def gen_forensic_flags(ocr_conf, marks, year, degraded=False):
    """
    Calculate forensic flags and verification status based on confidence and context
    """
    visual_scores = [m["confidence_score"] for m in marks] or [0.9]
    avg = round((ocr_conf + sum(visual_scores) / len(visual_scores)) / 2, 2)
    if degraded:
        avg = round(max(0.35, avg - 0.12), 2)
    verification = "verified" if avg >= 0.8 else "uncertain"
    uncertainty = "none" if avg >= 0.8 else "pending_reconstruction"
    alerts = []
    if ocr_conf < 0.7:
        alerts.append("LOW_OCR_CONFIDENCE")
    if any(m["confidence_score"] < 0.6 for m in marks):
        alerts.append("LOW_VISUAL_CONFIDENCE")
    if any("intervención" in " ".join(m.get("text", [])).lower() for m in marks):
        alerts.append("FRANCOIST_ADMINISTRATIVE_MARK")
    if 1936 <= year <= 1939 and any(m["mark_type"] in {"censorship_stamp", "crossed_out_text"} for m in marks):
        alerts.append("POLITICALLY_SENSITIVE_CONTENT")
    if verification == "uncertain":
        alerts.append("HUMAN_REVIEW_NEEDED")
    return {
        "confidence_score": avg,
        "verification_status": verification,
        "uncertainty_state": uncertainty,
        "alerts": sorted(set(alerts)),
        "human_review_required": verification == "uncertain"
    }


def build_retrieval_text(title, summary, entities, marks, ocr_text):
    """
    Construct a dense text representation for indexing and retrieval
    """
    entity_txt = ", ".join(e["text"] for e in entities) if entities else "none"
    mark_txt = ", ".join(m["mark_type"] for m in marks) if marks else "none"
    return (
        f"{title}. {summary}. OCR text: {ocr_text} "
        f"Entities: {entity_txt}. Detected marks: {mark_txt}."
    )


def validate_record(record):
    """
    Perform schema validation on a generated record
    """
    errors = []
    for key in REQUIRED_TOP:
        if key not in record:
            errors.append(f"Missing top-level field: {key}")
    forensic = record.get("forensic_flags", {})
    for key in REQUIRED_FORENSIC:
        if key not in forensic:
            errors.append(f"Missing forensic_flags field: {key}")
    prov = record.get("provenance", {})
    for key in REQUIRED_PROVENANCE:
        if key not in prov:
            errors.append(f"Missing provenance field: {key}")
    if record.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if record.get("unit_type") not in {"page", "region"}:
        errors.append("unit_type must be page or region")
    ff = record.get("forensic_flags", {})
    if ff.get("verification_status") not in {"verified", "uncertain"}:
        errors.append("verification_status invalid")
    conf = ff.get("confidence_score")
    if conf is None or not (0 <= conf <= 1):
        errors.append("forensic confidence_score must be between 0 and 1")
    ocr_conf = record.get("ocr", {}).get("confidence_score")
    if ocr_conf is None or not (0 <= ocr_conf <= 1):
        errors.append("ocr confidence_score must be between 0 and 1")
    return errors


def generate_record(idx):
    """
    Generate a complete synthetic archival record with rich metadata
    """
    year = random.randint(1934, 1953)
    period = choose_period(year)
    program_type = weighted_choice(PROGRAM_TYPES_BY_PERIOD[period])
    topics = random.sample(TOPICS_BY_TYPE[program_type], k=min(4, len(TOPICS_BY_TYPE[program_type])))
    date_created = random_date(year)

    degraded = random.random() < (0.18 if year >= 1948 else 0.08)
    region_id = None if random.random() < 0.83 else f"r{random.randint(1, 3):02d}"
    unit_type = "region" if region_id else "page"

    doc_num = f"{idx:03d}"
    page_num = f"p{random.randint(1, 6):03d}"
    title = TITLE_MAP[program_type]
    ocr_text = gen_ocr_text(program_type, topics)
    ocr_low, ocr_high = (0.45, 0.78) if degraded else (0.74, 0.98)
    ocr_conf = round(random.uniform(ocr_low, ocr_high), 2)
    entities = gen_entities(program_type, year)
    marks = gen_visual_marks(year, program_type, degraded=degraded)

    short_description = f"{title.lower()} fechado en {year}, con temática de {', '.join(topics[:2])}."
    retrieval_text = build_retrieval_text(title, short_description, entities, marks, ocr_text)
    forensic_flags = gen_forensic_flags(ocr_conf, marks, year, degraded=degraded)

    record = {
        "schema_version": "1.0",
        "record_id": f"rec_{year}_{doc_num}_{page_num}",
        "document_id": f"doc_{year}_{doc_num}",
        "page_id": page_num,
        "region_id": region_id,
        "unit_type": unit_type,
        "source_type": "radio_script",
        "collection_id": COLLECTION_ID,
        "collection_name": COLLECTION_NAME,
        "title": title,
        "date_created": date_created,
        "language": random.choice([["es"], ["es", "ca"]]),
        "creator": ["Radio Barcelona"],
        "contributors": [random.choice(PEOPLE)],
        "rights": "unknown",
        "physical_metadata": {
            "file_name": f"doc_{year}_{doc_num}_{page_num}.jpg",
            "file_path": f"/archive/uab/doc_{year}_{doc_num}/{page_num}.jpg",
            "file_format": "image/jpeg",
            "mime_type": "image/jpeg",
            "width": 2480,
            "height": 3508,
            "dpi": 300
        },
        "archival_metadata": {
            "fonds": "Radio Barcelona",
            "series": "Guiones de emisión",
            "box": f"RB-{year}-{random.randint(1, 12):02d}",
            "folder": f"Carpeta {year}",
            "call_number": f"RB-GUI-{year}-{doc_num}",
            "institution": INSTITUTION,
            "repository": REPOSITORY
        },
        "retrieval_text": retrieval_text,
        "ocr": {
            "full_text": ocr_text if not degraded else ocr_text.replace(".", " ... [zona dañada].", 1),
            "keywords": topics,
            "language": "es",
            "engine": "synthetic_ocr_v1",
            "confidence_score": ocr_conf
        },
        "summary": {
            "short_description": short_description if not degraded else short_description + " Página con degradación física apreciable.",
            "keywords": topics[:4]
        },
        "entities": entities,
        "visual_marks": marks,
        "linked_text_spans": [],
        "ai_metadata": {
            "detected_topics": topics[:3],
            "scene_description": f"Archival {unit_type} with typed radio content and possible manual interventions.",
            "material_observations": random.sample([
                "ink contrast variation",
                "paper yellowing",
                "minor fold",
                "stamp impression",
                "manual crossing out",
                "later archive handling mark",
                "text loss in central region",
                "staining"
            ], k=2),
            "model_outputs": {
                "vision_model": "synthetic_vision_v1",
                "entity_model": "synthetic_ner_v1"
            }
        },
        "forensic_flags": forensic_flags,
        "provenance": {
            "source_image_path": f"/archive/uab/doc_{year}_{doc_num}/{page_num}.jpg",
            "derived_from": [
                f"ocr_output_{year}_{doc_num}_{page_num}.json",
                f"vision_output_{year}_{doc_num}_{page_num}.json"
            ],
            "pipeline_version": "prototype_v2",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    }

    span_candidates = [
        "fragmento tachado",
        "leer únicamente la versión autorizada",
        "evítese toda improvisación",
        "orden revisado",
        "12 min",
        "autorizado para emisión"
    ]
    for i, mark in enumerate(marks[:2], start=1):
        record["linked_text_spans"].append({
            "span_id": f"s{i:03d}",
            "text": random.choice(span_candidates),
            "span_type": "ocr_span",
            "bbox": make_bbox(),
            "related_mark_ids": [mark["mark_id"]],
            "confidence_score": round(random.uniform(0.72, 0.94), 2)
        })

    return record


def main():
    """
    Orchestrate the generation and validation of synthetic archival records
    """
    records = []
    errors = []
    for idx in range(1, N_RECORDS + 1):
        rec = generate_record(idx)
        rec_errors = validate_record(rec)
        if rec_errors:
            errors.append({"record_id": rec.get("record_id"), "errors": rec_errors})
        else:
            records.append(rec)

    os.makedirs(os.path.join(BASE_PATH, "metadata"), exist_ok=True)  # Ensure metadata directory exists

    if EXPORT_JSONL:
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")  # Write line-delimited JSON

    if EXPORT_JSON:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)  # Write formatted JSON array

    print(json.dumps({
        "records_written": len(records),
        "validation_errors": len(errors),
        "jsonl": OUTPUT_JSONL if EXPORT_JSONL else None,
        "json": OUTPUT_JSON if EXPORT_JSON else None,
        "sample_record_ids": [r["record_id"] for r in records[:5]]
    }, ensure_ascii=False, indent=2))  # Log generation summary

    if errors:
        print(json.dumps(errors[:5], ensure_ascii=False, indent=2))  # Log sample validation errors


if __name__ == "__main__":
    main()
