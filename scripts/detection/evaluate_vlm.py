# -*- coding: utf-8 -*-
"""
Evaluation pipeline for VLM transcription performance

This module assesses the quality of the VLM transcription by calculating Character Error Rate (CER) and Word Error Rate (WER) across different document types. It evaluates printed text rendered in archive-style fonts and handwritten text from the IAM dataset, providing a comprehensive benchmark for the model's transcription accuracy and confidence calibration
"""

import os
import glob
import json
import random
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import metrics as M
from src.detection.vlm.qwen import QwenVLM

PATH_FILE_FONT = os.path.join(BASE_PATH, "data", "assets", "fonts", "SpecialElite-Regular.ttf")
PATH_DIR_IAM_LIB = os.path.join(BASE_PATH, "data", "iam", "library")

# Period radio-script-style Spanish lines for printed transcription benchmarking
PRINTED_LINES = [
    "Emisión autorizada por la Junta de Censura de Radio Barcelona",
    "Programa de la sección deportiva del domingo por la tarde",
    "Se ruega al público guardar silencio durante la transmisión",
    "Noticias del frente y partes de guerra del Estado Mayor",
    "Concierto de música sinfónica dirigido por el maestro Toldrá",
    "Anuncio de los productos de la casa comercial patrocinadora",
    "Boletín informativo de las nueve de la noche hora peninsular",
    "Charla sobre higiene y salud pública a cargo del doctor",
    "Retransmisión del discurso pronunciado en el Palacio Nacional",
    "Sección de variedades con la actuación de artistas invitados",
    "El locutor saluda a los oyentes y presenta el programa de hoy",
    "Cuadro de honor de los donantes para la suscripción benéfica",
    "Tiempo probable para mañana en la región de Cataluña",
    "Lectura de los telegramas recibidos en nuestra redacción",
    "Despedida y cierre de la emisión hasta el día siguiente",
    "Servicio religioso transmitido desde la catedral de Barcelona",
    "Cotizaciones de la bolsa y mercados de la jornada de hoy",
    "Cuento infantil narrado para los pequeños radioyentes",
    "Resumen de las crónicas de sociedad de la semana pasada",
    "Aviso oficial del gobierno civil de la provincia",
]

PAPER_RGB = (245, 242, 235)  # Simulated archival paper color
INK_RGB = (30, 28, 25)  # Simulated faded ink color

def get_artifact_dir(phase: str, model_name: str, artifact_type: str) -> str:
    """
    Retrieve the standardized directory path for model artifacts
    """
    path = os.path.join(BASE_PATH, "outputs", phase, model_name, artifact_type)
    os.makedirs(path, exist_ok=True)
    return path

def render_printed(text, font_size=40, pad=24):
    """
    Render a text string in the archive font on a paper-toned background
    """
    font = ImageFont.truetype(PATH_FILE_FONT, font_size)
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    x0, y0, x1, y1 = dummy.textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), PAPER_RGB)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill=INK_RGB)
    return img

def iam_pairs(n, seed=0):
    """
    Sample ground truth handwriting pairs from the IAM library
    """
    pngs = glob.glob(os.path.join(PATH_DIR_IAM_LIB, "**", "*.png"), recursive=True)
    random.Random(seed).shuffle(pngs)
    out = []
    for p in pngs:
        txt = p[:-4] + ".txt"
        if not os.path.exists(txt):
            continue
        with open(txt, "r", encoding="utf-8") as f:
            gt = f.read().strip()
        if gt:
            out.append((Image.open(p).convert("RGB"), gt))
        if len(out) >= n:
            break
    return out

def _summary(rows):
    """
    Calculate aggregate error metrics for a collection of transcriptions
    """
    if not rows:
        return {}
    cer = np.mean([M.cer(g, h) for g, h in rows])
    cern = np.mean([M.cer(g, h, normalized=True) for g, h in rows])
    wer = np.mean([M.wer(g, h) for g, h in rows])
    wern = np.mean([M.wer(g, h, normalized=True) for g, h in rows])
    exact = np.mean([1.0 if M.normalize_text(g) == M.normalize_text(h) else 0.0 for g, h in rows])
    return {
        "n": len(rows),
        "cer": float(cer),
        "cer_norm": float(cern),
        "wer": float(wer),
        "wer_norm": float(wern),
        "exact_match": float(exact)
    }

def main(args):
    """
    Execute the VLM transcription evaluation pipeline
    """
    vlm = QwenVLM(device_map="auto")
    samples = []  # Store metadata for each transcription attempt

    def _read(crop):
        res = vlm.transcribe_region(crop, max_new_tokens=args.max_new_tokens)
        return res[0], res[1]  # Extract text and confidence score

    printed = random.Random(0).sample(PRINTED_LINES, min(args.printed_samples, len(PRINTED_LINES))) \
        if args.printed_samples <= len(PRINTED_LINES) else PRINTED_LINES
    
    for gt in printed:
        hyp, conf = _read(render_printed(gt))
        samples.append(("printed", gt, hyp, conf))

    for img, gt in iam_pairs(args.iam_samples):
        hyp, conf = _read(img)
        samples.append(("handwritten", gt, hyp, conf))

    pr_rows = [(g, h) for layer, g, h, _ in samples if layer == "printed"]
    hw_rows = [(g, h) for layer, g, h, _ in samples if layer == "handwritten"]

    confs = [c for *_, c in samples]
    cers = [M.cer(g, h, normalized=True) for _, g, h, _ in samples]
    edges = np.linspace(0.0, 1.0, 11)
    bin_center, mean_cer, count = [], [], []
    
    for i in range(10):
        sel = [j for j, c in enumerate(confs) if edges[i] <= c < edges[i + 1] or (i == 9 and c == 1.0)]
        bin_center.append(float((edges[i] + edges[i + 1]) / 2))
        count.append(len(sel))
        mean_cer.append(float(np.mean([cers[j] for j in sel])) if sel else 0.0)

    out = {
        "stage": "vlm",
        "model": args.model,
        "note": "handwritten=IAM (English benchmark); printed=rendered period-Spanish",
        "printed": _summary(pr_rows),
        "handwritten": _summary(hw_rows),
        "overall": _summary(pr_rows + hw_rows),
        "confidence_vs_cer": {"bin_center": bin_center, "mean_cer": mean_cer, "count": count},
        "samples": [{"layer": l, "confidence": float(c), "cer_norm": M.cer(g, h, normalized=True)}
                    for l, g, h, c in samples],
    }
    
    out_dir = get_artifact_dir("detection", args.model, "metrics")
    out_path = os.path.join(out_dir, "vlm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"[*] Wrote {out_path}")
    pr, hw = out["printed"], out["handwritten"]
    print(f"[*] printed CER={pr.get('cer'):.3f} (norm {pr.get('cer_norm'):.3f}) | "
          f"handwritten CER={hw.get('cer'):.3f} (norm {hw.get('cer_norm'):.3f})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate Qwen3-VL transcription quality (CER/WER)")
    p.add_argument("--model", default="vlm_qwen", help="Model name for paths")
    p.add_argument("--printed-samples", type=int, default=20)
    p.add_argument("--iam-samples", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=128)
    main(p.parse_args())
