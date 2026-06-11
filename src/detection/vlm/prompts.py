# -*- coding: utf-8 -*-
"""
Spanish prompts for the Qwen3-VL reader

Region transcription is style-agnostic (the same prompt reads printed and handwritten crops - the layer is already known from segmentation, so we do not ask the model to classify the style, only to transcribe) The noise prompt asks for a short prose description of marks/stamps/strikethroughs
"""

import os

# Routes
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT_TRANSCRIBE = (
    "Transcribe ABSOLUTAMENTE TODO el texto visible en esta imagen, sin importar su "
    "estilo (impreso o manuscrito). No clasifiques el texto, solo transcríbelo. No uses "
    "secuencias repetitivas de tabulaciones ni espacios. Devuelve únicamente el texto "
    "transcrito, sin comentarios ni explicaciones. "
    "IMPORTANTE: esta instrucción no debe aparecer en tu respuesta bajo ningún concepto. "
    "Si no hay texto visible, responde con una cadena vacía."
)

PROMPT_NOISE = (
    "Esta imagen contiene únicamente ruido visual: manchas, sellos, tachaduras y marcas. "
    "Descríbelo brevemente en español (tipo de marca, color y posición aproximada). "
    "No transcribas texto; solo describe las marcas."
)
