# -*- coding: utf-8 -*-
"""
Embedded word-based text generator for printed and strict-mode handwriting regions

Uses a compact embedded Spanish word list to produce natural-looking text with
no external file dependency
"""

import os
import random

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Common Spanish words covering articles, prepositions, nouns, verbs, adjectives
VOCAB_SPANISH = [
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "en", "con", "por", "para", "a", "al", "del", "sin", "sobre",
    "que", "y", "o", "pero", "si", "como", "cuando", "donde", "porque", "aunque",
    "es", "era", "fue", "son", "han", "hay", "ser", "estar", "tener", "hacer",
    "poder", "decir", "ir", "ver", "dar", "saber", "querer", "llegar", "pasar",
    "hablar", "llevar", "dejar", "seguir", "encontrar", "llamar", "venir", "pensar",
    "ciudad", "casa", "hombre", "mujer", "año", "día", "tiempo", "vida", "manera",
    "parte", "mundo", "caso", "cosa", "lugar", "estado", "momento", "forma", "tipo",
    "gobierno", "país", "familia", "trabajo", "punto", "mano", "orden", "agua",
    "nombre", "persona", "gente", "historia", "palabra", "grupo", "guerra", "fuerza",
    "gran", "nuevo", "mismo", "otro", "mucho", "poco", "todo", "cada", "varios",
    "primero", "último", "alto", "bajo", "largo", "propio", "segundo", "general",
    "nacional", "local", "público", "social", "político", "militar", "civil",
    "se", "le", "lo", "me", "te", "nos", "su", "sus", "mi", "mis", "tu", "tus",
    "este", "esta", "estos", "estas", "ese", "esa", "aquel", "aquella",
    "no", "más", "ya", "también", "bien", "siempre", "nunca", "solo", "así",
    "entonces", "después", "antes", "ahora", "aquí", "allí", "muy", "tan",
    "Barcelona", "Madrid", "España", "radio", "prensa", "decreto", "artículo",
    "señor", "señora", "doctor", "comandante", "presidente", "ministro",
]

PROB_COMMA = 0.06
PROB_PERIOD = 0.04

def generate_fill_text(n_words):
    """
    Generate a natural-looking Spanish text string of approximately n_words words
    """
    words = [random.choice(VOCAB_SPANISH) for _ in range(max(1, n_words))]
    words[0] = words[0].capitalize()
    result = []
    
    for i, w in enumerate(words):
        if i > 0 and random.random() < PROB_COMMA:
            result[-1] += ","
        elif i > 0 and random.random() < PROB_PERIOD:
            result[-1] += "."
            w = w.capitalize()
        result.append(w)
        
    return " ".join(result)
