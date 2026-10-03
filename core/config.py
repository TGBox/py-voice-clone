# -*- coding: utf-8 -*-
"""
Application configuration and path constants.
"""

from pathlib import Path
import os
import sys

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data storage directories
DATA_DIR = BASE_DIR / "data"
PROFILES_DIR = DATA_DIR / "profiles"
OUTPUTS_DIR = DATA_DIR / "outputs"
RECORDINGS_DIR = DATA_DIR / "recordings"
MODELS_DIR = BASE_DIR / "models"
XTTS_MODEL_DIR = MODELS_DIR / "xtts_v2"

# Ensure all primary runtime directories exist
for path in [DATA_DIR, PROFILES_DIR, OUTPUTS_DIR, RECORDINGS_DIR, MODELS_DIR, XTTS_MODEL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# HuggingFace repository ID for XTTS-v2
HF_XTTS_REPO = "coqui/XTTS-v2"

# Supported languages in XTTS v2 with German display names
SUPPORTED_LANGUAGES = {
    "de": "Deutsch",
    "en": "Englisch",
    "fr": "Französisch",
    "es": "Spanisch",
    "it": "Italienisch",
    "pt": "Portugiesisch",
    "pl": "Polnisch",
    "tr": "Türkisch",
    "ru": "Russisch",
    "nl": "Niederländisch",
    "cs": "Tschechisch",
    "ar": "Arabisch",
    "zh-cn": "Chinesisch",
    "ja": "Japanisch",
    "ko": "Koreanisch",
    "hu": "Ungarisch",
    "hi": "Hindi",
}

# Audio reference limits
MIN_REF_DURATION_SEC = 2.0
RECOMMENDED_MIN_DURATION_SEC = 4.0
RECOMMENDED_MAX_DURATION_SEC = 15.0
MAX_REF_DURATION_SEC = 30.0

# Synthesis defaults
DEFAULT_TEMPERATURE = 0.75
DEFAULT_REPETITION_PENALTY = 2.0
DEFAULT_SPEED = 1.0
DEFAULT_LENGTH_PENALTY = 1.0
DEFAULT_SAMPLE_RATE = 24000
