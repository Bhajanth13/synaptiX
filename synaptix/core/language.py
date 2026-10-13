"""Supported languages, ordered roughly by number of speakers.

Codes are ISO 639 where possible. `sarvam` is the Sarvam AI TTS locale; None means
Sarvam has no voice for it and we fall back to `tts_fallback`.
"""

LANGUAGES = {
    "hi":  {"name": "Hindi",     "script": "Devanagari", "sarvam": "hi-IN"},
    "en":  {"name": "English",   "script": "Latin",      "sarvam": "en-IN"},
    "bn":  {"name": "Bengali",   "script": "Bengali",    "sarvam": "bn-IN"},
    "mr":  {"name": "Marathi",   "script": "Devanagari", "sarvam": "mr-IN"},
    "te":  {"name": "Telugu",    "script": "Telugu",     "sarvam": "te-IN"},
    "ta":  {"name": "Tamil",     "script": "Tamil",      "sarvam": "ta-IN"},
    "gu":  {"name": "Gujarati",  "script": "Gujarati",   "sarvam": "gu-IN"},
    "kn":  {"name": "Kannada",   "script": "Kannada",    "sarvam": "kn-IN"},
    "ml":  {"name": "Malayalam", "script": "Malayalam",  "sarvam": "ml-IN"},
    "or":  {"name": "Odia",      "script": "Odia",       "sarvam": "od-IN"},
    "pa":  {"name": "Punjabi",   "script": "Gurmukhi",   "sarvam": "pa-IN"},
    "mwr": {"name": "Marwari",   "script": "Devanagari", "sarvam": None, "tts_fallback": "hi"},
}

# Unicode block ranges used for script detection.
_SCRIPT_RANGES = [
    ((0x0900, 0x097F), "hi"),  # Devanagari: hi / mr / mwr are ambiguous, default hi
    ((0x0980, 0x09FF), "bn"),
    ((0x0A00, 0x0A7F), "pa"),
    ((0x0A80, 0x0AFF), "gu"),
    ((0x0B00, 0x0B7F), "or"),
    ((0x0B80, 0x0BFF), "ta"),
    ((0x0C00, 0x0C7F), "te"),
    ((0x0C80, 0x0CFF), "kn"),
    ((0x0D00, 0x0D7F), "ml"),
]


def detect(text: str) -> str:
    """Guess the language from the script. Devanagari defaults to Hindi; pass lang explicitly for mr/mwr."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for (lo, hi), code in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    return max(counts, key=counts.get) if counts else "en"


def tts_locale(lang: str) -> tuple[str, bool]:
    """Return (sarvam_locale, is_fallback)."""
    info = LANGUAGES.get(lang, LANGUAGES["en"])
    if info["sarvam"]:
        return info["sarvam"], False
    return LANGUAGES[info.get("tts_fallback", "en")]["sarvam"], True
