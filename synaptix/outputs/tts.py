"""Text to speech.

Default: `RuleTTS`, a hard-coded algorithm with no model, no network and no recordings:
    text --(g2p.py: Unicode script rules)--> phonemes --(formant.py: formant synthesis)--> WAV
It works for every supported language because Indian scripts share one phonetic layout.
Marwari uses the Hindi rules (Devanagari, final-schwa deletion).

Optional: `SarvamTTS` (Sarvam AI Bulbul cloud voices) for natural speech when online.
Clips are cached, and if Sarvam is unreachable it falls back to RuleTTS.
"""

import base64
import hashlib
import json
import os
import urllib.request
from pathlib import Path

from synaptix.core.language import tts_locale
from synaptix.outputs.formant import synth, write_wav
from synaptix.outputs.g2p import to_phonemes

CACHE_DIR = Path(os.environ.get("SYNAPTIX_CACHE", Path.home() / ".cache" / "synaptix" / "tts"))


def _cache_path(cache_dir: Path, *parts: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (hashlib.sha256("|".join(parts).encode()).hexdigest()[:24] + ".wav")


class RuleTTS:
    """Hard-coded, fully offline TTS."""

    def __init__(self, rate: float = 1.0, cache_dir: Path = CACHE_DIR):
        self.rate, self.cache_dir = rate, Path(cache_dir)

    def phonemes(self, text: str, lang: str) -> list[str]:
        return to_phonemes(text, lang)

    def synthesize(self, text: str, lang: str) -> Path:
        path = _cache_path(self.cache_dir, "rule", str(self.rate), lang, text)
        if not path.exists():
            write_wav(synth(self.phonemes(text, lang), self.rate), path)
        return path


class SarvamTTS:
    """Optional natural voices via Sarvam AI (needs internet and SARVAM_API_KEY)."""

    URL = "https://api.sarvam.ai/text-to-speech"

    def __init__(self, api_key: str | None = None, model: str = "bulbul:v3", speaker: str = "shubh",
                 cache_dir: Path = CACHE_DIR, timeout: float = 20):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model, self.speaker, self.timeout, self.cache_dir = model, speaker, timeout, Path(cache_dir)
        self.fallback = RuleTTS(cache_dir=cache_dir)

    def synthesize(self, text: str, lang: str) -> Path:
        locale, _ = tts_locale(lang)
        path = _cache_path(self.cache_dir, self.model, self.speaker, locale, text)
        if path.exists():
            return path
        if self.api_key:
            try:
                body = json.dumps({"text": text[:2500], "language_code": locale, "model": self.model,
                                   "speaker": self.speaker, "output_audio_codec": "wav"}).encode()
                req = urllib.request.Request(self.URL, data=body, method="POST", headers={
                    "api-subscription-key": self.api_key, "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    path.write_bytes(base64.b64decode(json.load(r)["audios"][0]))
                return path
            except (OSError, KeyError, ValueError):
                pass
        return self.fallback.synthesize(text, lang)


def get_tts(engine: str = "rule"):
    return SarvamTTS() if engine == "sarvam" else RuleTTS()
