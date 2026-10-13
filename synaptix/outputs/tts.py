"""Text to speech.

Primary: Sarvam AI Bulbul (https://docs.sarvam.ai), natural Indian voices in 11 languages.
It is a cloud API, so it needs internet and SARVAM_API_KEY. To stay offline-first:
  - every clip is cached on disk by (text, language, speaker), so verified answer cards can be
    pre-rendered once while online (`synaptix prerender`) and replayed offline forever;
  - if Sarvam is unreachable, we fall back to espeak-ng (robotic, but local).
Marwari has no Sarvam voice yet, so it is spoken with the Hindi voice.
"""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from synaptix.core.language import tts_locale

SARVAM_URL = "https://api.sarvam.ai/text-to-speech"
CACHE_DIR = Path(os.environ.get("SYNAPTIX_CACHE", Path.home() / ".cache" / "synaptix" / "tts"))


class SarvamTTS:
    def __init__(self, api_key: str | None = None, model: str = "bulbul:v3", speaker: str = "shubh",
                 cache_dir: Path = CACHE_DIR, timeout: float = 20):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model, self.speaker, self.timeout = model, speaker, timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str, locale: str) -> Path:
        key = hashlib.sha256(f"{self.model}|{self.speaker}|{locale}|{text}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{key}.wav"

    def synthesize(self, text: str, lang: str) -> Path:
        """Return a WAV file path for text spoken in lang (cache -> Sarvam -> espeak-ng)."""
        locale, _ = tts_locale(lang)
        path = self._cache_path(text, locale)
        if path.exists():
            return path
        if self.api_key:
            try:
                path.write_bytes(self._sarvam(text, locale))
                return path
            except OSError:
                pass  # offline or API error: fall through to local voice
        return espeak(text, lang, path)

    def _sarvam(self, text: str, locale: str) -> bytes:
        body = json.dumps({"text": text[:2500], "language_code": locale, "model": self.model,
                           "speaker": self.speaker, "output_audio_codec": "wav"}).encode()
        req = urllib.request.Request(SARVAM_URL, data=body, method="POST", headers={
            "api-subscription-key": self.api_key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.load(r)
        return base64.b64decode(data["audios"][0])


_ESPEAK_VOICES = {"hi": "hi", "en": "en-in", "bn": "bn", "mr": "mr", "te": "te", "ta": "ta",
                  "gu": "gu", "kn": "kn", "ml": "ml", "or": "or", "pa": "pa", "mwr": "hi"}


def espeak(text: str, lang: str, out: Path) -> Path:
    exe = shutil.which("espeak-ng")
    if not exe:
        raise RuntimeError("No TTS available: set SARVAM_API_KEY (online) or install espeak-ng (offline).")
    subprocess.run([exe, "-v", _ESPEAK_VOICES.get(lang, "en"), "-w", str(out), text], check=True)
    return out
