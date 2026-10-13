"""Hard-coded formant speech synthesizer (Klatt-style, pure numpy + stdlib).

Each phoneme is built from rules: vowels and voiced sounds are a glottal pulse train passed
through three resonators at the phoneme's formant frequencies; stops are a short silence plus
a filtered noise burst; fricatives are band-limited noise. No recordings, no model, no network.
It sounds robotic, but it is fully offline, tiny and deterministic.
"""

import math
import wave
from pathlib import Path

import numpy as np

FS = 16000

# (F1, F2, F3) in Hz, duration in ms
VOWELS = {"a": ((650, 1200, 2500), 90), "aa": ((750, 1250, 2550), 160),
          "i": ((350, 2100, 2800), 90), "ii": ((300, 2300, 3000), 160),
          "u": ((380, 900, 2400), 90), "uu": ((320, 800, 2300), 160),
          "e": ((450, 1900, 2600), 140), "o": ((480, 900, 2500), 140),
          "ai": ((600, 1700, 2600), 170), "au": ((600, 950, 2500), 170)}

# voiced continuants: formants, duration ms, amplitude
SONORANTS = {"m": ((250, 1000, 2200), 70, 0.5), "n": ((250, 1500, 2500), 70, 0.5),
             "nn": ((250, 1400, 2200), 70, 0.5), "ng": ((250, 1800, 2400), 70, 0.5),
             "ny": ((250, 2000, 2700), 70, 0.5), "l": ((360, 1300, 2600), 60, 0.7),
             "r": ((450, 1300, 1600), 35, 0.7), "y": ((300, 2200, 3000), 60, 0.7),
             "v": ((300, 1000, 2300), 60, 0.6)}

# stops: burst centre Hz, voiced?, aspirated?
STOPS = {"k": (2500, False, False), "kh": (2500, False, True), "g": (2500, True, False), "gh": (2500, True, True),
         "c": (3500, False, False), "ch": (3500, False, True), "j": (3500, True, False), "jh": (3500, True, True),
         "tt": (2800, False, False), "tth": (2800, False, True), "dd": (2800, True, False), "ddh": (2800, True, True),
         "t": (4000, False, False), "th": (4000, False, True), "d": (4000, True, False), "dh": (4000, True, True),
         "p": (1000, False, False), "ph": (1000, False, True), "b": (1000, True, False), "bh": (1000, True, True)}

# fricatives: noise band (low, high) Hz, duration ms, voiced?
FRICATIVES = {"s": ((4500, 7500), 110, False), "z": ((4500, 7500), 100, True),
              "sh": ((2200, 5000), 120, False), "f": ((1200, 6000), 100, False), "h": ((300, 4000), 70, False)}

PAUSES = {"_": 150, "__": 320, ".": 40}

_rng = np.random.default_rng(7)


def _ms(ms):
    return int(FS * ms / 1000)


def _resonate(x: np.ndarray, f: float, bw: float) -> np.ndarray:
    c = -math.exp(-2 * math.pi * bw / FS)
    b = 2 * math.exp(-math.pi * bw / FS) * math.cos(2 * math.pi * f / FS)
    a = 1 - b - c
    y = np.empty_like(x)
    y1 = y2 = 0.0
    for i, v in enumerate(x.tolist()):
        y0 = a * v + b * y1 + c * y2
        y[i] = y0
        y2, y1 = y1, y0
    return y


def _formants(x, fmts, bws=(80, 100, 150)):
    out = x
    for f, bw in zip(fmts, bws):
        out = _resonate(out, f, bw)
    return out


class _Glottis:
    """Pulse-train source with a falling pitch across the utterance."""

    def __init__(self, f0_start=135.0, f0_end=100.0, total=1):
        self.phase, self.f0_start, self.f0_end, self.total, self.done = 0.0, f0_start, f0_end, max(total, 1), 0

    def __call__(self, n):
        out = np.empty(n)
        for i in range(n):
            f0 = self.f0_start + (self.f0_end - self.f0_start) * min(self.done / self.total, 1)
            self.phase += f0 / FS
            if self.phase >= 1:
                self.phase -= 1
            out[i] = (1 - self.phase) ** 3 - 0.25  # sharp glottal closure, decaying open phase
            self.done += 1
        return out


def _noise(n, lo, hi):
    x = _rng.standard_normal(n)
    centre, bw = (lo + hi) / 2, (hi - lo)
    return _resonate(x, centre, bw) * 0.3


def _fade(x, ms=6):
    k = min(_ms(ms), len(x) // 2)
    if k:
        ramp = np.linspace(0, 1, k)
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


def synth(phonemes: list[str], rate: float = 1.0) -> np.ndarray:
    est = sum(_ms(VOWELS.get(p, (None, 80))[1]) for p in phonemes)
    glottis = _Glottis(total=int(est / rate))
    pieces = []
    for p in phonemes:
        if p in PAUSES:
            pieces.append(np.zeros(_ms(PAUSES[p] / rate)))
            continue
        if p in VOWELS:
            fm, dur = VOWELS[p]
            pieces.append(_fade(_formants(glottis(_ms(dur / rate)), fm)))
        elif p in SONORANTS:
            fm, dur, amp = SONORANTS[p]
            pieces.append(_fade(_formants(glottis(_ms(dur / rate)), fm) * amp))
        elif p in STOPS:
            centre, voiced, aspirated = STOPS[p]
            closure = _formants(glottis(_ms(45 / rate)), (250, 800, 2000)) * 0.15 if voiced else np.zeros(_ms(45 / rate))
            burst = _fade(_noise(_ms(18), centre * 0.7, centre * 1.3), 3)
            seg = [closure, burst]
            if aspirated:
                seg.append(_fade(_noise(_ms(55 / rate), 300, 4000) * 0.6))
            pieces.append(np.concatenate(seg))
        elif p in FRICATIVES:
            (lo, hi), dur, voiced = FRICATIVES[p]
            n = _ms(dur / rate)
            x = _noise(n, lo, hi)
            if voiced:
                x = x + _formants(glottis(n), (250, 1500, 2500)) * 0.3
            pieces.append(_fade(x))
    if not pieces:
        return np.zeros(0)
    y = np.concatenate(pieces)
    y = np.diff(y, prepend=0.0)  # lip radiation: brighter, more speech-like
    peak = np.abs(y).max() or 1.0
    return (y / peak * 0.85).astype(np.float32)


def write_wav(samples: np.ndarray, path: Path) -> Path:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    return path
