"""Rule-based grapheme-to-phoneme conversion (no model, no data files).

Indian scripts are phonetic and share the same Unicode layout: the same offset inside each
script's block is the same sound (क U+0915, ಕ U+0C95, க U+0B95 ... are all "k"). So one table of
offsets covers Hindi, Marwari, Marathi, Bengali, Gujarati, Punjabi, Odia, Tamil, Telugu, Kannada
and Malayalam. Rules:
  - a consonant carries an inherent "a" unless followed by a vowel sign (matra) or virama;
  - Hindi-belt languages drop the inherent "a" at the end of a word (schwa deletion);
  - anusvara/chandrabindu -> n, visarga -> h.
English uses a small set of letter and digraph rules; it is intentionally rough.
"""

import re

SCRIPT_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "r i",
          0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o", 0x14: "au"}
MATRAS = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "r i",
          0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o", 0x4C: "au"}
CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
              0x1A: "c", 0x1B: "ch", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
              0x1F: "tt", 0x20: "tth", 0x21: "dd", 0x22: "ddh", 0x23: "nn",
              0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
              0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
              0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "r", 0x35: "v",
              0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h"}
VIRAMA, ANUSVARA, CHANDRABINDU, VISARGA = 0x4D, 0x02, 0x01, 0x03

SCHWA_DELETING = {"hi", "mwr", "mr", "pa", "gu"}

PAUSE_SHORT, PAUSE_LONG, WORD_GAP = "_", "__", "."

_EN_DIGRAPHS = [("tch", "c"), ("sh", "sh"), ("ch", "c"), ("th", "th"), ("ph", "f"), ("ck", "k"),
                ("ng", "ng"), ("ee", "ii"), ("ea", "ii"), ("oo", "uu"), ("ai", "e"), ("ay", "e"),
                ("ou", "au"), ("ow", "au"), ("oa", "o"), ("qu", "k v"), ("wh", "v")]
_EN_LETTERS = {"a": "a", "b": "b", "c": "k", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h",
               "i": "i", "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "o": "o", "p": "p",
               "q": "k", "r": "r", "s": "s", "t": "t", "u": "a", "v": "v", "w": "v", "x": "k s",
               "y": "y", "z": "z"}
_EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _block_offset(ch: str) -> int | None:
    cp = ord(ch)
    for base in SCRIPT_BLOCKS:
        if base <= cp < base + 0x80:
            return cp - base
    return None


def _indic_word(word: str, lang: str) -> list[str]:
    out: list[str] = []
    pending = False  # last consonant still owes its inherent "a"
    for ch in word:
        off = _block_offset(ch)
        if off is None:
            continue
        if off in CONSONANTS:
            if pending:
                out.append("a")
            out.extend(CONSONANTS[off].split())
            pending = True
        elif off in MATRAS:
            out.extend(MATRAS[off].split())
            pending = False
        elif off == VIRAMA:
            pending = False
        elif off in VOWELS:
            if pending:
                out.append("a")
            out.extend(VOWELS[off].split())
            pending = False
        elif off in (ANUSVARA, CHANDRABINDU):
            if pending:
                out.append("a")
                pending = False
            out.append("n")
        elif off == VISARGA:
            if pending:
                out.append("a")
                pending = False
            out.append("h")
    if pending and lang not in SCHWA_DELETING:
        out.append("a")
    return out


def _english_word(word: str) -> list[str]:
    w = word.lower()
    if w.isdigit():
        return [p for d in w for p in _english_word(_EN_DIGITS[int(d)]) + [WORD_GAP]]
    if len(w) > 2 and w.endswith("e") and not w.endswith("ee"):
        w = w[:-1]  # silent final e
    out, i = [], 0
    while i < len(w):
        for g, ph in _EN_DIGRAPHS:
            if w.startswith(g, i):
                out.extend(ph.split())
                i += len(g)
                break
        else:
            if w[i] in _EN_LETTERS:
                out.extend(_EN_LETTERS[w[i]].split())
            i += 1
    if out and w.endswith("y") and len(w) > 1:
        out[-1] = "ii"
    return out


def to_phonemes(text: str, lang: str) -> list[str]:
    """Text -> flat phoneme list with pause markers ('_' short, '__' long, '.' between words)."""
    out: list[str] = []
    for tok in re.findall(r"[\wऀ-ൿ]+|[,;:]|[.!?।]", text):
        if tok in ",;:":
            out.append(PAUSE_SHORT)
        elif tok in ".!?।":
            out.append(PAUSE_LONG)
        else:
            indic = any(_block_offset(c) is not None for c in tok)
            ph = _indic_word(tok, lang) if indic else _english_word(tok)
            if ph:
                out.extend(ph)
                out.append(WORD_GAP)
    return out
