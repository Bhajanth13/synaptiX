import wave

from synaptix.outputs.g2p import to_phonemes
from synaptix.outputs.tts import RuleTTS


def strip(ph):
    return [p for p in ph if p not in {".", "_", "__"}]


def test_hindi_schwa_deletion():
    assert strip(to_phonemes("कमल", "hi")) == ["k", "a", "m", "a", "l"]


def test_kannada_keeps_final_vowel():
    assert strip(to_phonemes("ಕಮಲ", "kn")) == ["k", "a", "m", "a", "l", "a"]


def test_matra_and_virama():
    # सभा = s a bh aa ; क्या = k y aa (virama joins k+y)
    assert strip(to_phonemes("सभा", "hi")) == ["s", "a", "bh", "aa"]
    assert strip(to_phonemes("क्या", "hi")) == ["k", "y", "aa"]


def test_same_rules_across_scripts():
    assert strip(to_phonemes("सभा", "hi")) == strip(to_phonemes("ಸಭಾ", "kn"))


def test_english_rules():
    assert strip(to_phonemes("ship", "en")) == ["sh", "i", "p"]


def test_wav_is_written(tmp_path):
    path = RuleTTS(cache_dir=tmp_path).synthesize("ग्राम सभा क्या है?", "hi")
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnframes() > 8000
