from datetime import date

import numpy as np

from synaptix import Pipeline
from synaptix.core import language, router
from synaptix.inputs.sign.classifier import SignClassifier

pipe = Pipeline()


def test_language_detect():
    assert language.detect("ग्राम सभा क्या है") == "hi"
    assert language.detect("ಗ್ರಾಮ ಸಭೆ ಎಂದರೇನು") == "kn"
    assert language.detect("what is a gram sabha") == "en"


def test_marwari_falls_back_to_hindi_voice():
    assert language.tts_locale("mwr") == ("hi-IN", True)
    assert language.tts_locale("kn") == ("kn-IN", False)


def test_router():
    assert router.route("who do i complain to about the broken road") == "jurisdiction"
    assert router.route("what is zero hour") == "jargon"
    assert router.route("my employer pays less than minimum wage") == "legal"
    assert router.route("when is the next gram sabha") == "events"


def test_jargon_english():
    a = pipe.ask("what is mplads", lang="en")
    assert a.skill == "jargon" and "MPLADS" in a.text and a.sources


def test_jargon_hindi_verified_translation():
    a = pipe.ask("ग्राम सभा क्या है")
    assert a.lang == "hi" and "बैठक" in a.text


def test_jargon_marwari_uses_hindi_text():
    a = pipe.ask("ग्राम सभा क्या है", lang="mwr")
    assert "बैठक" in a.text


def test_jurisdiction_national_highway():
    a = pipe.ask("who is responsible for the national highway", lang="en")
    assert a.skill == "jurisdiction" and "MP" in a.text and "Sarpanch and MLA have no power" in a.text


def test_legal_escalates():
    a = pipe.ask("my contractor did not pay minimum wage", lang="en")
    assert a.skill == "legal" and a.escalate_to


def test_unknown_legal_question_does_not_guess():
    a = pipe.ask("legal rules for launching satellites", lang="en", skill="legal")
    assert "don't have verified information" in a.text


def test_events_upcoming_only():
    a = pipe.skills["events"].answer(None, "gram sabha", today=date(2027, 2, 1))
    assert "2027-01-26" not in a.text and "2027-05-01" in a.text


def test_sign_classifier_roundtrip():
    rng = np.random.default_rng(0)
    base = {g: rng.normal(size=(20, 225)) for g in ["LAND", "NOTICE", "WHAT"]}
    seqs, labels = [], []
    for g, b in base.items():
        for _ in range(5):
            seqs.append(b + rng.normal(scale=0.05, size=b.shape))
            labels.append(g)
    clf = SignClassifier().fit(seqs, labels)
    assert clf.predict(base["NOTICE"])[0] == "NOTICE"


def test_signs_entry_point():
    a = pipe.ask_signs(["NOTICE", "SUMMONS", "COURT", "MEANING"])
    assert a.skill == "legal"


def test_slm_rewrite_that_changes_numbers_is_rejected():
    from synaptix.llm.slm import SLM

    slm = SLM(model_path="/nonexistent")
    slm.llm = object()  # pretend a model is loaded
    slm._chat = lambda *a, **k: "Each MP gets some money."
    src = "Each MP can recommend up to Rs 5 crore a year."
    assert slm.simplify(src, "q") == src
    slm._chat = lambda *a, **k: "Each MP can suggest up to Rs 5 crore every year."
    assert slm.simplify(src, "q") != src
