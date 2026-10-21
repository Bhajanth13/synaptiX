import pytest

from synaptix import Pipeline
from synaptix.cli import main
from synaptix.core import location as L
from synaptix.core import router
from synaptix.kb.store import KnowledgeBase, tokens

KB = KnowledgeBase()


def test_tokenizer_keeps_indic_words():
    assert tokens("ಪಹಣಿ ಎಂದರೇನು") == ["ಪಹಣಿ"]
    assert tokens("मज़दूरी नहीं मिली")[0] == "मज़दूरी"


def test_router_kannada():
    assert router.route("ನನಗೆ ಕೂಲಿ ಸಿಗಲಿಲ್ಲ") == "legal"
    assert router.route("ರಸ್ತೆ ಸರಿಮಾಡುವುದು ಯಾರ ಜವಾಬ್ದಾರಿ") == "jurisdiction"


def test_resolve():
    a = L.resolve(KB, "Mysore")
    assert a == L.resolve(KB, "mysuru") and (a.state, a.district) == ("karnataka", "mysuru")
    assert L.resolve(KB, state="Karnataka").district is None
    with pytest.raises(L.UnknownLocation):
        L.resolve(KB, "Atlantis")


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPTIX_CONFIG", str(tmp_path / "c.yaml"))
    assert not L.load_config(KB)
    L.save_config(L.resolve(KB, "Mysuru"))
    assert L.load_config(KB).district == "mysuru"
    (tmp_path / "c.yaml").write_text("state: karnataka\ndistrict: gone\n")  # stale
    assert L.load_config(KB) == L.Location()
    L.clear_config()
    assert not (tmp_path / "c.yaml").exists()


def test_escalation_names_district_offices():
    a = Pipeline(location="Mysuru").ask("my contractor did not pay minimum wage", lang="en")
    assert "Mysuru" in a.escalate_to and "Legal Services Authority" in a.escalate_to
    assert Pipeline().ask("my contractor did not pay minimum wage", lang="en").escalate_to == "District labour office / DLSA"


def test_state_card_only_with_location():
    a = Pipeline().ask("what is the pahani record", lang="en", skill="legal", district="Mysuru")
    assert any("Karnataka Land Revenue Act" in s for s in a.sources)
    b = Pipeline().ask("what is the pahani record", lang="en", skill="legal")
    assert not any("Karnataka" in s for s in b.sources)


def test_jurisdiction_with_district():
    a = Pipeline().ask("who handles police complaints", lang="en", district="Mysuru")
    assert "In Mysuru district" in a.text and "Superintendent of Police" in a.text


def test_unknown_district_falls_back():
    a = Pipeline().ask("who handles police complaints", lang="en", district="Atlantis")
    assert "Unknown district" in a.note and "In Atlantis" not in a.text and "Superintendent of Police" in a.text


def test_kannada_state_cards():
    for c in KB.states["karnataka"]["legal_cards"]:
        if not c.get("summary_kn"):
            continue
        kw = next(k for k in c["keywords"].split() if any("ಀ" <= ch <= "೿" for ch in k))
        a = Pipeline().ask(kw + " ಕಾನೂನು", district="Mysuru")
        assert a.lang == "kn" and any(c["source"] in s for s in a.sources), c["id"]


def test_cli(capsys):
    assert main(["districts"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 32 and out[-1] == "31 districts"
    assert main(["ask", "x", "--district", "Atlantis"]) == 2
