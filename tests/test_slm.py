import pytest

from synaptix import Pipeline
from synaptix.llm import rewrites
from synaptix.llm.guard import check_rewrite
from synaptix.llm.rewrites import RewriteStore, key
from synaptix.llm.slm import FEW_SHOT, SLM
from synaptix.kb.store import KnowledgeBase

KB = KnowledgeBase()
FIR = next(e for e in KB.glossary if e["term"] == "FIR")
FIR_SRC, FIR_ALIASES = FIR["plain"], FIR["aliases"]
LEGAL_AID = next(c for c in KB.legal_cards if c["id"] == "free-legal-aid")["summary"]
GOOD_FIR = ("An FIR is the first written report of a serious crime, made at a police station. "
            "The police must register it and give you a free copy. You can file it at any police "
            "station; it is then sent to the right one (Zero FIR).")


@pytest.fixture(autouse=True)
def _clear_cache():
    rewrites.CACHE.clear()
    yield
    rewrites.CACHE.clear()


def make(tmp_path, reply):
    slm = SLM(model_path="/nonexistent", store=RewriteStore(tmp_path / "none.yaml"))
    slm.llm = object()
    slm.calls = []

    def chat(user, *a, **k):
        slm.calls.append(user)
        return reply

    slm._chat = chat
    return slm


def reasons(rewrite, source=FIR_SRC, term="FIR", aliases=FIR_ALIASES):
    return check_rewrite(source, rewrite, term, aliases).reasons


def test_garbled_entity_rejected(tmp_path):
    bad = "A POLICE is the first written report of a serious crime."
    r = reasons(bad)
    assert "term_missing" in r and "new_entity:POLICE" in r
    assert make(tmp_path, bad).simplify(FIR_SRC, "q", term="FIR", aliases=FIR_ALIASES) == FIR_SRC


def test_childish_and_added_facts_rejected():
    assert "banned_phrase" in reasons("Police are like helpers who catch bad guys.")
    assert not check_rewrite(LEGAL_AID, LEGAL_AID + " A lawyer is someone who helps people with their legal problems.").ok


def test_good_rewrite_accepted(tmp_path):
    assert check_rewrite(FIR_SRC, GOOD_FIR, "FIR", FIR_ALIASES).ok
    assert make(tmp_path, GOOD_FIR).simplify(FIR_SRC, "q", term="FIR", aliases=FIR_ALIASES) == GOOD_FIR


def test_length_and_coverage():
    r = reasons("An FIR is a report.")
    assert "too_short" in r and "dropped_facts" in r
    assert "too_long" in reasons(FIR_SRC + " " + FIR_SRC)


def test_meta_prefix_cleaned(tmp_path):
    slm = make(tmp_path, 'Sure! Here is the rewrite: "' + GOOD_FIR + '"')
    assert slm.simplify(FIR_SRC, "q", term="FIR", aliases=FIR_ALIASES) == GOOD_FIR


def test_few_shots_pass_guard():
    for user, ans in FEW_SHOT:
        term, text = user.removeprefix("Term: ").split("\nText: ")
        assert check_rewrite(text, ans, term).ok, term


def test_cache_means_one_model_call(tmp_path):
    slm = make(tmp_path, GOOD_FIR)
    slm.simplify(FIR_SRC, "q", term="FIR", aliases=FIR_ALIASES)
    slm.simplify(FIR_SRC, "other", term="FIR", aliases=FIR_ALIASES)
    assert len(slm.calls) == 1


def write_store(path, source, text, term):
    import yaml
    path.write_text(yaml.safe_dump({"entries": [{"key": key(source, term), "text": text, "term": term}]}))
    return RewriteStore(path)


def test_prerendered_without_model(tmp_path):
    slm = SLM(model_path="/nonexistent", store=write_store(tmp_path / "r.yaml", FIR_SRC, GOOD_FIR, "FIR"))
    assert not slm.available
    assert slm.simplify(FIR_SRC, "q", term="FIR") == GOOD_FIR
    assert slm.simplify(FIR_SRC + " Edited.", "q", term="FIR") == FIR_SRC + " Edited."  # stale


def test_question_never_sent_to_model(tmp_path):
    slm = make(tmp_path, GOOD_FIR)
    slm.simplify(FIR_SRC, "what is police", term="FIR", aliases=FIR_ALIASES)
    assert "what is police" not in slm.calls[0]
    assert "police" in slm.calls[0].lower()  # only via the verified text
    assert slm.calls[0].startswith("Term: FIR")


def test_glosses_guard(tmp_path):
    slm = make(tmp_path, "Hello there friend")
    assert slm.glosses_to_sentence(["LAND", "NOTICE", "MEANING"]) == "land notice meaning"
    slm = make(tmp_path, "What is the meaning of a land notice?")
    assert slm.glosses_to_sentence(["LAND", "NOTICE", "MEANING", "WHAT"]).endswith("?")


def test_no_match_falls_back_to_closest_entry():
    a = Pipeline().ask("what is police", lang="en")
    assert a.text.startswith("Closest verified entry: FIR")


def test_banned_word_allowed_when_in_source():
    from synaptix.llm.guard import check_rewrite
    src = "Women, children and workers can get a free lawyer from the Legal Services Authority."
    assert "banned_phrase" not in check_rewrite(src, "Women, children and workers can get a free lawyer from the Legal Services Authority.").reasons
    assert "banned_phrase" in check_rewrite("A lawyer helps you in court.", "A lawyer helps kids in court.").reasons
