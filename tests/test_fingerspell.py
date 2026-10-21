from pathlib import Path

import numpy as np
import pytest

from synaptix.inputs.sign.fingerspell import DEFAULT_WEIGHTS, FingerspellModel, Speller, hand_features, mirror

rng = np.random.default_rng(0)
H1, H2 = rng.random((21, 2)) * 0.2 + 0.2, rng.random((21, 2)) * 0.2 + 0.6


def test_features_ignore_position_and_size():
    a = hand_features([H1])
    b = hand_features([(H1 - H1[0]) * 0.5 + 0.7])
    assert np.allclose(a, b, atol=1e-5)


def test_two_hands_ordered_left_to_right():
    assert np.allclose(hand_features([H2, H1]), hand_features([H1, H2]))


def test_mirror_twice_is_identity():
    for hands in ([H1], [H1, H2]):
        f = hand_features(hands)
        assert np.allclose(mirror(mirror(f)), f, atol=1e-6)


def test_mirror_matches_flipped_image():
    flipped = [np.column_stack([1 - h[:, 0], h[:, 1]]) for h in (H1, H2)]
    assert np.allclose(mirror(hand_features([H1, H2]))[:84], hand_features(flipped)[:84], atol=1e-5)


def test_speller_holds_letters_and_spaces():
    sp = Speller(hold=3, gap=2)
    for x in ["A"] * 4 + [None] + ["B"] * 3 + [None] * 2 + ["C"] * 3:
        sp.update(x)
    assert sp.text == "AB C"


@pytest.mark.skipif(not Path(DEFAULT_WEIGHTS).exists(), reason="fingerspelling weights missing")
def test_model_loads_and_predicts():
    m = FingerspellModel()
    assert len(m.labels) == 36 and "A" in m.labels and "0" in m.labels and "9" in m.labels
    letter, conf = m.predict([H1], min_conf=0.0)
    assert letter in m.labels and 0 <= conf <= 1
    assert m.predict([]) == (None, 0.0)
