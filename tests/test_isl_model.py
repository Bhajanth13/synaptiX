from pathlib import Path

import numpy as np
import pytest

from synaptix.inputs.sign.isl_model import DEFAULT_WEIGHTS, ISLModel, clean_label, features, gloss_to_word
from synaptix.inputs.sign.live import segment

pytestmark = pytest.mark.skipif(not Path(DEFAULT_WEIGHTS).exists(), reason="pretrained ISL weights missing")


def test_labels_and_prediction_shape():
    m = ISLModel()
    assert len(m.labels) == 263 and "POLICE" in m.labels and "COURT" in m.labels
    seq = np.random.default_rng(0).random((40, 225))
    gloss, conf = m.predict(seq, min_conf=0.0)
    assert gloss in m.labels and 0 <= conf <= 1


def test_features_fixed_shape_for_any_length():
    for n in (5, 40, 300):
        assert features(np.random.default_rng(n).random((n, 75, 2))).shape == (48, 108)


def test_label_cleaning():
    assert clean_label("4. sad") == "SAD" and clean_label("83.big large") == "BIGLARGE"
    assert gloss_to_word("STREETORROAD") == "road" and gloss_to_word("POLICE") == "police"


def test_segment_splits_on_hand_free_gaps():
    hand = np.zeros(225); hand[0] = 0.5
    none = np.zeros(225)
    frames = np.array([hand] * 12 + [none] * 10 + [hand] * 15)
    assert [len(s) for s in segment(frames)] == [12, 15]
