"""The Python module against the C++ library's own behaviour and hand-checked cases."""

from pathlib import Path

import numpy as np
import pytest

import waferedge as we

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "data" / "waferlens_sample.wmap"
RULES = ROOT / "config" / "rules.txt"


def disc(n, fail=None):
    """An n x n round wafer: 1 inside the circle, 0 outside; fail(r, c) marks fails (2)."""
    r, c = np.mgrid[0:n, 0:n]
    centre = (n - 1) / 2
    inside = (r - centre) ** 2 + (c - centre) ** 2 <= (n / 2) ** 2
    m = inside.astype(np.uint8)
    if fail is not None:
        m[inside & fail(r, c)] = 2
    return m


def test_features_count_dies_and_fails_like_numpy():
    m = disc(41, lambda r, c: (r + c) % 7 == 0)
    f = we.features(m)
    assert f["dies"] == np.count_nonzero(m)
    assert f["fails"] == np.count_nonzero(m >= 2)
    # Zones, sectors and rings partition the wafer.
    for part in ("zone", "sector", "ring"):
        assert f[f"{part}_dies"].sum() == f["dies"]
        assert f[f"{part}_fails"].sum() == f["fails"]


def test_batch_equals_one_by_one():
    rng = np.random.default_rng(7)
    stack = np.stack([disc(30) * (1 + (rng.random((30, 30)) < 0.1)) for _ in range(20)])
    stack = stack.astype(np.uint8)
    batch = we.features_batch(stack)
    sig = we.signals(stack)
    assert sig.shape == (20, len(we.SIGNALS))
    for i, m in enumerate(stack):
        one = we.features(m)
        assert batch["dies"][i] == one["dies"]
        assert batch["fails"][i] == one["fails"]
        assert (batch["zone_fails"][i] == one["zone_fails"]).all()
        assert (batch["sector_fails"][i] == one["sector_fails"]).all()
        assert (batch["ring_fails"][i] == one["ring_fails"]).all()
        assert np.array_equal(sig[i], we.signals(m))


def test_maps_are_read_in_place_or_refused():
    m = disc(20)
    with pytest.raises(TypeError, match="uint8"):
        we.features(m.astype(np.int32))
    with pytest.raises(TypeError, match="contiguous"):
        we.features(np.asfortranarray(disc(20, lambda r, c: c < 3))[:, ::2])
    with pytest.raises(TypeError, match="2-D"):
        we.features(m.reshape(1, 20, 20))
    with pytest.raises(TypeError, match="3-D"):
        we.features_batch(m)
    # A transposed view isn't C-contiguous either; the copy that fixes it gives the same counts
    # as the transposed map itself would.
    t = disc(21, lambda r, c: r < 5).T
    with pytest.raises(TypeError):
        we.features(t)
    assert we.features(np.ascontiguousarray(t))["fails"] == np.count_nonzero(t >= 2)


def test_clusters_by_hand():
    m = np.ones((8, 8), dtype=np.uint8)
    m[1, 1] = m[2, 2] = 2  # diagonal neighbours: one cluster
    m[5:7, 5:8] = 3  # a 2 x 3 block, any fail bin counts
    labels, clusters = we.clusters(m)
    assert labels.shape == (8, 8) and labels.dtype == np.int32
    assert [c["size"] for c in clusters] == [2, 6]
    assert labels[1, 1] == labels[2, 2] == 0
    assert (labels[5:7, 5:8] == 1).all()
    assert (labels[m < 2] == -1).all()
    assert clusters[1]["min_row"] == 5 and clusters[1]["max_col"] == 7


def test_hough_finds_a_scratch():
    m = disc(41)
    m[20, 5:36] = 2  # a horizontal scratch
    line = we.hough_line(m)
    assert line["votes"] == 31
    assert line["angle"] == 90


def test_classifier_on_the_waferlens_sample():
    data = we.read_wmap(str(SAMPLE))
    assert len(data["maps"]) == len(data["truth"]) == data["wafer_id"].shape[0] > 0
    rules = we.Classifier.load(str(RULES))
    one_by_one = [rules.classify(m) for m in data["maps"]]
    assert set(one_by_one) <= set(we.PATTERNS)
    # Batches need equal shapes: group the sample by shape.
    by_shape = {}
    for i, m in enumerate(data["maps"]):
        by_shape.setdefault(m.shape, []).append(i)
    for idx in by_shape.values():
        stack = np.stack([data["maps"][i] for i in idx])
        assert rules.classify_batch(stack) == [one_by_one[i] for i in idx]
    # The same as the C++ tool: `waferedge-classify eval tests/data/waferlens_sample.wmap
    # config/rules.txt all` reports accuracy 0.542 (13 of 24; the sample is mostly patterned
    # wafers, the hard cases).
    right = sum(p == t for p, t in zip(one_by_one, data["truth"]))
    assert (right, len(one_by_one)) == (13, 24)
    # Wafer 14's signals as the C++ tool prints them (--csv, 6 significant digits).
    first = int(np.flatnonzero(data["wafer_id"] == 14)[0])
    expected = [0.0914634, 0.664865, 1.53067, 1.38238, 0, 1.17143, 0.0833333, 0.00762195,
                0.879646, 2.18816, 3.64444, 0.276811, -0.787799]
    assert we.signals(data["maps"][first]) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert "rule" in rules.explain(data["maps"][0]).lower()


def test_classifier_text_round_trip_and_errors():
    rules = we.Classifier.defaults()
    again = we.Classifier.parse(rules.to_text())
    assert again.to_text() == rules.to_text()
    with pytest.raises(ValueError):
        we.Classifier.parse("not a rule\n")
    with pytest.raises(ValueError):
        we.Classifier.load("/nonexistent/rules.txt")
    with pytest.raises(ValueError):
        we.read_wmap("/nonexistent/maps.wmap")


def test_module_metadata():
    assert we.backend() in ("avx2", "scalar")
    assert we.PATTERNS[0] == "none" and len(we.PATTERNS) == 9
    assert len(we.SIGNALS) == 13
    assert we.__version__.count(".") == 2
