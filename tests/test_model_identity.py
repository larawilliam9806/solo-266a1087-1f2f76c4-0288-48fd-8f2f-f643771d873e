"""The ONNX model must produce identical per-sequence statistics when padded
into a batch versus run alone."""
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ort = pytest.importorskip("onnxruntime")

MODEL = Path(__file__).resolve().parent.parent / "models" / "anomaly_model.onnx"


def test_model_exists():
    assert MODEL.exists(), "run `python scripts/make_model.py` first"


def test_padded_batch_matches_single():
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    rows = [rng.normal(size=n).astype(np.float32) for n in (1, 3, 7, 16, 31)]
    t_max = max(len(r) for r in rows)
    padded = np.zeros((len(rows), t_max), dtype=np.float32)
    for i, r in enumerate(rows):
        padded[i, : len(r)] = r

    names = [o.name for o in sess.get_outputs()]
    batch_out = dict(zip(names, sess.run(None, {"values": padded})))

    for i, r in enumerate(rows):
        single = dict(zip(names, sess.run(None, {"values": r[None, :]})))
        for key in ("energy", "abs_sum", "peak_sq"):
            a = float(batch_out[key][i])
            b = float(single[key][0])
            assert abs(a - b) <= 1e-5 * max(1.0, abs(b)), (key, i, a, b)
