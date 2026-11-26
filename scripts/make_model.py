"""Generate the local example anomaly-scoring ONNX model.

The model is intentionally padding-safe: every reduction has an f(0) == 0
contribution, so zero-padding shorter sequences to the batch maximum length
does not change the per-sequence outputs.  The serving layer normalises the
summed statistics by the *true* sequence length (which it knows), yielding
scores identical to unpadded single-request inference.

Outputs (all shape [batch]):
    energy   : sum_t x_t ** 2
    abs_sum  : sum_t |x_t|
    peak_sq  : max_t x_t ** 2

Run:
    python scripts/make_model.py [models/anomaly_model.onnx]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

OPSET = 17


def build_model() -> "onnx.ModelProto":
    X = helper.make_tensor_value_info("values", TensorProto.FLOAT, ["batch", "time"])

    energy = helper.make_tensor_value_info("energy", TensorProto.FLOAT, ["batch"])
    abs_sum = helper.make_tensor_value_info("abs_sum", TensorProto.FLOAT, ["batch"])
    peak_sq = helper.make_tensor_value_info("peak_sq", TensorProto.FLOAT, ["batch"])
    x_sq_info = helper.make_tensor_value_info("x_sq", TensorProto.FLOAT, ["batch", "time"])
    x_abs_info = helper.make_tensor_value_info("x_abs", TensorProto.FLOAT, ["batch", "time"])

    axes = helper.make_tensor("reduce_axes", TensorProto.INT64, [1], np.array([1], dtype=np.int64))

    nodes = [
        helper.make_node("Mul", ["values", "values"], ["x_sq"], name="square"),
        helper.make_node("Abs", ["values"], ["x_abs"], name="abs"),
        helper.make_node(
            "ReduceSum", ["x_sq", "reduce_axes"], ["energy_2d"], keepdims=1, name="sum_sq"
        ),
        helper.make_node(
            "ReduceSum", ["x_abs", "reduce_axes"], ["abs_sum_2d"], keepdims=1, name="sum_abs"
        ),
        helper.make_node(
            "ReduceMax", ["x_sq"], ["peak_2d"], keepdims=1, axes=[1], name="max_sq"
        ),
        helper.make_node("Squeeze", ["energy_2d", "reduce_axes"], ["energy"], name="sq1"),
        helper.make_node("Squeeze", ["abs_sum_2d", "reduce_axes"], ["abs_sum"], name="sq2"),
        helper.make_node("Squeeze", ["peak_2d", "reduce_axes"], ["peak_sq"], name="sq3"),
    ]

    graph = helper.make_graph(
        nodes,
        "vibration_anomaly_stats",
        [X],
        [energy, abs_sum, peak_sq, x_sq_info, x_abs_info],
        initializer=[axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="gsb-vibration-example",
        model_version=1,
    )
    model.doc_string = "Padding-safe vibration anomaly statistics extractor."
    onnx.checker.check_model(model)
    return model


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "models" / "anomaly_model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    onnx.save(model, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")

    # Quick self-check: padded batch row must match direct single inference.
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available; skipping runtime self-check")
        return 0

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    a = np.array([0.5, -1.5, 2.0, 0.25], dtype=np.float32)
    b = np.array([3.0, 1.0], dtype=np.float32)
    e_a = float(np.sum(a**2))
    e_b = float(np.sum(b**2))
    padded = np.zeros((2, 4), dtype=np.float32)
    padded[0, : a.shape[0]] = a
    padded[1, : b.shape[0]] = b
    e, ab, pk = sess.run(["energy", "abs_sum", "peak_sq"], {"values": padded})
    assert abs(e[0] - e_a) < 1e-6, (e[0], e_a)
    assert abs(e[1] - e_b) < 1e-6, (e[1], e_b)
    single = sess.run(["energy"], {"values": b[None, :]})[0][0]
    assert single == e[1], (single, e[1])
    print("self-check OK: padded batch outputs match single inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
