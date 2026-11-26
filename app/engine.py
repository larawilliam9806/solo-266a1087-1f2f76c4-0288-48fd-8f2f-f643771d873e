"""ONNX Runtime CPU inference wrapper.

The ONNX graph returns padding-safe sufficient statistics.  Final per-request
scores are normalised here with each request's *true* length, which is why
batched padded inference is numerically identical to running each sequence on
its own.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("anomaly_batcher.engine")


class ModelLoadError(RuntimeError):
    pass


class OnnxAnomalyEngine:
    def __init__(
        self,
        model_path: Path | str,
        *,
        intra_op_threads: int = 1,
        score_scale: float = 1.0,
        inference_delay_ms: float = 0.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.score_scale = float(score_scale)
        self.inference_delay_ms = float(inference_delay_ms)
        self.intra_op_threads = max(1, int(intra_op_threads))
        self.session: Any = None
        self.input_name = "values"
        self.output_names: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            # Last-resort: regenerate the example model if `onnx` is available.
            logger.warning("model missing at %s, attempting regeneration", self.model_path)
            try:
                from scripts.make_model import main as make_main  # type: ignore

                make_main(["make_model.py", str(self.model_path)])
            except Exception as exc:  # pragma: no cover - environment dependent
                raise ModelLoadError(
                    f"ONNX model not found at {self.model_path} and could not be regenerated: {exc}"
                ) from exc

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ModelLoadError("onnxruntime is not installed") from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.intra_op_threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ModelLoadError(f"failed to load ONNX model {self.model_path}: {exc}") from exc

        inputs = self.session.get_inputs()
        if inputs:
            self.input_name = inputs[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(
            "loaded ONNX model %s inputs=%s outputs=%s",
            self.model_path,
            [(i.name, i.shape) for i in inputs],
            self.output_names,
        )

    # ------------------------------------------------------------------
    # Runs inside a worker thread.  Must not touch asyncio / loop state.
    # ------------------------------------------------------------------
    def infer(self, rows: list[np.ndarray]) -> list[dict[str, float]]:
        if self.inference_delay_ms > 0:
            time.sleep(self.inference_delay_ms / 1000.0)

        batch_size = len(rows)
        lengths = np.asarray([len(r) for r in rows], dtype=np.int64)
        max_len = int(lengths.max()) if batch_size else 0
        padded = np.zeros((batch_size, max_len), dtype=np.float32)
        for i, row in enumerate(rows):
            padded[i, : row.shape[0]] = row

        raw_outputs = self.session.run(None, {self.input_name: padded})
        outputs = dict(zip(self.output_names, raw_outputs))

        # Prefer per-element outputs and reduce over each true length in a
        # fixed order: this makes batched padded inference bit-identical to
        # single-sequence inference.
        x_sq_all = outputs.get("x_sq")
        x_abs_all = outputs.get("x_abs")
        energy_graph = np.asarray(outputs.get("energy", raw_outputs[0]), dtype=np.float64).reshape(-1)
        abs_graph = (
            np.asarray(outputs["abs_sum"], dtype=np.float64).reshape(-1)
            if "abs_sum" in outputs
            else np.zeros(batch_size)
        )
        peak_graph = (
            np.asarray(outputs["peak_sq"], dtype=np.float64).reshape(-1)
            if "peak_sq" in outputs
            else np.zeros(batch_size)
        )

        results: list[dict[str, float]] = []
        for i in range(batch_size):
            true_len = int(lengths[i])
            length = max(1, true_len)
            if x_sq_all is not None:
                sq_row = np.asarray(x_sq_all[i, :true_len], dtype=np.float64)
                energy = float(np.sum(sq_row))
                peak_sq = float(np.max(sq_row)) if sq_row.size else 0.0
            else:
                energy = max(0.0, float(energy_graph[i]))
                peak_sq = max(0.0, float(peak_graph[i]))
            if x_abs_all is not None:
                abs_row = np.asarray(x_abs_all[i, :true_len], dtype=np.float64)
                abs_sum = float(np.sum(abs_row))
            else:
                abs_sum = float(abs_graph[i])

            rms = float(np.sqrt(max(0.0, energy) / length))
            mean_abs = float(abs_sum / length)
            peak = float(np.sqrt(max(0.0, peak_sq)))
            raw_score = rms
            # Bounded, monotonic normalisation -> score in [0, 1).
            score = raw_score / (raw_score + self.score_scale)
            results.append(
                {
                    "score": score,
                    "anomaly_score": score,
                    "raw_score": raw_score,
                    "rms": rms,
                    "mean_abs": mean_abs,
                    "peak": peak,
                    "energy": energy,
                    "sequence_length": true_len,
                }
            )
        return results

    def warmup(self) -> None:
        try:
            dummy = np.zeros((1, 4), dtype=np.float32)
            self.session.run(None, {self.input_name: dummy})
        except Exception:  # pragma: no cover
            logger.debug("warmup failed", exc_info=True)

    def infer_single(self, values: np.ndarray) -> dict[str, float]:
        return self.infer([values])[0]
