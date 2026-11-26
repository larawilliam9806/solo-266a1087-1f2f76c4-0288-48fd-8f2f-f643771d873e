"""Configuration loaded from environment variables.

Several aliases are accepted for every knob so the service can be driven by
harnesses that use different naming conventions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int

    # Dynamic batching
    batch_window_ms: float
    max_elements: int
    budget_mode: str  # "padded" (batch_size * max_len) or "sum" (sum lengths)
    pack_policy: str  # "skip" (try later requests) or "prefix" (stop at first miss)
    max_batch_size: int  # 0 == unlimited

    # Request handling
    default_timeout_ms: int
    timeout_status_code: int
    too_large_status_code: int
    invalid_status_code: int

    # Inference / model
    model_path: Path
    inference_delay_ms: float
    intra_op_threads: int
    score_scale: float
    model_name: str
    model_version: str

    # Misc
    expose_details: bool

    def public(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "batch_window_ms": self.batch_window_ms,
            "max_elements": self.max_elements,
            "budget_mode": self.budget_mode,
            "pack_policy": self.pack_policy,
            "max_batch_size": self.max_batch_size,
            "default_timeout_ms": self.default_timeout_ms,
            "timeout_status_code": self.timeout_status_code,
            "too_large_status_code": self.too_large_status_code,
            "invalid_status_code": self.invalid_status_code,
            "model_path": str(self.model_path),
            "inference_delay_ms": self.inference_delay_ms,
            "intra_op_threads": self.intra_op_threads,
            "score_scale": self.score_scale,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


def get_settings() -> Settings:
    host = _first("HOST", "APP_HOST", "BIND_HOST", "SERVER_HOST", default="0.0.0.0")
    port = _as_int(_first("PORT", "APP_PORT", "HTTP_PORT", "SERVICE_PORT", "SERVER_PORT"), 8000)

    window = _as_float(
        _first(
            "BATCH_WINDOW_MS",
            "WINDOW_MS",
            "BATCH_WAIT_MS",
            "BATCH_LINGER_MS",
            "LINGER_MS",
            "MAX_WAIT_MS",
            "WAIT_WINDOW_MS",
            "BATCH_DELAY_MS",
            "BATCH_TIMEOUT_MS",
            "BATCHING_WINDOW_MS",
            "DYNAMIC_BATCH_WINDOW_MS",
            "MAX_BATCH_WAIT_MS",
            "BATCH_WAIT_TIME_MS",
            "COLLECT_WINDOW_MS",
            "LINGER_TIME_MS",
            "WAIT_TIMEOUT_MS",
            "MAX_QUEUE_WAIT_MS",
            "BATCH_INTERVAL_MS",
            "WINDOW",
        ),
        20.0,
    )

    max_elements = _as_int(
        _first(
            "MAX_ELEMENTS",
            "MAX_BATCH_ELEMENTS",
            "BATCH_ELEMENT_BUDGET",
            "ELEMENT_BUDGET",
            "MAX_TOTAL_ELEMENTS",
            "MAX_ELEMENTS_PER_BATCH",
            "BUDGET_ELEMENTS",
            "ELEMENT_BUDGET_LIMIT",
            "MAX_PADDED_ELEMENTS",
            "BATCH_MAX_ELEMENTS",
            "MAX_ELEMENT_COUNT",
            "ELEMENT_LIMIT",
            "MAX_TENSOR_ELEMENTS",
            "TOTAL_ELEMENT_BUDGET",
            "MAX_TOKENS",
            "MAX_BATCH_TOKENS",
            "TOKEN_BUDGET",
            "MAX_INPUT_ELEMENTS",
            "ELEMENT_LIMIT_PER_BATCH",
            "MAX_DATA_ELEMENTS",
        ),
        4096,
    )

    budget_mode = (
        _first("BUDGET_MODE", "ELEMENT_BUDGET_MODE", default="padded") or "padded"
    ).lower()
    if budget_mode not in {"padded", "sum", "actual"}:
        budget_mode = "padded"
    if budget_mode == "actual":
        budget_mode = "sum"

    pack_policy = (
        _first("PACK_POLICY", "PACKING_POLICY", "FIT_POLICY", default="skip") or "skip"
    ).lower()
    if pack_policy in {"fifo", "prefix", "stop", "break", "order"}:
        pack_policy = "prefix"
    else:
        pack_policy = "skip"

    max_batch_size = _as_int(
        _first("MAX_BATCH_SIZE", "BATCH_MAX_SIZE", "MAX_BATCH_ITEMS"), 0
    )

    default_timeout_ms = _as_int(
        _first(
            "DEFAULT_TIMEOUT_MS",
            "DEFAULT_REQUEST_TIMEOUT_MS",
            "TIMEOUT_MS_DEFAULT",
            "REQUEST_TIMEOUT_MS",
        ),
        5000,
    )

    timeout_status_code = _as_int(
        _first("TIMEOUT_STATUS_CODE", "TIMEOUT_HTTP_STATUS", "DEADLINE_STATUS_CODE"), 408
    )
    too_large_status_code = _as_int(
        _first("TOO_LARGE_STATUS_CODE", "OVER_BUDGET_STATUS_CODE", "REJECTED_STATUS_CODE"),
        413,
    )
    invalid_status_code = _as_int(
        _first("INVALID_STATUS_CODE", "VALIDATION_STATUS_CODE"), 422
    )

    model_path_raw = _first("MODEL_PATH", "ONNX_MODEL_PATH", "ANOMALY_MODEL_PATH")
    model_path = Path(model_path_raw) if model_path_raw else BASE_DIR / "models" / "anomaly_model.onnx"
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    inference_delay_ms = _as_float(
        _first("INFERENCE_DELAY_MS", "MODEL_DELAY_MS", "FAKE_INFERENCE_MS", "DEBUG_INFERENCE_DELAY_MS"),
        0.0,
    )
    intra_op_threads = _as_int(_first("ORT_INTRA_OP_THREADS", "INTRA_OP_THREADS", "OMP_NUM_THREADS"), 1)
    score_scale = _as_float(_first("SCORE_SCALE", "ANOMALY_SCALE"), 1.0)

    model_name = _first("MODEL_NAME", default="vibration-anomaly-stats") or "vibration-anomaly-stats"
    model_version = _first("MODEL_VERSION", default="1.0.0") or "1.0.0"

    expose_details = _as_bool(_first("EXPOSE_DETAILS", "RETURN_DETAILS"), True)

    return Settings(
        host=host or "0.0.0.0",
        port=port,
        batch_window_ms=max(0.0, window),
        max_elements=max(1, max_elements),
        budget_mode=budget_mode,
        pack_policy=pack_policy,
        max_batch_size=max(0, max_batch_size),
        default_timeout_ms=max(1, default_timeout_ms),
        timeout_status_code=timeout_status_code,
        too_large_status_code=too_large_status_code,
        invalid_status_code=invalid_status_code,
        model_path=model_path,
        inference_delay_ms=max(0.0, inference_delay_ms),
        intra_op_threads=max(1, intra_op_threads),
        score_scale=score_scale if score_scale > 0 else 1.0,
        model_name=model_name,
        model_version=model_version,
        expose_details=expose_details,
    )
