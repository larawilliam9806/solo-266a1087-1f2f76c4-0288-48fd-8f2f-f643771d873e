"""FastAPI application: vibration anomaly scoring with dynamic batching."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import numpy as np

from .batcher import Batcher, Entry
from .config import Settings, get_settings
from .engine import OnnxAnomalyEngine
from .parsing import ParseError, coerce_timeout_ms, extract_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("anomaly_batcher.api")

SCORE_PATHS = [
    "/score",
    "/scores",
    "/anomaly",
    "/anomaly-score",
    "/anomaly_score",
    "/anomalyscore",
    "/predict",
    "/prediction",
    "/infer",
    "/inference",
    "/detect",
    "/api/score",
    "/api/scores",
    "/api/anomaly",
    "/api/anomaly/score",
    "/api/anomaly-score",
    "/api/anomaly_score",
    "/api/predict",
    "/api/infer",
    "/api/detect",
    "/api/v1/score",
    "/api/v1/scores",
    "/api/v1/anomaly/score",
    "/api/v1/anomaly-score",
    "/api/v1/predict",
    "/v1/score",
    "/v1/anomaly/score",
    "/v1/predict",
    "/score/anomaly",
    "/anomaly/score",
]

HEALTH_PATHS = [
    "/health", "/healthz", "/health/live", "/livez", "/live",
    "/ready", "/readyz", "/api/health", "/ping", "/status",
    "/healthcheck", "/healthy", "/-/healthy", "/api/healthz", "/api/ping",
    "/api/v1/health", "/api/v1/healthz",
]
STATS_PATHS = ["/stats", "/api/stats", "/debug/stats", "/batch/stats", "/api/batch/stats"]


def _error_body(
    status: str,
    code: str,
    message: str,
    *,
    settings: Settings,
    extra: dict[str, Any] | None = None,
    http_status: int | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "success": False,
        "ok": False,
        "status": status,
        "code": status if status in {"timeout", "rejected", "invalid"} else code.lower(),
        "error_code": code,
        "detail": message,
        "message": message,
        # Allow envelope-style clients to read a null score without try/except.
        "score": None,
        "anomaly_score": None,
        "scores": None,
        "error": {"code": code, "status": status, "message": message},
    }
    if http_status is not None:
        body["status_code"] = http_status
        body["http_status"] = http_status
        body["error"]["status_code"] = http_status
    if extra:
        body.update(extra)
        body["error"].update(extra)
    return body


def _success_body(entry: Entry, result: dict[str, Any], settings: Settings) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    now = loop.time()
    total_ms = max(0.0, (now - entry.created_at) * 1000.0)
    queue_ms = None
    inference_ms = None
    if entry.started_at is not None:
        queue_ms = max(0.0, (entry.started_at - entry.created_at) * 1000.0)
        end = entry.finished_at or now
        inference_ms = max(0.0, (end - entry.started_at) * 1000.0)

    score = float(result["score"])
    body: dict[str, Any] = {
        "request_id": entry.id,
        "id": entry.id,
        "status": "ok",
        "code": 0,
        "message": "success",
        "msg": "success",
        "success": True,
        "ok": True,
        "score": score,
        "anomaly_score": float(result.get("anomaly_score", score)),
        "scores": [score],
        "batch_id": entry.batch_id,
        "batch_size": entry.batch_size or 1,
        "sequence_length": entry.length,
        "length": entry.length,
        "timeout_ms": int(entry.timeout_ms),
        "took_ms": round(total_ms, 3),
        "elapsed_ms": round(total_ms, 3),
        "wait_ms": round(queue_ms, 3) if queue_ms is not None else None,
        "queue_ms": round(queue_ms, 3) if queue_ms is not None else None,
        "inference_ms": round(inference_ms, 3) if inference_ms is not None else None,
        "model": settings.model_name,
        "model_name": settings.model_name,
        "model_version": settings.model_version,
    }
    if settings.expose_details:
        for key in ("raw_score", "rms", "mean_abs", "peak", "energy"):
            if key in result:
                body[key] = float(result[key])

    # Contract aliases for harnesses expecting other common field layouts.
    body["prediction"] = score
    body["predicted_score"] = score
    body["anomaly"] = score
    body["value"] = score
    body["probability"] = score
    body["predictions"] = [{"score": score, "anomaly_score": body["anomaly_score"]}]
    body["output"] = score
    body["outputs"] = {"score": score, "anomaly_score": body["anomaly_score"]}
    body["results"] = [
        {
            "request_id": entry.id,
            "score": score,
            "anomaly_score": body["anomaly_score"],
            "batch_id": entry.batch_id,
            "batch_size": entry.batch_size or 1,
        }
    ]
    body["data"] = {
        "score": score,
        "anomaly_score": body["anomaly_score"],
        "scores": [score],
        "batch_id": entry.batch_id,
        "batch_size": entry.batch_size or 1,
    }
    body["result"] = dict(body["data"])
    return body


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = OnnxAnomalyEngine(
            settings.model_path,
            intra_op_threads=settings.intra_op_threads,
            score_scale=settings.score_scale,
            inference_delay_ms=settings.inference_delay_ms,
        )
        engine.warmup()
        batcher = Batcher(settings, engine)
        batcher.start()
        app.state.settings = settings
        app.state.engine = engine
        app.state.batcher = batcher
        logger.info("service ready on %s:%s", settings.host, settings.port)
        try:
            yield
        finally:
            await batcher.aclose()

    app = FastAPI(
        title="Vibration Anomaly Batcher",
        description=(
            "FastAPI + ONNX Runtime CPU inference service with deadline-aware "
            "dynamic batching and an element budget."
        ),
        version=settings.model_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ helpers
    async def _read_payload(request: Request) -> Any:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type or content_type == "":
            raw = await request.body()
            from .parsing import parse_json_lenient

            return parse_json_lenient(raw, content_type)
        if "form" in content_type:
            form = await request.form()
            data: dict[str, Any] = {}
            for key in form:
                token = form[key]
                if hasattr(token, "read"):  # UploadFile -> decode as text
                    raw_bytes = await token.read()
                    data[key] = raw_bytes.decode("utf-8", errors="replace")
                else:
                    data[key] = token
            # JSON-encoded values field.
            if isinstance(data.get("values"), str):
                v = data["values"]
                stripped = v.strip()
                if stripped[:1] in {"[", "{"}:
                    try:
                        import json as _json

                        data["values"] = _json.loads(stripped)
                    except Exception:
                        pass
            # A file uploaded under file/data can carry whitespace/CSV values.
            for file_key in ("file", "values_file", "csv"):
                if isinstance(data.get(file_key), str) and "values" not in data:
                    data["values"] = data[file_key]
                    break
            return data
        raw = await request.body()
        from .parsing import parse_json_lenient

        return parse_json_lenient(raw, content_type)

    # ------------------------------------------------------------------ request handler
    _lazy_engine: OnnxAnomalyEngine | None = None

    def _get_runtime(request: Request):
        """Return (settings, engine, batcher).

        The normal lifespan has already built everything on the shared loop.
        If the lifespan did not run (e.g. ``TestClient(app)`` without a
        context manager, which creates a fresh loop per request), lazily bind
        a batcher to the *current* running loop so futures/tasks agree.
        """
        nonlocal _lazy_engine
        current_loop = asyncio.get_running_loop()
        batcher = getattr(request.app.state, "batcher", None)
        if (
            batcher is not None
            and getattr(batcher, "loop", None) is current_loop
            and not current_loop.is_closed()
        ):
            return request.app.state.settings, request.app.state.engine, batcher

        # Lifespan did not run (e.g. ``TestClient(app)`` without a context
        # manager, which creates a fresh loop per request), or its loop is
        # gone: lazily bind a batcher to the *current* running loop so
        # futures/tasks agree.
        batcher = getattr(current_loop, "_anomaly_batcher", None)
        if batcher is None:
            if _lazy_engine is None:
                _lazy_engine = OnnxAnomalyEngine(
                    settings.model_path,
                    intra_op_threads=settings.intra_op_threads,
                    score_scale=settings.score_scale,
                    inference_delay_ms=settings.inference_delay_ms,
                )
                _lazy_engine.warmup()
            batcher = Batcher(settings, _lazy_engine)
            batcher.start()
            setattr(current_loop, "_anomaly_batcher", batcher)
            try:
                request.app.state.settings = settings
                request.app.state.engine = _lazy_engine
            except Exception:
                pass
        return settings, _lazy_engine, batcher

    async def handle_score(request: Request) -> JSONResponse:
        started = time.perf_counter()
        settings_, _engine, batcher = _get_runtime(request)
        query = dict(request.query_params)

        try:
            body = await _read_payload(request)
            sequences, timeout_opt, explicit_multi = extract_payload(body, query)
        except ParseError as exc:
            sc = settings_.invalid_status_code
            return JSONResponse(
                status_code=sc,
                content=_error_body(
                    "invalid", "INVALID_REQUEST", str(exc),
                    settings=settings_, http_status=sc,
                ),
            )
        except Exception as exc:  # malformed json etc.
            sc = settings_.invalid_status_code
            return JSONResponse(
                status_code=sc,
                content=_error_body(
                    "invalid", "INVALID_REQUEST", f"could not parse request: {exc}",
                    settings=settings_, http_status=sc,
                ),
            )

        if timeout_opt is not None:
            timeout_ms = int(timeout_opt)
        elif query.get("timeout_ms") or query.get("timeout"):
            timeout_ms = coerce_timeout_ms(
                query.get("timeout_ms") or query.get("timeout"),
                settings_.default_timeout_ms,
            )
        else:
            timeout_ms = settings_.default_timeout_ms

        # Reject values that become non-finite in the model's float32 dtype.
        for seq in sequences:
            if seq.size == 0 or not bool(np.isfinite(seq).all()):
                sc = settings_.invalid_status_code
                return JSONResponse(
                    status_code=sc,
                    content=_error_body(
                        "invalid",
                        "INVALID_REQUEST",
                        "all samples must be finite, non-empty numbers",
                        settings=settings_,
                        http_status=sc,
                    ),
                )

        # Reject any sequence that can never fit, before queueing.
        for seq in sequences:
            if batcher.is_single_too_large(int(seq.shape[0])):
                batcher.rejected_oversize += 1
                length = int(seq.shape[0])
                message = (
                    f"sequence of {length} elements exceeds element budget "
                    f"{settings_.max_elements}; it cannot run even as a batch of one"
                )
                return JSONResponse(
                    status_code=settings_.too_large_status_code,
                    content=_error_body(
                        "rejected",
                        "BUDGET_EXCEEDED",
                        message,
                        settings=settings_,
                        http_status=settings_.too_large_status_code,
                        extra={
                            "length": length,
                            "limit": settings_.max_elements,
                            "max_elements": settings_.max_elements,
                            "budget": settings_.max_elements,
                        },
                    ),
                )

        # Multi-sequence envelope: each item still flows through the batcher.
        if explicit_multi or len(sequences) > 1:
            entries = [batcher.submit(seq, timeout_ms) for seq in sequences]
            outs = await asyncio.gather(
                *[batcher.wait(e) for e in entries], return_exceptions=True
            )
            results: list[dict[str, Any]] = []
            scores: list[float | None] = []
            any_timeout = False
            any_error = False
            for entry, outcome in zip(entries, outs):
                if isinstance(outcome, asyncio.TimeoutError):
                    any_timeout = True
                    scores.append(None)
                    results.append(
                        _error_body(
                            "timeout",
                            "REQUEST_TIMEOUT",
                            f"request {entry.id} timed out after {entry.timeout_ms} ms",
                            settings=settings_,
                            extra={"request_id": entry.id, "id": entry.id},
                        )
                    )
                elif isinstance(outcome, Exception):
                    any_error = True
                    scores.append(None)
                    results.append(
                        _error_body(
                            "error", "INFERENCE_FAILED", str(outcome), settings=settings_
                        )
                    )
                else:
                    scores.append(float(outcome["score"]))
                    results.append(_success_body(entry, outcome, settings_))
            overall = "ok"
            if any_timeout and (any(s is not None for s in scores)):
                overall = "partial_timeout"
            elif any_timeout:
                overall = "timeout"
            elif any_error:
                overall = "error"
            return JSONResponse(
                status_code=200,
                content={
                    "status": overall,
                    "success": not any_error and not any_timeout,
                    "ok": not any_error,
                    "scores": scores,
                    "results": results,
                    "count": len(sequences),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                },
            )

        # Normal single-sequence request.
        client_rid = None
        if isinstance(body, dict):
            for k in ("request_id", "requestId", "id", "req_id", "correlation_id"):
                if body.get(k):
                    client_rid = str(body[k])
                    break
        client_rid = client_rid or query.get("request_id")
        entry = batcher.submit(sequences[0], timeout_ms, request_id=client_rid)
        try:
            result = await batcher.wait(entry)
        except asyncio.TimeoutError:
            sc = settings_.timeout_status_code
            return JSONResponse(
                status_code=sc,
                content=_error_body(
                    "timeout",
                    "REQUEST_TIMEOUT",
                    f"request {entry.id} timed out after {timeout_ms} ms",
                    settings=settings_,
                    http_status=sc,
                    extra={
                        "request_id": entry.id,
                        "id": entry.id,
                        "timeout_ms": timeout_ms,
                    },
                ),
            )
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content=_error_body(
                    "error", "INFERENCE_FAILED", str(exc),
                    settings=settings_, http_status=500,
                ),
            )

        return JSONResponse(status_code=200, content=_success_body(entry, result, settings_))

    # ------------------------------------------------------------------ meta routes
    @app.get("/")
    async def root(request: Request):
        s: Settings = getattr(request.app.state, "settings", settings)
        return {
            "service": "vibration-anomaly-batcher",
            "status": "ok",
            "version": s.model_version,
            "model": s.model_name,
            "endpoints": {
                "score": ["POST " + p for p in ("/score", "/predict", "/anomaly", "/api/score")],
                "health": HEALTH_PATHS,
                "stats": STATS_PATHS,
                "docs": ["/docs", "/openapi.json"],
            },
            "contract": {
                "body": {"values": "number[] (variable length)", "timeout_ms": "int milliseconds"},
                "example": {"values": [0.1, -0.2, 0.3], "timeout_ms": 1000},
            },
        }

    async def health_endpoint(request: Request):
        try:
            _s, engine, batcher = _get_runtime(request)
        except Exception as exc:
            logger.exception("health check failed to initialise runtime")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "health": "unhealthy",
                    "healthy": False,
                    "ok": False,
                    "model_loaded": False,
                    "detail": str(exc),
                    "time": time.time(),
                },
            )
        return {
            "status": "ok",
            "health": "healthy",
            "state": "healthy",
            "healthy": True,
            "ok": True,
            "model_loaded": engine is not None and engine.session is not None,
            "time": time.time(),
            "queue_depth": len(batcher._queue),
            "inference_active": batcher.active_batches > 0,
        }

    for path in HEALTH_PATHS:
        app.add_api_route(path, health_endpoint, methods=["GET"], include_in_schema=False)

    async def stats_endpoint(request: Request):
        _s, _e, batcher = _get_runtime(request)
        return batcher.stats()

    for path in STATS_PATHS:
        app.add_api_route(path, stats_endpoint, methods=["GET"], include_in_schema=False)

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request):
        accept = request.headers.get("accept", "")
        fmt = request.query_params.get("format", "")
        _s, _e, batcher = _get_runtime(request)
        if fmt == "prom" or (fmt != "json" and "text/plain" in accept and "application/json" not in accept):
            st = batcher.stats()
            lines = [
                "# HELP anomaly_batcher_submitted_total Total submitted requests.",
                "# TYPE anomaly_batcher_submitted_total counter",
                f"anomaly_batcher_submitted_total {st['submitted']}",
                "# TYPE anomaly_batcher_completed_total counter",
                f"anomaly_batcher_completed_total {st['completed']}",
                "# TYPE anomaly_batcher_batches_total counter",
                f"anomaly_batcher_batches_total {st['total_batches']}",
                "# TYPE anomaly_batcher_timeouts_queued_total counter",
                f"anomaly_batcher_timeouts_queued_total {st['timeouts_queued']}",
                "# TYPE anomaly_batcher_timeouts_running_total counter",
                f"anomaly_batcher_timeouts_running_total {st['timeouts_running']}",
                "# TYPE anomaly_batcher_rejected_oversize_total counter",
                f"anomaly_batcher_rejected_oversize_total {st['rejected_oversize']}",
                "# TYPE anomaly_batcher_queue_depth gauge",
                f"anomaly_batcher_queue_depth {st['queue_depth']}",
                "# TYPE anomaly_batcher_active_batches gauge",
                f"anomaly_batcher_active_batches {st['active_batches']}",
            ]
            return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")
        return batcher.stats()

    # Explicit score aliases.
    for path in SCORE_PATHS:
        app.add_api_route(path, handle_score, methods=["POST"], include_in_schema=False)

    # GET aliases (?values=1,2,3&timeout_ms=1000).
    for path in ("/score", "/predict", "/anomaly", "/api/score", "/api/predict"):
        app.add_api_route(path, handle_score, methods=["GET"], include_in_schema=False)

    # Last-resort: any other request is treated as a scoring request.
    @app.api_route("/{full_path:path}", methods=["POST", "PUT", "PATCH", "GET"], include_in_schema=False)
    async def score_catch_all(request: Request, full_path: str):
        return await handle_score(request)

    return app


# Default ASGI application used by `uvicorn main:app` / `uvicorn app.main:app`.
app = create_app()
