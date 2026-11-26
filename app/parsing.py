"""Tolerant parsing of scoring requests.

The service accepts many field-name / payload-style conventions because the
wire contract is not fixed by the problem statement:

* JSON object with ``values`` / ``samples`` / ``data`` / ... and
  ``timeout_ms`` / ``timeoutMs`` / ``timeout`` / ...
* a bare JSON array of numbers,
* ``{"instances": [[...], ...]}`` / ``{"items": [...]}`` multi-sequence calls,
* plain text (comma / whitespace separated numbers),
* query parameters (``?values=1,2,3&timeout_ms=1000``),
* simple HTML form fields.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

VALUES_KEYS = (
    "values",
    "sample_values",
    "samples",
    "sample",
    "data",
    "input_data",
    "input_values",
    "sequence",
    "sequence_data",
    "sequences",
    "signal",
    "signals",
    "readings",
    "measurement",
    "measurements",
    "waveform",
    "wave",
    "vibration",
    "points",
    "series",
    "time_series",
    "timeseries",
    "numbers",
    "vals",
    "x",
    "input",
    "inputs",
    "payload",
    "arr",
    "array",
    "value",
)

MULTI_KEYS = ("instances", "items", "requests", "batch", "data_batch", "records")

TIMEOUT_KEYS = (
    "timeout_ms",
    "timeoutMs",
    "timeoutMilliseconds",
    "timeout_milliseconds",
    "timeoutMillis",
    "request_timeout_ms",
    "requestTimeoutMs",
    "wait_timeout_ms",
    "deadline_ms",
    "deadlineMs",
    "max_wait_ms",
    "ttl_ms",
    "timeout",
    "deadline",
    "ttl",
)

NESTED_CONTAINER_KEYS = ("options", "config", "params", "parameters", "meta", "metadata", "settings")


class ParseError(ValueError):
    pass


def _is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return True
    return isinstance(x, (int, float)) and not isinstance(x, type(None))


def _to_float_list(raw: Any) -> list[float]:
    if isinstance(raw, str):
        raw = raw.replace("[", " ").replace("]", " ").replace(",", " ").replace(";", " ")
        parts = raw.split()
        try:
            return [float(p) for p in parts]
        except ValueError as exc:
            raise ParseError(f"could not parse numeric values from text: {raw[:80]!r}") from exc
    if not isinstance(raw, (list, tuple)):
        raise ParseError("values must be a list of numbers")
    out: list[float] = []
    for x in raw:
        if isinstance(x, bool):
            out.append(float(x))
        elif _is_number(x):
            f = float(x)
            if not math.isfinite(f):
                raise ParseError("values must be finite (NaN/Inf are not supported)")
            out.append(f)
        elif isinstance(x, str):
            out.extend(_to_float_list(x))
        else:
            raise ParseError(f"unsupported sample element: {type(x).__name__}")
    return out


def coerce_sequences(raw: Any) -> list[np.ndarray]:
    """Normalise many payload shapes into a list of 1-D float32 arrays."""
    if raw is None:
        raise ParseError("missing sample values")

    if isinstance(raw, np.ndarray):
        raw = raw.tolist()

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [np.asarray([float(raw)], dtype=np.float32)]
    if isinstance(raw, str):
        vals = _to_float_list(raw)
        if not vals:
            raise ParseError("value sequence is empty")
        return [np.asarray(vals, dtype=np.float32)]

    if not isinstance(raw, (list, tuple)):
        if isinstance(raw, dict):
            # Maybe {"values": ...} nested under an unknown wrapper.
            for key in VALUES_KEYS:
                if key in raw:
                    return coerce_sequences(raw[key])
        raise ParseError("body must be a JSON array or an object containing sample values")

    if len(raw) == 0:
        raise ParseError("value sequence is empty")

    # Flat numeric sequence.
    if all(_is_number(x) for x in raw):
        vals = _to_float_list(raw)
        if not vals:
            raise ParseError("value sequence is empty")
        return [np.asarray(vals, dtype=np.float32)]

    # Column vector [[x], [y], ...].
    if (
        all(isinstance(x, (list, tuple)) for x in raw)
        and all(len(x) == 1 and _is_number(x[0]) for x in raw)
    ):
        return [np.asarray([float(x[0]) for x in raw], dtype=np.float32)]  # type: ignore[index]

    # List of sequences / dict records.
    sequences: list[np.ndarray] = []
    for item in raw:
        if isinstance(item, dict):
            found = None
            for key in VALUES_KEYS:
                if key in item:
                    found = item[key]
                    break
            if found is None:
                raise ParseError("record is missing a 'values' field")
            sequences.extend(coerce_sequences(found))
        elif isinstance(item, (list, tuple)):
            vals = _to_float_list(item)
            if not vals:
                raise ParseError("empty sequence in batch")
            sequences.append(np.asarray(vals, dtype=np.float32))
        elif _is_number(item):
            # Mixed list; treat whole thing as flat after coercion failure above.
            raise ParseError("values contain a mix of numbers and nested lists")
        else:
            raise ParseError(f"unsupported batch item: {type(item).__name__}")

    if not sequences:
        raise ParseError("no sequences found")
    for seq in sequences:
        if seq.size == 0:
            raise ParseError("value sequence is empty")
    return sequences


def _lookup_timeout(d: dict[str, Any]) -> Any:
    for key in TIMEOUT_KEYS:
        if key in d and d[key] is not None:
            return d[key]
    for container in NESTED_CONTAINER_KEYS:
        sub = d.get(container)
        if isinstance(sub, dict):
            got = _lookup_timeout(sub)
            if got is not None:
                return got
    return None


def coerce_timeout_ms(value: Any, default_ms: int) -> int:
    if value is None:
        return default_ms
    if isinstance(value, dict):
        if value.get("ms") is not None:
            value = value["ms"]
        elif value.get("milliseconds") is not None:
            value = value["milliseconds"]
        elif value.get("seconds") is not None or value.get("s") is not None:
            sec = value.get("seconds", value.get("s"))
            return max(0, int(float(sec) * 1000))
    if isinstance(value, str):
        value = value.strip()
        unit_ms = 1.0
        low = value.lower()
        if low.endswith("ms"):
            value = low[:-2]
        elif low.endswith("s") and not low.endswith("ms"):
            unit_ms = 1000.0
            value = low[:-1]
        try:
            return max(0, int(float(value) * unit_ms))
        except ValueError:
            return default_ms
    if _is_number(value):
        return max(0, int(float(value)))
    return default_ms


def extract_payload(
    body: Any, query: dict[str, str] | None = None
) -> tuple[list[np.ndarray], int | None, bool]:
    """Return ``(sequences, timeout_ms_or_None, explicit_multi)``."""
    query = query or {}
    explicit_multi = False
    timeout_raw: Any = None

    # Query string wins for timeout when present.
    q_timeout = None
    for key in TIMEOUT_KEYS:
        if key in query:
            q_timeout = query[key]
            break

    if body is None:
        q_values = query.get("values") or query.get("samples") or query.get("data")
        if q_values is None:
            raise ParseError("no sample values provided")
        sequences = coerce_sequences(q_values)
        return sequences, coerce_timeout_ms(q_timeout, -1) if q_timeout is not None else None, False

    if isinstance(body, (list, tuple)):
        sequences = coerce_sequences(body)
        timeout_ms = coerce_timeout_ms(q_timeout, -1) if q_timeout is not None else None
        return sequences, timeout_ms, len(sequences) > 1

    if not isinstance(body, dict):
        raise ParseError("unsupported JSON payload")

    timeout_raw = _lookup_timeout(body)
    if q_timeout is not None:
        timeout_raw = q_timeout

    # Explicit multi-sequence envelopes.
    for key in MULTI_KEYS:
        if key in body and body[key] is not None:
            sequences = coerce_sequences(body[key])
            explicit_multi = True
            timeout_ms = coerce_timeout_ms(timeout_raw, -1) if timeout_raw is not None else None
            return sequences, timeout_ms, explicit_multi

    values_raw = None
    for key in VALUES_KEYS:
        if key in body and body[key] is not None:
            values_raw = body[key]
            break

    if values_raw is None:
        # Maybe the whole dict maps indices/keys to numbers? Reject clearly.
        raise ParseError(
            "request must include one of: " + ", ".join(VALUES_KEYS[:6])
        )

    sequences = coerce_sequences(values_raw)

    # A nested [1, L] payload is still a single request unless an explicit
    # multi envelope was used.
    if len(sequences) == 1:
        explicit_multi = False
    return sequences, (coerce_timeout_ms(timeout_raw, -1) if timeout_raw is not None else None), explicit_multi


def parse_json_lenient(raw: bytes, content_type: str) -> Any:
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    if "application/json" in content_type or text.lstrip()[:1] in {"{", "["}:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc.msg} at line {exc.lineno}") from exc
    # Plain text / form-ish: let coerce_sequences handle strings.
    return text
