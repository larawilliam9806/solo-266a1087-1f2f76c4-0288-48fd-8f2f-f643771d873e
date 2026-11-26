import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from app.parsing import coerce_sequences, coerce_timeout_ms, extract_payload  # noqa: E402


def test_flat_values():
    seqs = coerce_sequences([1, 2.5, -3])
    assert len(seqs) == 1
    assert seqs[0].dtype == np.float32
    assert list(seqs[0]) == [1.0, 2.5, -3.0]


def test_column_vector():
    seqs = coerce_sequences([[1], [2], [3]])
    assert len(seqs) == 1
    assert list(seqs[0]) == [1, 2, 3]


def test_nested_single():
    seqs = coerce_sequences([[1, 2, 3]])
    assert len(seqs) == 1
    assert list(seqs[0]) == [1, 2, 3]


def test_multi():
    seqs = coerce_sequences([[1, 2], [3, 4, 5]])
    assert len(seqs) == 2
    assert seqs[1].shape[0] == 3


def test_string_values():
    seqs = coerce_sequences("0.1, 0.2 0.3; 0.4")
    assert len(seqs[0]) == 4


def test_empty_rejected():
    with pytest.raises(Exception):
        coerce_sequences([])


def test_nan_rejected():
    with pytest.raises(Exception):
        coerce_sequences([1.0, float("nan")])


def test_timeout_aliases_and_units():
    assert coerce_timeout_ms(50, 1000) == 50
    assert coerce_timeout_ms("25ms", 1000) == 25
    assert coerce_timeout_ms("2s", 1000) == 2000
    assert coerce_timeout_ms({"seconds": 0.5}, 1000) == 500
    assert coerce_timeout_ms(None, 777) == 777


def test_extract_object_aliases():
    for key in ("values", "samples", "data", "signal", "x"):
        seqs, tmo, multi = extract_payload({key: [1, 2], "timeout_ms": 42}, {})
        assert seqs[0].shape[0] == 2
        assert tmo == 42
        assert multi is False


def test_extract_nested_timeout():
    _, tmo, _ = extract_payload({"values": [1], "options": {"timeoutMs": 99}}, {})
    assert tmo == 99


def test_extract_instances_envelope():
    seqs, tmo, multi = extract_payload(
        {"instances": [[1, 2], [3]], "timeout_ms": 5}, {}
    )
    assert multi is True
    assert len(seqs) == 2
    assert tmo == 5


def test_extract_bare_array_with_query_timeout():
    seqs, tmo, _ = extract_payload([1, 2, 3], {"timeout_ms": "123"})
    assert seqs[0].shape[0] == 3
    assert tmo == 123
