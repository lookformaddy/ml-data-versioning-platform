"""Hashing tests: determinism, sensitivity, schema inference."""
from mdvp import hashing
from mdvp.models import DType
from tests.conftest import SCHEMA, rows_v1


def test_row_hash_deterministic_across_key_order():
    row_a = {"customer_id": "C001", "age": 21, "region": "north"}
    row_b = {"region": "north", "age": 21, "customer_id": "C001"}
    assert hashing.row_hash(row_a) == hashing.row_hash(row_b)


def test_row_hash_sensitive_to_value_change():
    row = {"customer_id": "C001", "age": 21}
    changed = dict(row, age=22)
    assert hashing.row_hash(row) != hashing.row_hash(changed)


def test_manifest_hash_changes_when_row_changes(v1):
    new_rows = [dict(r) for r in v1.rows]
    new_rows[0]["age"] = 99
    new_hashes = hashing.hash_rows(new_rows, "customer_id")
    assert hashing.manifest_hash(new_hashes, SCHEMA) != v1.manifest_hash


def test_manifest_hash_changes_when_schema_changes(v1):
    schema = [f for f in SCHEMA if f.name != "region"]
    assert hashing.schema_hash(schema) != hashing.schema_hash(SCHEMA)
    assert hashing.manifest_hash(v1.row_hashes, schema) != v1.manifest_hash


def test_hash_rows_rejects_missing_primary_key():
    rows = [{"age": 21}]
    try:
        hashing.hash_rows(rows, "customer_id")
    except ValueError as exc:
        assert "missing primary key" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hash_rows_rejects_duplicate_primary_key():
    rows = [{"customer_id": "C001", "age": 21}, {"customer_id": "C001", "age": 22}]
    try:
        hashing.hash_rows(rows, "customer_id")
    except ValueError as exc:
        assert "duplicate primary key" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_infer_schema_types():
    rows = [
        {"id": "a", "n": 3, "x": 1.5, "flag": True, "opt": None},
        {"id": "b", "n": 4, "x": 2.5, "flag": False, "opt": None},
    ]
    schema = {f.name: f for f in hashing.infer_schema(rows)}
    assert schema["id"].dtype is DType.STRING
    assert schema["n"].dtype is DType.INTEGER
    assert schema["x"].dtype is DType.NUMBER
    assert schema["flag"].dtype is DType.BOOLEAN
    assert schema["opt"].nullable is True


def test_code_hash_deterministic_and_sensitive():
    files = {"a.py": "x = 1\n", "b.py": "y = 2\n"}
    assert hashing.code_hash(files) == hashing.code_hash(dict(reversed(list(files.items()))))
    assert hashing.code_hash(files) != hashing.code_hash({"a.py": "x = 2\n", "b.py": "y = 2\n"})
