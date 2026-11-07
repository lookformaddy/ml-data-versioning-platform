"""Content hashing for dataset snapshots.

All hashing is deterministic: rows are serialized as canonical JSON
(sorted keys, compact separators, UTF-8) before SHA-256, so the same
logical snapshot always produces the same hash regardless of dict
insertion order or process. A version's manifest hash pins the sorted
row hashes together with the schema hash, so any row or schema change
changes the manifest hash.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from mdvp.models import DType, SchemaField


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically for hashing / manifests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_hash(row: dict) -> str:
    """SHA-256 of one row's canonical JSON."""
    return sha256_hex(canonical_json(row))


def schema_hash(schema: list[SchemaField]) -> str:
    """SHA-256 of the schema, ordered by column name."""
    fields = sorted((f.to_dict() for f in schema), key=lambda d: d["name"])
    return sha256_hex(canonical_json(fields))


def manifest_hash(row_hashes: dict[str, str], schema: list[SchemaField]) -> str:
    """SHA-256 pinning the whole snapshot: sorted row hashes + schema."""
    ordered = [row_hashes[key] for key in sorted(row_hashes)]
    payload = canonical_json({"rows": ordered, "schema": schema_hash(schema)})
    return sha256_hex(payload)


def hash_rows(rows: list[dict], primary_key: str) -> dict[str, str]:
    """Map primary-key value -> row hash. Raises on missing/duplicate keys."""
    hashes: dict[str, str] = {}
    for row in rows:
        if primary_key not in row:
            raise ValueError(f"row is missing primary key column {primary_key!r}: {row!r}")
        key = str(row[primary_key])
        if key in hashes:
            raise ValueError(f"duplicate primary key value {key!r} in snapshot")
        hashes[key] = row_hash(row)
    return hashes


def infer_dtype(values: list) -> DType:
    """Infer a column's DType from its non-null sample values."""
    seen = {type(v) for v in values if v is not None}
    if not seen:
        return DType.STRING
    if seen <= {bool}:
        return DType.BOOLEAN
    if seen <= {int} and not (bool in seen):
        return DType.INTEGER
    if seen <= {int, float} and not (bool in seen):
        return DType.NUMBER
    return DType.STRING


def infer_schema(rows: list[dict]) -> list[SchemaField]:
    """Infer a schema from rows (column order = first-seen order)."""
    columns: list[str] = []
    values: dict[str, list] = {}
    for row in rows:
        for name, value in row.items():
            if name not in values:
                columns.append(name)
                values[name] = []
            values[name].append(value)
    fields = []
    for name in columns:
        col_values = values[name]
        fields.append(
            SchemaField(
                name=name,
                dtype=infer_dtype(col_values),
                nullable=any(v is None for v in col_values),
            )
        )
    return fields


def chain_hash(steps: list[dict]) -> str:
    """SHA-256 over a list of canonical-JSON transform-step dicts."""
    return sha256_hex(canonical_json(steps))


def code_hash(file_contents: dict[str, str]) -> str:
    """SHA-256 pinning a set of source files (path -> content)."""
    ordered = {path: file_contents[path] for path in sorted(file_contents)}
    return sha256_hex(canonical_json(ordered))
