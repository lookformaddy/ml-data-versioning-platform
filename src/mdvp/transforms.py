"""Deterministic, pure-stdlib dataset transforms.

Every transform is specified by a JSON-serializable ``params`` dict
(so the transform chain can be hashed into a reproducibility manifest
and replayed exactly) and returns both the output rows and the
column-level lineage: each output column mapped to the
``(source version id, source column)`` pairs it derives from.

Supported ops:
* ``filter``    — keep rows matching one predicate
  (``{"column", "op", "value"}``; ops ``== != < <= > >= in "not in"``).
* ``project``   — keep a subset of columns, in order.
* ``join``      — inner / left join against a second snapshot on a key.
* ``aggregate`` — group-by with sum / mean / count / min / max.
"""
from __future__ import annotations

from mdvp.models import DatasetVersion, TransformStep

_OPS = ("==", "!=", "<", "<=", ">", ">=", "in", "not in")
_AGGS = ("sum", "mean", "count", "min", "max")


def _matches(row: dict, column: str, op: str, value) -> bool:
    if column not in row:
        raise ValueError(f"filter column {column!r} not present in row")
    actual = row[column]
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    if op == "<":
        return actual < value
    if op == "<=":
        return actual <= value
    if op == ">":
        return actual > value
    if op == ">=":
        return actual >= value
    if op == "in":
        return actual in value
    if op == "not in":
        return actual not in value
    raise ValueError(f"unknown filter op {op!r}")


def filter_rows(version: DatasetVersion, column: str, op: str, value) -> tuple[list[dict], TransformStep]:
    if op not in _OPS:
        raise ValueError(f"unknown filter op {op!r}; expected one of {_OPS}")
    kept = [dict(r) for r in version.rows if _matches(r, column, op, value)]
    lineage = {c: [(version.version_id, c)] for c in version.columns}
    step = TransformStep(
        op="filter",
        params={"column": column, "op": op, "value": value},
        input_version_ids=[version.version_id],
        column_lineage=lineage,
    )
    return kept, step


def project_rows(version: DatasetVersion, columns: list[str]) -> tuple[list[dict], TransformStep]:
    unknown = [c for c in columns if c not in version.columns]
    if unknown:
        raise ValueError(f"project columns not in schema: {unknown}")
    projected = [{c: r[c] for c in columns} for r in version.rows]
    lineage = {c: [(version.version_id, c)] for c in columns}
    step = TransformStep(
        op="project",
        params={"columns": list(columns)},
        input_version_ids=[version.version_id],
        column_lineage=lineage,
    )
    return projected, step


def join_rows(
    left: DatasetVersion,
    right: DatasetVersion,
    on: str,
    how: str = "inner",
) -> tuple[list[dict], TransformStep]:
    if how not in ("inner", "left"):
        raise ValueError(f"unsupported join type {how!r}; expected 'inner' or 'left'")
    if on not in left.columns or on not in right.columns:
        raise ValueError(f"join key {on!r} must exist in both snapshots")
    right_index: dict = {}
    for row in right.rows:
        right_index.setdefault(row[on], []).append(row)
    right_extra = [c for c in right.columns if c != on]
    out: list[dict] = []
    for lrow in left.rows:
        matches = right_index.get(lrow[on], [])
        if not matches:
            if how == "left":
                merged = dict(lrow)
                for c in right_extra:
                    merged[c] = None
                out.append(merged)
            continue
        for rrow in matches:
            merged = dict(lrow)
            for c in right_extra:
                # Right-side collision: keep the left value, prefix right's.
                merged[c if c not in merged else f"right_{c}"] = rrow[c]
            out.append(merged)
    lineage: dict[str, list[tuple[str, str]]] = {}
    for c in left.columns:
        lineage[c] = [(left.version_id, c)]
    for c in right_extra:
        key = c if c not in left.columns else f"right_{c}"
        lineage[key] = [(right.version_id, c)]
    step = TransformStep(
        op="join",
        params={"on": on, "how": how, "right_version_id": right.version_id},
        input_version_ids=[left.version_id, right.version_id],
        column_lineage=lineage,
    )
    return out, step


def aggregate_rows(
    version: DatasetVersion,
    group_by: list[str],
    aggregations: dict[str, str],
) -> tuple[list[dict], TransformStep]:
    unknown_groups = [c for c in group_by if c not in version.columns]
    if unknown_groups:
        raise ValueError(f"group-by columns not in schema: {unknown_groups}")
    for col, agg in aggregations.items():
        if col not in version.columns:
            raise ValueError(f"aggregation column {col!r} not in schema")
        if agg not in _AGGS:
            raise ValueError(f"unknown aggregation {agg!r}; expected one of {_AGGS}")
    groups: dict[tuple, list[dict]] = {}
    for row in version.rows:
        key = tuple(row[c] for c in group_by)
        groups.setdefault(key, []).append(row)
    out: list[dict] = []
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        rows = groups[key]
        record = {c: v for c, v in zip(group_by, key)}
        for col, agg in aggregations.items():
            values = [r[col] for r in rows if r.get(col) is not None]
            out_col = f"{col}_{agg}"
            if agg == "count":
                record[out_col] = len(values)
            elif not values:
                record[out_col] = None
            elif agg == "sum":
                record[out_col] = sum(values)
            elif agg == "mean":
                record[out_col] = sum(values) / len(values)
            elif agg == "min":
                record[out_col] = min(values)
            elif agg == "max":
                record[out_col] = max(values)
        out.append(record)
    lineage: dict[str, list[tuple[str, str]]] = {}
    for c in group_by:
        lineage[c] = [(version.version_id, c)]
    for col, agg in aggregations.items():
        lineage[f"{col}_{agg}"] = [(version.version_id, col)]
    step = TransformStep(
        op="aggregate",
        params={"group_by": list(group_by), "aggregations": dict(aggregations)},
        input_version_ids=[version.version_id],
        column_lineage=lineage,
    )
    return out, step
