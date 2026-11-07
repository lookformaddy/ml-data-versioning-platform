"""Drift comparison between dataset versions (pure stdlib).

* PSI (population stability index) for numeric columns via
  reference-quantile bins, and for categorical columns via value
  proportions. Standard rule of thumb: PSI < 0.1 stable, 0.1–0.25
  watch, > 0.25 drifted.
* Two-sample Kolmogorov–Smirnov statistic for numeric columns: the
  maximum absolute CDF difference over the combined sorted values.
* Schema drift: added / removed / type-changed columns, plus a
  rename heuristic — a removed column and an added column of the same
  type whose value multisets are identical are reported as a possible
  rename rather than a breaking change.
"""
from __future__ import annotations

import math
from collections import Counter

from mdvp.models import DatasetVersion, DType, DriftReport, SchemaDrift, SchemaField

_EPS = 1e-4
PSI_STABLE = 0.1
PSI_WATCH = 0.25
KS_STABLE = 0.1
KS_WATCH = 0.2


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sample")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _psi_from_proportions(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(e, _EPS)
        a = max(a, _EPS)
        total += (a - e) * math.log(a / e)
    return total


def psi_numeric(reference: list[float], current: list[float], bins: int = 10) -> float:
    """PSI between two numeric samples using reference-quantile bins."""
    if not reference or not current:
        raise ValueError("PSI needs non-empty reference and current samples")
    ref_sorted = sorted(reference)
    edges = [_quantile(ref_sorted, i / bins) for i in range(bins + 1)]
    # Deduplicate edges (low-cardinality data collapses bins).
    deduped = [edges[0]]
    for edge in edges[1:]:
        if edge > deduped[-1]:
            deduped.append(edge)
    if len(deduped) < 2:
        return 0.0  # constant column: no distribution to shift
    expected: list[float] = []
    actual: list[float] = []
    for lo, hi in zip(deduped[:-1], deduped[1:]):
        last = hi == deduped[-1]
        e = sum(1 for v in reference if (lo <= v < hi) or (last and v == hi))
        a = sum(1 for v in current if (lo <= v < hi) or (last and v == hi))
        expected.append(e / len(reference))
        actual.append(a / len(current))
    return _psi_from_proportions(expected, actual)


def psi_categorical(reference: list, current: list) -> float:
    """PSI between two categorical samples via value proportions."""
    if not reference or not current:
        raise ValueError("PSI needs non-empty reference and current samples")
    ref_counts = Counter(str(v) for v in reference)
    cur_counts = Counter(str(v) for v in current)
    keys = set(ref_counts) | set(cur_counts)
    expected = [ref_counts.get(k, 0) / len(reference) for k in keys]
    actual = [cur_counts.get(k, 0) / len(current) for k in keys]
    return _psi_from_proportions(expected, actual)


def ks_statistic(reference: list[float], current: list[float]) -> float:
    """Two-sample KS statistic: max |CDF_ref - CDF_cur|."""
    if not reference or not current:
        raise ValueError("KS needs non-empty reference and current samples")
    ref_sorted = sorted(reference)
    cur_sorted = sorted(current)
    pooled = sorted(set(ref_sorted) | set(cur_sorted))
    max_diff = 0.0
    for value in pooled:
        cdf_ref = sum(1 for v in ref_sorted if v <= value) / len(ref_sorted)
        cdf_cur = sum(1 for v in cur_sorted if v <= value) / len(cur_sorted)
        max_diff = max(max_diff, abs(cdf_ref - cdf_cur))
    return max_diff


def schema_drift(
    old_schema: list[SchemaField],
    new_schema: list[SchemaField],
    old_rows: list[dict] | None = None,
    new_rows: list[dict] | None = None,
) -> SchemaDrift:
    """Compare two schemas; detect renames via identical value multisets."""
    old = {f.name: f for f in old_schema}
    new = {f.name: f for f in new_schema}
    added = [n for n in new if n not in old]
    removed = [n for n in old if n not in new]
    type_changed = [
        n for n in old if n in new and old[n].dtype != new[n].dtype
    ]
    renames: list[dict] = []
    if old_rows is not None and new_rows is not None:
        remaining_added = list(added)
        remaining_removed = list(removed)
        for old_name in list(removed):
            old_values = sorted(str(r.get(old_name)) for r in old_rows)
            for new_name in list(remaining_added):
                if old[old_name].dtype != new[new_name].dtype:
                    continue
                new_values = sorted(str(r.get(new_name)) for r in new_rows)
                if old_values == new_values:
                    renames.append({"from": old_name, "to": new_name})
                    remaining_added.remove(new_name)
                    remaining_removed.remove(old_name)
                    break
        added, removed = remaining_added, remaining_removed
    breaking = bool(removed or type_changed)
    return SchemaDrift(
        columns_added=added,
        columns_removed=removed,
        columns_type_changed=type_changed,
        possible_renames=renames,
        breaking=breaking,
    )


def compare_versions(
    reference: DatasetVersion,
    current: DatasetVersion,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> DriftReport:
    """Distribution + schema comparison between two versions.

    Numeric columns default to every shared column whose dtype is
    integer/number; categorical columns default to every shared string
    or boolean column.
    """
    shared = [c for c in reference.columns if c in current.columns]
    ref_by_dtype = {f.name: f.dtype for f in reference.schema}
    if numeric_columns is None:
        numeric_columns = [
            c for c in shared if ref_by_dtype[c] in (DType.INTEGER, DType.NUMBER)
        ]
    if categorical_columns is None:
        categorical_columns = [
            c for c in shared if ref_by_dtype[c] in (DType.STRING, DType.BOOLEAN)
        ]
    per_column: list[dict] = []
    max_psi = 0.0
    max_ks = 0.0
    for col in numeric_columns:
        ref_values = [float(r[col]) for r in reference.rows if r.get(col) is not None]
        cur_values = [float(r[col]) for r in current.rows if r.get(col) is not None]
        if not ref_values or not cur_values:
            continue
        psi = psi_numeric(ref_values, cur_values)
        ks = ks_statistic(ref_values, cur_values)
        max_psi = max(max_psi, psi)
        max_ks = max(max_ks, ks)
        per_column.append(
            {"column": col, "kind": "numeric", "psi": round(psi, 4), "ks": round(ks, 4)}
        )
    for col in categorical_columns:
        ref_values = [r[col] for r in reference.rows if r.get(col) is not None]
        cur_values = [r[col] for r in current.rows if r.get(col) is not None]
        if not ref_values or not cur_values:
            continue
        psi = psi_categorical(ref_values, cur_values)
        max_psi = max(max_psi, psi)
        per_column.append({"column": col, "kind": "categorical", "psi": round(psi, 4)})
    schema = schema_drift(reference.schema, current.schema, reference.rows, current.rows)
    if max_psi >= PSI_WATCH or max_ks >= KS_WATCH or schema.breaking:
        verdict = "drifted"
    elif max_psi >= PSI_STABLE or max_ks >= KS_STABLE:
        verdict = "watch"
    else:
        verdict = "stable"
    return DriftReport(
        reference_version_id=reference.version_id,
        current_version_id=current.version_id,
        per_column=per_column,
        max_psi=round(max_psi, 4),
        max_ks=round(max_ks, 4),
        verdict=verdict,
    )
