"""Drift tests: PSI / KS on identical, shifted, and categorical data;
schema drift including rename detection."""
import pytest

from mdvp.drift import (
    compare_versions,
    ks_statistic,
    psi_categorical,
    psi_numeric,
    schema_drift,
)
from mdvp.models import Dataset
from tests.conftest import SCHEMA, TODAY, rows_v1


def test_psi_identical_distributions_is_zero():
    values = [float(i) for i in range(100)]
    assert psi_numeric(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_large_shift_is_large():
    reference = [float(i) for i in range(100)]
    shifted = [float(i) + 500 for i in range(100)]
    assert psi_numeric(reference, shifted) > 0.25


def test_psi_categorical_shift():
    reference = ["a"] * 90 + ["b"] * 10
    current = ["a"] * 50 + ["b"] * 50
    assert psi_categorical(reference, current) > 0.25
    assert psi_categorical(reference, reference) == pytest.approx(0.0, abs=1e-9)


def test_ks_identical_is_zero():
    values = [float(i) for i in range(50)]
    assert ks_statistic(values, values) == pytest.approx(0.0)


def test_ks_shifted_is_large():
    reference = [float(i) for i in range(100)]
    shifted = [float(i) + 50 for i in range(100)]
    assert ks_statistic(reference, shifted) > 0.4


def test_ks_rejects_empty_samples():
    with pytest.raises(ValueError):
        ks_statistic([], [1.0])


def test_compare_identical_versions_is_stable(registry, v1):
    report = compare_versions(v1, v1)
    assert report.verdict == "stable"
    assert report.max_psi == 0.0
    assert report.max_ks == 0.0


def test_compare_shifted_version_is_drifted(registry, v1):
    rows = [dict(r) for r in rows_v1()]
    for row in rows:
        row["monthly_spend"] = row["monthly_spend"] + 500.0
    v2 = registry.commit("churn", rows, SCHEMA, branch="main",
                         created_by="a.chen", now=TODAY)
    report = compare_versions(v1, v2)
    assert report.verdict == "drifted"
    spend = next(c for c in report.per_column if c["column"] == "monthly_spend")
    assert spend["psi"] > 0.25


def test_compare_small_shift_is_watch_or_stable(registry, v1):
    # A 2% shift needs a decent sample for PSI to be stable: build a
    # 200-row snapshot rather than the 10-row fixture.
    base = [
        {"customer_id": f"C{i:04d}", "age": 20 + (i % 60),
         "monthly_spend": 50.0 + (i % 200), "region": "north", "churned": i % 2}
        for i in range(200)
    ]
    registry.create_dataset(
        Dataset(dataset_id="big", name="Big", owner="a.chen",
                primary_key="customer_id", row_count_floor=5),
        now=TODAY,
    )
    ref = registry.commit("big", base, SCHEMA, branch="main",
                          created_by="a.chen", now=TODAY)
    shifted = [dict(r, monthly_spend=round(r["monthly_spend"] * 1.02, 2))
               for r in base]
    cur = registry.commit("big", shifted, SCHEMA, branch="main",
                          created_by="a.chen", now=TODAY)
    report = compare_versions(ref, cur)
    assert report.verdict in ("stable", "watch")
    assert report.max_psi < 0.25


def test_schema_drift_detects_added_removed_type_changed():
    from mdvp.models import DType, SchemaField

    new_schema = [
        SchemaField("customer_id", DType.STRING, nullable=False),
        SchemaField("age", DType.NUMBER),  # was INTEGER
        SchemaField("monthly_spend", DType.NUMBER),
        SchemaField("tenure_months", DType.INTEGER),  # added
        SchemaField("churned", DType.INTEGER),
        # region removed
    ]
    drift = schema_drift(SCHEMA, new_schema)
    assert drift.columns_added == ["tenure_months"]
    assert drift.columns_removed == ["region"]
    assert drift.columns_type_changed == ["age"]
    assert drift.breaking is True


def test_schema_drift_rename_detection():
    from mdvp.models import DType, SchemaField

    new_schema = [
        SchemaField("customer_id", DType.STRING, nullable=False),
        SchemaField("age", DType.INTEGER),
        SchemaField("monthly_spend", DType.NUMBER),
        SchemaField("area", DType.STRING),  # renamed from region
        SchemaField("churned", DType.INTEGER),
    ]
    rows = rows_v1()
    renamed = [
        {**{k: v for k, v in r.items() if k != "region"}, "area": r["region"]}
        for r in rows
    ]
    drift = schema_drift(SCHEMA, new_schema, rows, renamed)
    assert drift.possible_renames == [{"from": "region", "to": "area"}]
    assert drift.columns_added == []
    assert drift.columns_removed == []
    assert drift.breaking is False
