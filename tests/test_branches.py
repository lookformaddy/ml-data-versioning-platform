"""Branch / merge tests: clean merges, conflict detection, branch rules."""
import pytest

from mdvp.registry import MergeConflictError
from tests.conftest import SCHEMA, TODAY, rows_v1


def _changed(rows, key, **updates):
    out = [dict(r) for r in rows]
    for row in out:
        if row["customer_id"] == key:
            row.update(updates)
    return out


def test_branch_and_clean_merge(registry, v1):
    registry.create_branch("churn", "experiment", "1.0.0")
    exp_rows = _changed(rows_v1(), "C001", age=50)
    exp_rows.append({"customer_id": "C011", "age": 31, "monthly_spend": 61.0,
                     "region": "east", "churned": 0})
    v_exp = registry.commit("churn", exp_rows, SCHEMA, branch="experiment",
                            created_by="b.okafor", now=TODAY)
    assert v_exp.version_id == "1.1.0"
    merged = registry.merge("churn", "experiment", "main",
                            created_by="a.chen", now=TODAY)
    assert merged.version_id == "1.2.0"
    assert merged.row_count == 11
    by_id = {r["customer_id"]: r for r in merged.rows}
    assert by_id["C001"]["age"] == 50
    assert registry.head("churn", "main").version_id == "1.2.0"


def test_merge_conflict_on_overlapping_row_edit(registry, v1):
    registry.create_branch("churn", "experiment", "1.0.0")
    registry.commit("churn", _changed(rows_v1(), "C001", age=50), SCHEMA,
                    branch="experiment", created_by="b.okafor", now=TODAY)
    registry.commit("churn", _changed(rows_v1(), "C001", age=60), SCHEMA,
                    branch="main", created_by="a.chen", now=TODAY)
    with pytest.raises(MergeConflictError) as excinfo:
        registry.merge("churn", "experiment", "main", created_by="a.chen", now=TODAY)
    assert excinfo.value.conflicts == ["C001"]
    # Fail-closed: no merged version was created on main.
    assert registry.head("churn", "main").version_id == "1.2.0"


def test_merge_same_value_both_sides_is_not_a_conflict(registry, v1):
    registry.create_branch("churn", "experiment", "1.0.0")
    registry.commit("churn", _changed(rows_v1(), "C001", age=50), SCHEMA,
                    branch="experiment", created_by="b.okafor", now=TODAY)
    registry.commit("churn", _changed(rows_v1(), "C001", age=50), SCHEMA,
                    branch="main", created_by="a.chen", now=TODAY)
    merged = registry.merge("churn", "experiment", "main",
                            created_by="a.chen", now=TODAY)
    assert merged.row_count == 10
    assert {r["customer_id"]: r for r in merged.rows}["C001"]["age"] == 50


def test_merge_source_deletion_applies_when_target_untouched(registry, v1):
    registry.create_branch("churn", "experiment", "1.0.0")
    exp_rows = [r for r in rows_v1() if r["customer_id"] != "C002"]
    registry.commit("churn", exp_rows, SCHEMA, branch="experiment",
                    created_by="b.okafor", now=TODAY)
    merged = registry.merge("churn", "experiment", "main",
                            created_by="a.chen", now=TODAY)
    assert merged.row_count == 9
    assert "C002" not in {r["customer_id"] for r in merged.rows}


def test_duplicate_branch_rejected(registry, v1):
    registry.create_branch("churn", "experiment", "1.0.0")
    with pytest.raises(ValueError, match="already exists"):
        registry.create_branch("churn", "experiment", "1.0.0")


def test_merge_same_head_rejected(registry, v1):
    registry.create_branch("churn", "experiment", "1.0.0")
    with pytest.raises(ValueError, match="same version"):
        registry.merge("churn", "experiment", "main", created_by="a.chen", now=TODAY)


def test_merge_unknown_branch_rejected(registry, v1):
    with pytest.raises(ValueError, match="unknown branch"):
        registry.merge("churn", "ghost", "main", created_by="a.chen", now=TODAY)
