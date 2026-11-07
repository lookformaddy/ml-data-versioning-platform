"""Registry tests: commits, change summaries, semantic versioning,
immutability, validation."""
import pytest

from mdvp.models import Stage
from mdvp.registry import DatasetRegistry
from tests.conftest import SCHEMA, TODAY, rows_v1


def test_first_commit_is_1_0_0(v1):
    assert v1.version_id == "1.0.0"
    assert v1.parent_version_id is None
    assert v1.change_summary is None
    assert v1.row_count == 10
    assert v1.stage is Stage.EXPERIMENTAL
    assert len(v1.manifest_hash) == 64


def test_row_change_bumps_minor_with_change_summary(registry, v1):
    rows = [dict(r) for r in rows_v1()]
    rows[0]["age"] = 99
    rows.append({"customer_id": "C011", "age": 31, "monthly_spend": 61.0,
                 "region": "east", "churned": 0})
    v2 = registry.commit("churn", rows, SCHEMA, branch="main",
                         created_by="a.chen", now=TODAY)
    assert v2.version_id == "1.1.0"
    assert v2.change_summary.rows_added == 1
    assert v2.change_summary.rows_changed == 1
    assert v2.change_summary.changed_keys == ["C001"]
    assert v2.change_summary.breaking is False


def test_breaking_schema_change_bumps_major(registry, v1):
    schema = [f for f in SCHEMA if f.name != "region"]
    rows = [{k: v for k, v in r.items() if k != "region"} for r in rows_v1()]
    v2 = registry.commit("churn", rows, schema, branch="main",
                         created_by="a.chen", now=TODAY)
    assert v2.version_id == "2.0.0"
    assert v2.change_summary.columns_removed == ["region"]
    assert v2.change_summary.breaking is True


def test_additive_schema_change_bumps_minor(registry, v1):
    from mdvp.models import DType, SchemaField

    schema = SCHEMA + [SchemaField("tenure_months", DType.INTEGER)]
    rows = [dict(r, tenure_months=12) for r in rows_v1()]
    v2 = registry.commit("churn", rows, schema, branch="main",
                         created_by="a.chen", now=TODAY)
    assert v2.version_id == "1.1.0"
    assert v2.change_summary.columns_added == ["tenure_months"]
    assert v2.change_summary.breaking is False


def test_no_change_commit_rejected(registry, v1):
    with pytest.raises(ValueError, match="no changes"):
        registry.commit("churn", rows_v1(), SCHEMA, branch="main",
                        created_by="a.chen", now=TODAY)


def test_versions_are_immutable(registry, v1):
    rows = [dict(r) for r in rows_v1()]
    rows[0]["age"] = 99
    registry.commit("churn", rows, SCHEMA, branch="main",
                    created_by="a.chen", now=TODAY)
    stored_v1 = registry.get_version("churn", "1.0.0")
    assert stored_v1.rows[0]["age"] == 21
    assert stored_v1.manifest_hash == v1.manifest_hash


def test_duplicate_dataset_rejected(registry):
    from mdvp.models import Dataset

    with pytest.raises(ValueError, match="already exists"):
        registry.create_dataset(
            Dataset(dataset_id="churn", name="Dup", owner="a.chen"), now=TODAY
        )


def test_commit_to_unknown_branch_rejected(registry, v1):
    with pytest.raises(ValueError, match="unknown branch"):
        registry.commit("churn", rows_v1(), SCHEMA, branch="nope",
                        created_by="a.chen", now=TODAY)


def test_list_versions_sorted_by_semver(registry, v1):
    rows = [dict(r) for r in rows_v1()]
    rows[0]["age"] = 99
    registry.commit("churn", rows, SCHEMA, branch="main",
                    created_by="a.chen", now=TODAY)
    ids = [v.version_id for v in registry.list_versions("churn")]
    assert ids == ["1.0.0", "1.1.0"]


def test_schema_inferred_when_not_given(registry):
    v = registry.commit("churn", rows_v1(), None, branch="main",
                        created_by="a.chen", now=TODAY)
    assert [f.name for f in v.schema] == [
        "customer_id", "age", "monthly_spend", "region", "churned"
    ]


def test_production_version_none_until_promoted(registry, v1):
    assert registry.production_version("churn") is None
