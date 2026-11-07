"""Transform tests: filter / project / join / aggregate + lineage."""
import pytest

from mdvp.registry import DatasetRegistry
from mdvp.transforms import aggregate_rows, filter_rows, join_rows, project_rows
from tests.conftest import SCHEMA, TODAY, rows_v1


@pytest.fixture()
def other_registry(registry):
    from mdvp.models import Dataset, DType, SchemaField

    registry.create_dataset(
        Dataset(dataset_id="regions", name="Regions", owner="a.chen",
                primary_key="region"), now=TODAY,
    )
    registry.commit(
        "regions",
        [
            {"region": "north", "manager": "m1", "budget": 1000},
            {"region": "south", "manager": "m2", "budget": 2000},
            {"region": "east", "manager": "m3", "budget": 3000},
            {"region": "west", "manager": "m4", "budget": 4000},
        ],
        [SchemaField("region", DType.STRING), SchemaField("manager", DType.STRING),
         SchemaField("budget", DType.INTEGER)],
        branch="main", created_by="a.chen", now=TODAY,
    )
    return registry


def test_filter_keeps_matching_rows_and_lineage(v1):
    rows, step = filter_rows(v1, "churned", "==", 1)
    assert len(rows) == 5
    assert all(r["churned"] == 1 for r in rows)
    assert step.op == "filter"
    assert step.column_lineage["age"] == [("1.0.0", "age")]
    assert step.input_version_ids == ["1.0.0"]


def test_filter_operators(v1):
    rows, _ = filter_rows(v1, "age", ">", 25)
    assert len(rows) == 5
    rows, _ = filter_rows(v1, "region", "in", ["north", "south"])
    assert len(rows) == 5
    rows, _ = filter_rows(v1, "region", "not in", ["north", "south"])
    assert len(rows) == 5


def test_filter_unknown_op_rejected(v1):
    with pytest.raises(ValueError, match="unknown filter op"):
        filter_rows(v1, "age", "~=", 25)


def test_filter_unknown_column_rejected(v1):
    with pytest.raises(ValueError, match="not present"):
        filter_rows(v1, "nope", "==", 1)


def test_project_subset_and_order(v1):
    rows, step = project_rows(v1, ["region", "customer_id"])
    assert rows[0] == {"region": "south", "customer_id": "C001"}
    assert set(step.column_lineage) == {"region", "customer_id"}


def test_project_unknown_column_rejected(v1):
    with pytest.raises(ValueError, match="not in schema"):
        project_rows(v1, ["nope"])


def test_join_inner(other_registry, v1):
    left = other_registry.get_version("churn", "1.0.0")
    right = other_registry.get_version("regions", "1.0.0")
    rows, step = join_rows(left, right, on="region", how="inner")
    assert len(rows) == 10
    first = next(r for r in rows if r["customer_id"] == "C001")  # region south
    assert first["manager"] == "m2"
    assert first["budget"] == 2000
    assert step.column_lineage["budget"] == [("1.0.0", "budget")]
    assert step.column_lineage["age"] == [("1.0.0", "age")]


def test_join_left_keeps_unmatched(other_registry, v1):
    left = other_registry.get_version("churn", "1.0.0")
    right = other_registry.get_version("regions", "1.0.0")
    # Drop the west region from the right side: left join keeps those rows.
    from mdvp.models import Dataset, DType, SchemaField

    other_registry.create_dataset(
        Dataset(dataset_id="regions2", name="R2", owner="a.chen",
                primary_key="region"), now=TODAY,
    )
    other_registry.commit(
        "regions2",
        [{"region": "north", "manager": "m1"}],
        [SchemaField("region", DType.STRING), SchemaField("manager", DType.STRING)],
        branch="main", created_by="a.chen", now=TODAY,
    )
    right2 = other_registry.get_version("regions2", "1.0.0")
    rows, _ = join_rows(left, right2, on="region", how="left")
    assert len(rows) == 10
    west = [r for r in rows if r["region"] == "west"]
    assert west and west[0]["manager"] is None


def test_join_unknown_key_rejected(other_registry, v1):
    left = other_registry.get_version("churn", "1.0.0")
    right = other_registry.get_version("regions", "1.0.0")
    with pytest.raises(ValueError, match="must exist in both"):
        join_rows(left, right, on="nope")


def test_aggregate_group_by(v1):
    rows, step = aggregate_rows(
        v1, group_by=["region"],
        aggregations={"monthly_spend": "mean", "customer_id": "count"},
    )
    assert len(rows) == 4
    north = next(r for r in rows if r["region"] == "north")
    assert north["customer_id_count"] == 2
    assert north["monthly_spend_mean"] == pytest.approx((54.0 + 58.0) / 2)
    assert step.column_lineage["monthly_spend_mean"] == [("1.0.0", "monthly_spend")]
    assert step.column_lineage["region"] == [("1.0.0", "region")]


def test_aggregate_unknown_agg_rejected(v1):
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate_rows(v1, group_by=["region"], aggregations={"age": "median"})
