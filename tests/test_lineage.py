"""Lineage tests: transform recording, column origins, impact, training runs."""
import pytest

from mdvp.lineage import LineageGraph
from mdvp.models import TrainingRun
from mdvp.transforms import aggregate_rows, filter_rows, project_rows
from tests.conftest import TODAY, rows_v1


def test_record_transform_and_column_origin(v1):
    graph = LineageGraph()
    _rows, step = filter_rows(v1, "churned", "==", 1)
    graph.record_transform("1.0.0-churned", step)
    assert graph.column_sources("1.0.0-churned", "age") == {("1.0.0", "age")}
    assert graph.column_origin("1.0.0-churned", "age") == {("1.0.0", "age")}


def test_column_origin_walks_through_chained_transforms(registry, v1):
    graph = LineageGraph()
    _rows, step1 = filter_rows(v1, "churned", "==", 1)
    graph.record_transform("1.0.0-churned", step1)
    # Project the filtered snapshot: column lineage points at the
    # filtered version, which itself points at 1.0.0.
    from mdvp.models import DatasetVersion

    filtered_version = DatasetVersion(
        version_id="1.0.0-churned", dataset_id="churn", parent_version_id="1.0.0",
        branch="main", schema=v1.schema,
        rows=[r for r in v1.rows if r["churned"] == 1],
        row_hashes={}, manifest_hash="x", change_summary=None,
    )
    _rows, step2 = project_rows(filtered_version, ["age", "region"])
    graph.record_transform("1.0.0-churned-proj", step2)
    assert graph.column_origin("1.0.0-churned-proj", "age") == {("1.0.0", "age")}


def test_aggregate_column_origin(v1):
    graph = LineageGraph()
    _rows, step = aggregate_rows(
        v1, group_by=["region"], aggregations={"monthly_spend": "mean"}
    )
    graph.record_transform("1.0.0-by-region", step)
    assert graph.column_origin("1.0.0-by-region", "monthly_spend_mean") == {
        ("1.0.0", "monthly_spend")
    }


def test_training_run_links_dataset_to_model(v1):
    graph = LineageGraph()
    run = TrainingRun(
        run_id="run-1", dataset_id="churn", dataset_version_id="1.0.0",
        dataset_manifest_hash=v1.manifest_hash, transform_chain=[],
        seed=42, code_hash="abc", model_version="churn-model@1.0.0",
    )
    graph.record_training_run(run)
    assert "run-1" in graph.impact("1.0.0")
    assert "churn-model@1.0.0" in graph.impact("1.0.0")
    assert "1.0.0" in graph.upstream("churn-model@1.0.0")
    assert "run-1" in graph.upstream("churn-model@1.0.0")


def test_impact_is_transitive(registry, v1):
    graph = LineageGraph()
    _rows, step = filter_rows(v1, "churned", "==", 1)
    graph.record_transform("1.0.0-churned", step)
    run = TrainingRun(
        run_id="run-2", dataset_id="churn", dataset_version_id="1.0.0-churned",
        dataset_manifest_hash="x", transform_chain=[step],
        seed=42, code_hash="abc", model_version="churn-model@2.0.0",
    )
    graph.record_training_run(run)
    downstream = graph.impact("1.0.0")
    assert "1.0.0-churned" in downstream
    assert "run-2" in downstream
    assert "churn-model@2.0.0" in downstream


def test_impact_of_unknown_node_is_empty():
    assert LineageGraph().impact("nope") == []
