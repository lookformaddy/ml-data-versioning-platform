"""Pytest configuration: put the src layout on the import path so plain
``pytest`` passes from the project root, with or without an install."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdvp.models import Dataset, DType, SchemaField  # noqa: E402
from mdvp.registry import DatasetRegistry  # noqa: E402

TODAY = "2026-11-07"

SCHEMA = [
    SchemaField("customer_id", DType.STRING, nullable=False),
    SchemaField("age", DType.INTEGER),
    SchemaField("monthly_spend", DType.NUMBER),
    SchemaField("region", DType.STRING),
    SchemaField("churned", DType.INTEGER),
]


def rows_v1() -> list[dict]:
    return [
        {"customer_id": f"C{i:03d}", "age": 20 + i, "monthly_spend": 50.0 + i,
         "region": ["north", "south", "east", "west"][i % 4], "churned": i % 2}
        for i in range(1, 11)
    ]


@pytest.fixture()
def registry() -> DatasetRegistry:
    reg = DatasetRegistry()
    reg.create_dataset(
        Dataset(dataset_id="churn", name="Churn Features", owner="a.chen",
                primary_key="customer_id", row_count_floor=5),
        now=TODAY,
    )
    return reg


@pytest.fixture()
def v1(registry: DatasetRegistry):
    return registry.commit(
        "churn", rows_v1(), SCHEMA, branch="main",
        created_by="a.chen", notes="initial", now=TODAY,
    )
