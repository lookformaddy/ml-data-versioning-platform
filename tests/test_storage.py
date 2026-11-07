"""Storage tests: chunked persistence, integrity, size, retention."""
import pytest

from mdvp.storage import FileStore
from tests.conftest import SCHEMA, TODAY, rows_v1


@pytest.fixture()
def store(tmp_path):
    return FileStore(tmp_path / "store")


def test_save_and_load_round_trip(store, v1):
    store.save_version(v1)
    loaded = store.load_version("churn", "1.0.0")
    assert loaded.rows == v1.rows
    assert loaded.manifest_hash == v1.manifest_hash
    assert loaded.schema == v1.schema
    assert loaded.stage == v1.stage


def test_save_and_load_dataset_descriptor(store, registry):
    dataset = registry.get_dataset("churn")
    store.save_dataset(dataset)
    descriptor = store.load_dataset("churn")
    assert descriptor["dataset_id"] == "churn"
    assert descriptor["primary_key"] == "customer_id"
    assert descriptor["row_count_floor"] == 5
    assert store.load_dataset("ghost") is None


def test_chunked_storage_many_rows(store, registry):
    rows = [
        {"customer_id": f"C{i:05d}", "age": 20 + (i % 60),
         "monthly_spend": 50.0 + i, "region": "north", "churned": i % 2}
        for i in range(250)
    ]
    version = registry.commit("churn", rows, SCHEMA, branch="main",
                              created_by="a.chen", now=TODAY)
    manifest = store.save_version(version)
    assert len(manifest["chunks"]) == 3  # 100 + 100 + 50
    loaded = store.load_version("churn", version.version_id)
    assert loaded.row_count == 250
    assert loaded.rows == rows


def test_chunk_integrity_check(store, v1):
    store.save_version(v1)
    chunk_files = list((store.root / "churn" / "chunks").glob("*.jsonl"))
    assert len(chunk_files) == 1
    chunk_files[0].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check"):
        store.load_version("churn", "1.0.0")


def test_load_unknown_version_raises(store):
    with pytest.raises(KeyError):
        store.load_version("churn", "9.9.9")


def test_size_accounting(store, registry, v1):
    store.save_version(v1)
    size = store.dataset_size_bytes("churn")
    assert size > 0
    assert store.total_size_bytes() == size
    assert store.version_size_bytes("churn", "1.0.0") > 0
    report = store.dataset_report("churn")
    assert report["version_count"] == 1
    assert report["stored_versions"] == ["1.0.0"]


def test_retention_keeps_production_and_latest(store, registry, v1):
    store.save_version(v1)
    for i in range(4):
        rows = [dict(r) for r in rows_v1()]
        rows[0]["age"] = 30 + i
        version = registry.commit("churn", rows, SCHEMA, branch="main",
                                  created_by="a.chen", now=TODAY)
        store.save_version(version)
    assert store.list_stored_versions("churn") == [
        "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"
    ]
    result = store.apply_retention(
        "churn", keep_latest=2, keep_version_ids={"1.0.0"}
    )
    assert result["deleted"] == ["1.1.0", "1.2.0"]
    assert result["bytes_reclaimed"] > 0
    assert store.list_stored_versions("churn") == ["1.0.0", "1.3.0", "1.4.0"]


def test_retention_never_deletes_production(store, registry, v1):
    store.save_version(v1)
    rows = [dict(r) for r in rows_v1()]
    rows[0]["age"] = 99
    v2 = registry.commit("churn", rows, SCHEMA, branch="main",
                         created_by="a.chen", now=TODAY)
    store.save_version(v2)
    result = store.apply_retention(
        "churn", keep_latest=1, keep_version_ids={"1.1.0"}
    )
    assert result["deleted"] == ["1.0.0"]
    assert store.list_stored_versions("churn") == ["1.1.0"]
