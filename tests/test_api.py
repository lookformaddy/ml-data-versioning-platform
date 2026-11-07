"""API tests. Auto-skips when FastAPI is not installed (runs in CI).

The service keeps global in-memory state, so every test uses its own
dataset id (uuid-suffixed) to stay independent.
"""
import uuid

import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
from fastapi.testclient import TestClient  # noqa: E402

from mdvp.service import app  # noqa: E402
from tests.conftest import rows_v1  # noqa: E402


@pytest.fixture()
def ds_id():
    return f"api-churn-{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def client():
    return TestClient(app)


def _create_dataset(client, ds_id):
    resp = client.post("/datasets", json={
        "dataset_id": ds_id,
        "name": "Churn Features",
        "owner": "a.chen",
        "primary_key": "customer_id",
        "row_count_floor": 5,
    })
    assert resp.status_code == 201, resp.text


def _commit(client, ds_id, rows, branch="main"):
    resp = client.post(
        f"/datasets/{ds_id}/versions",
        json={"rows": rows, "branch": branch, "created_by": "a.chen"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_create_dataset_and_commit_version(client, ds_id):
    _create_dataset(client, ds_id)
    version = _commit(client, ds_id, rows_v1())
    assert version["version_id"] == "1.0.0"
    assert version["row_count"] == 10
    assert len(version["manifest_hash"]) == 64


def test_commit_duplicate_dataset_conflict(client, ds_id):
    _create_dataset(client, ds_id)
    resp = client.post("/datasets", json={
        "dataset_id": ds_id, "name": "Dup", "owner": "a.chen",
    })
    assert resp.status_code == 400


def test_commit_no_change_rejected(client, ds_id):
    _create_dataset(client, ds_id)
    _commit(client, ds_id, rows_v1())
    resp = client.post(
        f"/datasets/{ds_id}/versions",
        json={"rows": rows_v1(), "branch": "main", "created_by": "a.chen"},
    )
    assert resp.status_code == 400


def test_drift_endpoint(client, ds_id):
    _create_dataset(client, ds_id)
    _commit(client, ds_id, rows_v1())
    shifted = [dict(r, monthly_spend=r["monthly_spend"] + 500.0) for r in rows_v1()]
    _commit(client, ds_id, shifted)
    resp = client.post(
        f"/datasets/{ds_id}/drift",
        json={"reference_version_id": "1.0.0", "current_version_id": "1.1.0"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["verdict"] == "drifted"


def test_branch_merge_and_promotion_flow(client, ds_id):
    _create_dataset(client, ds_id)
    _commit(client, ds_id, rows_v1())
    assert client.post(
        f"/datasets/{ds_id}/branches",
        json={"name": "experiment", "from_version_id": "1.0.0"},
    ).status_code == 201
    exp_rows = [dict(r) for r in rows_v1()]
    exp_rows[0]["age"] = 50
    _commit(client, ds_id, exp_rows, branch="experiment")
    merged = client.post(
        f"/datasets/{ds_id}/merge",
        json={"source_branch": "experiment", "target_branch": "main",
              "created_by": "a.chen"},
    )
    assert merged.status_code == 200, merged.text
    merged_id = merged.json()["version_id"]
    # Promote merged version to validated (quorum 1).
    promo = client.post(
        f"/datasets/{ds_id}/versions/{merged_id}/promotions",
        json={"request_id": f"promo-{ds_id}", "target_stage": "validated",
              "requested_by": "a.chen"},
    )
    assert promo.status_code == 201, promo.text
    assert all(g["passed"] for g in promo.json()["gate_results"])
    decide = client.post(
        f"/promotions/promo-{ds_id}/decide",
        json={"approver": "d.kim", "decision": "approve"},
    )
    assert decide.json()["state"] == "approved"
    applied = client.post(f"/promotions/promo-{ds_id}/apply")
    assert applied.status_code == 200, applied.text
    assert applied.json()["stage"] == "validated"


def test_promotion_sod_enforced_via_api(client, ds_id):
    _create_dataset(client, ds_id)
    _commit(client, ds_id, rows_v1())
    client.post(
        f"/datasets/{ds_id}/versions/1.0.0/promotions",
        json={"request_id": f"promo-sod-{ds_id}", "target_stage": "validated",
              "requested_by": "a.chen"},
    )
    resp = client.post(
        f"/promotions/promo-sod-{ds_id}/decide",
        json={"approver": "a.chen", "decision": "approve"},
    )
    assert resp.status_code == 400


def test_training_run_and_manifest_verify(client, ds_id):
    _create_dataset(client, ds_id)
    _commit(client, ds_id, rows_v1())
    run = client.post(
        f"/datasets/{ds_id}/training-runs",
        json={"run_id": f"run-{ds_id}", "dataset_version_id": "1.0.0",
              "seed": 42, "code_hash": "deadbeef",
              "model_version": "churn-model@1.0.0", "created_by": "b.okafor"},
    )
    assert run.status_code == 201, run.text
    assert len(run.json()["manifest_hash"]) == 64
    verify = client.post(
        f"/training-runs/run-{ds_id}/verify",
        json={"dataset_version_id": "1.0.0", "transform_chain": [],
              "code_hash": "deadbeef", "seed": 42},
    )
    assert verify.json()["verified"] is True
    tampered = client.post(
        f"/training-runs/run-{ds_id}/verify",
        json={"dataset_version_id": "1.0.0", "transform_chain": [],
              "code_hash": "deadbeef", "seed": 7},
    )
    assert tampered.json()["verified"] is False


def test_audit_verify_endpoint(client):
    resp = client.get("/audit/verify")
    assert resp.json()["valid"] is True
