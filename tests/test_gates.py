"""Gate tests: schema gates, promotion flow, SoD, quorum, quarantine."""
import pytest

from mdvp import gates as gates_mod
from mdvp.gates import (
    apply_promotion,
    assert_trainable,
    evaluate_promotion,
    quarantine,
    record_approval,
    request_promotion,
    run_schema_gates,
)
from mdvp.models import Stage
from tests.conftest import SCHEMA, TODAY, rows_v1

SECRET = "test-secret"


def _promote(registry, v1, target, approvers, requester="a.chen", apply=False):
    dataset = registry.get_dataset("churn")
    request = request_promotion(
        registry, dataset, v1, target, requester,
        f"req-{target.value}-{id(v1)}", now=TODAY,
    )
    for approver in approvers:
        record_approval(request, approver, "approve", secret=SECRET, now=TODAY)
    if apply:
        apply_promotion(registry, request)
    return request


def test_schema_gates_pass_on_clean_version(v1):
    results = run_schema_gates(v1, expected_columns=["customer_id", "age"],
                               row_count_floor=5)
    assert all(g.passed for g in results)
    assert {g.gate_id for g in results} == {
        "row_count_floor", "expected_columns", "non_nullable_columns"
    }


def test_schema_gates_fail_on_row_count_floor(registry, v1):
    rows = rows_v1()[:2]
    small = registry.commit("churn", rows, SCHEMA, branch="main",
                            created_by="a.chen", now=TODAY)
    results = run_schema_gates(small, row_count_floor=5)
    floor = next(g for g in results if g.gate_id == "row_count_floor")
    assert floor.passed is False
    assert floor.severity == "blocking"


def test_schema_gates_fail_on_missing_column(registry, v1):
    schema = [f for f in SCHEMA if f.name != "region"]
    rows = [{k: v for k, v in r.items() if k != "region"} for r in rows_v1()]
    v2 = registry.commit("churn", rows, schema, branch="main",
                         created_by="a.chen", now=TODAY)
    results = run_schema_gates(v2, expected_columns=["region"])
    missing = next(g for g in results if g.gate_id == "expected_columns")
    assert missing.passed is False


def test_schema_gates_fail_on_null_in_non_nullable(registry, v1):
    rows = [dict(r) for r in rows_v1()]
    rows[0]["customer_id"] = None
    # Hashing needs a string key; use a sentinel row instead: rebuild
    # with a null customer_id is impossible (PK missing), so test the
    # gate directly on a hand-built version with a null non-nullable col.
    from mdvp.models import DatasetVersion

    version = DatasetVersion(
        version_id="9.9.9", dataset_id="churn", parent_version_id=None,
        branch="main", schema=SCHEMA, rows=rows,
        row_hashes={}, manifest_hash="x", change_summary=None,
    )
    version.rows[0] = dict(rows[0], age=None)
    # age is nullable in the fixture schema; make a non-nullable copy.
    from mdvp.models import DType, SchemaField

    version.schema = [
        SchemaField(f.name, f.dtype, nullable=(f.name != "age")) for f in SCHEMA
    ]
    results = run_schema_gates(version, row_count_floor=1)
    nulls = next(g for g in results if g.gate_id == "non_nullable_columns")
    assert nulls.passed is False
    assert "age" in nulls.message


def test_promotion_to_validated_needs_one_approver(registry, v1):
    request = _promote(registry, v1, Stage.VALIDATED, ["d.kim"])
    assert request.state == "approved"
    applied = apply_promotion(registry, request)
    assert applied.stage is Stage.VALIDATED
    assert request.state == "applied"


def test_promotion_to_production_needs_two_approvers(registry, v1):
    dataset = registry.get_dataset("churn")
    request = request_promotion(
        registry, dataset, v1, Stage.PRODUCTION, "a.chen",
        "req-prod", now=TODAY,
    )
    # Production from experimental is not an allowed transition.
    assert any(
        not g["passed"] and g["gate_id"] == "stage_transition"
        for g in request.gate_results
    )
    _promote(registry, v1, Stage.VALIDATED, ["d.kim"], apply=True)
    request2 = _promote(registry, v1, Stage.PRODUCTION, ["d.kim"])
    assert request2.state == "pending"  # quorum is 2
    record_approval(request2, "e.santos", "approve", secret=SECRET, now=TODAY)
    assert request2.state == "approved"
    assert apply_promotion(registry, request2).stage is Stage.PRODUCTION


def test_separation_of_duties_enforced(registry, v1):
    dataset = registry.get_dataset("churn")
    request = request_promotion(
        registry, dataset, v1, Stage.VALIDATED, "a.chen", "req-sod", now=TODAY
    )
    with pytest.raises(ValueError, match="separation of duties"):
        record_approval(request, "a.chen", "approve", secret=SECRET, now=TODAY)


def test_duplicate_approver_rejected(registry, v1):
    request = _promote(registry, v1, Stage.VALIDATED, ["d.kim"])
    with pytest.raises(ValueError, match="already decided"):
        record_approval(request, "d.kim", "approve", secret=SECRET, now=TODAY)


def test_rejection_ends_request(registry, v1):
    dataset = registry.get_dataset("churn")
    request = request_promotion(
        registry, dataset, v1, Stage.VALIDATED, "a.chen", "req-rej", now=TODAY
    )
    record_approval(request, "d.kim", "reject", secret=SECRET, now=TODAY)
    assert request.state == "rejected"
    with pytest.raises(ValueError, match="already rejected"):
        record_approval(request, "e.santos", "approve", secret=SECRET, now=TODAY)
    with pytest.raises(ValueError, match="must be approved"):
        apply_promotion(registry, request)


def test_apply_blocked_when_gates_fail(registry, v1):
    rows = rows_v1()[:2]
    small = registry.commit("churn", rows, SCHEMA, branch="main",
                            created_by="a.chen", now=TODAY)
    dataset = registry.get_dataset("churn")
    request = request_promotion(
        registry, dataset, small, Stage.VALIDATED, "a.chen",
        "req-small", now=TODAY,
    )
    record_approval(request, "d.kim", "approve", secret=SECRET, now=TODAY)
    assert request.state == "approved"
    with pytest.raises(ValueError, match="blocking gates failed"):
        apply_promotion(registry, request)


def test_drift_gate_blocks_production_on_large_drift(registry, v1):
    _promote(registry, v1, Stage.VALIDATED, ["d.kim"], apply=True)
    request = _promote(registry, v1, Stage.PRODUCTION, ["d.kim", "e.santos"])
    apply_promotion(registry, request)
    assert registry.production_version("churn").version_id == "1.0.0"
    rows = [dict(r) for r in rows_v1()]
    for row in rows:
        row["monthly_spend"] = row["monthly_spend"] + 500.0
    v2 = registry.commit("churn", rows, SCHEMA, branch="main",
                         created_by="a.chen", now=TODAY)
    _promote(registry, v2, Stage.VALIDATED, ["d.kim"], apply=True)
    dataset = registry.get_dataset("churn")
    prod_request = request_promotion(
        registry, dataset, v2, Stage.PRODUCTION, "a.chen",
        "req-drift", now=TODAY,
    )
    drift_gate = next(
        g for g in prod_request.gate_results if g["gate_id"] == "drift_vs_production"
    )
    assert drift_gate["passed"] is False


def test_quarantine_blocks_training_and_promotion(registry, v1):
    quarantine(v1)
    assert v1.stage is Stage.QUARANTINED
    with pytest.raises(ValueError, match="quarantined"):
        assert_trainable(v1)
    dataset = registry.get_dataset("churn")
    results = evaluate_promotion(registry, dataset, v1, Stage.VALIDATED)
    assert any(
        not g.passed and g.gate_id == "not_quarantined" for g in results
    )


def test_approval_signature_is_deterministic():
    sig1 = gates_mod.sign_decision("req-1", "d.kim", "approve", SECRET)
    sig2 = gates_mod.sign_decision("req-1", "d.kim", "approve", SECRET)
    assert sig1 == sig2
    assert sig1 != gates_mod.sign_decision("req-2", "d.kim", "approve", SECRET)
    assert sig1 != gates_mod.sign_decision("req-1", "e.santos", "approve", SECRET)
