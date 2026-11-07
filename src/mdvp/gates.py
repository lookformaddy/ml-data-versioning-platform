"""Schema gates, promotion gates, and signed approvals.

Fail-closed throughout:
* A version that fails a *blocking* schema gate is quarantined and
  can neither be promoted nor used in a training run.
* Promotion (experimental -> validated -> production) requires every
  blocking gate to pass *and* a signed approval quorum with separation
  of duties (approver != requester). Production requires two distinct
  approvers; validated requires one.
* Signatures are SHA-256 over the request, approver, decision, and a
  shared secret, so a decision cannot be replayed onto another
  request. This is tamper-evidence, not PKI.
"""
from __future__ import annotations

import os

from mdvp.drift import compare_versions
from mdvp.models import (
    Dataset,
    DatasetVersion,
    GateResult,
    PromotionApproval,
    PromotionRequest,
    Stage,
)
from mdvp.registry import DatasetRegistry

SECRET = os.environ.get("MDVP_SIGNING_SECRET", "dev-only-signing-secret")

# Drift thresholds for promotion vs the current production version.
PSI_BLOCK = 0.25
KS_BLOCK = 0.20
PSI_WARN = 0.10
KS_WARN = 0.10

REQUIRED_APPROVERS = {Stage.VALIDATED: 1, Stage.PRODUCTION: 2}

ALLOWED_TRANSITIONS = {
    Stage.EXPERIMENTAL: {Stage.VALIDATED},
    Stage.VALIDATED: {Stage.PRODUCTION},
}


def sign_decision(
    request_id: str, approver: str, decision: str, secret: str = SECRET
) -> str:
    from mdvp.hashing import sha256_hex

    return sha256_hex(f"{request_id}|{approver}|{decision}|{secret}")


def run_schema_gates(
    version: DatasetVersion,
    *,
    expected_columns: list[str] | None = None,
    row_count_floor: int = 1,
) -> list[GateResult]:
    """Blocking schema gates for one version; empty pass = clean."""
    results: list[GateResult] = []
    if version.stage is Stage.QUARANTINED:
        results.append(
            GateResult(
                gate_id="not_quarantined",
                passed=False,
                severity="blocking",
                message="version is quarantined",
            )
        )
    if version.row_count < row_count_floor:
        results.append(
            GateResult(
                gate_id="row_count_floor",
                passed=False,
                severity="blocking",
                message=(
                    f"row count {version.row_count} below floor {row_count_floor}"
                ),
                detail={"row_count": version.row_count, "floor": row_count_floor},
            )
        )
    else:
        results.append(
            GateResult(
                gate_id="row_count_floor",
                passed=True,
                severity="blocking",
                message=f"row count {version.row_count} meets floor {row_count_floor}",
                detail={"row_count": version.row_count, "floor": row_count_floor},
            )
        )
    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in version.columns]
        results.append(
            GateResult(
                gate_id="expected_columns",
                passed=not missing,
                severity="blocking",
                message=(
                    f"missing expected columns: {missing}"
                    if missing
                    else "all expected columns present"
                ),
                detail={"missing": missing},
            )
        )
    null_violations = []
    for field in version.schema:
        if not field.nullable:
            bad = sum(1 for r in version.rows if r.get(field.name) is None)
            if bad:
                null_violations.append(f"{field.name} ({bad} nulls)")
    results.append(
        GateResult(
            gate_id="non_nullable_columns",
            passed=not null_violations,
            severity="blocking",
            message=(
                f"nulls in non-nullable columns: {null_violations}"
                if null_violations
                else "no nulls in non-nullable columns"
            ),
            detail={"violations": null_violations},
        )
    )
    return results


def evaluate_promotion(
    registry: DatasetRegistry,
    dataset: Dataset,
    version: DatasetVersion,
    target_stage: Stage,
) -> list[GateResult]:
    """All promotion gates for ``version`` -> ``target_stage``."""
    results: list[GateResult] = []
    if target_stage not in ALLOWED_TRANSITIONS.get(version.stage, set()):
        results.append(
            GateResult(
                gate_id="stage_transition",
                passed=False,
                severity="blocking",
                message=(
                    f"cannot promote from {version.stage.value} "
                    f"to {target_stage.value}"
                ),
            )
        )
    else:
        results.append(
            GateResult(
                gate_id="stage_transition",
                passed=True,
                severity="blocking",
                message=f"{version.stage.value} -> {target_stage.value} is allowed",
            )
        )
    # Schema gates always run, even on an invalid transition, so the
    # report is complete (e.g. a quarantined version is flagged even
    # though its transition is already blocked).
    results.extend(
        run_schema_gates(
            version, row_count_floor=dataset.row_count_floor
        )
    )
    production = registry.production_version(dataset.dataset_id)
    if target_stage is Stage.PRODUCTION and production is not None:
        if production.version_id == version.version_id:
            results.append(
                GateResult(
                    gate_id="drift_vs_production",
                    passed=False,
                    severity="blocking",
                    message="version is already the production version",
                )
            )
        else:
            report = compare_versions(production, version)
            drift_ok = report.max_psi < PSI_BLOCK and report.max_ks < KS_BLOCK
            results.append(
                GateResult(
                    gate_id="drift_vs_production",
                    passed=drift_ok,
                    severity="blocking" if not drift_ok else "warning",
                    message=(
                        f"drift vs production {production.version_id}: "
                        f"max PSI {report.max_psi}, max KS {report.max_ks} "
                        f"({report.verdict})"
                    ),
                    detail={
                        "max_psi": report.max_psi,
                        "max_ks": report.max_ks,
                        "verdict": report.verdict,
                    },
                )
            )
            if drift_ok and (
                report.max_psi >= PSI_WARN or report.max_ks >= KS_WARN
            ):
                results.append(
                    GateResult(
                        gate_id="drift_watch",
                        passed=True,
                        severity="warning",
                        message=(
                            f"drift is within limits but elevated "
                            f"(PSI {report.max_psi}, KS {report.max_ks})"
                        ),
                    )
                )
    return results


def request_promotion(
    registry: DatasetRegistry,
    dataset: Dataset,
    version: DatasetVersion,
    target_stage: Stage,
    requested_by: str,
    request_id: str,
    now: str = "",
) -> PromotionRequest:
    gates = evaluate_promotion(registry, dataset, version, target_stage)
    return PromotionRequest(
        request_id=request_id,
        dataset_id=dataset.dataset_id,
        version_id=version.version_id,
        target_stage=target_stage,
        requested_by=requested_by,
        gate_results=[g.to_dict() for g in gates],
        created_at=now,
    )


def record_approval(
    request: PromotionRequest,
    approver: str,
    decision: str,
    *,
    secret: str = SECRET,
    reason: str = "",
    now: str = "",
) -> PromotionApproval:
    """Record one signed decision. Enforces separation of duties."""
    if decision not in ("approve", "reject"):
        raise ValueError(f"decision must be 'approve' or 'reject', got {decision!r}")
    if approver == request.requested_by:
        raise ValueError(
            "separation of duties: the requester cannot approve their own promotion"
        )
    if any(a.approver == approver for a in request.approvals):
        raise ValueError(f"approver {approver!r} has already decided this request")
    if request.state in ("approved", "rejected", "applied"):
        raise ValueError(f"request {request.request_id} is already {request.state}")
    approval = PromotionApproval(
        approval_id=f"{request.request_id}:{approver}",
        dataset_id=request.dataset_id,
        version_id=request.version_id,
        target_stage=request.target_stage,
        requested_by=request.requested_by,
        approver=approver,
        decision=decision,
        signature=sign_decision(request.request_id, approver, decision, secret),
        decided_at=now,
        reason=reason,
    )
    request.approvals.append(approval)
    if decision == "reject":
        request.state = "rejected"
        return approval
    quorum = REQUIRED_APPROVERS.get(request.target_stage, 1)
    distinct_approvers = {
        a.approver for a in request.approvals if a.decision == "approve"
    }
    if len(distinct_approvers) >= quorum:
        request.state = "approved"
    return approval


def apply_promotion(
    registry: DatasetRegistry, request: PromotionRequest
) -> DatasetVersion:
    """Apply an approved promotion whose blocking gates all passed."""
    if request.state != "approved":
        raise ValueError(
            f"cannot apply promotion in state {request.state!r}; "
            "it must be approved first"
        )
    blocking_failures = [
        g for g in request.gate_results
        if not g["passed"] and g["severity"] == "blocking"
    ]
    if blocking_failures:
        raise ValueError(
            "cannot apply promotion: blocking gates failed: "
            + ", ".join(g["gate_id"] for g in blocking_failures)
        )
    version = registry.get_version(request.dataset_id, request.version_id)
    version.stage = request.target_stage
    request.state = "applied"
    return version


def quarantine(version: DatasetVersion) -> None:
    """Fail-closed quarantine: a version that failed a schema gate."""
    version.stage = Stage.QUARANTINED


def assert_trainable(version: DatasetVersion) -> None:
    """Fail-closed guard: only non-quarantined versions train models."""
    if version.stage is Stage.QUARANTINED:
        raise ValueError(
            f"version {version.version_id} is quarantined and cannot be "
            "used for training"
        )
