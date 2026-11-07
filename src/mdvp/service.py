"""FastAPI surface for the ML Data Versioning Platform.

The service is a thin HTTP wrapper over the pure-stdlib core: it owns
the in-memory registry, the lineage graph, the audit log, and the
promotion requests, stamps timestamps, and translates domain
ValueErrors into 400 responses. All decision logic lives in the core
modules, so the API adds no second source of truth.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mdvp import gates as gates_mod
from mdvp import repro as repro_mod
from mdvp.audit import AuditLog
from mdvp.drift import compare_versions
from mdvp.gates import SECRET
from mdvp.lineage import LineageGraph
from mdvp.models import (
    Dataset,
    DatasetVersion,
    DType,
    PromotionRequest,
    SchemaField,
    Stage,
    TrainingRun,
)
from mdvp.registry import DatasetRegistry, MergeConflictError
from mdvp.transforms import aggregate_rows, filter_rows, join_rows, project_rows


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


app = FastAPI(title="ML Data Versioning Platform", version="1.0.0")

# In-memory stores (a real deployment backs these with a database).
registry = DatasetRegistry()
lineage = LineageGraph()
audit = AuditLog()
promotions: dict[str, PromotionRequest] = {}
runs: dict[str, TrainingRun] = {}
manifests: dict[str, dict] = {}


def _version_out(version: DatasetVersion) -> dict:
    return {
        "version_id": version.version_id,
        "dataset_id": version.dataset_id,
        "parent_version_id": version.parent_version_id,
        "branch": version.branch,
        "schema": [f.to_dict() for f in version.schema],
        "manifest_hash": version.manifest_hash,
        "row_count": version.row_count,
        "stage": version.stage.value,
        "created_at": version.created_at,
        "created_by": version.created_by,
        "notes": version.notes,
        "change_summary": (
            version.change_summary.to_dict() if version.change_summary else None
        ),
    }


class CreateDatasetRequest(BaseModel):
    dataset_id: str
    name: str
    owner: str
    primary_key: str = "id"
    description: str = ""
    row_count_floor: int = 1
    retention_keep_latest: int = 5


class CommitRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rows: list[dict]
    # Aliased: "schema" shadows a BaseModel attribute name.
    schema_: list[dict] | None = Field(default=None, alias="schema")
    branch: str = "main"
    created_by: str = ""
    notes: str = ""


class BranchRequest(BaseModel):
    name: str
    from_version_id: str


class MergeRequest(BaseModel):
    source_branch: str
    target_branch: str
    created_by: str = ""


class DriftRequest(BaseModel):
    reference_version_id: str
    current_version_id: str


class PromoteRequest(BaseModel):
    request_id: str
    target_stage: Stage
    requested_by: str


class DecideRequest(BaseModel):
    approver: str
    decision: str
    reason: str = ""


class TrainingRunRequest(BaseModel):
    run_id: str
    dataset_version_id: str
    seed: int = 42
    code_hash: str
    model_version: str
    created_by: str = ""
    transform_chain: list[dict] = []
    metrics: dict = {}


class VerifyRequest(BaseModel):
    dataset_version_id: str
    transform_chain: list[dict] = []
    code_hash: str
    seed: int


def _get_version_or_404(dataset_id: str, version_id: str) -> DatasetVersion:
    try:
        return registry.get_version(dataset_id, version_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "datasets": len(registry.datasets)}


@app.post("/datasets", status_code=201)
def create_dataset(payload: CreateDatasetRequest) -> dict:
    dataset = Dataset(
        dataset_id=payload.dataset_id,
        name=payload.name,
        owner=payload.owner,
        primary_key=payload.primary_key,
        description=payload.description,
        row_count_floor=payload.row_count_floor,
        retention_keep_latest=payload.retention_keep_latest,
    )
    try:
        registry.create_dataset(dataset, now=_now())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit.append(_now(), payload.owner, "dataset_created", payload.dataset_id, "", "")
    return {"dataset_id": dataset.dataset_id, "name": dataset.name}


@app.get("/datasets/{dataset_id}/versions")
def list_versions(dataset_id: str, branch: str | None = None) -> dict:
    try:
        versions = registry.list_versions(dataset_id, branch)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"dataset_id": dataset_id, "versions": [_version_out(v) for v in versions]}


@app.post("/datasets/{dataset_id}/versions", status_code=201)
def commit_version(dataset_id: str, payload: CommitRequest) -> dict:
    schema = None
    if payload.schema_ is not None:
        try:
            schema = [
                SchemaField(
                    name=f["name"], dtype=DType(f["dtype"]), nullable=f.get("nullable", True)
                )
                for f in payload.schema_
            ]
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid schema: {exc}") from exc
    try:
        version = registry.commit(
            dataset_id,
            payload.rows,
            schema,
            branch=payload.branch,
            created_by=payload.created_by,
            notes=payload.notes,
            now=_now(),
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit.append(_now(), payload.created_by, "version_committed", dataset_id,
                 version.version_id, f"manifest={version.manifest_hash[:12]}")
    return _version_out(version)


@app.get("/datasets/{dataset_id}/versions/{version_id}")
def get_version(dataset_id: str, version_id: str) -> dict:
    return _version_out(_get_version_or_404(dataset_id, version_id))


@app.post("/datasets/{dataset_id}/branches", status_code=201)
def create_branch(dataset_id: str, payload: BranchRequest) -> dict:
    try:
        registry.create_branch(dataset_id, payload.name, payload.from_version_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"dataset_id": dataset_id, "branch": payload.name,
            "from_version_id": payload.from_version_id}


@app.post("/datasets/{dataset_id}/merge")
def merge_branches(dataset_id: str, payload: MergeRequest) -> dict:
    try:
        version = registry.merge(
            dataset_id, payload.source_branch, payload.target_branch,
            created_by=payload.created_by, now=_now(),
        )
    except MergeConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit.append(_now(), payload.created_by, "branch_merged", dataset_id,
                 version.version_id, f"{payload.source_branch} -> {payload.target_branch}")
    return _version_out(version)


@app.post("/datasets/{dataset_id}/drift")
def drift(dataset_id: str, payload: DriftRequest) -> dict:
    reference = _get_version_or_404(dataset_id, payload.reference_version_id)
    current = _get_version_or_404(dataset_id, payload.current_version_id)
    return compare_versions(reference, current).to_dict()


@app.post("/datasets/{dataset_id}/versions/{version_id}/promotions", status_code=201)
def request_promotion(dataset_id: str, version_id: str, payload: PromoteRequest) -> dict:
    try:
        dataset = registry.get_dataset(dataset_id)
        version = registry.get_version(dataset_id, version_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request = gates_mod.request_promotion(
        registry, dataset, version, payload.target_stage,
        payload.requested_by, payload.request_id, now=_now(),
    )
    promotions[request.request_id] = request
    return {
        "request_id": request.request_id,
        "state": request.state,
        "gate_results": request.gate_results,
    }


@app.post("/promotions/{request_id}/decide")
def decide_promotion(request_id: str, payload: DecideRequest) -> dict:
    request = promotions.get(request_id)
    if request is None:
        raise HTTPException(status_code=404, detail=f"unknown promotion {request_id!r}")
    try:
        approval = gates_mod.record_approval(
            request, payload.approver, payload.decision,
            secret=SECRET, reason=payload.reason, now=_now(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "request_id": request.request_id,
        "state": request.state,
        "approver": approval.approver,
        "decision": approval.decision,
    }


@app.post("/promotions/{request_id}/apply")
def apply_promotion(request_id: str) -> dict:
    request = promotions.get(request_id)
    if request is None:
        raise HTTPException(status_code=404, detail=f"unknown promotion {request_id!r}")
    try:
        version = gates_mod.apply_promotion(registry, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit.append(_now(), request.requested_by, "promotion_applied",
                 request.dataset_id, version.version_id, version.stage.value)
    return {"request_id": request_id, "state": request.state,
            "version_id": version.version_id, "stage": version.stage.value}


@app.post("/datasets/{dataset_id}/training-runs", status_code=201)
def create_training_run(dataset_id: str, payload: TrainingRunRequest) -> dict:
    version = _get_version_or_404(dataset_id, payload.dataset_version_id)
    try:
        gates_mod.assert_trainable(version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    from mdvp.models import TransformStep

    chain = [
        TransformStep(
            op=step.get("op", ""),
            params=step.get("params", {}),
            input_version_ids=step.get("input_version_ids", []),
            column_lineage={
                col: [tuple(pair) for pair in srcs]
                for col, srcs in step.get("column_lineage", {}).items()
            },
        )
        for step in payload.transform_chain
    ]
    run = TrainingRun(
        run_id=payload.run_id,
        dataset_id=dataset_id,
        dataset_version_id=payload.dataset_version_id,
        dataset_manifest_hash=version.manifest_hash,
        transform_chain=chain,
        seed=payload.seed,
        code_hash=payload.code_hash,
        model_version=payload.model_version,
        created_by=payload.created_by,
        created_at=_now(),
        metrics=payload.metrics,
    )
    runs[run.run_id] = run
    manifest = repro_mod.build_manifest(run, now=_now())
    manifests[run.run_id] = {
        "manifest": manifest,
        "transform_chain": [s.to_dict() for s in chain],
    }
    lineage.record_training_run(run)
    audit.append(_now(), payload.created_by, "training_run_recorded",
                 dataset_id, version.version_id, run.run_id)
    return {
        "run_id": run.run_id,
        "model_version": run.model_version,
        "dataset_version_id": run.dataset_version_id,
        "manifest_hash": manifest.manifest_hash,
    }


@app.post("/training-runs/{run_id}/verify")
def verify_training_run(run_id: str, payload: VerifyRequest) -> dict:
    record = manifests.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    version = _get_version_or_404(
        runs[run_id].dataset_id, payload.dataset_version_id
    )
    ok, message = repro_mod.verify_manifest(
        record["manifest"], version, payload.transform_chain,
        payload.code_hash, payload.seed,
    )
    return {"run_id": run_id, "verified": ok, "message": message}


@app.get("/lineage/{version_id}/impact")
def version_impact(version_id: str) -> dict:
    return {"version_id": version_id, "downstream": lineage.impact(version_id)}


@app.get("/audit/verify")
def verify_audit() -> dict:
    ok, message = audit.verify()
    return {"valid": ok, "message": message, "entries": len(audit.entries)}
