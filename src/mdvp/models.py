"""Core domain models.

Design notes
------------
* Everything is a plain dataclass with no I/O, so the domain logic is
  testable without a database or a network.
* A dataset version is the unit of versioning: one immutable snapshot
  of rows plus its schema, identified by a content hash (manifest
  hash). Nothing about a published version is ever mutated in place;
  a change ships as a new version.
* Rows are plain ``dict[str, Any]`` values keyed by a dataset-level
  primary-key column (default ``"id"``). Types are deliberately kept
  JSON-shaped (str / int / float / bool / None) so snapshots serialize
  losslessly and hash deterministically.
* Timestamps are plain ISO strings; the service layer stamps them. The
  core never calls ``datetime.now`` except through an explicit ``now``
  argument, which keeps tests deterministic.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Stage(enum.Enum):
    """Promotion stage of a dataset version."""

    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    PRODUCTION = "production"
    QUARANTINED = "quarantined"  # fail-closed: failed a schema gate


class DType(enum.Enum):
    """Column types tracked by the schema (JSON-shaped)."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class SchemaField:
    """One column of a dataset schema."""

    name: str
    dtype: DType
    nullable: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype.value, "nullable": self.nullable}


@dataclass
class ChangeSummary:
    """What changed between a version and its parent."""

    rows_added: int = 0
    rows_removed: int = 0
    rows_changed: int = 0
    changed_keys: list[str] = field(default_factory=list)
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    columns_type_changed: list[str] = field(default_factory=list)
    possible_renames: list[dict] = field(default_factory=list)  # {from, to}
    breaking: bool = False  # removed column or type change

    def to_dict(self) -> dict:
        return {
            "rows_added": self.rows_added,
            "rows_removed": self.rows_removed,
            "rows_changed": self.rows_changed,
            "changed_keys": self.changed_keys,
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "columns_type_changed": self.columns_type_changed,
            "possible_renames": self.possible_renames,
            "breaking": self.breaking,
        }


@dataclass
class Dataset:
    """A registered dataset (the versioned entity)."""

    dataset_id: str
    name: str
    owner: str
    primary_key: str = "id"
    description: str = ""
    created_at: str = ""
    retention_keep_latest: int = 5  # versions kept per branch by retention
    row_count_floor: int = 1  # promotion gate: minimum rows

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.dataset_id.strip():
            errors.append("dataset_id is required")
        if not self.name.strip():
            errors.append("name is required")
        if not self.owner.strip():
            errors.append("owner is required")
        if not self.primary_key.strip():
            errors.append("primary_key is required")
        if self.retention_keep_latest < 1:
            errors.append("retention_keep_latest must be >= 1")
        if self.row_count_floor < 0:
            errors.append("row_count_floor must be >= 0")
        return errors


@dataclass
class DatasetVersion:
    """One immutable snapshot of a dataset."""

    version_id: str  # semantic version, e.g. "1.2.0"
    dataset_id: str
    parent_version_id: str | None
    branch: str
    schema: list[SchemaField]
    rows: list[dict]
    row_hashes: dict[str, str]  # primary-key value -> SHA-256 row hash
    manifest_hash: str  # SHA-256 over sorted row hashes + schema hash
    change_summary: ChangeSummary | None
    primary_key: str = "id"
    stage: Stage = Stage.EXPERIMENTAL
    created_at: str = ""
    created_by: str = ""
    notes: str = ""

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def columns(self) -> list[str]:
        return [f.name for f in self.schema]

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.version_id.strip():
            errors.append("version_id is required")
        if not self.branch.strip():
            errors.append("branch is required")
        for row in self.rows:
            if not isinstance(row, dict):
                errors.append("every row must be a dict")
                break
        return errors


@dataclass
class TransformStep:
    """One recorded, deterministic transform in a derivation chain.

    ``params`` is a JSON-serializable spec (see :mod:`mdvp.transforms`)
    so the chain can be hashed and replayed exactly.
    """

    op: str  # "filter" | "project" | "join" | "aggregate"
    params: dict
    input_version_ids: list[str]
    # output column -> list of (source version id, source column)
    column_lineage: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "params": self.params,
            "input_version_ids": self.input_version_ids,
            "column_lineage": {
                col: [[v, c] for v, c in srcs]
                for col, srcs in self.column_lineage.items()
            },
        }


@dataclass
class TrainingRun:
    """A training run pinned to one dataset version + transform chain."""

    run_id: str
    dataset_id: str
    dataset_version_id: str
    dataset_manifest_hash: str
    transform_chain: list[TransformStep]
    seed: int
    code_hash: str
    model_version: str
    created_by: str = ""
    created_at: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class ReproManifest:
    """Reproducibility manifest for one training run.

    ``manifest_hash`` pins seed + dataset version hash + transform
    chain hash + code hash, so ``verify`` can re-derive it and report
    match/mismatch.
    """

    run_id: str
    dataset_version_id: str
    dataset_manifest_hash: str
    transform_chain_hash: str
    code_hash: str
    seed: int
    manifest_hash: str
    created_at: str = ""


@dataclass
class PromotionApproval:
    """A signed promotion approval (one approver, one decision)."""

    approval_id: str
    dataset_id: str
    version_id: str
    target_stage: Stage
    requested_by: str
    approver: str
    decision: str  # "approve" | "reject"
    signature: str  # SHA-256 over approval + approver + decision
    decided_at: str = ""
    reason: str = ""


@dataclass
class PromotionRequest:
    """A promotion request: gates + signed approvals."""

    request_id: str
    dataset_id: str
    version_id: str
    target_stage: Stage
    requested_by: str
    state: str = "pending"  # pending | approved | rejected | applied
    approvals: list[PromotionApproval] = field(default_factory=list)
    gate_results: list[dict] = field(default_factory=list)
    created_at: str = ""


@dataclass
class GateResult:
    """Outcome of one promotion / schema gate."""

    gate_id: str
    passed: bool
    severity: str  # "blocking" | "warning"
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class DriftReport:
    """Distribution comparison between two dataset versions."""

    reference_version_id: str
    current_version_id: str
    per_column: list[dict] = field(default_factory=list)
    max_psi: float = 0.0
    max_ks: float = 0.0
    verdict: str = "stable"  # stable | watch | drifted

    def to_dict(self) -> dict:
        return {
            "reference_version_id": self.reference_version_id,
            "current_version_id": self.current_version_id,
            "per_column": self.per_column,
            "max_psi": self.max_psi,
            "max_ks": self.max_ks,
            "verdict": self.verdict,
        }


@dataclass
class SchemaDrift:
    """Schema comparison between two dataset versions."""

    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    columns_type_changed: list[str] = field(default_factory=list)
    possible_renames: list[dict] = field(default_factory=list)
    breaking: bool = False

    def to_dict(self) -> dict:
        return {
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "columns_type_changed": self.columns_type_changed,
            "possible_renames": self.possible_renames,
            "breaking": self.breaking,
        }


@dataclass
class AuditEntry:
    """One hash-chained audit record."""

    seq: int
    timestamp: str
    actor: str
    action: str
    dataset_id: str
    version_id: str
    detail: str
    prev_hash: str
    entry_hash: str
