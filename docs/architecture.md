# Architecture — ML Data Versioning Platform

## Overview

```
snapshot (rows + schema)
        │
        ▼
┌──────────────┐   SHA-256 per row, manifest hash over sorted row hashes + schema
│   hashing    │   (canonical JSON: sorted keys, compact separators)
└──────┬───────┘
       ▼
┌──────────────┐   immutable DatasetVersion, automatic semver,
│   registry   │   branches as pointers, merge-lite via common ancestor
└──────┬───────┘
       │                ┌────────────┐
       ├───────────────▶│ transforms │  filter / project / join / aggregate,
       │                │            │  JSON specs, column-level lineage
       │                └─────┬──────┘
       │                      ▼
       │                ┌────────────┐
       │                │  lineage   │  version → run → model graph,
       │                │            │  column origins, BFS impact
       │                └────────────┘
       │
       ▼
┌──────────────┐   PSI (quantile bins / proportions), KS statistic,
│    drift     │   schema drift + rename heuristic
└──────┬───────┘
       ▼
┌──────────────┐   schema gates → quarantine (fail-closed)
│    gates     │   promotion gates → signed approvals (SoD, quorum)
└──────┬───────┘
       ▼
┌──────────────┐   seed + dataset hash + transform-chain hash + code hash
│    repro     │   verify re-derives and reports match/mismatch
└──────┬───────┘
       ▼
┌──────────────┐   chunked content-addressed blobs + manifest,
│   storage    │   size accounting, retention (keep production + heads)
└──────┬───────┘
       ▼
┌──────────────┐   FastAPI service + CLI (thin wrappers over the core)
│ service/cli  │   audit log (SHA-256 hash-chained, append-only)
└──────────────┘
```

## Modules (`src/mdvp/`)

| Module | Responsibility |
| ------ | -------------- |
| `models.py` | Plain dataclasses: `Dataset`, `DatasetVersion`, `ChangeSummary`, `TransformStep`, `TrainingRun`, `ReproManifest`, `PromotionRequest`/`PromotionApproval`, `GateResult`, `DriftReport`, `SchemaDrift`, `AuditEntry`. No I/O. |
| `hashing.py` | Canonical JSON, SHA-256 row/schema/manifest hashes, schema inference, transform-chain and code hashing. |
| `registry.py` | `DatasetRegistry`: create dataset, commit (diff + auto semver), branches, merge-lite with `MergeConflictError`. |
| `transforms.py` | Deterministic filter / project / join / aggregate over row dicts; returns rows + `TransformStep` with column lineage. |
| `drift.py` | PSI (numeric + categorical), KS statistic, schema drift with rename heuristic, `compare_versions`. |
| `lineage.py` | `LineageGraph`: version/run/model edges, column origins (transitive), BFS impact/upstream. |
| `gates.py` | Schema gates, promotion evaluation, signed approvals (SoD, quorum), quarantine, trainable guard. |
| `repro.py` | `build_manifest` / `verify_manifest` for training runs. |
| `storage.py` | `FileStore`: chunked blobs + manifest, integrity checks, size accounting, retention. |
| `audit.py` | `AuditLog`: SHA-256 hash-chained append-only log with `verify()`. |
| `service.py` | FastAPI app: thin HTTP wrapper; domain errors → 400, merge conflicts → 409. |
| `cli.py` | `argparse` CLI over a persistent `FileStore`. |

## Data model

- **Dataset** — `dataset_id`, owner, `primary_key` column, `row_count_floor`
  (promotion gate), `retention_keep_latest`.
- **DatasetVersion** — `version_id` (semver, dataset-global), `parent_version_id`,
  `branch`, `schema`, `rows`, `row_hashes` (PK → SHA-256), `manifest_hash`,
  `change_summary`, `stage` (`experimental` / `validated` / `production` /
  `quarantined`).
- **Branch** — a name → head version id pointer. History is read from parent
  links, not from version numbers.
- **TransformStep** — `op`, JSON `params`, `input_version_ids`, `column_lineage`
  (output column → `[(source version, source column)]`).
- **TrainingRun** — pins `dataset_version_id` + `dataset_manifest_hash` +
  transform chain + `seed` + `code_hash` → `model_version`.
- **ReproManifest** — the pinned values plus `manifest_hash` over them.

## Versioning rules

1. First commit of a dataset is `1.0.0`.
2. A commit diffs against its branch head: rows added / removed / changed
   (by primary key) and schema added / removed / type-changed columns.
3. Breaking schema change (column removed or type changed) → major bump;
   any other change → minor bump. The bump is taken from the dataset-wide
   latest version, so branches never collide on a version number.
4. An identical snapshot is rejected (no-change commit).
5. Versions are immutable: a commit copies rows; nothing mutates a
   published version in place.

## Merge rules

1. Find the common ancestor of the two branch heads (walk parent links).
2. Diff each side against the ancestor (added / removed / changed row keys).
3. A key changed on both sides to *different* content is a conflict:
   `MergeConflictError` lists the keys and no version is created.
4. Otherwise the merged snapshot starts from the target head, applies
   source-side additions/changes, and applies source deletions for keys
   the target did not touch. Schema is the union; source wins on type
   conflicts (surfaced as a breaking bump).

## Drift thresholds

- PSI < 0.10 stable, 0.10–0.25 watch, ≥ 0.25 drifted (per column; the
  report's `max_psi` drives the verdict).
- KS < 0.10 stable, 0.10–0.20 watch, ≥ 0.20 drifted.
- Promotion to production blocks on PSI ≥ 0.25 or KS ≥ 0.20 vs the
  current production version; elevated-but-passing drift is a warning.

## Promotion gates

| Gate | Severity | Rule |
| ---- | -------- | ---- |
| `stage_transition` | blocking | Only `experimental → validated` and `validated → production`. |
| `row_count_floor` | blocking | `row_count >= dataset.row_count_floor`. |
| `expected_columns` | blocking | All expected columns present (when specified). |
| `non_nullable_columns` | blocking | No nulls in non-nullable columns. |
| `not_quarantined` | blocking | Quarantined versions fail every gate. |
| `drift_vs_production` | blocking | PSI/KS vs production below thresholds (production only). |
| `drift_watch` | warning | Elevated but passing drift. |

Approvals: validated needs 1 distinct approver, production needs 2.
The requester can never approve their own request; a rejection ends the
request; a decision is signed over `(request_id, approver, decision, secret)`.

## Storage layout

```
<store-root>/<dataset_id>/
    versions/<version_id>/manifest.json
    chunks/<chunk_sha256>.jsonl     # 100 rows per chunk, content-addressed
```

`manifest.json` pins the schema, row hashes, manifest hash, stage,
change summary, and chunk list (hash + rows + bytes). Loading a version
re-verifies every chunk hash. Retention deletes version manifests beyond
`keep_latest` newest per dataset, always keeping the production version
and branch heads; chunks are content-addressed and deliberately left in
place (a full GC pass is a separate, explicit operation).

## Failure modes (fail-closed)

- Quarantined version → cannot be promoted, cannot train (`assert_trainable`).
- Merge conflict → no merged version is created.
- Blocking gate failure → promotion cannot be applied, even after quorum.
- Manifest mismatch → `verify` reports the exact field that changed.
- Tampered chunk → `load_version` raises on the integrity check.
- Tampered audit entry → `verify()` reports the first broken link.

## Testing

- Core tests run on pure stdlib: `pytest` from the project root (no
  install needed; `tests/conftest.py` puts `src/` on the path).
- API tests use `pytest.importorskip("fastapi")` and skip gracefully
  when FastAPI is not installed; they run in CI where `requirements.txt`
  is installed.
- The demo (`scripts/demo.py`) exercises the full flow end to end with
  seeded synthetic data and asserts the key invariants.
