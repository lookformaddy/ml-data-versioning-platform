# ML Data Versioning Platform

A **versioned dataset registry for ML teams**: every training dataset
is an immutable, content-hashed snapshot with a semantic version,
branch/merge history, column-level lineage, drift comparison against
production, reproducibility manifests, and gated promotion — the
machinery that makes "which data trained this model?" answerable in
an audit, and "can we reproduce this run?" answerable by re-deriving
a hash.

Most ML teams can tell you which model is in production. Far fewer
can say, for a given model version, exactly which dataset version
trained it, which transform chain derived it, whether the data has
drifted since, and who approved promoting that data to production.
When a model misbehaves in production, that gap turns a data question
into a guessing game. This platform closes it with deterministic,
explainable machinery: no black-box scoring, no network calls, no
real training dependency.

## What it does

- **Versioned dataset registry** — content-hashed snapshots (SHA-256
  per row + manifest hash over the sorted row hashes and schema).
  Versions are immutable; a change ships as a new version with a
  change summary (rows added / removed / changed, columns added /
  removed / type-changed). Semantic versioning is automatic: a
  breaking schema change (column removed or type changed) bumps
  major, any other change bumps minor. No-change commits are
  rejected.
- **Branch / merge-lite** — `main` plus named branches. A merge finds
  the common ancestor, diffs each side, and fails closed on
  conflicts: a row key changed on both sides to different content
  blocks the merge and no merged version is created. Identical edits
  on both sides are not conflicts; source deletions apply when the
  target left the row untouched.
- **Lineage** — dataset version → training run → model version links,
  plus column-level lineage for derived datasets. Transforms
  (filter, project, join, aggregate) are recorded as deterministic,
  pure-stdlib operations with serializable specs, so a column can be
  traced back to its raw inputs (`column_origin`) and the downstream
  blast radius of a version can be listed (`impact`).
- **Drift vs version** — distribution comparison between any two
  versions: PSI (numeric via reference-quantile bins, categorical via
  value proportions) and a two-sample KS statistic, pure stdlib.
  Schema drift detection covers added / removed / type-changed
  columns and a rename heuristic (same type + identical value
  multiset). Versions failing schema gates are quarantined
  fail-closed: they cannot be promoted or used for training.
- **Reproducibility manifests** — seed + dataset version hash +
  transform-chain hash + code hash pinned per training run.
  `verify-manifest` re-derives the hash and reports match/mismatch,
  catching silent changes to data, transforms, or code.
- **Promotion gates** — `experimental → validated → production`,
  gated on schema checks, drift thresholds vs the production version
  (PSI < 0.25, KS < 0.20 blocking), and row-count floors. Approvals
  are SHA-256 signed over the request, approver, and decision, with
  separation of duties (approver ≠ requester) and a quorum of 1 for
  validated, 2 for production.
- **Storage** — local file store with chunked blobs (100 rows/chunk,
  content-addressed by chunk hash, so identical chunks are stored
  once) + manifest, size accounting, and a retention policy that
  prunes old non-production versions per branch while always keeping
  the production version and branch heads.
- **FastAPI service + CLI + demo script**, deterministic core (no
  network, no keys).

## Architecture

See [docs/architecture.md](docs/architecture.md).

```
snapshot (rows + schema) ──▶ content hashing (row SHA-256, manifest SHA-256)
        │                              │
        ▼                              ▼
registry (immutable versions, semver, branches, merge-lite)
        │                              │
        ▼                              ▼
transforms (filter/project/join/agg) ──▶ lineage graph (version → run → model)
        │                              │
        ▼                              ▼
drift (PSI + KS, schema drift) ──▶ gates (schema, drift vs prod, row floor)
        │                              │
        ▼                              ▼
promotion (signed approvals, SoD, quorum) + reproducibility manifests
        │                              │
        ▼                              ▼
file store (chunked blobs + manifest, retention) + hash-chained audit log
```

## How to run

```bash
pip install -r requirements.txt
make test          # pytest suite (core tests run on stdlib only)
make demo          # end-to-end demo: version → branch → merge → drift →
                   #   quarantine → promote → lineage → manifest verify → retention
uvicorn mdvp.service:app --host 0.0.0.0 --port 8000   # with PYTHONPATH=src
```

CLI (persists snapshots in a local file store):

```bash
export MDVP_STORE=./mdvp-store
PYTHONPATH=src python -m mdvp.cli create churn --owner a.chen --primary-key customer_id
PYTHONPATH=src python -m mdvp.cli commit churn snapshot.jsonl --created-by a.chen
PYTHONPATH=src python -m mdvp.cli drift churn 1.0.0 1.1.0
PYTHONPATH=src python -m mdvp.cli promote churn 1.1.0 production \
  --request-id promo-001 --requested-by a.chen --approver d.kim --approver e.santos
```

## API surface

| Method | Endpoint | Purpose |
| ------ | -------- | ------- |
| `GET` | `/health` | Liveness + dataset count |
| `POST` | `/datasets` | Register a dataset |
| `POST` | `/datasets/{id}/versions` | Commit a snapshot (auto semver) |
| `GET` | `/datasets/{id}/versions`, `/datasets/{id}/versions/{vid}` | Inventory |
| `POST` | `/datasets/{id}/branches` | Create a branch |
| `POST` | `/datasets/{id}/merge` | Merge branches (409 on conflict) |
| `POST` | `/datasets/{id}/drift` | PSI/KS report between two versions |
| `POST` | `/datasets/{id}/versions/{vid}/promotions` | Request a promotion (gates evaluated) |
| `POST` | `/promotions/{id}/decide`, `/promotions/{id}/apply` | Signed approvals + apply |
| `POST` | `/datasets/{id}/training-runs` | Record a run + reproducibility manifest |
| `POST` | `/training-runs/{id}/verify` | Re-derive and verify the manifest |
| `GET` | `/lineage/{version_id}/impact` | Downstream blast radius |
| `GET` | `/audit/verify` | Hash-chain verification |

## Key design decisions and trade-offs

- **Content hashing over row counts.** A version is identified by
  what it *contains*, not by a label someone typed. The trade-off is
  cost: hashing is O(rows) per commit, which is fine for snapshot
  sizes this platform targets and is exactly what makes tamper
  detection and reproducibility cheap.
- **Dataset-global semver, not per-branch semver.** Version ids are
  unique across the dataset, so two branches committing from the same
  parent never collide on a number. The trade-off is that a branch's
  versions are not contiguous — branch history is read from parent
  links, not from version numbers.
- **Fail-closed everywhere.** Quarantined versions cannot train or
  promote; a merge conflict creates nothing; a promotion with a
  failed blocking gate cannot be applied even after quorum; a
  manifest mismatch is a hard failure, not a warning. A versioning
  system that fails open manufactures false reproducibility.
- **Deterministic, serializable transforms.** Transform specs are
  JSON, so the transform chain hashes stably and replays exactly.
  The trade-off is expressiveness — only filter / project / join /
  aggregate are supported, deliberately; arbitrary code cannot be
  hashed or replayed.
- **Pure-stdlib core, FastAPI at the edge.** All decision logic is
  testable without a server, a database, or a network. The service is
  a thin wrapper, so the API can never become a second source of
  truth. The trade-off is in-memory state: durable persistence beyond
  the file store (a real database, plus durable audit storage) is the
  explicit next step.
- **Signatures, not just names.** A promotion decision is a hash over
  the request, approver, and decision, so a decision cannot be
  replayed onto another request. This is tamper-evidence, not PKI —
  a production deployment would use real key-based signatures and a
  key-management service.
- **Chunked, content-addressed blobs.** Identical chunks across
  versions are stored once, so near-identical snapshots are cheap.
  Retention prunes version manifests, not chunks, by design: orphaned
  chunks are harmless and a full garbage-collection pass is a
  separate, explicit operation.

## What it demonstrates

ML dataset versioning end to end: content-hashed immutable snapshots
with automatic semantic versioning, branch/merge with fail-closed
conflict detection, column-level lineage and impact analysis, PSI/KS
drift and schema-drift detection, fail-closed quarantine, signed
multi-approver promotion gates with separation of duties,
reproducibility manifests with re-derivation verification, and a
chunked content-addressed file store with retention — built as a
typed, tested, enterprise-shaped Python codebase with a FastAPI
service, CLI, Docker image, and CI definition.
