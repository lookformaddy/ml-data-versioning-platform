"""CLI for the ML Data Versioning Platform.

Operates on a local file store (``--store`` or ``MDVP_STORE`` env,
default ``./mdvp-store``) so dataset snapshots persist across
invocations. Commands:

    create     register a dataset
    commit     commit a JSONL snapshot file as a new version
    list       list versions of a dataset
    diff       row/schema change summary between two versions
    drift      PSI/KS drift report between two versions
    branch     create a branch from a version
    merge      merge one branch into another (fail-closed on conflicts)
    promote    request a promotion and record signed approvals
    verify     verify a training run's reproducibility manifest
    store      storage report (versions, bytes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from mdvp import gates as gates_mod
from mdvp import repro as repro_mod
from mdvp.drift import compare_versions
from mdvp.hashing import code_hash, infer_schema
from mdvp.models import Dataset, Stage
from mdvp.registry import DatasetRegistry, MergeConflictError
from mdvp.storage import FileStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_path(args) -> str:
    return args.store or os.environ.get("MDVP_STORE", "./mdvp-store")


def _load_registry(store: FileStore) -> DatasetRegistry:
    """Rebuild an in-memory registry from the file store's manifests."""
    registry = DatasetRegistry()
    for dataset_dir in sorted(store.root.iterdir()) if store.root.exists() else []:
        if not dataset_dir.is_dir():
            continue
        dataset_id = dataset_dir.name
        versions = store.list_stored_versions(dataset_id)
        descriptor = store.load_dataset(dataset_id)
        if descriptor is not None:
            dataset = Dataset(
                dataset_id=descriptor["dataset_id"],
                name=descriptor["name"],
                owner=descriptor["owner"],
                primary_key=descriptor.get("primary_key", "id"),
                description=descriptor.get("description", ""),
                retention_keep_latest=descriptor.get("retention_keep_latest", 5),
                row_count_floor=descriptor.get("row_count_floor", 1),
            )
            dataset.created_at = descriptor.get("created_at", "")
            registry.create_dataset(dataset, now=dataset.created_at)
        elif versions:
            first = store.load_version(dataset_id, versions[0])
            dataset = Dataset(
                dataset_id=dataset_id,
                name=dataset_id,
                owner=first.created_by or "unknown",
                primary_key=first.primary_key,
            )
            registry.create_dataset(dataset, now=first.created_at)
        else:
            continue
        # Re-attach versions in semver order; branch heads from manifests.
        for version_id in sorted(
            versions, key=lambda v: tuple(int(p) for p in v.split("."))
        ):
            version = store.load_version(dataset_id, version_id)
            registry.versions[dataset_id][version_id] = version
            registry.branches[dataset_id].setdefault(version.branch, version_id)
            # A later version on the same branch supersedes the head.
            registry.branches[dataset_id][version.branch] = version_id
    return registry


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def cmd_create(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    dataset = Dataset(
        dataset_id=args.dataset_id,
        name=args.name or args.dataset_id,
        owner=args.owner,
        primary_key=args.primary_key,
        description=args.description,
        row_count_floor=args.row_count_floor,
    )
    registry.create_dataset(dataset, now=_now())
    store.save_dataset(dataset)
    print(f"created dataset {dataset.dataset_id} (primary key: {dataset.primary_key})")
    return 0


def cmd_commit(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    rows = _read_jsonl(args.file)
    schema = infer_schema(rows)
    try:
        version = registry.commit(
            args.dataset_id, rows, schema,
            branch=args.branch, created_by=args.created_by,
            notes=args.notes, now=_now(),
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    store.save_version(version)
    change = version.change_summary
    print(f"committed {args.dataset_id}@{version.version_id} "
          f"on branch {args.branch} ({version.row_count} rows, "
          f"manifest {version.manifest_hash[:12]})")
    if change:
        print(f"  +{change.rows_added} rows / -{change.rows_removed} rows / "
              f"~{change.rows_changed} changed; breaking={change.breaking}")
    return 0


def cmd_list(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    try:
        versions = registry.list_versions(args.dataset_id, args.branch)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for version in versions:
        print(f"{version.version_id}  {version.branch:12} {version.stage.value:12} "
              f"{version.row_count:6} rows  {version.manifest_hash[:12]}  "
              f"{version.created_by or '-'}")
    return 0


def cmd_diff(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    try:
        old = registry.get_version(args.dataset_id, args.from_version)
        new = registry.get_version(args.dataset_id, args.to_version)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    from mdvp.drift import schema_drift

    drift = schema_drift(old.schema, new.schema, old.rows, new.rows)
    print(json.dumps(drift.to_dict(), indent=2))
    return 0


def cmd_drift(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    try:
        reference = registry.get_version(args.dataset_id, args.reference)
        current = registry.get_version(args.dataset_id, args.current)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(compare_versions(reference, current).to_dict(), indent=2))
    return 0


def cmd_branch(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    try:
        registry.create_branch(args.dataset_id, args.name, args.from_version)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"branch {args.name} created from {args.from_version}")
    return 0


def cmd_merge(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    try:
        version = registry.merge(
            args.dataset_id, args.source, args.target,
            created_by=args.created_by, now=_now(),
        )
    except MergeConflictError as exc:
        print(f"merge blocked: {exc}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    store.save_version(version)
    print(f"merged {args.source} into {args.target}: "
          f"new version {version.version_id} ({version.row_count} rows)")
    return 0


def cmd_promote(args) -> int:
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    try:
        dataset = registry.get_dataset(args.dataset_id)
        version = registry.get_version(args.dataset_id, args.version)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    target = Stage(args.target)
    request = gates_mod.request_promotion(
        registry, dataset, version, target, args.requested_by,
        args.request_id, now=_now(),
    )
    for gate in request.gate_results:
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"[{mark}] {gate['gate_id']}: {gate['message']}")
    for approver in args.approver:
        try:
            gates_mod.record_approval(
                request, approver, "approve", secret=args.secret, now=_now()
            )
            print(f"approval recorded: {approver} (state={request.state})")
        except ValueError as exc:
            print(f"approval error ({approver}): {exc}", file=sys.stderr)
            return 1
    if request.state == "approved":
        try:
            gates_mod.apply_promotion(registry, request)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        store.save_version(version)
        print(f"promoted {args.dataset_id}@{args.version} to {target.value}")
    else:
        print(f"promotion {args.request_id} is {request.state}: "
              "needs more approvals or gates failed")
        return 1
    return 0


def cmd_verify(args) -> int:
    # A manifest JSON file is expected: {"manifest": {...},
    # "transform_chain": [...], "code_hash": ..., "seed": ...}
    with open(args.manifest_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    store = FileStore(_store_path(args))
    registry = _load_registry(store)
    from mdvp.models import ReproManifest

    manifest = ReproManifest(**payload["manifest"])
    try:
        version = registry.get_version(args.dataset_id, manifest.dataset_version_id)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ok, message = repro_mod.verify_manifest(
        manifest, version, payload.get("transform_chain", []),
        payload.get("code_hash", ""), payload.get("seed", 0),
    )
    print(f"verified={ok}: {message}")
    return 0 if ok else 1


def cmd_store(args) -> int:
    store = FileStore(_store_path(args))
    print(json.dumps(store.dataset_report(args.dataset_id), indent=2))
    print(f"total store bytes: {store.total_size_bytes()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdvp", description=__doc__)
    parser.add_argument("--store", default=None, help="file store root")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="register a dataset")
    p.add_argument("dataset_id")
    p.add_argument("--name", default="")
    p.add_argument("--owner", required=True)
    p.add_argument("--primary-key", default="id")
    p.add_argument("--description", default="")
    p.add_argument("--row-count-floor", type=int, default=1)
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("commit", help="commit a JSONL snapshot as a new version")
    p.add_argument("dataset_id")
    p.add_argument("file")
    p.add_argument("--branch", default="main")
    p.add_argument("--created-by", default="")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_commit)

    p = sub.add_parser("list", help="list versions of a dataset")
    p.add_argument("dataset_id")
    p.add_argument("--branch", default=None)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("diff", help="schema change summary between two versions")
    p.add_argument("dataset_id")
    p.add_argument("from_version")
    p.add_argument("to_version")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("drift", help="PSI/KS drift report between two versions")
    p.add_argument("dataset_id")
    p.add_argument("reference")
    p.add_argument("current")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("branch", help="create a branch from a version")
    p.add_argument("dataset_id")
    p.add_argument("name")
    p.add_argument("from_version")
    p.set_defaults(func=cmd_branch)

    p = sub.add_parser("merge", help="merge one branch into another")
    p.add_argument("dataset_id")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--created-by", default="")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("promote", help="request + approve a promotion")
    p.add_argument("dataset_id")
    p.add_argument("version")
    p.add_argument("target", choices=["validated", "production"])
    p.add_argument("--request-id", required=True)
    p.add_argument("--requested-by", required=True)
    p.add_argument("--approver", action="append", default=[])
    p.add_argument("--secret", default=gates_mod.SECRET)
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("verify", help="verify a reproducibility manifest")
    p.add_argument("dataset_id")
    p.add_argument("manifest_file")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("store", help="storage report for a dataset")
    p.add_argument("dataset_id")
    p.set_defaults(func=cmd_store)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
