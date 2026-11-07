"""Local file store: chunked blobs + manifest, size accounting,
retention.

Layout under the store root::

    <root>/<dataset_id>/versions/<version_id>/
        manifest.json        # version metadata + chunk list + hashes
        chunks/<chunk_hash>.jsonl   # content-addressed row chunks

Rows are written in fixed-size chunks (default 100 rows), each chunk
file named by its content hash, so identical chunks across versions
are stored once. Retention prunes old non-production versions per
branch (keeping at least the production version and the branch heads),
and reports bytes reclaimed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mdvp.hashing import canonical_json, sha256_hex
from mdvp.models import DatasetVersion, DType, SchemaField, Stage

CHUNK_SIZE = 100


class FileStore:
    """Content-addressed local file store for dataset versions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------
    def _version_dir(self, dataset_id: str, version_id: str) -> Path:
        return self.root / dataset_id / "versions" / version_id

    def _chunk_path(self, dataset_id: str, chunk_hash: str) -> Path:
        return self.root / dataset_id / "chunks" / f"{chunk_hash}.jsonl"

    # -- dataset descriptors ----------------------------------------
    def save_dataset(self, dataset) -> None:
        """Persist a dataset descriptor (so the CLI can reload it)."""
        dataset_dir = self.root / dataset.dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "owner": dataset.owner,
            "primary_key": dataset.primary_key,
            "description": dataset.description,
            "created_at": dataset.created_at,
            "retention_keep_latest": dataset.retention_keep_latest,
            "row_count_floor": dataset.row_count_floor,
        }
        (dataset_dir / "dataset.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def load_dataset(self, dataset_id: str) -> dict | None:
        path = self.root / dataset_id / "dataset.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- writes -------------------------------------------------------
    def save_version(self, version: DatasetVersion) -> dict:
        """Persist a version; returns the manifest dict that was written."""
        version_dir = self._version_dir(version.dataset_id, version.version_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[dict] = []
        for start in range(0, len(version.rows), CHUNK_SIZE):
            chunk_rows = version.rows[start : start + CHUNK_SIZE]
            payload = "\n".join(canonical_json(r) for r in chunk_rows)
            chunk_hash = sha256_hex(payload)
            chunk_path = self._chunk_path(version.dataset_id, chunk_hash)
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            if not chunk_path.exists():
                chunk_path.write_text(payload, encoding="utf-8")
            chunks.append(
                {
                    "hash": chunk_hash,
                    "rows": len(chunk_rows),
                    "bytes": len(payload.encode("utf-8")),
                }
            )
        manifest = {
            "version_id": version.version_id,
            "dataset_id": version.dataset_id,
            "parent_version_id": version.parent_version_id,
            "branch": version.branch,
            "primary_key": version.primary_key,
            "schema": [f.to_dict() for f in version.schema],
            "row_hashes": version.row_hashes,
            "manifest_hash": version.manifest_hash,
            "stage": version.stage.value,
            "created_at": version.created_at,
            "created_by": version.created_by,
            "notes": version.notes,
            "change_summary": (
                version.change_summary.to_dict() if version.change_summary else None
            ),
            "chunks": chunks,
            "row_count": version.row_count,
        }
        (version_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest

    # -- reads --------------------------------------------------------
    def load_version(self, dataset_id: str, version_id: str) -> DatasetVersion:
        version_dir = self._version_dir(dataset_id, version_id)
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"no stored version {version_id!r} for dataset {dataset_id!r}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows: list[dict] = []
        for chunk in manifest["chunks"]:
            chunk_path = self._chunk_path(dataset_id, chunk["hash"])
            text = chunk_path.read_text(encoding="utf-8")
            if sha256_hex(text) != chunk["hash"]:
                raise ValueError(f"chunk {chunk['hash']} failed integrity check")
            rows.extend(json.loads(line) for line in text.splitlines() if line.strip())
        schema = [
            SchemaField(name=f["name"], dtype=DType(f["dtype"]), nullable=f["nullable"])
            for f in manifest["schema"]
        ]
        from mdvp.models import ChangeSummary

        change = manifest.get("change_summary")
        change_summary = ChangeSummary(**change) if change else None
        return DatasetVersion(
            version_id=manifest["version_id"],
            dataset_id=manifest["dataset_id"],
            parent_version_id=manifest["parent_version_id"],
            branch=manifest["branch"],
            primary_key=manifest.get("primary_key", "id"),
            schema=schema,
            rows=rows,
            row_hashes=manifest["row_hashes"],
            manifest_hash=manifest["manifest_hash"],
            change_summary=change_summary,
            stage=Stage(manifest["stage"]),
            created_at=manifest["created_at"],
            created_by=manifest["created_by"],
            notes=manifest["notes"],
        )

    def list_stored_versions(self, dataset_id: str) -> list[str]:
        versions_dir = self.root / dataset_id / "versions"
        if not versions_dir.exists():
            return []
        return sorted(p.name for p in versions_dir.iterdir() if p.is_dir())

    # -- size accounting ----------------------------------------------
    def dataset_size_bytes(self, dataset_id: str) -> int:
        total = 0
        dataset_dir = self.root / dataset_id
        if not dataset_dir.exists():
            return 0
        for path in dataset_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def total_size_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def version_size_bytes(self, dataset_id: str, version_id: str) -> int:
        version_dir = self._version_dir(dataset_id, version_id)
        if not version_dir.exists():
            return 0
        return sum(p.stat().st_size for p in version_dir.rglob("*") if p.is_file())

    # -- retention ----------------------------------------------------
    def apply_retention(
        self,
        dataset_id: str,
        keep_latest: int,
        *,
        keep_version_ids: set[str],
    ) -> dict:
        """Delete stored versions beyond ``keep_latest`` (newest first),
        never deleting ``keep_version_ids`` (production + branch heads).
        Returns {"deleted": [...], "bytes_reclaimed": int}."""
        stored = self.list_stored_versions(dataset_id)
        # Newest-first by semantic version; oldest pruned first.
        ordered = sorted(
            stored, key=lambda v: tuple(int(p) for p in v.split(".")), reverse=True
        )
        kept: set[str] = set(keep_version_ids)
        kept.update(ordered[:keep_latest])
        deleted: list[str] = []
        reclaimed = 0
        for version_id in ordered:
            if version_id in kept:
                continue
            reclaimed += self.version_size_bytes(dataset_id, version_id)
            version_dir = self._version_dir(dataset_id, version_id)
            for path in sorted(version_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            version_dir.rmdir()
            deleted.append(version_id)
        # Orphaned chunks (no remaining manifest references them) stay
        # on disk by design: they are content-addressed and may be
        # shared with a kept version; a full GC pass is out of scope.
        return {"deleted": sorted(deleted), "bytes_reclaimed": reclaimed}

    def dataset_report(self, dataset_id: str) -> dict:
        versions = self.list_stored_versions(dataset_id)
        return {
            "dataset_id": dataset_id,
            "stored_versions": versions,
            "version_count": len(versions),
            "total_bytes": self.dataset_size_bytes(dataset_id),
        }
