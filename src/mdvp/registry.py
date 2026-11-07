"""Versioned dataset registry: immutable snapshots, semantic
versioning, branch/merge-lite.

* A commit diffs the new snapshot against its branch head: rows
  added / removed / changed (by primary key), plus schema added /
  removed / type-changed columns. An identical snapshot is rejected
  (no-change commit).
* Semantic versioning is automatic: a breaking schema change (column
  removed or type changed) bumps major, any other change bumps minor.
  The first commit of a dataset is ``1.0.0``.
* Branches are named pointers at a version. A merge finds the common
  ancestor of the two branch heads, diffs each side against it, and
  fails closed on conflicts: a row key changed on both sides with
  different resulting content is a conflict and no merged version is
  created. Non-conflicting changes merge deterministically (source
  additions/changes win; source deletions apply unless the target
  also changed that key).
"""
from __future__ import annotations

from mdvp import hashing
from mdvp.drift import schema_drift
from mdvp.models import (
    ChangeSummary,
    Dataset,
    DatasetVersion,
    SchemaField,
    Stage,
)

MAIN = "main"


class MergeConflictError(ValueError):
    """Raised when a merge has overlapping row edits; carries the keys."""

    def __init__(self, conflicts: list[str]) -> None:
        super().__init__(f"merge conflicts on row keys: {conflicts}")
        self.conflicts = conflicts


def _bump(version_id: str, breaking: bool) -> str:
    major, minor, _patch = (int(part) for part in version_id.split("."))
    if breaking:
        return f"{major + 1}.0.0"
    return f"{major}.{minor + 1}.0"


def _diff_rows(
    old_hashes: dict[str, str], new_hashes: dict[str, str]
) -> tuple[int, int, list[str]]:
    added = [k for k in new_hashes if k not in old_hashes]
    removed = [k for k in old_hashes if k not in new_hashes]
    changed = sorted(
        k for k in old_hashes if k in new_hashes and old_hashes[k] != new_hashes[k]
    )
    return len(added), len(removed), changed


class DatasetRegistry:
    """In-memory registry of datasets and their immutable versions."""

    def __init__(self) -> None:
        self.datasets: dict[str, Dataset] = {}
        self.versions: dict[str, dict[str, DatasetVersion]] = {}
        # dataset_id -> branch name -> head version id
        self.branches: dict[str, dict[str, str]] = {}

    # -- datasets -----------------------------------------------------
    def create_dataset(self, dataset: Dataset, now: str = "") -> Dataset:
        errors = dataset.validate()
        if errors:
            raise ValueError(f"invalid dataset: {errors}")
        if dataset.dataset_id in self.datasets:
            raise ValueError(f"dataset {dataset.dataset_id!r} already exists")
        dataset.created_at = now
        self.datasets[dataset.dataset_id] = dataset
        self.versions[dataset.dataset_id] = {}
        self.branches[dataset.dataset_id] = {MAIN: None}
        return dataset

    def get_dataset(self, dataset_id: str) -> Dataset:
        if dataset_id not in self.datasets:
            raise KeyError(f"unknown dataset {dataset_id!r}")
        return self.datasets[dataset_id]

    # -- versions -----------------------------------------------------
    def commit(
        self,
        dataset_id: str,
        rows: list[dict],
        schema: list[SchemaField] | None = None,
        *,
        branch: str = MAIN,
        created_by: str = "",
        notes: str = "",
        now: str = "",
    ) -> DatasetVersion:
        dataset = self.get_dataset(dataset_id)
        if branch not in self.branches[dataset_id]:
            raise ValueError(f"unknown branch {branch!r} for dataset {dataset_id!r}")
        if schema is None:
            schema = hashing.infer_schema(rows)
        row_hashes = hashing.hash_rows(rows, dataset.primary_key)
        parent_id = self.branches[dataset_id].get(branch)
        existing = self.versions[dataset_id]
        if parent_id is None:
            version_id = "1.0.0"
            change = None
        else:
            parent = existing[parent_id]
            schema_change = schema_drift(parent.schema, schema, parent.rows, rows)
            added, removed, changed_keys = _diff_rows(parent.row_hashes, row_hashes)
            if (
                added == 0
                and removed == 0
                and not changed_keys
                and not schema_change.columns_added
                and not schema_change.columns_removed
                and not schema_change.columns_type_changed
            ):
                raise ValueError("no changes vs branch head: refusing no-change commit")
            change = ChangeSummary(
                rows_added=added,
                rows_removed=removed,
                rows_changed=len(changed_keys),
                changed_keys=changed_keys,
                columns_added=schema_change.columns_added,
                columns_removed=schema_change.columns_removed,
                columns_type_changed=schema_change.columns_type_changed,
                possible_renames=schema_change.possible_renames,
                breaking=schema_change.breaking,
            )
            # Version ids are dataset-global, so bump from the dataset-wide
            # latest version: two branches committing from the same parent
            # must not collide on the same semver.
            latest = max(
                existing.values(),
                key=lambda v: tuple(int(p) for p in v.version_id.split(".")),
            )
            version_id = _bump(latest.version_id, schema_change.breaking)
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            parent_version_id=parent_id,
            branch=branch,
            primary_key=dataset.primary_key,
            schema=list(schema),
            rows=[dict(r) for r in rows],
            row_hashes=row_hashes,
            manifest_hash=hashing.manifest_hash(row_hashes, schema),
            change_summary=change,
            stage=Stage.EXPERIMENTAL,
            created_at=now,
            created_by=created_by,
            notes=notes,
        )
        self.versions[dataset_id][version_id] = version
        self.branches[dataset_id][branch] = version_id
        return version

    def get_version(self, dataset_id: str, version_id: str) -> DatasetVersion:
        self.get_dataset(dataset_id)
        if version_id not in self.versions[dataset_id]:
            raise KeyError(f"unknown version {version_id!r} for dataset {dataset_id!r}")
        return self.versions[dataset_id][version_id]

    def list_versions(
        self, dataset_id: str, branch: str | None = None
    ) -> list[DatasetVersion]:
        self.get_dataset(dataset_id)
        versions = list(self.versions[dataset_id].values())
        if branch is not None:
            versions = [v for v in versions if v.branch == branch]
        return sorted(versions, key=lambda v: tuple(int(p) for p in v.version_id.split(".")))

    def head(self, dataset_id: str, branch: str = MAIN) -> DatasetVersion | None:
        self.get_dataset(dataset_id)
        head_id = self.branches[dataset_id].get(branch)
        return self.versions[dataset_id][head_id] if head_id else None

    def production_version(self, dataset_id: str) -> DatasetVersion | None:
        self.get_dataset(dataset_id)
        production = [
            v for v in self.versions[dataset_id].values() if v.stage is Stage.PRODUCTION
        ]
        if not production:
            return None
        return sorted(
            production, key=lambda v: tuple(int(p) for p in v.version_id.split("."))
        )[-1]

    # -- branches / merge ---------------------------------------------
    def create_branch(
        self, dataset_id: str, name: str, from_version_id: str
    ) -> str:
        self.get_dataset(dataset_id)
        if name in self.branches[dataset_id]:
            raise ValueError(f"branch {name!r} already exists for dataset {dataset_id!r}")
        self.get_version(dataset_id, from_version_id)
        self.branches[dataset_id][name] = from_version_id
        return name

    def _ancestor_chain(self, dataset_id: str, version_id: str) -> list[str]:
        chain = []
        current: str | None = version_id
        while current is not None:
            chain.append(current)
            current = self.versions[dataset_id][current].parent_version_id
        return chain

    def _common_ancestor(
        self, dataset_id: str, source_head: str, target_head: str
    ) -> str | None:
        target_chain = set(self._ancestor_chain(dataset_id, target_head))
        for version_id in self._ancestor_chain(dataset_id, source_head):
            if version_id in target_chain:
                return version_id
        return None

    def merge(
        self,
        dataset_id: str,
        source_branch: str,
        target_branch: str,
        *,
        created_by: str = "",
        now: str = "",
    ) -> DatasetVersion:
        """Merge ``source_branch`` into ``target_branch`` (fail-closed).

        Returns the merged version created on the target branch.
        Raises :class:`MergeConflictError` when both sides changed the
        same row key to different content; no version is created.
        """
        dataset = self.get_dataset(dataset_id)
        for branch in (source_branch, target_branch):
            if branch not in self.branches[dataset_id]:
                raise ValueError(f"unknown branch {branch!r} for dataset {dataset_id!r}")
        source_head = self.branches[dataset_id][source_branch]
        target_head = self.branches[dataset_id][target_branch]
        if source_head == target_head:
            raise ValueError("branches are already at the same version: nothing to merge")
        ancestor_id = self._common_ancestor(dataset_id, source_head, target_head)
        if ancestor_id is None:
            raise ValueError("no common ancestor: cannot merge unrelated histories")
        ancestor = self.versions[dataset_id][ancestor_id]
        source = self.versions[dataset_id][source_head]
        target = self.versions[dataset_id][target_head]
        _s_added, _s_removed, s_changed = _diff_rows(ancestor.row_hashes, source.row_hashes)
        _t_added, _t_removed, t_changed = _diff_rows(ancestor.row_hashes, target.row_hashes)
        source_changed_keys = set(s_changed) | {
            k for k in source.row_hashes if k not in ancestor.row_hashes
        }
        target_changed_keys = set(t_changed) | {
            k for k in target.row_hashes if k not in ancestor.row_hashes
        }
        conflicts = sorted(
            k
            for k in source_changed_keys & target_changed_keys
            if source.row_hashes.get(k) != target.row_hashes.get(k)
        )
        if conflicts:
            raise MergeConflictError(conflicts)
        # Apply source-side changes onto the target snapshot.
        merged_rows: dict[str, dict] = {str(r[dataset.primary_key]): dict(r) for r in target.rows}
        for key, row in ((str(r[dataset.primary_key]), r) for r in source.rows):
            if key in source_changed_keys:
                merged_rows[key] = dict(row)
        for key in ancestor.row_hashes:
            if key not in source.row_hashes and key not in target_changed_keys:
                merged_rows.pop(key, None)  # source deleted it, target untouched
        rows = [merged_rows[k] for k in sorted(merged_rows)]
        # Schema: union of both schemas; source wins on type conflicts
        # (merge conflicts on schema are surfaced as a breaking bump).
        merged_schema: dict[str, SchemaField] = {f.name: f for f in target.schema}
        for field in source.schema:
            merged_schema[field.name] = field
        schema = [merged_schema[name] for name in sorted(merged_schema)]
        # No-op merge: the merged snapshot is identical to the target
        # head (e.g. both sides made the same edit). There is nothing
        # new to commit; return the target head.
        merged_hashes = hashing.hash_rows(rows, dataset.primary_key)
        if (
            merged_hashes == target.row_hashes
            and hashing.schema_hash(schema) == hashing.schema_hash(target.schema)
        ):
            return target
        version = self.commit(
            dataset_id,
            rows,
            schema,
            branch=target_branch,
            created_by=created_by,
            notes=f"merge {source_branch} ({source_head}) into {target_branch}",
            now=now,
        )
        return version
