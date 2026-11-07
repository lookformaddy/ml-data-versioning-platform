"""Reproducibility manifests for training runs.

A manifest pins, for one training run:
* the dataset version id and its manifest hash,
* the transform-chain hash (canonical JSON of the recorded steps),
* the code hash (SHA-256 over the pinned source files),
* the random seed.

``verify_manifest`` re-derives the hash from the live inputs and
reports match/mismatch, so a run can be re-checked long after the
fact — and any silent change to the data, the transform chain, or the
code is caught.
"""
from __future__ import annotations

from mdvp import hashing
from mdvp.models import DatasetVersion, ReproManifest, TrainingRun


def build_manifest(run: TrainingRun, now: str = "") -> ReproManifest:
    chain_hash = hashing.chain_hash([step.to_dict() for step in run.transform_chain])
    payload = {
        "run_id": run.run_id,
        "dataset_version_id": run.dataset_version_id,
        "dataset_manifest_hash": run.dataset_manifest_hash,
        "transform_chain_hash": chain_hash,
        "code_hash": run.code_hash,
        "seed": run.seed,
    }
    manifest_hash = hashing.sha256_hex(hashing.canonical_json(payload))
    return ReproManifest(
        run_id=run.run_id,
        dataset_version_id=run.dataset_version_id,
        dataset_manifest_hash=run.dataset_manifest_hash,
        transform_chain_hash=chain_hash,
        code_hash=run.code_hash,
        seed=run.seed,
        manifest_hash=manifest_hash,
        created_at=now,
    )


def verify_manifest(
    manifest: ReproManifest,
    version: DatasetVersion,
    transform_chain: list[dict],
    code_hash_value: str,
    seed: int,
) -> tuple[bool, str]:
    """Re-derive the manifest hash from live inputs; report match/mismatch."""
    if version.version_id != manifest.dataset_version_id:
        return False, (
            f"dataset version mismatch: manifest pins "
            f"{manifest.dataset_version_id}, got {version.version_id}"
        )
    if version.manifest_hash != manifest.dataset_manifest_hash:
        return False, (
            "dataset manifest hash mismatch: the pinned snapshot's "
            "content has changed"
        )
    chain_hash = hashing.chain_hash(transform_chain)
    if chain_hash != manifest.transform_chain_hash:
        return False, "transform chain hash mismatch: the derivation changed"
    if code_hash_value != manifest.code_hash:
        return False, "code hash mismatch: training code changed"
    if seed != manifest.seed:
        return False, f"seed mismatch: manifest pins {manifest.seed}, got {seed}"
    payload = {
        "run_id": manifest.run_id,
        "dataset_version_id": manifest.dataset_version_id,
        "dataset_manifest_hash": manifest.dataset_manifest_hash,
        "transform_chain_hash": manifest.transform_chain_hash,
        "code_hash": manifest.code_hash,
        "seed": manifest.seed,
    }
    recomputed = hashing.sha256_hex(hashing.canonical_json(payload))
    if recomputed != manifest.manifest_hash:
        return False, "manifest hash mismatch: the manifest itself was altered"
    return True, "manifest verified: data, transforms, code, and seed all match"
