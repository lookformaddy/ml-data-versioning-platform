"""Reproducibility manifest tests: build, verify, tamper detection."""
from mdvp import repro as repro_mod
from mdvp.hashing import code_hash
from mdvp.models import TrainingRun
from mdvp.transforms import filter_rows
from tests.conftest import TODAY


def _run(v1, seed=42, code="train.py"):
    _, step = filter_rows(v1, "churned", "==", 1)
    return TrainingRun(
        run_id="run-1", dataset_id="churn", dataset_version_id=v1.version_id,
        dataset_manifest_hash=v1.manifest_hash, transform_chain=[step],
        seed=seed, code_hash=code_hash({"train.py": code}),
        model_version="churn-model@1.0.0", created_by="b.okafor",
        created_at=TODAY,
    )


def test_build_and_verify_manifest(v1):
    run = _run(v1)
    manifest = repro_mod.build_manifest(run, now=TODAY)
    assert len(manifest.manifest_hash) == 64
    ok, message = repro_mod.verify_manifest(
        manifest, v1, [s.to_dict() for s in run.transform_chain],
        run.code_hash, run.seed,
    )
    assert ok is True
    assert "all match" in message


def test_verify_fails_on_seed_change(v1):
    run = _run(v1)
    manifest = repro_mod.build_manifest(run, now=TODAY)
    ok, message = repro_mod.verify_manifest(
        manifest, v1, [s.to_dict() for s in run.transform_chain],
        run.code_hash, 7,
    )
    assert ok is False
    assert "seed mismatch" in message


def test_verify_fails_on_code_change(v1):
    run = _run(v1)
    manifest = repro_mod.build_manifest(run, now=TODAY)
    ok, message = repro_mod.verify_manifest(
        manifest, v1, [s.to_dict() for s in run.transform_chain],
        code_hash({"train.py": "different code"}), run.seed,
    )
    assert ok is False
    assert "code hash mismatch" in message


def test_verify_fails_on_transform_chain_change(v1):
    run = _run(v1)
    manifest = repro_mod.build_manifest(run, now=TODAY)
    _, other_step = filter_rows(v1, "churned", "==", 0)
    ok, message = repro_mod.verify_manifest(
        manifest, v1, [other_step.to_dict()], run.code_hash, run.seed,
    )
    assert ok is False
    assert "transform chain hash mismatch" in message


def test_verify_fails_on_dataset_content_change(registry, v1):
    from tests.conftest import SCHEMA, rows_v1

    run = _run(v1)
    manifest = repro_mod.build_manifest(run, now=TODAY)
    rows = [dict(r) for r in rows_v1()]
    rows[0]["age"] = 99
    v2 = registry.commit("churn", rows, SCHEMA, branch="main",
                         created_by="a.chen", now=TODAY)
    ok, message = repro_mod.verify_manifest(
        manifest, v2, [s.to_dict() for s in run.transform_chain],
        run.code_hash, run.seed,
    )
    assert ok is False
    assert "version mismatch" in message


def test_verify_fails_on_tampered_manifest(v1):
    run = _run(v1)
    manifest = repro_mod.build_manifest(run, now=TODAY)
    manifest.manifest_hash = "0" * 64
    ok, message = repro_mod.verify_manifest(
        manifest, v1, [s.to_dict() for s in run.transform_chain],
        run.code_hash, run.seed,
    )
    assert ok is False
    assert "manifest hash mismatch" in message
