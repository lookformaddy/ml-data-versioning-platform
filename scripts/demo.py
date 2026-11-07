"""End-to-end demo: version a synthetic customer-churn feature
dataset, branch + merge, compare drift, quarantine a bad version,
promote with signed approvals, record a training run with a
reproducibility manifest, verify it (then tamper and re-verify),
and report storage + retention.

All data is synthetic (seeded); no network calls, no API keys. Run with:
    PYTHONPATH=src python scripts/demo.py
"""
from __future__ import annotations

import json
import random
import tempfile

from mdvp import gates as gates_mod
from mdvp import repro as repro_mod
from mdvp.audit import AuditLog
from mdvp.drift import compare_versions
from mdvp.gates import quarantine
from mdvp.hashing import code_hash
from mdvp.lineage import LineageGraph
from mdvp.models import Dataset, DType, SchemaField, Stage, TrainingRun
from mdvp.registry import DatasetRegistry, MergeConflictError
from mdvp.storage import FileStore
from mdvp.transforms import aggregate_rows, filter_rows

TODAY = "2026-11-07"
SECRET = "demo-signing-secret"
SEED = 42
REGIONS = ["north", "south", "east", "west"]


def _synthetic_rows(n: int, seed: int, spend_shift: float = 0.0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "customer_id": f"C{i:05d}",
                "age": rng.randint(18, 80),
                "tenure_months": rng.randint(0, 120),
                "monthly_spend": round(rng.uniform(20.0, 300.0) + spend_shift, 2),
                "region": rng.choice(REGIONS),
                "churned": 1 if rng.random() < 0.18 else 0,
            }
        )
    return rows


SCHEMA = [
    SchemaField("customer_id", DType.STRING, nullable=False),
    SchemaField("age", DType.INTEGER),
    SchemaField("tenure_months", DType.INTEGER),
    SchemaField("monthly_spend", DType.NUMBER),
    SchemaField("region", DType.STRING),
    SchemaField("churned", DType.INTEGER),
]


def main() -> None:
    registry = DatasetRegistry()
    lineage = LineageGraph()
    audit = AuditLog()
    store = FileStore(tempfile.mkdtemp(prefix="mdvp-demo-"))

    # 1. Register the dataset and commit v1.0.0 (600 synthetic rows).
    dataset = Dataset(
        dataset_id="customer-churn-features",
        name="Customer Churn Features",
        owner="a.chen",
        primary_key="customer_id",
        description="Synthetic churn-model training features (demo data).",
        row_count_floor=100,
    )
    registry.create_dataset(dataset, now=TODAY)
    v1 = registry.commit(
        "customer-churn-features", _synthetic_rows(600, SEED), SCHEMA,
        branch="main", created_by="a.chen", notes="initial snapshot", now=TODAY,
    )
    store.save_version(v1)
    audit.append(TODAY, "a.chen", "version_committed", dataset.dataset_id,
                 v1.version_id, f"manifest={v1.manifest_hash[:12]}")
    print(f"committed {v1.version_id}: {v1.row_count} rows, "
          f"manifest {v1.manifest_hash[:12]}")

    # 2. Branch "experiment": raise spend on 40 rows, add 20 new rows.
    registry.create_branch("customer-churn-features", "experiment", v1.version_id)
    exp_rows = [dict(r) for r in v1.rows]
    rng = random.Random(SEED + 1)
    for row in rng.sample(exp_rows, 40):
        row["monthly_spend"] = round(row["monthly_spend"] * 1.25, 2)
    exp_rows.extend(_synthetic_rows(20, SEED + 2))
    # Re-key the new rows so they do not collide with v1 ids.
    for j, row in enumerate(exp_rows[600:], start=601):
        row["customer_id"] = f"C{j + 600:05d}"
    v_exp = registry.commit(
        "customer-churn-features", exp_rows, SCHEMA,
        branch="experiment", created_by="b.okafor",
        notes="spend uplift experiment", now=TODAY,
    )
    store.save_version(v_exp)
    change = v_exp.change_summary
    print(f"branch experiment -> {v_exp.version_id}: +{change.rows_added} rows, "
          f"~{change.rows_changed} changed, breaking={change.breaking}")

    # 3. Merge experiment back into main (no conflicts: main untouched).
    merged = registry.merge(
        "customer-churn-features", "experiment", "main",
        created_by="a.chen", now=TODAY,
    )
    store.save_version(merged)
    print(f"merged experiment into main: {merged.version_id} "
          f"({merged.row_count} rows)")

    # 4. Drift: merged version vs v1.0.0 (spend shifted +25% on 40 rows).
    report = compare_versions(v1, merged)
    print(f"drift v1.0.0 -> {merged.version_id}: max PSI {report.max_psi}, "
          f"max KS {report.max_ks}, verdict={report.verdict}")
    for col in report.per_column:
        print(f"  {col['column']:15} {col['kind']:11} PSI {col.get('psi', 0):.4f}"
              + (f"  KS {col['ks']:.4f}" if "ks" in col else ""))

    # 5. A bad version: drop the region column (breaking) and 550 rows.
    bad_rows = [
        {k: v for k, v in r.items() if k != "region"} for r in v1.rows[:50]
    ]
    bad_schema = [f for f in SCHEMA if f.name != "region"]
    v_bad = registry.commit(
        "customer-churn-features", bad_rows, bad_schema,
        branch="main", created_by="a.chen", notes="broken extract", now=TODAY,
    )
    store.save_version(v_bad)
    gates = gates_mod.run_schema_gates(
        v_bad, expected_columns=[f.name for f in SCHEMA],
        row_count_floor=dataset.row_count_floor,
    )
    failed = [g for g in gates if not g.passed and g.severity == "blocking"]
    print(f"bad version {v_bad.version_id}: {len(failed)} blocking gate failures "
          f"({', '.join(g.gate_id for g in failed)})")
    quarantine(v_bad)
    print(f"bad version quarantined: stage={v_bad.stage.value}")
    try:
        gates_mod.assert_trainable(v_bad)
    except ValueError as exc:
        print(f"fail-closed training guard: {exc}")

    # 6. Promote the merged version: experimental -> validated -> production.
    for target, approvers in ((Stage.VALIDATED, ["d.kim"]),
                              (Stage.PRODUCTION, ["d.kim", "e.santos"])):
        request = gates_mod.request_promotion(
            registry, dataset, merged, target, "a.chen",
            f"promo-{merged.version_id}-{target.value}", now=TODAY,
        )
        blocking = [g for g in request.gate_results
                    if not g["passed"] and g["severity"] == "blocking"]
        print(f"promotion to {target.value}: {len(blocking)} blocking gate failures")
        try:
            gates_mod.record_approval(request, "a.chen", "approve", secret=SECRET, now=TODAY)
        except ValueError as exc:
            print(f"separation of duties enforced: {exc}")
        for approver in approvers:
            gates_mod.record_approval(request, approver, "approve", secret=SECRET, now=TODAY)
        gates_mod.apply_promotion(registry, request)
        print(f"promoted {merged.version_id} to {target.value} "
              f"(state={request.state}, {len(request.approvals)} approvals)")
        audit.append(TODAY, "a.chen", "promotion_applied", dataset.dataset_id,
                     merged.version_id, target.value)

    # 7. Derived dataset via transforms + column lineage.
    filtered, step_filter = filter_rows(merged, "churned", "==", 1)
    agg_rows, step_agg = aggregate_rows(
        merged, group_by=["region"],
        aggregations={"monthly_spend": "mean", "customer_id": "count"},
    )
    print(f"transforms: filter churned==1 -> {len(filtered)} rows; "
          f"aggregate by region -> {len(agg_rows)} rows")
    lineage.record_transform(f"{merged.version_id}-churned", step_filter)
    lineage.record_transform(f"{merged.version_id}-by-region", step_agg)
    origins = lineage.column_origin(
        f"{merged.version_id}-by-region", "monthly_spend_mean"
    )
    print(f"column origin of monthly_spend_mean: {sorted(origins)}")
    print(f"downstream impact of {merged.version_id}: "
          f"{lineage.impact(merged.version_id)}")

    # 8. Training run + reproducibility manifest; verify, then tamper.
    code = code_hash({"train.py": "def train(seed):\n    return seed\n",
                      "features.py": "FEATURES = ['age', 'tenure_months']\n"})
    run = TrainingRun(
        run_id="run-001",
        dataset_id=dataset.dataset_id,
        dataset_version_id=merged.version_id,
        dataset_manifest_hash=merged.manifest_hash,
        transform_chain=[step_filter],
        seed=SEED,
        code_hash=code,
        model_version="churn-model@1.0.0",
        created_by="b.okafor",
        created_at=TODAY,
        metrics={"auc": 0.81},
    )
    lineage.record_training_run(run)
    manifest = repro_mod.build_manifest(run, now=TODAY)
    print(f"training run {run.run_id}: manifest {manifest.manifest_hash[:12]}")
    ok, message = repro_mod.verify_manifest(
        manifest, merged, [s.to_dict() for s in run.transform_chain], code, SEED
    )
    print(f"verify manifest: {ok} ({message})")
    assert ok
    ok_tampered, message_tampered = repro_mod.verify_manifest(
        manifest, merged, [s.to_dict() for s in run.transform_chain], code, 7
    )
    print(f"verify with tampered seed: {ok_tampered} ({message_tampered})")
    assert not ok_tampered

    # 9. Storage accounting + retention.
    report_store = store.dataset_report(dataset.dataset_id)
    print(f"storage: {report_store['version_count']} stored versions, "
          f"{report_store['total_bytes']} bytes")
    keep = {registry.head(dataset.dataset_id, "main").version_id,
            registry.production_version(dataset.dataset_id).version_id}
    retention = store.apply_retention(
        dataset.dataset_id, keep_latest=2, keep_version_ids=keep
    )
    print(f"retention: pruned {retention['deleted']}, "
          f"reclaimed {retention['bytes_reclaimed']} bytes")

    ok_audit, audit_msg = audit.verify()
    print(f"audit: valid={ok_audit} ({audit_msg})")
    assert ok_audit
    print("\ndemo OK: versioned, branched, merged, drift-checked, quarantined, "
          "promoted, lineage-tracked, and reproducibly manifested.")


if __name__ == "__main__":
    main()
