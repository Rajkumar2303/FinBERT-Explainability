"""Verify completed results against frozen test rows and prediction exports."""
import json
from pathlib import Path

import numpy as np

from finbert_explainability.phase2.artifacts import digest, load_split, read_json, verify_manifest, verify_run
from finbert_explainability.phase2.metrics import metrics, summarize


if __name__ == "__main__":
    root = Path("artifacts/phase2")
    output = root / "execution/final_verification.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_manifest("artifacts/phase1")
    assert manifest == read_json(root / "audit/audit.json")["phase1_manifest_sha256"]
    frozen = load_split("artifacts/phase1", "test")
    test_hash = digest("artifacts/phase1/test.jsonl")
    cfg = read_json("configs/phase2.json")
    checks = []
    for kind in ["baseline", "bert", "finbert", "reference"]:
        run = root / ("reference" if kind == "reference" else f"{kind}-evaluation")
        verify_run(run)
        summary = read_json(run / ("reference.json" if kind == "reference" else "summary.json"))
        assert summary["phase1_manifest_sha256"] == manifest
        assert summary["test_sha256"] == test_hash
        reports = []
        for seed in ([None] if kind == "reference" else cfg["seeds"]):
            path = run if seed is None else run / f"seed-{seed}"
            predictions = [json.loads(line) for line in (path / "test_predictions.jsonl").read_text().splitlines()]
            assert len(predictions) == len(frozen) == 727
            for pred, row in zip(predictions, frozen):
                assert pred["row_id"] == row["row_id"] and pred["true_label"] == row["label"]
                assert pred["split"] == "test" and pred["is_allagree"] == row["is_allagree"]
                probabilities = [pred["probabilities"][k] for k in ["negative", "neutral", "positive"]]
                assert np.isfinite(probabilities).all() and np.isclose(sum(probabilities), 1)
                assert int(np.argmax(probabilities)) == pred["predicted_label"]
            computed = metrics([p["true_label"] for p in predictions], [p["predicted_label"] for p in predictions])
            saved = read_json(path / "test_metrics.json")
            for key, value in computed.items():
                assert saved[key] == value, (kind, seed, key)
            reports.append(computed)
        if kind != "reference":
            calculated_summary = summarize(reports)
            for key, value in calculated_summary.items():
                assert summary["metrics"][key] == value, (kind, key)
            verify_run(root / kind)
            assert read_json(root / kind / "config.json") == cfg
        else:
            assert summary["independent_evaluation"] is False
            assert summary["original_id2label"] == {"0": "positive", "1": "negative", "2": "neutral"}
        checks.append({"kind": kind, "prediction_sets_verified": len(reports), "rows_per_set": len(frozen),
                       "same_frozen_test": True, "metrics_recomputed": True})
    result = {"status": "passed", "phase1_manifest_sha256": manifest, "test_sha256": test_hash,
              "all_original_configs_preserved": True, "checks": checks}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
