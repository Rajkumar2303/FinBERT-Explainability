from pathlib import Path

from huggingface_hub import hf_hub_download

from ..data import LABELS
from .io import digest, load_predictions, read_json, require_listed, verify_manifest


def read_rows(path):
    import json
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError(f"Empty/duplicate dataset rows: {path}")
    if any(r["label"] not in [0, 1, 2] or not isinstance(r["is_allagree"], bool) for r in rows):
        raise ValueError(f"Invalid label/subgroup metadata: {path}")
    return rows


def load_inputs(cfg):
    p1, p2 = Path(cfg["phase1_dir"]), Path(cfg["phase2_dir"])
    verify_manifest(p1, "manifest.json")
    manifest, test_hash = digest(p1 / "manifest.json"), digest(p1 / "test.jsonl")
    if read_json(p1 / "metadata.json")["label_mapping"] != {str(i): n for i, n in enumerate(LABELS)}:
        raise ValueError("Phase 1 label mapping mismatch")
    test, validation = read_rows(p1 / "test.jsonl"), read_rows(p1 / "validation.jsonl")
    if any(r["split"] != "test" for r in test) or any(r["split"] != "validation" for r in validation):
        raise ValueError("Frozen split name mismatch")
    models, evidence = [], {"phase1_manifest_sha256": manifest, "test_sha256": test_hash,
                           "raw_saved_logits_available": False, "validation_predictions_checked": {}, "manifests": {}}
    for kind in ("baseline", "bert", "finbert"):
        run, evaluation = p2 / kind, p2 / f"{kind}-evaluation"
        print(f"Verifying sealed {kind} artifacts...", flush=True)
        verify_manifest(run, "run_manifest.json")
        verify_manifest(evaluation, "run_manifest.json")
        require_listed(run, ["selection.json", "config.json"])
        require_listed(evaluation, ["summary.json"])
        selected, saved_cfg = read_json(run / "selection.json"), read_json(run / "config.json")
        summary = read_json(evaluation / "summary.json")
        if selected["kind"] != kind or selected["seeds"] != cfg["seeds"] or saved_cfg["seeds"] != cfg["seeds"]:
            raise ValueError(f"Model/seed protocol mismatch: {kind}")
        for obj in (selected, summary):
            if obj["phase1_manifest_sha256"] != manifest or obj["test_sha256"] != test_hash:
                raise ValueError(f"Frozen input fingerprint mismatch: {kind}")
        if summary["selection_run_manifest_sha256"] != digest(run / "run_manifest.json"):
            raise ValueError(f"Selection manifest mismatch: {kind}")
        evidence["manifests"][kind] = digest(run / "run_manifest.json")
        for seed in cfg["seeds"]:
            folder, prefix = run / f"seed-{seed}", f"seed-{seed}"
            checkpoint = folder / ("model.joblib" if kind == "baseline" else "model")
            required = [f"{prefix}/validation_predictions.jsonl", f"{prefix}/validation_metrics.json"]
            required += [f"{prefix}/model.joblib"] if kind == "baseline" else [f"{prefix}/model/config.json", f"{prefix}/model/model.safetensors", f"{prefix}/model/tokenizer.json", f"{prefix}/training_record.json"]
            require_listed(run, required)
            require_listed(evaluation, [f"{prefix}/test_predictions.jsonl", f"{prefix}/test_metrics.json"])
            p = load_predictions(evaluation / prefix / "test_predictions.jsonl", test)
            vp = load_predictions(folder / "validation_predictions.jsonl", validation)
            if kind != "baseline":
                checkpoint_cfg = read_json(checkpoint / "config.json")
                if checkpoint_cfg["id2label"] != {str(i): n for i, n in enumerate(LABELS)}:
                    raise ValueError(f"Checkpoint class mapping mismatch: {checkpoint}")
            models.append({"key": f"{kind}-{seed}", "kind": kind, "seed": seed, "checkpoint": str(checkpoint.resolve()),
                "probabilities": p, "max_length": saved_cfg["max_length"], "batch_size": saved_cfg["per_device_batch_size"],
                "saved_metrics": read_json(evaluation / prefix / "test_metrics.json")})
            evidence["validation_predictions_checked"][f"{kind}-{seed}"] = len(vp)
    ref = p2 / "reference"
    verify_manifest(ref, "run_manifest.json")
    require_listed(ref, ["reference.json", "test_predictions.jsonl", "test_metrics.json"])
    reference = read_json(ref / "reference.json")
    if reference["independent_evaluation"] is not False or reference["phase1_manifest_sha256"] != manifest or reference["test_sha256"] != test_hash:
        raise ValueError("Reference status or frozen inputs mismatch")
    spec = reference["model"]
    if spec["id"] != "ProsusAI/finbert":
        raise ValueError("Unexpected external reference")
    # Require the exact already-cached checkpoint; never download or modify it.
    config_path = Path(hf_hub_download(spec["id"], "config.json", revision=spec["revision"],
        cache_dir=reference["config"]["cache_dir"], local_files_only=True))
    if read_json(config_path)["id2label"] != reference["original_id2label"]:
        raise ValueError("Reference label mapping mismatch")
    models.append({"key": "reference", "kind": "reference", "seed": None,
        "checkpoint": str(config_path.parent.resolve()), "probabilities": load_predictions(ref / "test_predictions.jsonl", test),
        "max_length": reference["config"]["max_length"], "batch_size": reference["config"]["per_device_batch_size"],
        "saved_metrics": read_json(ref / "test_metrics.json")})
    evidence["reference_warning"] = reference["warning"]
    return test, models, evidence
