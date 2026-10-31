import os
from pathlib import Path
import time
import warnings

import joblib
import torch

from ..pipeline import write_json
from .artifacts import digest, load_split, new_output, read_json, seal, verify_manifest, verify_run
from .baseline import tune_baseline
from .common import close_logger, export_predictions, logger_for, runtime
from .metrics import summarize
from .settings import Settings

REFERENCE_WARNING = ("NON-INDEPENDENT EXTERNAL REFERENCE: ProsusAI/finbert was fine-tuned on Financial PhraseBank. "
                     "The saved test split may overlap its original training data; these results must not be treated as an independent held-out comparison.")


def code_fingerprint():
    import hashlib
    root = Path(__file__).parent
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(root.glob("*.py")))).hexdigest()


def training_gate(approved, smoke_receipt):
    if not approved:
        raise PermissionError("Full fine-tuning requires explicit user approval; pass --approve-training only after approval")
    receipt = read_json(smoke_receipt)
    if receipt.get("status") != "passed" or receipt.get("code_fingerprint") != code_fingerprint():
        raise ValueError("A passing CPU smoke receipt for the current Phase 2 code is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one GPU: 8 x 2 accumulation = effective batch 16")
    if not torch.cuda.is_available():
        raise RuntimeError("Full fine-tuning is configured for CUDA; use smoke for CPU testing")


def train_workflow(cfg, kind, output, *, approved=False, smoke_receipt="artifacts/phase2/smoke-final/smoke_receipt.json"):
    if kind not in {"baseline", "bert", "finbert"}:
        raise ValueError("Reference models must never enter fine-tuning")
    if kind != "baseline":
        training_gate(approved, smoke_receipt)
    manifest = verify_manifest(cfg.phase1_dir)
    # Deliberately no test loader in selection/training.
    train, validation = (load_split(cfg.phase1_dir, s) for s in ("train", "validation"))
    out = new_output(output, cfg.phase1_dir)
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "runtime.json", runtime())
    logger = logger_for(out)
    start = time.perf_counter()
    try:
        logger.info("Starting %s; test is not loaded for fitting/selection", kind)
        if kind == "baseline":
            winner = tune_baseline(train, validation, cfg, out, logger)
        else:
            from .neural import tune_neural
            winner = tune_neural(train, validation, cfg, cfg.models[kind], out, logger)
        write_json(out / "selection.json", {"kind": kind, "winner": winner, "seeds": cfg.seeds,
            "selection_metric": "validation_macro_f1", "tie_break": "first configured candidate",
            "phase1_manifest_sha256": manifest, "test_sha256": digest(Path(cfg.phase1_dir) / "test.jsonl"),
            "code_fingerprint": code_fingerprint(), "test_used_for_selection": False,
            "elapsed_seconds": time.perf_counter()-start})
        if verify_manifest(cfg.phase1_dir) != manifest:
            raise RuntimeError("Phase 1 artifacts changed")
        logger.info("Selection frozen; three final seed checkpoints saved")
    finally:
        close_logger(logger)
    seal(out)
    return out


def compare_evaluations(runs, output, phase1="artifacts/phase1"):
    """Collect frozen results without ranking/selecting models on test scores."""
    independent, references, hashes = {}, [], set()
    for run in map(Path, runs):
        verify_run(run)
        reference = (run / "reference.json").exists()
        report = read_json(run / ("reference.json" if reference else "summary.json"))
        hashes.add((report["phase1_manifest_sha256"], report["test_sha256"]))
        if reference:
            references.append(report)
        else:
            if report["kind"] in independent:
                raise ValueError("Duplicate workflow in comparison")
            independent[report["kind"]] = report
    if len(hashes) != 1:
        raise ValueError("All comparisons must use the same frozen Phase 1 test set")
    if not {"baseline", "bert", "finbert"}.issubset(independent):
        raise ValueError("Comparison requires baseline, bert, and finbert evaluations")
    out = new_output(output, phase1)
    write_json(out / "comparison.json", {"independent_workflows": independent,
        "non_independent_references": references, "warning": REFERENCE_WARNING if references else None,
        "test_used_for_model_selection": False})
    seal(out)
    return out


def evaluate_workflow(run, output, *, cpu=False):
    # Test becomes available only after a completed and sealed selection run.
    run = Path(run)
    verify_run(run)
    selection = read_json(run / "selection.json")
    cfg = Settings.load(run / "config.json")
    if verify_manifest(cfg.phase1_dir) != selection["phase1_manifest_sha256"]:
        raise ValueError("Evaluation must use the exact Phase 1 artifacts used for selection")
    if digest(Path(cfg.phase1_dir) / "test.jsonl") != selection["test_sha256"]:
        raise ValueError("Test set changed")
    test = load_split(cfg.phase1_dir, "test")
    out = new_output(output, cfg.phase1_dir)
    write_json(out / "evaluation_record.json", {"event": "evaluate_frozen_models", "selection": selection,
        "test_rows": len(test), "device": "cpu" if cpu or selection["kind"] == "baseline" else "cuda"})
    reports = []
    for seed in selection["seeds"]:
        path = out / f"seed-{seed}"
        path.mkdir()
        if selection["kind"] == "baseline":
            model = joblib.load(run / f"seed-{seed}" / "model.joblib")
            probabilities = model.predict_proba([r["sentence"] for r in test])
        else:
            from .neural import predict_model
            probabilities, _ = predict_model(test, {"id": str(run / f"seed-{seed}" / "model")}, cfg, cpu=cpu, local=True)
        reports.append(export_predictions(path, "test", test, probabilities))
    summary = summarize(reports)
    summary["subgroups"] = {key: summarize([r["subgroups"][key] for r in reports])
                             for key in ("allagree", "not_allagree") if all(r["subgroups"][key] is not None for r in reports)}
    write_json(out / "summary.json", {"kind": selection["kind"], "seeds": selection["seeds"],
        "phase1_manifest_sha256": selection["phase1_manifest_sha256"], "test_sha256": selection["test_sha256"],
        "selection_run_manifest_sha256": digest(run / "run_manifest.json"),
        "metrics": summary, "runtime": runtime(),
        "note": "Sample standard deviation across independent fits, not a confidence interval. Deterministic lbfgs baseline can have zero variance."})
    seal(out)
    return out


def reference_workflow(cfg, output, *, cpu=False, local_spec=None):
    warnings.warn(REFERENCE_WARNING, UserWarning)
    manifest = verify_manifest(cfg.phase1_dir)
    test = load_split(cfg.phase1_dir, "test")
    out = new_output(output, cfg.phase1_dir)
    write_json(out / "evaluation_record.json", {"event": "external_reference_inference", "warning": REFERENCE_WARNING,
        "test_rows": len(test), "device": "cpu" if cpu else "cuda"})
    from .neural import predict_model
    spec = local_spec or cfg.models["reference"]
    probabilities, mapping = predict_model(test, spec, cfg, cpu=cpu, local=local_spec is not None)
    report = export_predictions(out, "test", test, probabilities)
    write_json(out / "reference.json", {"warning": REFERENCE_WARNING, "independent_evaluation": False,
        "original_id2label": mapping, "model": spec, "config": cfg.to_dict(),
        "phase1_manifest_sha256": manifest, "test_sha256": digest(Path(cfg.phase1_dir) / "test.jsonl"),
        "metrics": report, "runtime": runtime(), "smoke_only": local_spec is not None,
        "seed_aggregation": "Not applicable: fixed external checkpoint, no fine-tuning or search"})
    seal(out)
    return out
