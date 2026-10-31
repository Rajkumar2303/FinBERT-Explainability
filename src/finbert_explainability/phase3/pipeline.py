import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from ..data import LABELS
from ..pipeline import seed_everything
from ..phase2.common import runtime
from ..phase2.metrics import metrics, summarize
from .calibration import calibration, mean_sd
from .errors import select_errors
from .inference import predict
from .inputs import load_inputs
from .io import digest, protected_output, read_json, snapshot, write_csv, write_json, write_jsonl
from .perturbations import TYPES, generate
from .plots import plots


def fingerprint():
    return hashlib.sha256(b"".join(p.name.encode()+p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()


def validate_config(cfg):
    if len(cfg["seeds"]) != 3 or len(set(cfg["seeds"])) != 3 or cfg["bins"] != [10, 15]:
        raise ValueError("Require three distinct seeds and ECE bins [10,15]")
    if cfg["monetary_scale"] <= 0 or cfg["monetary_scale"] == 1 or not np.isfinite(cfg["monetary_scale"]):
        raise ValueError("Monetary scale must be finite, positive and not one")
    if cfg["error_examples"] != 20 or not 0 < cfg["confident_error_threshold"] <= 1:
        raise ValueError("Require 20 error examples and a confidence threshold in (0,1]")
    if not 0 < cfg["probability_tolerance"] <= 1e-4:
        raise ValueError("Invalid checkpoint probability tolerance")
    if not cfg["company_names"] or not cfg["replacement_company"].strip():
        raise ValueError("Company perturbation rules must be configured")


def subgroup_report(rows, probabilities, cfg):
    output = {}
    y = np.array([r["label"] for r in rows])
    unanimous = np.array([r["is_allagree"] for r in rows])
    for name, mask in [("allagree", unanimous), ("non_allagree", ~unanimous)]:
        counts = np.bincount(y[mask], minlength=3)
        warnings = []
        if mask.sum() < cfg["small_subgroup_count"]:
            warnings.append("small_subgroup")
        warnings += [f"small_class:{label}:{counts[i]}" for i, label in enumerate(LABELS) if counts[i] <= cfg["small_class_count"]]
        output[name] = {"n": int(mask.sum()), "class_counts": dict(zip(LABELS, counts.tolist())),
            "warnings": warnings, "metrics": metrics(y[mask], probabilities[mask].argmax(axis=1)) if mask.any() else None}
    return output


def robustness_metrics(records):
    result = {}
    for kind in TYPES:
        subset = [r for r in records if r["type"] == kind]
        visible = [r for r in subset if r["model_visible_change"]]
        def average(items, key):
            return float(np.mean([r[key] for r in items])) if items else None
        result[kind] = {"n": len(subset), "visible_n": len(visible),
            "flip_rate": average(subset, "class_flip"), "visible_flip_rate": average(visible, "class_flip"),
            "mean_confidence_change": average(subset, "confidence_change"),
            "mean_original_class_probability_change": average(subset, "original_class_probability_change"),
            "perturbed_labels_known": False}
    return result


def analyze(cfg, rows, models, evidence, out, cpu=False, smoke=False):
    y = np.array([r["label"] for r in rows])
    variants = generate(rows, cfg)
    eligible = [r for r in variants if r["eligible"]]
    write_jsonl(out / "perturbation_registry.jsonl", variants)
    write_csv(out / "perturbation_registry.csv", variants)
    write_json(out / "perturbation_coverage.json", {t: {"total": len(rows),
        "eligible": sum(r["eligible"] for r in variants if r["type"] == t),
        "skipped_reasons": {reason: sum(r["reason"] == reason for r in variants if r["type"] == t)
            for reason in sorted({r["reason"] for r in variants if r["type"] == t and not r["eligible"]})}}
        for t in TYPES})
    texts = [r["sentence"] for r in rows] + [r["perturbed_text"] for r in eligible]
    index = {r["row_id"]: i for i, r in enumerate(rows)}
    reports, prediction_audit, per_seed_rows, class_rows, calibration_rows, subgroup_rows = {}, {}, [], [], [], []
    probability_sets = {}
    for spec in models:
        key = spec["key"]
        print(f"Analyzing {key}: {len(rows)} originals, {len(eligible)} perturbations; inference only", flush=True)
        path = out / key
        path.mkdir()
        p = spec["probabilities"]
        classification = metrics(y, p.argmax(axis=1))
        if not smoke:
            for name, value in classification.items():
                if spec["saved_metrics"][name] != value:
                    raise ValueError(f"Recomputed saved classification mismatch: {key}/{name}")
        c = calibration(y, p, cfg["bins"])
        groups = subgroup_report(rows, p, cfg)
        start = time.perf_counter()
        if not cpu:
            torch.cuda.reset_peak_memory_stats()
        actual, logits, representations, truncated = predict(spec, texts, cpu=cpu)
        difference = float(np.max(np.abs(actual[:len(rows)]-p)))
        if difference > cfg["probability_tolerance"] or not np.array_equal(actual[:len(rows)].argmax(axis=1), p.argmax(axis=1)):
            raise ValueError(f"Checkpoint/saved-prediction inconsistency for {key}: max probability difference {difference}")
        prediction_audit[key] = {"max_probability_difference": difference, "classes_identical": True,
            "inference_seconds": time.perf_counter()-start, "original_truncated_count": sum(truncated[:len(rows)]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if not cpu else 0,
            "recomputed_logits": logits is not None, "saved_raw_logits_available": False}
        np.savez_compressed(path / "recomputed_inference.npz", probabilities=actual,
            **({"canonical_logits": logits} if logits is not None else {}))
        write_json(path / "inference_row_order.json", [{"row_id": r["row_id"], "type": "original"} for r in rows] +
                   [{"row_id": r["row_id"], "type": r["type"]} for r in eligible])
        perturbed = []
        for j, variant in enumerate(eligible):
            i, n = index[variant["row_id"]], len(rows)+j
            before, after = int(p[i].argmax()), int(actual[n].argmax())
            perturbed.append({**variant, "model": spec["kind"], "seed": spec["seed"],
                "original_label": int(y[i]), "original_predicted_class": before, "perturbed_predicted_class": after,
                "original_probabilities": dict(zip(LABELS, p[i].tolist())),
                "perturbed_probabilities": dict(zip(LABELS, actual[n].tolist())),
                "original_confidence": float(p[i].max()), "perturbed_confidence": float(actual[n].max()),
                "confidence_change": float(actual[n].max()-p[i].max()),
                "original_class_probability_change": float(actual[n,before]-p[i,before]),
                "class_flip": before != after, "model_visible_change": representations[i] != representations[n],
                "original_truncated": bool(truncated[i]), "perturbed_truncated": bool(truncated[n])})
        write_jsonl(path / "robustness_predictions.jsonl", perturbed)
        if perturbed:
            write_csv(path / "robustness_predictions.csv", perturbed)
        report = {"model": spec["kind"], "seed": spec["seed"], "classification": classification,
                  "calibration": c, "subgroups": groups, "robustness": robustness_metrics(perturbed)}
        reports[key] = report
        write_json(path / "metrics.json", report)
        write_csv(path / "confusion_matrix.csv", [{"true_label": LABELS[i], **dict(zip(LABELS, row))}
                  for i, row in enumerate(classification["confusion_matrix"])])
        # External reference remains in its own folder and JSON report, not pooled CSV/plots.
        if spec["kind"] != "reference":
            probability_sets[key] = p
            base = {"model": spec["kind"], "seed": spec["seed"]}
            per_seed_rows.append({**base, **{m: classification[m] for m in ("macro_f1", "accuracy")}})
            calibration_rows.append({**base, **{m: c[m] for m in ("ece_10", "ece_15", "brier", "nll", "mean_confidence")}})
            class_rows.extend({**base, "class": label, **values} for label, values in classification["per_class"].items())
            subgroup_rows.extend({**base, "subgroup": group, "n": values["n"], "class_counts": values["class_counts"],
                "macro_f1": values["metrics"]["macro_f1"] if values["metrics"] else None, "warnings": values["warnings"]}
                for group, values in groups.items())
    aggregate, robustness = {}, {}
    aggregate_rows = []
    for kind in ("baseline", "bert", "finbert"):
        seed_reports = [reports[f"{kind}-{s}"] for s in cfg["seeds"]]
        aggregate[kind] = {"classification": summarize([r["classification"] for r in seed_reports]),
            "calibration": {m: mean_sd([r["calibration"][m] for r in seed_reports]) for m in ("ece_10", "ece_15", "brier", "nll")},
            "subgroups": {g: {"n": seed_reports[0]["subgroups"][g]["n"],
                "class_counts": seed_reports[0]["subgroups"][g]["class_counts"], "warnings": seed_reports[0]["subgroups"][g]["warnings"],
                "macro_f1": mean_sd([r["subgroups"][g]["metrics"]["macro_f1"] for r in seed_reports])}
                for g in ("allagree", "non_allagree")}}
        robustness[kind] = {}
        for t in TYPES:
            values = [r["robustness"][t] for r in seed_reports]
            robustness[kind][t] = {"n": values[0]["n"], "visible_n_per_seed": [v["visible_n"] for v in values],
                **{m: mean_sd([v[m] for v in values]) if all(v[m] is not None for v in values) else None
                    for m in ("flip_rate", "visible_flip_rate", "mean_confidence_change", "mean_original_class_probability_change")}}
        for section, values in [("classification", aggregate[kind]["classification"]), ("calibration", aggregate[kind]["calibration"])]:
            aggregate_rows.extend({"model": kind, "section": section, "metric": m, "mean": v["mean"], "sample_sd": v["std"]}
                                  for m, v in values.items() if isinstance(v, dict) and "mean" in v)
    selected = select_errors(rows, probability_sets, cfg["error_examples"], cfg["confident_error_threshold"])
    if not smoke and len(selected) != cfg["error_examples"]:
        raise ValueError(f"Only {len(selected)} misclassified examples available for requested export")
    write_jsonl(out / "error_examples.jsonl", selected)
    write_csv(out / "human_annotation.csv", selected, list(selected[0]) if selected else ["row_id", "error_category", "interpretation", "comments"])
    error_detail = [{"row_id": r["row_id"], "true_label": r["true_label"], "model_seed": key, **value}
                    for r in selected for key, value in r["model_predictions"].items()]
    write_csv(out / "error_predictions.csv", error_detail, list(error_detail[0]) if error_detail else ["row_id", "model_seed"])
    summary = {"classification_and_calibration": aggregate, "robustness": robustness,
        "error_selection": {"n": len(selected), "confident_errors": sum(r["confident_error"] for r in selected),
            "disagreements": sum(r["bert_finbert_disagreement"] for r in selected),
            "policy": "Half-quota highest-confidence BERT/FinBERT paired-seed disagreements among errors; fill with highest-confidence errors. Not a representative random sample."},
        "external_reference": reports["reference"], "reference_warning": evidence["reference_warning"], "smoke_only": smoke}
    write_json(out / "summary.json", summary)
    write_json(out / "prediction_audit.json", prediction_audit)
    write_csv(out / "classification_per_seed.csv", per_seed_rows)
    write_csv(out / "per_class_per_seed.csv", class_rows)
    write_csv(out / "calibration_per_seed.csv", calibration_rows)
    write_csv(out / "subgroup_per_seed.csv", subgroup_rows)
    write_csv(out / "aggregate_metrics.csv", aggregate_rows)
    write_csv(out / "robustness_summary.csv", [{"model": kind, "type": t, **v} for kind, types in robustness.items() for t, v in types.items()])
    print("Rendering Phase 3 figures...", flush=True)
    plots(out, reports, aggregate, robustness, cfg["seeds"])
    return summary


def run(cfg, cpu=False, smoke=False):
    validate_config(cfg)
    if not smoke:
        receipt = read_json(cfg.get("smoke_receipt", "artifacts/phase3-smoke/smoke_receipt.json"))
        if receipt.get("status") != "passed" or receipt.get("code_fingerprint") != fingerprint():
            raise ValueError("Run CPU smoke on the current Phase 3 implementation first")
    roots = [cfg["phase1_dir"], cfg["phase2_dir"]]
    out, report = protected_output(cfg["output_dir"], roots, cfg["report_path"])
    start = time.perf_counter()
    write_json(out / "config.json", cfg)
    try:
        rows, models, evidence = load_inputs(cfg)
        roots.append(models[-1]["checkpoint"])
        print("Recording read-only input hashes...", flush=True)
        before = snapshot(roots)
        write_json(out / "input_hashes.json", before)
        write_json(out / "integrity.json", evidence)
        seed_everything(cfg["random_seed"])
        if cpu:
            torch.set_num_threads(2)
        if smoke:
            chosen = sorted([i for label in [0,1,2] for agreement in [True, False]
                for i in [j for j, r in enumerate(rows) if r["label"] == label and r["is_allagree"] == agreement][:2]])
            rows = [rows[i] for i in chosen]
            for spec in models:
                spec["probabilities"] = spec["probabilities"][chosen]
        result = analyze(cfg, rows, models, evidence, out, cpu=cpu, smoke=smoke)
        print("Verifying all earlier artifacts remain unchanged...", flush=True)
        if snapshot(roots) != before:
            raise RuntimeError("Earlier artifacts changed during Phase 3")
        write_json(out / "completion.json", {"status": "passed", "elapsed_seconds": time.perf_counter()-start,
            "all_prior_artifacts_unchanged": True, "runtime": runtime(), "code_fingerprint": fingerprint(),
            "test_rows_analyzed": len(rows), "device": "cpu" if cpu else "cuda", "smoke_only": smoke,
            "calibration_fitting": "none", "model_training": "none"})
        from .report import write_report
        write_report(out, report)
        if smoke:
            write_json(out / "smoke_receipt.json", {"status": "passed", "code_fingerprint": fingerprint(), "cpu_only": True,
                "models_checked": [m["key"] for m in models], "subset_rows": len(rows)})
        write_json(out / "manifest.json", {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob("*")) if p.is_file()})
        return out
    except BaseException as error:
        import traceback
        write_json(out / "failure.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
