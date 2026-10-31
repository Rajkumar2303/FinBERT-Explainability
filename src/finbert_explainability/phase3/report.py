from .io import read_json


def fmt(value):
    if value is None:
        return "N/A"
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def write_report(out, path):
    summary = read_json(out / "summary.json")
    aggregate = summary["classification_and_calibration"]
    coverage = read_json(out / "perturbation_coverage.json")
    completed = read_json(out / "completion.json")
    cfg = read_json(out / "config.json")
    audit = read_json(out / "prediction_audit.json")
    lines = ["# Phase 3 measured reliability analysis", "",
        "**CPU smoke subset only; not full experimental results.**" if summary["smoke_only"] else "All results below were measured on the unchanged frozen test set. No models, hyperparameters, calibration parameters, or splits were fitted or modified.", "",
        f"Analyzed {completed['test_rows_analyzed']} test rows with three seeds per independent workflow. Values with ± use sample SD (ddof=1); these are seed-variation summaries, not confidence intervals or significance tests.", "",
        "## Classification", "", "| Model | Macro-F1 | Accuracy |", "|---|---:|---:|"]
    for kind, r in aggregate.items():
        lines.append(f"| {kind} | {fmt(r['classification']['macro_f1'])} | {fmt(r['classification']['accuracy'])} |")
    lines += ["", "Per-seed confusion matrices, per-class precision/recall/F1/support and their three-seed summaries are saved in metrics JSON/CSV and the figures directory. They were recomputed from saved probabilities and checked against Phase 2 metrics.", "",
        "## Agreement subgroups", "", "| Subgroup | N | Negative / neutral / positive | Count warnings |", "|---|---:|---|---|"]
    for name, g in aggregate["bert"]["subgroups"].items():
        counts = g["class_counts"]
        lines.append(f"| {name} | {g['n']} | {counts['negative']} / {counts['neutral']} / {counts['positive']} | {', '.join(g['warnings']) or 'none'} |")
    lines += ["", "| Model | Allagree macro-F1 | Non-allagree macro-F1 |", "|---|---:|---:|"]
    for kind, r in aggregate.items():
        lines.append(f"| {kind} | {fmt(r['subgroups']['allagree']['macro_f1'])} | {fmt(r['subgroups']['non_allagree']['macro_f1'])} |")
    lines += ["", f"Subgroups below {cfg['small_subgroup_count']} rows and class counts at or below {cfg['small_class_count']} are flagged. Agreement is an observational subgroup, not a randomized intervention; composition and annotation ambiguity can affect its scores. Do not infer causation or precise population effects.", "",
        "## Raw probability calibration", "", "| Model | ECE (10 bins) | ECE (15 bins) | Multiclass Brier | NLL |", "|---|---:|---:|---:|---:|"]
    for kind, r in aggregate.items():
        lines.append(f"| {kind} | " + " | ".join(fmt(r["calibration"][m]) for m in ["ece_10", "ece_15", "brier", "nll"]) + " |")
    lines += ["", "ECE uses equal-width confidence bins and the sample-weighted absolute gap between top-label accuracy and confidence; intervals are [lower, upper), with 1 included in the last bin. Empty bins contribute zero and are omitted from reliability curves, with all bin counts saved. ECE depends on binning and finite-sample occupancy. NLL clips true-class probabilities below 1e-15. Brier is the mean sum of squared errors across three classes (range 0–2), not divided by two or by the number of classes. Lower is better for all four metrics.", "",
        "No temperature scaling was implemented or fitted. Calibration uses the original saved probabilities; raw logits were not present in Phase 2 prediction exports. Canonically ordered logits were recomputed read-only for neural checkpoints and stored in Phase 3 NPZ files, without claiming that log probabilities are raw logits. Baseline raw logits are not exported.", "",
        "Method references: [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html), [scikit-learn multiclass Brier convention](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html).", "",
        "## Deterministic robustness probes", "", "| Type | Eligible / test rows | Expected behavior |", "|---|---:|---|"]
    for kind, r in coverage.items():
        expected = "Possible sentiment change; no prescribed target class" if kind == "negation" else "Conditional invariance, subject to semantic review"
        lines.append(f"| {kind} | {r['eligible']} / {r['total']} | {expected} |")
    lines += ["", "| Model | Company visible flip rate | Numerical visible flip rate | Negation visible flip rate |", "|---|---:|---:|---:|"]
    for kind, types in summary["robustness"].items():
        lines.append(f"| {kind} | " + " | ".join(fmt(types[t]["visible_flip_rate"]) for t in ["company", "numerical", "negation"]) + " |")
    lines += ["", "Each variant retains original/perturbed text, rule edits, skip reasons, review flags, classes, probabilities, confidence changes, original-class probability changes, truncation flags and whether the actual model representation changed. Both all-eligible and model-visible denominators/flip rates are saved. Invisible changes are excluded only from the visible-rate column above; denominators can differ across tokenizers/models.", "",
        f"Company edits use the configured exact-name lexicon and pseudonym `{cfg['replacement_company']}`. It is a convenience lexicon chosen from text, not exhaustive entity recognition. Monetary edits multiply currency-prefixed amounts by {cfg['monetary_scale']}, preserving signs and relative ordering among those amounts; dates, percentages, and other unmarked numbers remain untouched. Ambiguous ranges/decimal-comma forms are skipped. Source spaced decimals are explicitly flagged when normalized. Unscaled quantities or magnitude-dependent meaning may invalidate invariance. Negation inserts 'not' at a supported auxiliary and skips existing negation; scope/grammar and financial sentiment remain unvalidated. Negating losses can improve sentiment, and negating a factual clause need not flip the class.", "",
        "These are sensitivity measurements, not adversarial accuracy or human-validated label changes. Every generated variant is marked for semantic/grammar review, and perturbed gold labels remain unknown. Some original source sentences have encoding/grammar issues. Low flip rates are not proof of semantic robustness; high negation flip rates are not automatically desirable.", "",
        "## Error review handoff", "",
        f"Exported {summary['error_selection']['n']} unique misclassified rows, including {summary['error_selection']['confident_errors']} errors at confidence ≥ {cfg['confident_error_threshold']} and {summary['error_selection']['disagreements']} rows with paired-seed BERT/FinBERT disagreements. Selection policy: {summary['error_selection']['policy']}", "",
        "`human_annotation.csv` contains original text, true labels, per-model predictions/confidence/probabilities and empty `error_category`, `interpretation`, `comments` fields. `error_predictions.csv` provides a long-form model-by-example view. No human annotations, error causes, or qualitative conclusions have been invented. The confidence-focused sample is not representative of all errors.", "",
        "## Non-independent external reference", ""]
    ref = summary["external_reference"]
    lines += [summary["reference_warning"], "",
        f"ProsusAI macro-F1={ref['classification']['macro_f1']:.4f}; accuracy={ref['classification']['accuracy']:.4f}; ECE10={ref['calibration']['ece_10']:.4f}; ECE15={ref['calibration']['ece_15']:.4f}; Brier={ref['calibration']['brier']:.4f}; NLL={ref['calibration']['nll']:.4f}. Single fixed checkpoint: no seed SD. It is excluded from the independent comparison figures and pooled CSVs; its metrics, perturbations and reliability figure are separate.", "",
        "## Integrity, artifacts and limits", "",
        f"All earlier artifact hashes remained unchanged. Inference reproduced every saved class; maximum absolute probability difference across models was {max(v['max_probability_difference'] for v in audit.values()):.3g}. Analysis completed in {completed['elapsed_seconds']:.1f} seconds on {completed['device']} using the existing environment.", "",
        "`integrity.json`, `input_hashes.json`, `prediction_audit.json`, and `completion.json` record input manifests, checkpoint mappings/reproduction and runtime. `manifest.json` seals new outputs. Charts are in `artifacts/phase3/figures` for the full run; per-seed metrics and perturbation predictions are in model/seed subdirectories. No earlier successful results were overwritten.", "",
        "Limits include the fixed 64-token cap, lexicon/rule coverage, unknown perturbed labels, small class/bin counts, only three training seeds, and possible semantic/pretraining overlap not removed by lexical duplicate grouping. This is a descriptive reanalysis of an already-evaluated frozen test set, not a new independent holdout. No significance, causality, or human error-category claims are made. No retraining, test-set calibration fitting, or Phase 4 Integrated Gradients work occurred.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
