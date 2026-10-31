# Phase 3 measured reliability analysis

All results below were measured on the unchanged frozen test set. No models, hyperparameters, calibration parameters, or splits were fitted or modified.

Analyzed 727 test rows with three seeds per independent workflow. Values with ± use sample SD (ddof=1); these are seed-variation summaries, not confidence intervals or significance tests.

## Classification

| Model | Macro-F1 | Accuracy |
|---|---:|---:|
| baseline | 0.6713 ± 0.0000 | 0.7442 ± 0.0000 |
| bert | 0.8476 ± 0.0034 | 0.8556 ± 0.0014 |
| finbert | 0.8339 ± 0.0093 | 0.8469 ± 0.0097 |

Per-seed confusion matrices, per-class precision/recall/F1/support and their three-seed summaries are saved in metrics JSON/CSV and the figures directory. They were recomputed from saved probabilities and checked against Phase 2 metrics.

## Agreement subgroups

| Subgroup | N | Negative / neutral / positive | Count warnings |
|---|---:|---|---|
| allagree | 338 | 50 / 209 / 79 | small_class:negative:50 |
| non_allagree | 389 | 41 / 223 / 125 | small_class:negative:41 |

| Model | Allagree macro-F1 | Non-allagree macro-F1 |
|---|---:|---:|
| baseline | 0.8163 ± 0.0000 | 0.5182 ± 0.0000 |
| bert | 0.9583 ± 0.0094 | 0.7500 ± 0.0142 |
| finbert | 0.9553 ± 0.0026 | 0.7262 ± 0.0156 |

Subgroups below 100 rows and class counts at or below 50 are flagged. Agreement is an observational subgroup, not a randomized intervention; composition and annotation ambiguity can affect its scores. Do not infer causation or precise population effects.

## Raw probability calibration

| Model | ECE (10 bins) | ECE (15 bins) | Multiclass Brier | NLL |
|---|---:|---:|---:|---:|
| baseline | 0.0701 ± 0.0000 | 0.0730 ± 0.0000 | 0.3674 ± 0.0000 | 0.6320 ± 0.0000 |
| bert | 0.1064 ± 0.0037 | 0.1088 ± 0.0028 | 0.2468 ± 0.0010 | 0.5115 ± 0.0039 |
| finbert | 0.0923 ± 0.0556 | 0.0944 ± 0.0570 | 0.2620 ± 0.0221 | 0.6022 ± 0.1669 |

ECE uses equal-width confidence bins and the sample-weighted absolute gap between top-label accuracy and confidence; intervals are [lower, upper), with 1 included in the last bin. Empty bins contribute zero and are omitted from reliability curves, with all bin counts saved. ECE depends on binning and finite-sample occupancy. NLL clips true-class probabilities below 1e-15. Brier is the mean sum of squared errors across three classes (range 0–2), not divided by two or by the number of classes. Lower is better for all four metrics.

No temperature scaling was implemented or fitted. Calibration uses the original saved probabilities; raw logits were not present in Phase 2 prediction exports. Canonically ordered logits were recomputed read-only for neural checkpoints and stored in Phase 3 NPZ files, without claiming that log probabilities are raw logits. Baseline raw logits are not exported.

Method references: [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html), [scikit-learn multiclass Brier convention](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_multiclass.html).

## Deterministic robustness probes

| Type | Eligible / test rows | Expected behavior |
|---|---:|---|
| company | 104 / 727 | Conditional invariance, subject to semantic review |
| negation | 375 / 727 | Possible sentiment change; no prescribed target class |
| numerical | 109 / 727 | Conditional invariance, subject to semantic review |

| Model | Company visible flip rate | Numerical visible flip rate | Negation visible flip rate |
|---|---:|---:|---:|
| baseline | 0.2019 ± 0.0000 | 0.0769 ± 0.0000 | 0.0640 ± 0.0000 |
| bert | 0.0128 ± 0.0222 | 0.0031 ± 0.0053 | 0.2255 ± 0.0216 |
| finbert | 0.0192 ± 0.0096 | 0.0000 ± 0.0000 | 0.1889 ± 0.0312 |

Each variant retains original/perturbed text, rule edits, skip reasons, review flags, classes, probabilities, confidence changes, original-class probability changes, truncation flags and whether the actual model representation changed. Both all-eligible and model-visible denominators/flip rates are saved. Invisible changes are excluded only from the visible-rate column above; denominators can differ across tokenizers/models.

Company edits use the configured exact-name lexicon and pseudonym `Example Holdings`. It is a convenience lexicon chosen from text, not exhaustive entity recognition. Monetary edits multiply currency-prefixed amounts by 2, preserving signs and relative ordering among those amounts; dates, percentages, and other unmarked numbers remain untouched. Ambiguous ranges/decimal-comma forms are skipped. Source spaced decimals are explicitly flagged when normalized. Unscaled quantities or magnitude-dependent meaning may invalidate invariance. Negation inserts 'not' at a supported auxiliary and skips existing negation; scope/grammar and financial sentiment remain unvalidated. Negating losses can improve sentiment, and negating a factual clause need not flip the class.

These are sensitivity measurements, not adversarial accuracy or human-validated label changes. Every generated variant is marked for semantic/grammar review, and perturbed gold labels remain unknown. Some original source sentences have encoding/grammar issues. Low flip rates are not proof of semantic robustness; high negation flip rates are not automatically desirable.

## Error review handoff

Exported 20 unique misclassified rows, including 20 errors at confidence ≥ 0.9 and 10 rows with paired-seed BERT/FinBERT disagreements. Selection policy: Half-quota highest-confidence BERT/FinBERT paired-seed disagreements among errors; fill with highest-confidence errors. Not a representative random sample.

`human_annotation.csv` contains original text, true labels, per-model predictions/confidence/probabilities and empty `error_category`, `interpretation`, `comments` fields. `error_predictions.csv` provides a long-form model-by-example view. No human annotations, error causes, or qualitative conclusions have been invented. The confidence-focused sample is not representative of all errors.

## Non-independent external reference

NON-INDEPENDENT EXTERNAL REFERENCE: ProsusAI/finbert was fine-tuned on Financial PhraseBank. The saved test split may overlap its original training data; these results must not be treated as an independent held-out comparison.

ProsusAI macro-F1=0.8797; accuracy=0.8900; ECE10=0.0241; ECE15=0.0316; Brier=0.1693; NLL=0.3036. Single fixed checkpoint: no seed SD. It is excluded from the independent comparison figures and pooled CSVs; its metrics, perturbations and reliability figure are separate.

## Integrity, artifacts and limits

All earlier artifact hashes remained unchanged. Inference reproduced every saved class; maximum absolute probability difference across models was 4.39e-07. Analysis completed in 64.9 seconds on cuda using the existing environment.

`integrity.json`, `input_hashes.json`, `prediction_audit.json`, and `completion.json` record input manifests, checkpoint mappings/reproduction and runtime. `manifest.json` seals new outputs. Charts are in `artifacts/phase3/figures` for the full run; per-seed metrics and perturbation predictions are in model/seed subdirectories. No earlier successful results were overwritten.

Limits include the fixed 64-token cap, lexicon/rule coverage, unknown perturbed labels, small class/bin counts, only three training seeds, and possible semantic/pretraining overlap not removed by lexical duplicate grouping. This is a descriptive reanalysis of an already-evaluated frozen test set, not a new independent holdout. No significance, causality, or human error-category claims are made. No retraining, test-set calibration fitting, or Phase 4 Integrated Gradients work occurred.
