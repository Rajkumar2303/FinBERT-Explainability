# Phase 3: read-only reliability analysis

Phase 3 reads the completed Phase 1/2 artifacts and pinned, locally cached
checkpoints. It does not train, tune, resplit, fit calibration parameters, or
change earlier artifacts. No Phase 4 attribution work is included.

Completed verification: **41 tests passed**, followed by a passing CPU smoke
test across all ten saved model instances and a passing full CUDA analysis of
727 test rows. The full run produced 11 figures, 588 eligible perturbations per
model instance, and 20 error examples with blank human-annotation fields. All
89 sealed output files were verified, and earlier input hashes were unchanged.
See [PHASE3_RESULTS.md](PHASE3_RESULTS.md) for measured results.

An initial test launch was blocked by an automatic approval-review usage limit;
that command did not execute. After resuming, the test, smoke and full-analysis
runs all completed successfully. There are no unresolved analysis failures.

## Reproduce in the existing environment

Run from the repository root. Dependencies are already present in `rag-env`;
do not reinstall PyTorch.

```powershell
conda activate rag-env
$env:PYTHONPATH = 'src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

python -m pytest -q -p no:cacheprovider --junitxml=artifacts/phase3-tests.xml
python -m finbert_explainability.phase3 smoke
python -m finbert_explainability.phase3 analyze
```

The smoke command forces CPU inference on 12 frozen test rows (two per class and
agreement group). It loads all ten saved model instances, checks probabilities,
and exercises exports and plots. It is not a separate test set or a source of
full-data conclusions. The full analysis requires a passing smoke receipt whose
source-code fingerprint matches the current Phase 3 implementation.

Both commands refuse to overwrite existing results. For a subsequent rerun use
new output/report paths and the corresponding new smoke receipt:

```powershell
python -m finbert_explainability.phase3 smoke --output artifacts/phase3-smoke-rerun
python -m finbert_explainability.phase3 analyze --output artifacts/phase3-rerun --report PHASE3_RESULTS_RERUN.md --smoke-receipt artifacts/phase3-smoke-rerun/smoke_receipt.json
```

Use `--cpu` for full CPU inference if desired. The default full run uses CUDA.
`--config configs/phase3.json` selects the analysis configuration. Only supported
arguments are exposed by `python -m finbert_explainability.phase3 --help`.

## Integrity and alignment

- Verify Phase 1 and all final Phase 2 training/evaluation manifests, required
  checkpoint and prediction entries, seed lists, class mappings and test hashes.
- Check every final validation and test prediction row: ID, label, subgroup,
  split, probability names/range/sum and predicted-class consistency.
- Recompute test classification metrics and compare them with Phase 2 outputs.
- Load saved models locally in evaluation/inference mode, with gradients disabled
  for neural models. Compare their original-test predictions with saved results;
  fail on class differences or probability differences above `1e-5`.
- Hash all files under Phase 1, Phase 2 and the cached reference checkpoint before
  and after analysis. Fail if any content or file inventory changes.
- Reject output/report paths inside, equal to, or containing protected Phase 1/2
  directories. Earlier results are never used as output locations.

All seeds remain fixed at 42, 123 and 2026. Original model length/batch settings
come from saved Phase 2 configs; Phase 3 does not alter them. `input_hashes.json`,
`integrity.json`, `prediction_audit.json`, `completion.json` and the new output
manifest record provenance and verification. Failures write `failure.json` where
the new output directory has already been created; failed runs remain visible.

## Metrics and plots

Classification is recomputed with canonical order negative/neutral/positive.
Confusion matrices use true labels as rows and predicted labels as columns.
Independent comparisons include TF-IDF, BERT and FinBERT. Three-seed summaries use
sample standard deviation (`ddof=1`), not a confidence interval. Per-class metrics
always use the same three labels, including zero support if encountered.

Agreement groups are measured separately. Class counts at or below 50 and groups
below 100 rows are flagged; differences are observational and are not evidence
of a causal effect of annotation agreement.

Calibration uses saved unmodified probabilities:

- Top-label ECE with 10 and 15 equal-width bins: sample-weighted absolute difference
  between bin accuracy and mean confidence. Bins are left-closed/right-open, with
  confidence 1 included in the last bin. Empty bins contribute zero.
- Multiclass Brier: mean sum of squared errors over all three classes, range 0–2.
- NLL: mean negative log true-class probability, clipped below at `1e-15`.

No temperature scaling is fitted, on validation or test data. Saved raw logits
were not present in Phase 2; new neural logits are obtained from read-only
inference, reordered canonically and saved in Phase 3 NPZ files. Baseline log
probabilities are not misrepresented as raw logits.

Figures in `artifacts/phase3/figures` include classification and per-class
comparisons, per-seed BERT/FinBERT confusion matrices, subgroup comparisons,
10/15-bin reliability diagrams, confidence histograms, calibration comparisons,
and robustness flip rates. ProsusAI's reliability figure and metrics are separate
and explicitly non-independent because its original training used PhraseBank.

## Perturbations and errors

Each original row receives an eligibility/skip record for each probe:

- **Company:** replace a configured, whole-token company name with a fixed
  pseudonym. The lexicon is limited and drawn from observed text, not exhaustive
  entity recognition. Identity-sensitive meaning can invalidate invariance.
- **Numerical:** multiply currency-prefixed amounts by two, preserving signs and
  ordering among changed amounts. Dates, percentages and other numbers remain
  untouched. Skip ambiguous ranges and decimal-comma forms; flag normalization
  of spaced decimal points and possible inconsistency with unscaled quantities.
- **Negation:** insert `not` after a supported auxiliary, avoiding existing
  negation and unsupported possession patterns. This is a possible sentiment
  change with no prescribed direction, not an automatic label reversal.

Every variant has unknown perturbed gold sentiment and explicit grammar/semantic
review flags. Original/perturbed text, classes, probabilities, confidence changes,
class flips, truncation and actual representation changes are exported. Report
both all-eligible and model-visible rates. An edit beyond the token cap or one
that leaves TF-IDF features unchanged is not counted in the visible denominator.
Flip rates are sensitivity measurements, not perturbed-label accuracy.

The 20-example error export prioritizes high-confidence errors and reserves half
the quota for paired-seed BERT/FinBERT disagreements. It is deliberately not a
representative random error sample. `human_annotation.csv` contains blank
`error_category`, `interpretation`, and `comments` fields. No human conclusions
are generated; the detailed probabilities are also in `error_predictions.csv`.

## Deliverable map

- `PHASE3_RESULTS.md`: automatically generated from measured full-run outputs.
- `artifacts/phase3/summary.json`, aggregate/per-seed CSVs: primary metrics.
- `<model>-<seed>/metrics.json`, `confusion_matrix.csv`: individual results.
- `<model>-<seed>/recomputed_inference.npz`, `inference_row_order.json`: numeric
  inference outputs and their row order.
- `perturbation_registry.jsonl/.csv`: all eligibility decisions and edits.
- `<model>-<seed>/robustness_predictions.jsonl/.csv`: detailed perturbation results.
- `reference/`: separate non-independent reference outputs.
- `human_annotation.csv`, `error_examples.jsonl`, `error_predictions.csv`: review handoff.
- `figures/`: standalone PNG charts.

This is a descriptive reanalysis of an already-evaluated test set. It does not
create a new independent holdout, establish statistical significance, or remove
all semantic/pretraining overlap. Limited perturbation coverage, unknown altered
labels, finite calibration bins and only three training seeds limit conclusions.
