# Phase 5: empirical IG and attention comparison

This phase compares the existing FinBERT seed-42 checkpoint's Integrated Gradients and final-layer CLS attention. It does not train, retune, select a new checkpoint, alter frozen data, or publish anything. Code is in `src/finbert_explainability/phase5`, tests in `tests/phase5`, and outputs in `artifacts/phase5`. Phases 1–4 remain read-only.

## Reproduction

Run from the project root with the existing rag-env. No package installation or PyTorch replacement is needed.

```powershell
$phase5Python = 'C:\Users\rajku\anaconda3\envs\rag-env\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
& $phase5Python -m finbert_explainability.phase5 run --help
& $phase5Python -m pytest tests/phase5 -q -p no:cacheprovider --basetemp artifacts/phase5/test-temp-new
& $phase5Python -m finbert_explainability.phase5 cpu-smoke --output artifacts/phase5/cpu-smoke
& $phase5Python -m finbert_explainability.phase5 run --config configs/phase5/cuda-smoke.json --smoke
& $phase5Python -m finbert_explainability.phase5 run --config configs/phase5.json
```

Smoke directories must be new. CPU smoke processes ten synthetic examples with a tiny randomly initialized BERT, using the real IG/attention/deletion functions without training. CUDA smoke uses the exact saved FinBERT and ten fixed Phase 4 examples, verifies all 727 predictions, reuses their validated IG records, and exercises attention, deletion, statistics, plots and integrity checks. Full execution uses all 727 frozen examples. Windows pytest temporary-directory permissions may require execution with normal user filesystem access outside the restricted sandbox.

After interruption, use exactly the same source, environment and configuration:

```powershell
& $phase5Python -m finbert_explainability.phase5 run --config configs/phase5.json --resume
```

Successful IG and example records are independently checksummed and reused. The complete run manifest is verified on subsequent resumes, with no overwrite. Partial invalid JSON is reported rather than silently removed; an incomplete NPZ is likewise a visible failure. Completed attention-only files are checked against recomputation if a later step was interrupted. A changed configuration, source, environment, or earlier-input snapshot requires a new output directory and report path. `progress.jsonl` records progress; per-example files bound memory use. Failure JSON includes the traceback and earlier-file preservation status.

## Inputs and frozen IG protocol

`configs/phase5.json` references the unchanged `configs/phase4.json` and sealed `artifacts/phase4/initial` run. This pins `artifacts/phase2/finbert/seed-42/model`, label mapping negative=0, neutral=1, positive=2, the Phase 1 manifest and test hashes, original tokenizer, and the Phase 4 implementation source hashes. The main checkpoint was not selected by test performance.

The Phase 4 `explain` implementation is imported directly, without copying or changing its attribution algorithm. It uses Captum LayerIntegratedGradients at `model.bert.embeddings`, predicted-class logits, a PAD-content baseline preserving special/padding positions and masks, and the fixed 50/100/200-step schedule. Completeness tolerances remain .001 absolute and .01 relative with a 1e-6 denominator floor. No test findings change these choices. Signed and absolute token scores, token IDs, attention masks, reference IDs, all integration attempts, and convergence flags are retained for every example.

The ten completed PAD records from Phase 4 are reused, not recalculated. Additional existing compatible Phase 4 runs can be listed in `additional_ig_runs` before starting a new run. Existing Phase 5 IG records are resumed before extraction/deletion. Reused records are checked against source hashes, exact IDs/text/labels, current tokenization, baseline IDs, predictions, logits, and convergence metadata. Every frozen prediction is reproduced against Phase 2 probabilities and Phase 3 logits before processing begins, using the existing 1e-5 / 1e-4 tolerances.

All earlier scientific artifacts, configurations, source, tests and documents are hashed before and after. Old pytest scratch directories with Windows ACL issues are explicitly listed as excluded from the hash inventory; no writes to them are performed. The user-edited Phase 3 annotation CSV is preserved and its pre-existing manifest mismatch remains disclosed. No earlier manifest is repaired or rewritten.

Unconverged IG records remain available and are listed in the results. Primary agreement and paired deletion comparisons use the converged subset; all-record sensitivity summaries are also exported. Excluding numerical failures can itself change the analyzed population, so counts and IDs are explicit. Attention mass summaries use all examples.

## Attention and token alignment

The saved model must dynamically verify as BERT with 12 layers and 12 heads. It runs in `eval()` mode with eager attention and `output_attentions=True`. Each tensor must have shape `(1, heads, sequence_length, sequence_length)`. The reported score is the final layer's CLS-query row averaged over its 12 heads. Attention-enabled logits/probabilities must match IG and prior inference. Padding attention and head row sums are checked.

This attention summary is not conditioned on a chosen output class, whereas IG targets the original predicted-class logit. Agreement therefore compares two different quantities; neither their agreement nor disagreement alone establishes an explanation's validity.

Each sentence's compressed NPZ retains the raw CLS-to-token row for **all layers and all heads**, active token IDs and positions, and the final-layer head mean. Padding is excluded from stored arrays; CLS, SEP and other special tokens remain. Complete query-key attention matrices can be enabled with `save_full_attention` in a fresh run, but are disabled by default to avoid unnecessary disk use. This does not omit the raw values needed to reproduce the chosen attention score. Only one sentence's large tensors are held at a time.

Filtering is identical for correlation and deletion rankings:

- Padding and all tokenizer special IDs are excluded.
- A WordPiece is punctuation if, after removing its leading `##`, every character has a Unicode category starting with `P`. Such tokens are excluded.
- Digits, numeric WordPieces, currency/math symbols and other content remain eligible. Decimal dots, commas and percent symbols categorized as punctuation are excluded.
- Original WordPieces are retained; no word aggregation alters comparison scores. Deleting a partial WordPiece is a known limitation.

Special-token analysis partitions each final-layer head's CLS attention mass into CLS, SEP, punctuation, ordinary content, other special tokens, and padding. The categories sum to one before filtering. Per-head and per-sentence values are exported. Attention on contextual special tokens is not by itself evidence of unfaithfulness.

## Agreement statistics

Primary comparisons correlate absolute signed-sum IG importance with attention. Signed IG is a separate sensitivity analysis. Spearman uses average ranks for ties; Kendall uses tau-b. Fewer than three eligible tokens and constant vectors produce an explicit undefined status, not zero. Summary JSON/CSV includes counts, means, medians, sample SDs, and percentile-bootstrap intervals. Correlation alone does not establish explanation faithfulness.

## Token masking and AUC

Rankings are fixed from the unmodified input. The target is its original predicted class at every budget, even after a label flip. Budgets are `[0, .05, .1, .2, .3, .5, .75, 1]`; count = `ceil(budget × eligible_count)` except zero. Actual masked counts/fractions are exported. Repeated counts and identical perturbations are deduplicated for deterministic batch inference. Token IDs are replaced by MASK without shifting positions or changing the attention mask.

Methods are descending absolute IG, descending attention, five random orderings (seeds 11/23/37/53/71, combined with row ID using SHA256), and positive-only signed IG. Score ties use original token position. The positive-only method ranks strictly positive IG scores and stops once exhausted. Saturation and actual coverage are recorded. Partial-area values are exported, but incomplete-coverage AUC is undefined and never compared as a full-coverage AUC. The other methods share all eligible tokens and have identical 100%-mask endpoints.

Each budget exports original-target probability and logit, their drops relative to unmodified input, full class probabilities/logits, mask positions, predicted class, and label-flip status. Drops may be negative and are not clipped. AUC uses trapezoidal integration over unique **actual** masked fractions; lower probability AUC means stronger suppression in this MASK intervention. Probability-drop and logit-drop AUCs are also exported. Label flips are relative to the original prediction, not the gold label.

Random-order mean and sample SD are computed within each sentence before between-sentence summaries. Random replicates are not treated as independent test samples. MASK replacement introduces distribution shift, and these are imperfect interventions rather than ground-truth explanation fidelity. Removing high absolute IG can remove inhibitory evidence; the positive-only analysis makes that distinction visible.

## Uncertainty, figures, and interpretation

The prespecified uncertainty method is 2,000 percentile-bootstrap resamples of sentences, seed 2042. Paired AUC differences resample the within-sentence differences. Curve intervals are pointwise and not multiplicity-adjusted. Heads, tokens and random-order replicates are not sampling units. Similar topics/phrasing can induce dependence, and a single frozen benchmark/model seed limits generalizability.

Figures include absolute/signed Spearman and Kendall distributions, probability/logit/flip deletion curves, AUC distributions and paired intervals, special-token/head attention distributions, token heatmaps, and example-level disagreement plots. Representative heatmaps select the first three correct and first three incorrect row IDs. Two separately labeled lowest-correlation cases show descriptive extremes; they are not claimed to be representative. Heatmap columns are independently scaled because attention and logit attribution have different units.

`PHASE5_RESULTS.md` is generated only from measured outputs. Main exports include `summary.json`, `correlations.csv`, `deletion.csv`, `auc.csv`, `attention_mass.csv`, `ig_convergence.csv`, per-example IG/analysis JSON, compressed attention NPZs, `figure_selection.json`, and output checksums. Runtime and peak CUDA allocated/reserved memory are recorded. Checkpoints are loaded once and never duplicated. The all-layer CLS attention storage is small; full attention matrices, if requested, are substantially larger. Full-run IG may take tens of minutes; measured runtime, not a projection, is reported at completion.

The comparison is related to, but does not reproduce all experiments from [Jain and Wallace (2019)](https://aclanthology.org/N19-1357/) and [Wiegreffe and Pinter (2019)](https://aclanthology.org/D19-1002/). Their claims depend on explanatory definitions, architectures, tasks and controls. This phase does not perform adversarial attention training, alternate-attention interventions, or multi-model-seed attention analyses. It cannot establish a universal claim that attention is or is not an explanation. Phase 6 and GitHub publication are outside scope.
