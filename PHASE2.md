# Phase 2: frozen-split model evaluation

The approved full GPU experiments and all four test evaluations are complete.
See [PHASE2_RESULTS.md](PHASE2_RESULTS.md) for actual metrics, timings, peak VRAM,
and the per-class comparison. Existing Phase 1 Python modules, dataset files,
and split assignments are unchanged. Outputs live under `artifacts/phase2`.

## Phase 1 audit

`artifacts/phase2/audit/audit.json` records a checksum verification of every file in
the Phase 1 manifest, a new duplicate search, initial sklearn split replay, two
grouped-split replays, and exact agreement with saved assignments. All passed.

The 2,199 moved rows consist of **2,143 singletons and 56 rows in duplicate groups**.
Once any crossing pair is found, the Phase 1 algorithm rebuilds every assignment
using its seeded greedy class-balancing objective. It does not minimize movement
from the initial split. The movement count is therefore not a duplicate count or
an indication of randomness/data loss. Initial crossings: 25; final crossings: 0.
No split changes are proposed or made.

| Split | Rows | Negative | Neutral | Positive | Allagree | Allagree share |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,391 | 422 | 2,015 | 954 | 1,592 | 46.95% |
| Validation | 728 | 91 | 432 | 205 | 334 | 45.88% |
| Test | 727 | 91 | 432 | 204 | 338 | 46.49% |

Overall class proportions are approximately 12.46% negative, 59.41% neutral,
28.13% positive; split proportions closely match. Within the allagree subgroup,
negative/neutral/positive counts are **210/970/412**, **43/212/79**, and **50/209/79**
respectively. Allagree is not a stratification variable: its negative share is
13.19%, 12.87%, and 14.79% across the splits. Both unanimous and non-unanimous
distributions are preserved in the audit, and both receive separate model metrics.

## Protocol

- Canonical labels: negative=0, neutral=1, positive=2. All workflows load the same
  checksum-verified JSONL files from `artifacts/phase1`. No resplitting occurs.
- TF-IDF + Logistic Regression: training-only vocabulary/IDF fitting, word
  n-grams (1,1)/(1,2), C in 0.1/1/10, up to 20,000 features, deterministic lbfgs.
  Validation macro-F1 selects the candidate. A convergence warning fails the run.
- `google-bert/bert-base-uncased` and `yiyanghkust/finbert-pretrain`: each uses its
  own pinned tokenizer and pretrained encoder with a newly initialized sentiment
  head, fine-tuned via Hugging Face Trainer. The legacy FinBERT config lacks
  `model_type`; explicit `BertConfig` loading handles this in Transformers 5.
- Shared search: learning rates **2e-5, 3e-5** crossed with **3, 4 epochs**, with
  the initial candidate 2e-5/3 epochs. Each of four candidates uses seed 42.
  Selection is solely by validation macro-F1, with configuration order resolving
  ties. This is a small single-seed search, not nested cross-validation.
- After selection, each model is initialized from its original pretrained source
  and fitted independently with **42, 123, 2026**. Seed 42 is deliberately rerun
  rather than reusing search weights. All training is on training rows only;
  validation never joins the fitting data.
- Maximum length **64**; batch **8**; accumulation **2**; one GPU, effective
  batch **16** (the final partial batch is smaller); FP16 training on CUDA;
  AdamW, weight decay 0.01, linear scheduler, no warmup. Validation and checkpoint
  saving happen each epoch; best validation macro-F1 checkpoint is restored.
  Evaluation/inference is FP32 for consistent exported probabilities.
- Fixed seeds, deterministic algorithms, eager attention, TF32 off, and zero
  DataLoader workers suit native Windows. Determinism is scoped to the same
  hardware/software stack; CUDA nondeterministic operations fail rather than
  silently relaxing the requirement.
- Test loading is a separate `evaluate` command, possible only after a completed,
  sealed selection run. Three final checkpoints are all evaluated; test scores
  never choose a seed, checkpoint, hyperparameter, or winning model. `compare`
  rejects reports whose saved test/Phase 1 fingerprints differ.
- Each seed exports accuracy, macro-F1, per-class precision/recall/F1/support,
  a 3x3 confusion matrix (rows=true, columns=predicted), and row-ID predictions
  with canonical class probabilities. Aggregation uses sample SD (`ddof=1`) over
  three independent fits. It is not a confidence interval. lbfgs can legitimately
  have zero variance despite three independent seeded fits.

`ProsusAI/finbert` is inference-only, with no search or fine-tuning. Its inspected
`id2label` is **0=positive, 1=negative, 2=neutral**; probabilities are reordered to
the canonical label order before scoring. Unknown mappings fail explicitly.
Its [model card](https://huggingface.co/ProsusAI/finbert) documents Financial
PhraseBank training: our test rows may overlap that training set. This is a
**non-independent external reference**, separated from the independent workflow
reports and prominently marked in its metadata and warning. Three-seed variance
does not apply to a single fixed external checkpoint.

## Environment and commands

Verified existing Windows `rag-env`: Python 3.11.15, PyTorch 2.12.1+cu126,
CUDA 12.6, RTX 4070 Laptop GPU (~8 GB), Transformers 5.13.0, Accelerate 1.15.0.
No packages or PyTorch builds were replaced. `configs/phase2.json` pins model
revisions and all experiment settings. Runtime versions are saved per run.

Run from the repository root in PowerShell:

```powershell
conda activate rag-env
$env:PYTHONPATH = 'src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q -p no:cacheprovider --junitxml=artifacts/phase2/tests.xml

# Fresh output directories are mandatory; existing runs are never overwritten.
python -m finbert_explainability.phase2 audit --output artifacts/phase2/audit-rerun
python -m finbert_explainability.phase2 smoke --output artifacts/phase2/smoke-rerun

# CPU baseline: selection first, test evaluation second.
python -m finbert_explainability.phase2 train baseline --output artifacts/phase2/baseline
python -m finbert_explainability.phase2 evaluate --run artifacts/phase2/baseline --output artifacts/phase2/baseline-evaluation --cpu
```

The module entry point needs no installation. Optional console registration, using
existing dependencies only: `python -m pip install -e . --no-deps --no-build-isolation`.
Then `finbert-experiment` is equivalent to `python -m finbert_explainability.phase2`.

**The following full GPU commands were executed after explicit user approval.**
The approval flag alone is not a substitute for obtaining that approval.
The code also verifies a successful CPU smoke receipt matching current Phase 2
source. The verified receipt is `artifacts/phase2/smoke-final/smoke_receipt.json`;
pass `--smoke-receipt` to use a new receipt after any code changes.

```powershell
python -m finbert_explainability.phase2 train bert --output artifacts/phase2/bert --approve-training
python -m finbert_explainability.phase2 train finbert --output artifacts/phase2/finbert --approve-training

# After both selection runs are frozen:
python -m finbert_explainability.phase2 evaluate --run artifacts/phase2/bert --output artifacts/phase2/bert-evaluation
python -m finbert_explainability.phase2 evaluate --run artifacts/phase2/finbert --output artifacts/phase2/finbert-evaluation

# External reference: inference only, not an independent result.
python -m finbert_explainability.phase2 reference --output artifacts/phase2/reference
python -m finbert_explainability.phase2 compare --runs artifacts/phase2/baseline-evaluation artifacts/phase2/bert-evaluation artifacts/phase2/finbert-evaluation artifacts/phase2/reference --output artifacts/phase2/comparison
```

Do not leave `HF_HUB_OFFLINE=1` set when first downloading pretrained weights.
Once cached, pinned artifacts can be reused offline. Configs/tokenizers have been
checked against the real upstream repositories. The pinned pretrained weights
are now cached from the completed approved experiments.

## Outputs and verification

Each training run saves effective config, runtime metadata, `run.log`, incremental
`search_records.json`, per-candidate validation predictions/metrics, and three
seed directories. Neural directories include validation checkpoints, best model
plus tokenizer, and `training_record.json` with Trainer arguments/log history.
`selection.json` records the winner, seed list, source-code fingerprint, and frozen
dataset hashes. `run_manifest.json` protects every completed run file. Evaluation
saves seed predictions, metrics, aggregated summaries, and an evaluation record.
`compare` collects results with the same test fingerprint without ranking them.

CPU smoke uses synthetic data and tiny randomly initialized local BERT MLMs,
including a config missing `model_type`. It exercises all four paths, checkpoint
save/reload, validation selection, three seeds, prediction export, and comparison.
It asserts identical weight hashes for repeated seed-42 training and distinct
weights across seeds. It forces CPU/FP32 even on a CUDA machine. Synthetic metrics
are plumbing checks, **not accuracy results for the requested pretrained models**.

The full pytest suite includes Phase 1 regression tests, label permutations,
train-only TF-IDF vocabulary, validation-only selection, metrics/sample SD,
checkpoint reload, artifact tampering, protected paths, config validation,
training approval gates, and mismatched-test comparison rejection. Tests need
permission to create temporary files; the managed sandbox previously blocked that
and the successful runs used approved filesystem access.

Completed verification: **26 tests passed** (`artifacts/phase2/tests.xml`), and
the final offline CPU smoke passed (`artifacts/phase2/smoke-final/smoke_receipt.json`).
Its source fingerprint matches the delivered implementation. A final Phase 1
manifest verification confirmed every saved artifact is unchanged. Both neural
smoke paths reproduced identical weights for repeated seed 42 and distinct
weights for the three independent seeds. The later approved CUDA experiments
also reproduced byte-identical seed-42 weights for both models; all real test
predictions and metrics passed `scripts/verify_phase2_results.py`.

## Planning estimates and limitations

These are estimates, not measured GPU benchmarks:

| Resource | Budget |
|---|---|
| Full neural runs | 4 search + 3 final fits per model; 14 fits total |
| Training work | 46–52 total epochs; about 9,752–11,024 optimizer steps |
| GPU time | Roughly 0.5–3 hours total; laptop power/thermals and I/O can increase this |
| VRAM | Plan for roughly 4–7 GB; actual peak must be measured after approval |
| System RAM | 16 GB recommended |
| Disk | Reserve 20–30 GB for caches, search/final models, and checkpoints |
| CPU baseline | Usually seconds to a few minutes on this small corpus |

Models run sequentially. `save_total_limit=1` can retain best and last checkpoints;
`save_only_model=True` avoids optimizer-state disk cost. Final selected models are
saved separately. This deliberately does **not** support exact optimizer-state
resume: interrupted runs remain incomplete and require a new output directory.
The completed run's measured memory/time requirements are in PHASE2_RESULTS.md;
the CPU smoke alone cannot validate CUDA FP16 behavior or laptop thermal performance.

The fixed 64-token cap can truncate long sentences. Three seeds give a limited
variance estimate; validation is reused for search and epoch selection. Phase 1
group splitting prevents detected lexical duplicates crossing splits, not every
semantic paraphrase or pretraining-corpus overlap. Do not alter the saved test set
or revisit tuning after seeing its scores. Actual achieved metrics are reported
separately in PHASE2_RESULTS.md, with the reference clearly marked non-independent.

## Measurement and reporting commands used for the approved run

`scripts/run_measured.py` calls the existing CLI without changing its settings or
implementation. Its first positional argument is a new telemetry JSON path; the
remaining arguments are the supported CLI command. For example:

```powershell
python scripts/run_measured.py artifacts/phase2/execution/bert.json train bert --config configs/phase2.json --output artifacts/phase2/bert --approve-training
python scripts/verify_phase2_results.py
python scripts/report_phase2.py
```

The existing outputs are successful runs and must not be overwritten. Use new
output/telemetry paths for any separately authorized rerun. The CUDA sanity,
console logs, per-fit resource measurements, and repeatability checks are saved
under `artifacts/phase2`; no Phase 3 implementation was started.
