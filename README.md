# FinBERT Explainability — Phase 1

Reproducible data preparation only. No model is loaded or trained.

## Run in the existing environment

```powershell
conda activate rag-env
# All dependencies are already available. Do not reinstall PyTorch.
$env:PYTHONPATH = 'src'
python -m finbert_explainability --config configs/phase1.json
# Optional: register the console command without resolving/installing dependencies.
python -m pip install -e . --no-deps --no-build-isolation
finbert-prepare --config configs/phase1.json --output artifacts/another-run

# Disable unrelated pytest plugins installed in this shared environment.
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest -q -p no:cacheprovider
```

Run from the repository root; configuration paths are relative to the working directory.
Output directories must be empty, protecting existing runs from accidental overwrite.
For offline reruns, pass `--archive path/to/FinancialPhraseBank-v1.0.zip`.
The SHA256 is checked for both downloads and local archives.

## Data and leakage policy

The [original Financial PhraseBank repository](https://huggingface.co/datasets/takala/financial_phrasebank)
is pinned to an immutable commit in `configs/phase1.json`. The original Latin-1 ZIP
is parsed into Hugging Face Datasets without executing its legacy remote Python
script. Labels are negative=0, neutral=1, positive=2. The dataset's
CC BY-NC-SA 3.0 license applies to the source and derived data.

1. Make a seeded, two-stage scikit-learn stratified 70/15/15 split.
2. Identify allagree membership using normalized sentence **and label**, validating
   that every unanimous key occurs in the 50agree source. This is a subgroup flag,
   not additional training examples. Stable row IDs retain the original row index.
3. Detect normalized exact duplicates (Unicode NFKC, casefold, collapsed whitespace)
   and near duplicates using character-within-word TF-IDF 3–5 grams and cosine
   similarity >= 0.90. Search all pairs in bounded query batches. Threshold is fixed
   before training and is not tuned on test performance.
4. If any pair crosses splits, regroup the entire corpus using connected components,
   including transitive chains and conflicting labels. Assign largest groups first
   with seeded tie ordering, minimizing incremental class-count error against
   70/15/15 targets. Keep every row exactly once. Exact quotas can be incompatible
   with intact groups, so report achieved ratios and deviations explicitly.
5. Audit all detected pairs against final assignments and fail if any cross splits.

Near-duplicate detection is a lexical heuristic: unrelated templates can be grouped,
and semantic paraphrases can be missed. Zero reported leakage means zero crossings
under this declared detector. Corpus-wide TF-IDF here is used only for decontamination;
it is not a feature transformer for a downstream model. Fit downstream preprocessing
on training data only, and keep the test split fixed.

## Artifacts

`artifacts/phase1/` contains:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: text, label, stable row ID,
  source index, unanimous subgroup flag, component ID, and split.
- `dataset/`: Hugging Face DatasetDict, load with `datasets.load_from_disk(...)`.
- `statistics.json`: class counts/proportions, split sizes/deviations, subgroup distributions.
- `leakage_report.json`: every duplicate pair, similarities, label conflicts,
  initial crossings, final audit, and moved-row count. Pair indices refer to source indices.
- `initial_assignments.json`: baseline stratified assignments for comparison.
- `config.json`, `metadata.json`: effective settings, source hash/revision, label mapping,
  algorithm, Python/package versions, and CUDA availability.
- `manifest.json`: SHA256 checksums for all other run files.

Reproducibility requires the same source, configuration, and package versions recorded
in metadata. JSONL content is stable across reruns; metadata records runtime differences.
PyTorch, NumPy, and Python seeds are set, with deterministic PyTorch algorithms enabled.

## Verified dataset run

The 4,846 examples contain 604 negative, 2,879 neutral, and 1,363 positive rows;
2,264 are in the unanimous subgroup. Final split counts are:

| Split | Rows | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 3,391 | 422 | 2,015 | 954 |
| Validation | 728 | 91 | 432 | 205 |
| Test | 727 | 91 | 432 | 204 |

The detector found 8 exact and 49 near-duplicate pairs. There were 25 crossing pairs
in the initial split and zero in the final split. Group reassignment moved 2,199 rows.
Python 3.11.15 and the existing PyTorch 2.12.1+cu126 installation were used; CUDA was
available. See saved metadata for all dependency versions.

Tests use a small offline archive and cover checksum validation, subgroup matching,
normalization, exact/near pairs, transitive grouping, split balance, group integrity,
repeatable JSONL outputs, full row coverage, manifests, and overwrite protection.
Verification completed in `rag-env`: **10 tests passed**. The managed sandbox blocked
pytest temporary-directory access; the successful run used approved filesystem access.
