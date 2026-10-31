# Phase 4: frozen-checkpoint Integrated Gradients

All commands run from the project root using the existing Windows rag-env. No dependency installation, model training, or checkpoint modification is needed. Captum 0.9.0 and CUDA PyTorch are already available. New implementation lives in `src/finbert_explainability/phase4`, tests in `tests/phase4`, configuration in `configs/phase4.json`, and artifacts in `artifacts/phase4`. Earlier phase files are read-only inputs. `PHASE4_RESULTS.md` is generated from measurements.

```powershell
$phase4Python = 'C:\Users\rajku\anaconda3\envs\rag-env\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
& $phase4Python -m finbert_explainability.phase4 --help
& $phase4Python -m finbert_explainability.phase4 run --help
& $phase4Python -m pytest tests/phase4 -q -p no:cacheprovider --basetemp artifacts/phase4/test-temp-new
& $phase4Python -m finbert_explainability.phase4 smoke --output artifacts/phase4/smoke
& $phase4Python -m finbert_explainability.phase4 smoke --cuda --output artifacts/phase4/cuda-smoke
& $phase4Python -m finbert_explainability.phase4 run --config configs/phase4.json
```

Each smoke output must be fresh. Smoke uses a tiny randomly initialized BERT only; it never trains or saves model weights. The main run validates that the interpreter is rag-env and refuses unavailable CUDA. `--cpu` is supported for diagnostic execution of the saved checkpoint, but is much slower.

## Fixed model and verification

The default primary checkpoint is `artifacts/phase2/finbert/seed-42/model`, derived from yiyanghkust/finbert-pretrain. Seed 42 was specified before explanation generation, not selected using test performance. Phase 2 selected its hyperparameters on validation only. Optional `model_kind: bert` and `model_seed: 123` or `2026` work through a copied Phase 4 configuration; choose new output/report paths. Never edit Phase 2 configuration.

Before attribution the CLI verifies the pinned Phase 1 manifest and test SHA256, the relevant complete Phase 2 training/evaluation manifests, model/tokenizer manifest membership, canonical negative=0 / neutral=1 / positive=2 mappings, and sample IDs. It verifies the Phase 3 manifest and its inference row order, probabilities, and logits. The sole documented exception is `human_annotation.csv`, edited by the user after sealing; its sealed/current hashes are reported, never repaired. Missing annotations still fail. Any other mismatch fails clearly.

Every frozen test prediction is recomputed in evaluation mode with float32 eager attention, then compared against Phase 2 probabilities and Phase 3 logits. Tolerances are 1e-5 probability and 1e-4 logit, with identical predicted classes required. The per-example attribution forward is checked again. Parameters have gradients disabled; Captum differentiates the intermediate embedding outputs. Dropout is disabled by `model.eval()`. All existing Phase 1–3 artifact bytes, source, tests, configurations and root documents are snapshotted before/after; authorized annotation edits are preserved in their current state.

## Attribution definition

The actual architecture must be `BertForSequenceClassification`. Captum `LayerIntegratedGradients` is attached to `model.bert.embeddings`, attributing its output (word + position + type embeddings, after LayerNorm, with dropout inactive). The forward returns the **original predicted class logit**, not its softmax probability. Probabilities are exported only as prediction context.

The PAD baseline replaces actual content token IDs while preserving inserted CLS/SEP, original padding, sequence length, attention masks and token types. Replaced content remains attended: hiding it would change the function along the integration path. The secondary baseline substitutes MASK at those same content positions. Both endpoints are valid model inputs, but are artificial references, not neutral natural-language sentences. Integration follows a straight line between complete embedding-layer outputs; it is not a discrete token-edit trajectory.

Gauss–Legendre quadrature starts at 50 steps and increases to 100 then 200 if either absolute residual exceeds .001 or relative residual exceeds .01. Residual is sum(attributions) - (input target logit - reference target logit). Relative residual uses max(abs(logit difference), 1e-6) as denominator. Captum delta is also retained. Every attempt is logged, including unresolved failures; numerical completeness is not proof of explanation faithfulness. Near-zero endpoint differences can produce large relative residuals despite small absolute errors.

Token signed score = sum over embedding dimensions. `absolute` = abs(signed score); `sum_absolute_embedding_dimensions` is separately exported and is not used for ranking. Padding/special tokens remain in the completeness sum and JSON; they are excluded from content rankings. Original WordPiece scores are preserved. Optional `words` entries merge only consecutive `##` continuation tokens; signed contributions are summed. Punctuation stays separate, and both absolute-of-signed-sum and sum-of-absolute-pieces are recorded. No original text spans are claimed for these reconstructed words.

## Selection, exports and sensitivity

Initial selection takes the five lowest row IDs classified correctly and five lowest misclassified row IDs. No confidence or attribution values enter selection. This is deliberately balanced for explanation inspection, not representative sampling. Ten PNG bar plots and an HTML token gallery show signed contributions, true/predicted labels, probability, integration steps, and delta. Green supports the target logit; red opposes it, regardless of whether the target is positive or negative sentiment.

The first two examples of each outcome group also receive MASK-reference attribution. Sensitivity exports Spearman correlation of absolute token rankings, top-5 Jaccard overlap, sign-change fractions, and convergence status. A small sample cannot establish that either reference is universally better.

Each immutable example JSON contains original sentence, row ID, canonical labels, tokens/IDs, mask, baseline IDs, logits/probabilities, signed/absolute scores, integration attempts, truncation flag, configuration, checkpoint path, and context fingerprint. Embedding-dimension attributions are computed but only their signed/absolute reductions are persisted. `token_attributions.csv`, `convergence.csv`, and sensitivity JSON/CSV permit downstream analysis. Runtime, throughput, peak CUDA allocated/reserved memory and environment versions are measured. Failure JSON files retain tracebacks; no failures are silently discarded.

## Resumable full-test mode

Create a new config under `configs/phase4/` with `output_dir` set to `artifacts/phase4/full-test` and `report_path` to `artifacts/phase4/full-test/PHASE4_RESULTS.md`, keeping all other fields fixed. Then:

```powershell
& $phase4Python -m finbert_explainability.phase4 run --config configs/phase4/full-test.json --full-test
# After interruption, same config/code/environment and same mode:
& $phase4Python -m finbert_explainability.phase4 run --config configs/phase4/full-test.json --full-test --resume
```

All per-example records include integrity hashes. Resume rejects changed configuration, source, inputs, or environment and never overwrites existing records. A completed run's manifest is verified and execution exits without rewriting. An interrupted partial JSON is reported as invalid; use a fresh run directory rather than silently discarding it. A changed code version also requires a fresh run directory. The original initial run is never expanded in place. All 727 rows can be processed one at a time with bounded integration batch size; no full-test run is automatic.

The default internal integration batch size is 4 to limit 8 GB GPU usage. If an OOM occurs, the failure is recorded; choose a smaller internal batch size in a new Phase 4 config/output. This does not modify the model or its Phase 2 training batch size. Expected initial output size is a few MB, excluding test scratch; full-test token JSON/CSV is likely tens to hundreds of MB, with no checkpoint duplication. Runtime estimates should be based on the measured initial throughput, with extra allowance for 100/200-step escalation.

References: [Captum LayerIntegratedGradients API](https://captum.ai/api/layer.html#layer-integrated-gradients), [Captum memory guidance](https://captum.ai/docs/faq), [Sundararajan et al., Integrated Gradients](https://arxiv.org/abs/1703.01365). Phase 5 attention comparison is outside this implementation.
