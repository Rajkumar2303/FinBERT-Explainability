# Read the historical commands

A CLI command is a way to call a workflow with explicit settings. In these commands, `python -m` selects a module; the following words select its action and arguments. A config path supplies recorded settings, while output paths identify where results go.

These are the exact supported historical commands, not commands executed for this course. Existing outputs must be preserved. A separately authorized reproduction needs fresh output locations, corresponding config paths, lawful data/model access, the recorded environment and passing smoke checks. Broad Phase 4/5 context fingerprints may require the archived source inventory.

Source: [audited inventory](../publication/commands.json). The validator runs the real argument parsers and stops before workflow dispatch. Parsing verifies supported syntax; it does not verify a fresh experiment would finish.

**Check:** Does a parser check run training? **Answer:** No. This validator blocks dispatch immediately after parsing, before the workflow can load a model.

## phase1

```powershell
python -m finbert_explainability --config configs/phase1.json
```

## phase2 audit

```powershell
python -m finbert_explainability.phase2 audit --phase1 artifacts/phase1 --output artifacts/phase2/audit
```

## phase2 smoke

```powershell
python -m finbert_explainability.phase2 smoke --output artifacts/phase2/smoke-final
```

## phase2 train baseline

```powershell
python -m finbert_explainability.phase2 train baseline --config configs/phase2.json --output artifacts/phase2/baseline --approve-training --smoke-receipt artifacts/phase2/smoke-final/smoke_receipt.json
```

## phase2 train bert

```powershell
python -m finbert_explainability.phase2 train bert --config configs/phase2.json --output artifacts/phase2/bert --approve-training --smoke-receipt artifacts/phase2/smoke-final/smoke_receipt.json
```

## phase2 train finbert

```powershell
python -m finbert_explainability.phase2 train finbert --config configs/phase2.json --output artifacts/phase2/finbert --approve-training --smoke-receipt artifacts/phase2/smoke-final/smoke_receipt.json
```

## phase2 evaluate baseline

```powershell
python -m finbert_explainability.phase2 evaluate --run artifacts/phase2/baseline --output artifacts/phase2/baseline-evaluation
```

## phase2 evaluate bert

```powershell
python -m finbert_explainability.phase2 evaluate --run artifacts/phase2/bert --output artifacts/phase2/bert-evaluation
```

## phase2 evaluate finbert

```powershell
python -m finbert_explainability.phase2 evaluate --run artifacts/phase2/finbert --output artifacts/phase2/finbert-evaluation
```

## phase2 reference

```powershell
python -m finbert_explainability.phase2 reference --config configs/phase2.json --output artifacts/phase2/reference
```

## phase2 compare

```powershell
python -m finbert_explainability.phase2 compare --runs artifacts/phase2/baseline-evaluation artifacts/phase2/bert-evaluation artifacts/phase2/finbert-evaluation artifacts/phase2/reference --output artifacts/phase2/comparison
```

## phase3 smoke

```powershell
python -m finbert_explainability.phase3 smoke --output artifacts/phase3-smoke
```

## phase3

```powershell
python -m finbert_explainability.phase3 analyze --config configs/phase3.json --smoke-receipt artifacts/phase3-smoke/smoke_receipt.json
```

## phase4 smoke

```powershell
python -m finbert_explainability.phase4 smoke --output artifacts/phase4/smoke
```

## phase4

```powershell
python -m finbert_explainability.phase4 run --config configs/phase4.json
```

## phase5 cpu

```powershell
python -m finbert_explainability.phase5 cpu-smoke --output artifacts/phase5/cpu-smoke
```

## phase5 cuda

```powershell
python -m finbert_explainability.phase5 run --config configs/phase5/cuda-smoke.json --smoke
```

## phase5

```powershell
python -m finbert_explainability.phase5 run --config configs/phase5.json
```

## phase6

```powershell
python -m finbert_explainability.phase6 audit --output artifacts/phase6/audit-final
```

## Before a future reproduction

Read the [training chapter](04_baselines_and_training.md) for the existing rag-env, CUDA, validation selection and measured resources. Trace every input path and every output path before executing a command in a new workspace.

**Check:** May you reuse old output directories because the flags parse? **Answer:** No. Syntax validity does not authorize overwriting completed evidence.
