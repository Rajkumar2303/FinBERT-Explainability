# Phase 2 actual experiment results

All reported test evaluations use the same frozen 727-row test set. Supervised models use three independent final fits (seeds 42, 123, 2026); ± denotes sample standard deviation (ddof=1). Test results were not used for hyperparameter, checkpoint, or seed selection.

| Model | Model type | Macro-F1 | Accuracy | Training/search workflow wall time |
|---|---|---:|---:|---|
| TF-IDF + Logistic Regression | Linear supervised baseline | 0.6713 ± 0.0000 | 0.7442 ± 0.0000 | 3.61 s |
| bert-base-uncased | Fine-tuned general BERT | 0.8476 ± 0.0034 | 0.8556 ± 0.0014 | 13.43 min |
| finbert-pretrain | Fine-tuned financial BERT | 0.8339 ± 0.0093 | 0.8469 ± 0.0097 | 15.46 min |
| ProsusAI/finbert | External reference; non-independent | 0.8797 | 0.8900 | No training; 2.48 s inference workflow |

ProsusAI/finbert is a fixed pretrained external reference with Financial PhraseBank in its original fine-tuning data. Overlap with our test rows may exist, so its evaluation is not independent or a fair held-out ranking against the other workflows. A single fixed reference has no three-seed SD. Reference time is inference/loading/export time, not training.

## Per-class test metrics

| Model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | negative | 0.8235 ± 0.0000 | 0.4615 ± 0.0000 | 0.5915 ± 0.0000 |
| TF-IDF + Logistic Regression | neutral | 0.7544 ± 0.0000 | 0.9028 ± 0.0000 | 0.8219 ± 0.0000 |
| TF-IDF + Logistic Regression | positive | 0.6855 ± 0.0000 | 0.5343 ± 0.0000 | 0.6006 ± 0.0000 |
| bert-base-uncased | negative | 0.8625 ± 0.0135 | 0.8718 ± 0.0168 | 0.8670 ± 0.0085 |
| bert-base-uncased | neutral | 0.8926 ± 0.0143 | 0.8758 ± 0.0171 | 0.8839 ± 0.0026 |
| bert-base-uncased | positive | 0.7797 ± 0.0200 | 0.8056 ± 0.0283 | 0.7919 ± 0.0031 |
| finbert-pretrain | negative | 0.8290 ± 0.0132 | 0.8498 ± 0.0277 | 0.8390 ± 0.0068 |
| finbert-pretrain | neutral | 0.8922 ± 0.0105 | 0.8681 ± 0.0083 | 0.8799 ± 0.0088 |
| finbert-pretrain | positive | 0.7659 ± 0.0194 | 0.8007 ± 0.0075 | 0.7828 ± 0.0122 |
| ProsusAI/finbert | negative | 0.8155 | 0.9231 | 0.8660 |
| ProsusAI/finbert | neutral | 0.9590 | 0.8657 | 0.9100 |
| ProsusAI/finbert | positive | 0.8077 | 0.9265 | 0.8630 |

## Validation-only selections

| Model | Selected hyperparameters | Validation macro-F1 |
|---|---|---:|
| TF-IDF + Logistic Regression | `{"C": 10.0, "ngram_range": [1, 1]}` | 0.7203 |
| bert-base-uncased | `{"epochs": 3, "learning_rate": 3e-05}` | 0.8380 |
| finbert-pretrain | `{"epochs": 4, "learning_rate": 2e-05}` | 0.8221 |

- bert-base-uncased: final three fits took 5.15 min; per-seed durations: 42: 1.68 min, 123: 1.71 min, 2026: 1.76 min.
- finbert-pretrain: final three fits took 7.01 min; per-seed durations: 42: 2.26 min, 123: 2.28 min, 2026: 2.47 min.

## Resource measurements

| Model | Peak allocated GiB | Peak reserved GiB | OOM adjustments |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.000 | 0.000 | 0 |
| bert-base-uncased | 2.079 | 2.348 | 0 |
| finbert-pretrain | 2.079 | 2.357 | 0 |
| ProsusAI/finbert | 0.437 | 0.488 | 0 |

Peaks use PyTorch CUDA max_memory_allocated/max_memory_reserved across each workflow and each individual fit. They exclude other applications, driver/context memory, and display VRAM. Workflow wall times include loading, selection, fitting, checkpoint/export writes, and integrity checks; per-fit wall times include model loading and validation/export. Trainer loop runtime (including in-loop validation/checkpointing) is also saved in every training_record.json. CPU baseline has zero CUDA tensor allocation.

Batch configuration stayed at 8 per device × 2 accumulation = effective batch 16 unless an adjustment is explicitly listed in resource records. FP16 training, length 64, pinned revisions, and original configs/phase2.json were used. A short preflight confirmed classifier updates and finite final gradients without OOM; early FP16 loss-scaling overflow/skip events were recorded in sanity logs.

## Integrity and completion

- Phase 1 manifest SHA256: `c5b8f2b32453b22c9f5038ba844b3cc5c025bce603e27f7899e7d62e00a8c782`
- Frozen test JSONL SHA256: `97ad15089bd04e0d61726a6218bfcc2bdb18a4bcc1f5e8a299e1da41607e36a9`
- All successful run manifests were verified; no Phase 1 artifact or successful prior run was overwritten.
- Repeated seed-42 CUDA fits produced byte-identical checkpoint weights for both BERT and FinBERT; checksums are saved in execution/*-repeatability.json.
- Existing rag-env and PyTorch 2.12.1+cu126 were used. No PyTorch reinstall and no Phase 3 work.
- Incomplete experiments: none.

Checkpoint models, selected hyperparameters, logs, per-seed predictions, confusion matrices, subgroup metrics, and mean/SD summaries are in artifacts/phase2/<model> and <model>-evaluation. The combined machine-readable table is artifacts/phase2/comparison/comparison.json; resource records are under artifacts/phase2/execution.

Limitations: three seeds give a limited variance estimate, validation is reused for search/epoch selection, text is truncated at 64 tokens, and lexical duplicate grouping cannot exclude all semantic/pretraining overlap. Do not retune on these test scores.
