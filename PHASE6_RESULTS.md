# Phase 6 results: validation and publication preparation

Phase 6 completed without model training, model inference, frozen-data changes, result changes, Git commits, remote creation, or publication. The existing rag-env and CUDA-enabled PyTorch were preserved. No packages were installed. PDF generation read ReportLab 4.4.9 from the bundled runtime with bytecode writes disabled; rendering used the existing pypdfium2 because Poppler was unavailable.

## Deliverables

- Five-page validation report: [PDF](output/pdf/finbert-model-validation.pdf) and [matching Markdown](artifacts/phase6/MODEL_VALIDATION_REPORT.md). All five rendered pages were inspected for layout, legibility and clipping.
- [GitHub README](publication/README.md), [model card](publication/MODEL_CARD.md), [reproducibility checklist](publication/REPRODUCIBILITY.md), and [license review](publication/LICENSE_REVIEW.md).
- GitHub-ready, allowlist-based repository copy: `artifacts/phase6/github-ready/`. Nothing was staged, committed, pushed or published.
- Pinned non-Torch dependencies, observed Torch constraints, report dependencies and machine-readable CLI commands under `publication/`.
- Audit implementation in `src/finbert_explainability/phase6/` and seven new tests in `tests/phase6/`.
- Metric/manifests audit, annotation summary, 40-case reliability inventory, subset sensitivity JSON/CSV and report-regeneration evidence in `artifacts/phase6/audit-final/`.

## Final scientific conclusions

General BERT mean macro-F1 is **0.8476 +/- 0.0034**, versus financial BERT **0.8339 +/- 0.0093**, and TF-IDF **0.6713 +/- 0.0000** (sample SD across three seeds). These results do not support a benchmark advantage for financial pretraining and are not a significance test. ProsusAI is a non-independent external reference.

BERT allagree/non-allagree macro-F1 is **0.9583/0.7500**. The large observational gap, small subgroup class counts, imperfect calibration and unvalidated perturbation labels limit deployment claims. Recommend supervised research or a monitored human-review pilot only, pending prospective temporal/domain validation and rights clearance.

Primary absolute IG-attention correlation is the **mean of 687 per-sentence Spearman coefficients**, comparing absolute signed-sum IG with final-layer CLS attention averaged over 12 heads on identically filtered content WordPieces: **0.2406**. It is not a pooled token correlation or proof of faithfulness.

| Population | N | Mean Spearman | IG AUC | Attention AUC | Random AUC | Paired IG-attention mean and 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Converged primary | 687 | 0.2406 | 0.6555 | 0.6608 | 0.7866 | -0.0053 [-0.0130, 0.0024] |
| All, includes unreliable IG | 727 | 0.2465 | 0.6353 | 0.6372 | 0.7778 | -0.0019 [-0.0097, 0.0061] |

Both intervals include zero. The main directional conclusion remains unchanged, but the numerical means and class mix change; no practical-materiality margin was specified. Both methods show stronger mean probability suppression than random under the prescribed MASK test, not universal superiority on every diagnostic or example. A null-including interval is not proof of equivalence.

The 40 unreliable cases contain **11 negative / 0 neutral / 29 positive** gold labels, **31 allagree**, **36 correct predictions**, and **2 truncated inputs**. Their mean eligible-token length is **28.6**, versus **24.7** for the converged subset. All fail the unchanged absolute 0.001 tolerance after 200 steps; all relative residuals remain below 1%. The maximum absolute residual is **0.015395**. Exclusion is not random. The unreliable-only paired result is diagnostic and cannot establish a method winner. All records remain saved and explicitly flagged; no thresholds or attribution settings were changed.

IG baseline choice materially affects rankings (four-example PAD/MASK Spearman range **0.1980-0.6443**, top-5 Jaccard **0.25-0.6667**). MASK deletion introduces distribution shift, may remove part of a WordPiece word, and does not prove causality. Special-token attention mass alone does not show unfaithfulness. The report discusses Jain & Wallace and Wiegreffe & Pinter without claiming to reproduce every control or settle the debate.

## Qualitative annotation status

All **20/20 user categories are completed**: entity context 7; financial context 4; implicit sentiment 4; negation 2; complex sentence 1; numerical reasoning 1; label ambiguity 1. Categories were counted as supplied, not recoded. Interpretations and comments in all 20 rows contain labeled AI-assisted drafts and are not independent human-validated causal assessments. Fourteen examples have at least 25% disagreement, not necessarily strictly more than 25%; nine are in the 34%-to-less-than-50% band. Exact per-row vote counts are unavailable. The confidence/disagreement-selected sample is not representative.

## Audit and tests

- **68 tests passed** in 23.40 seconds, including seven new Phase 6 tests; JUnit: `artifacts/phase6/tests.xml`.
- **30 input/output manifests, 3,025 checksum entries**, including saved smoke runs, were audited.
- **3,212 earlier files remain byte-identical**. Source configs, checkpoint weights, data and all earlier reports were preserved. The original root README remains intact; the new README is in the publication package.
- Numeric claims in all four earlier result reports were regenerated with their metric labels. Classification, per-class scores, calibration, subgroup and robustness metrics were recomputed from saved predictions; IG sensitivity and Phase 5 statistics were recomputed from saved records. CSV numeric cross-checks passed.
- Known exception: the authorized post-seal changes to Phase 3 `human_annotation.csv` differ from its original manifest. Both hashes are recorded; the manifest was not repaired. The historical report's blank-annotation statement is no longer current.
- **19 reproduction commands** parsed successfully through the real CLIs with workflow dispatch blocked. No training command was executed.
- The empty `.gitignore` directory was preserved at `artifacts/phase6/backups/original-gitignore-directory` and replaced with a working ignore file. Ignore checks cover weights, caches, JSONL data and environment secrets.
- Publication checks found no prohibited weights/datasets, no matching secret patterns, no missing README links, and no files over the 50 MiB package limit. Secret scans are heuristic, not a guarantee. Approximately **11.44 GiB across 66 weight files** are excluded; largest weight file **418.7 MiB**.
- Resolved authoring issues: numeric report regeneration initially flagged JSON/table ordering, not different values; the check now preserves metric/value association while allowing row/key order. A ReportLab style-name collision was corrected before a PDF was generated. Neither issue changed experimental outputs.

## Remaining manual tasks and blockers

1. Select the project code license and maintainer/contact; no license grant was fabricated.
2. Review exact pinned-model/data rights. Financial PhraseBank is listed CC BY-NC-SA 3.0, general BERT Apache-2.0, the current ProsusAI card CC BY-NC-SA 4.0, and financial pretraining lacks clear license metadata. Pinned-card retrieval was unavailable; current metadata does not prove historical revision rights.
3. Review any dataset-derived publication excerpts/aggregate outputs and attribution obligations. Raw text, weights and per-example data are excluded by default.
4. Arrange lawful access to private original artifacts for exact historical replay. The code-only package is not a self-contained reproduction of model weights. Confirm historical wheel/driver availability for a fresh environment; observed pins are not a complete platform-independent lock.
5. Legacy Phase 4/5 resume checks bind broad source/document inventories. Adding later phase files can invalidate those historical contexts even though all earlier files are unchanged. Use the original archived inventory for exact historical resumption or a separately configured new run; never rewrite old contexts/manifests to hide differences.
6. Review the final file inventory and automated secret-scan results, choose repository name/visibility, and explicitly approve publication. No GitHub remote is configured and no tracked files existed at inspection.

No unresolved numerical results were hidden: 40 IG records remain unreliable. No new training, inference, publication or external-directory modifications were performed. The pipeline and evidence support cautious research conclusions, not production certification.
