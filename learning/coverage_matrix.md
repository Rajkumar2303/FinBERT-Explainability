# Connect the six phases to what you can explain

Use this as a revision checklist. Start with the question, open the chapter, then follow its code and evidence links. A phase is covered when you can explain both its calculation and its limitation.

| Phase | Evidence and code | Main chapters | Skill to demonstrate | Practical check |
|---|---|---|---|---|
| 1: data construction | [data](../src/finbert_explainability/data.py), [splitting](../src/finbert_explainability/splitting.py), [leakage](../artifacts/phase1/leakage_report.json) | [02](02_dataset_and_preprocessing.md), [05](05_evaluation_and_validation.md) | Explain thresholds, connected components, moved rows and frozen hashes | Recalculate split class counts and explain 2199 without calling them duplicates |
| 2: model comparison | [neural](../src/finbert_explainability/phase2/neural.py), [baseline](../src/finbert_explainability/phase2/baseline.py), [comparison](../artifacts/phase2/comparison/comparison.json) | [01](01_foundations_and_theory.md), [03](03_model_architecture.md), [04](04_baselines_and_training.md), [05](05_evaluation_and_validation.md) | Trace tensors, optimizer, search and independent final fits | Identify seed-123 FinBERT's best epoch and canonical external labels |
| 3: reliability | [summary](../artifacts/phase3/summary.json), [calibration](../src/finbert_explainability/phase3/calibration.py), [perturbations](../src/finbert_explainability/phase3/perturbations.py) | [05](05_evaluation_and_validation.md), [06](06_calibration_and_robustness.md), [10](10_results_and_discussion.md) | Interpret subgroup, probability and perturbation evidence | Distinguish eligible/visible rates and current user categories from draft interpretations |
| 4: IG | [core](../src/finbert_explainability/phase4/core.py), [report](../PHASE4_RESULTS.md) | [07](07_integrated_gradients.md), [03](03_model_architecture.md) | Derive target/reference/completeness and explain WordPieces | Trace a saved example and compare PAD/MASK rankings |
| 5: attention comparison | [analysis](../src/finbert_explainability/phase5/analysis.py), [inference](../src/finbert_explainability/phase5/inference.py), [summary](../artifacts/phase5/full/summary.json) | [08](08_attention_mechanisms.md), [09](09_explainability_experiments.md) | Define exact correlation, intervention and uncertainty unit | Recalculate toy AUC and explain why random replicates do not multiply sample size |
| 6: audit/report/publication preparation | [audit](../src/finbert_explainability/phase6/audit.py), [report](../PHASE6_RESULTS.md), [sensitivity](../artifacts/phase6/audit-final/ig_sensitivity.json) | [00](00_project_overview.md), [09](09_explainability_experiments.md), [10](10_results_and_discussion.md), [11](11_improvements_and_future_research.md) | Distinguish integrity from validity and formulate bounded conclusions | Explain non-random exclusions, known annotation hash exception and release blockers |

## Check the connections

**Question:** Why does Phase 1 appear in both the dataset and evaluation chapters? **Answer:** The split is an engineering artifact, but its integrity determines what later scores mean.

**Question:** Why study Phase 6 after explanation experiments? **Answer:** The audit checks that reported claims match preserved evidence, including the cases that did not pass numerical checks. It cannot turn a limited experiment into a universal result.

For a full rehearsal, use the [60 interview questions](interview_preparation.md). For missing evidence, use the [open-question checklist](evidence_gaps.md). The [teaching review](teaching_review.md) describes coverage of the worked examples and the [validation output](validation_results.json) records automated checks.
