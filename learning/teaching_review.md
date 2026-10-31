# Review of the rewritten course

## What changed

Chapters 00 and 05 were rewritten first to establish the teaching style. The remaining chapters follow the same progression: project question, plain-language intuition, concrete example, steps, technical name, mathematics, actual implementation, alternatives, answered check, and advanced caveats.

The course contains 12 main chapters. Each has explicit learning objectives and at least five answered end-of-chapter questions. Each major teaching section includes a concept check. The supporting material includes 60 interview questions, a 30-minute mock interview, 147 glossary entries, references, reproduction commands, phase coverage and open evidence questions.

## Worked-example coverage

| Concept | Starting example | Full lesson |
|---|---|---|
| Financial sentiment | Profits rising versus losses rising | [00](00_project_overview.md) |
| Embeddings | Token ID as a lookup address, then a small vector | [01](01_foundations_and_theory.md) |
| Backpropagation | One weight, one loss and a calculated update | [01](01_foundations_and_theory.md) |
| Duplicate grouping | A–B–C links and keeping a group together | [02](02_dataset_and_preprocessing.md) |
| Pretraining versus fine-tuning | Learn language first, then learn the three-label task | [01](01_foundations_and_theory.md), [03](03_model_architecture.md) |
| Effective batch size | Two microbatches of 8 before one update | [04](04_baselines_and_training.md) |
| Accuracy, precision, recall and F1 | One real 727-row baseline confusion matrix | [05](05_evaluation_and_validation.md) |
| Macro-F1 | Calculate each of the three class scores before averaging | [05](05_evaluation_and_validation.md) |
| Calibration | Ten 90%-confidence predictions with seven correct | [06](06_calibration_and_robustness.md) |
| ECE | Four predictions; changing bins changes the answer | [06](06_calibration_and_robustness.md) |
| Brier and NLL | Positive truth with probabilities [0.1,0.2,0.7] | [06](06_calibration_and_robustness.md) |
| Gradients and IG | Score changes, then a linear path and an interaction | [07](07_integrated_gradients.md) |
| Attention | “Losses increased” before query/key/value matrices | [08](08_attention_mechanisms.md) |
| Spearman correlation | Ranked lists [1,2,3] and [1,3,2] | [09](09_explainability_experiments.md) |
| Deletion AUC | Trapezoids under a three-point probability curve | [09](09_explainability_experiments.md) |
| Bootstrap intervals | Repeated sentence draws with replacement, preserving pairs | [05](05_evaluation_and_validation.md), [09](09_explainability_experiments.md) |
| Results versus causes | BERT’s observed gap, followed by competing unproven explanations | [10](10_results_and_discussion.md), [11](11_improvements_and_future_research.md) |
| Future losses and ensembles | Weighted loss, softened targets and averaged probabilities | [11](11_improvements_and_future_research.md) |

**Check:** Are these all new experimental findings? **Answer:** No. Small calculations are labeled teaching examples. Actual findings are tied to saved records and their populations.

## Readability and prerequisite review

- Main concepts begin with a question or example before technical definitions.
- Advanced sections follow foundational lessons; chapter introductions link prerequisites.
- Gradients are explained before IG; attention mixing precedes Q/K/V; class counts precede metric formulas; resampling precedes bootstrap interval interpretation.
- Research hypotheses and improvement proposals remain explicitly unproven or unimplemented.
- The dense research roadmap is split into individual cards with steps, controls and acceptance criteria.
- Bibliographic records and measured comparison tables are retained as reference material after explanatory lessons.
- The glossary is a lookup aid, not a substitute for the full ten-stage lessons. Research cards use the foundations already taught rather than repeating ten headings for every variation.

## What the validation checks

[validation_results.json](validation_results.json) is the execution record. It reports ordered lesson stages, answered checks, required files, interview coverage, local links, basic Mermaid grammar, accidental control characters, selected numerical claims against artifacts, cross-phase consistency, supported CLI parsing and earlier-file hashes. It also records individual Markdown file sizes and aggregate counts.

The checks do not prove every sentence of prose, render Mermaid or mathematics, or establish that a future learner will understand every concept on a first reading. Source-linked manual review complements the automated assertions. External reference verification dates are retained honestly from earlier authoring; this rewrite does not claim fresh live verification of every page.

**Check:** Does a passing documentation check prove the research hypothesis? **Answer:** No. It checks the documentation’s structure and agreement with evidence. Scientific conclusions remain bounded by the experiments.

## Remaining questions

No requested major concept is intentionally omitted. The deepest topics—matrix calculus, optimizer dynamics, attribution axioms and uncertainty under dependent data—have introductory explanations and links for further study rather than full graduate-level proofs.

The project still cannot establish why BERT outperformed FinBERT, the numerical cause of all 40 IG failures, exact annotator vote distributions, or performance on a new independent deployment population. These need new evidence, not stronger wording. See [the evidence-gap checklist](evidence_gaps.md).

No training, model inference, experiment regeneration, commit or publication is part of this rewrite. Earlier code, configurations, reports and artifacts remain outside the editing scope.
