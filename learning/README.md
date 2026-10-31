# Learn the FinBERT Explainability project

Start with a sentence: **“The company’s losses fell.”** Why might a model call this positive? How would we know whether it is usually right? And if it highlights “losses,” what does that actually tell us?

This course builds the answers from small examples to the completed implementation. You need basic Python and some familiarity with training a classifier. You do **not** need to arrive knowing calculus, attention matrices or attribution methods. The foundations explain those before the later chapters use them.

## How a lesson works

Each main lesson begins with a project question, explains the idea in everyday language, and works through an example. Only then does it name the technical concept and introduce equations. The last part connects the idea to saved results, compares alternatives, checks your understanding and adds advanced caveats.

On your first pass, stop at each **Check your understanding** and answer aloud. On a second pass, follow the source links and work the arithmetic yourself. The research notebook in Chapter 11 is a reference library of future ideas; read a few cards at a time rather than trying to absorb the whole chapter in one sitting.

## Keep these kinds of claims separate

| Label | How to read it |
|---|---|
| Implementation fact | The source, configuration or saved model shows that we did this. |
| Measured finding | A saved experiment produced this value under its stated protocol. |
| Teaching example / toy | Small invented numbers or sentences used to explain an idea. They are not experiment results. |
| Interpretation | A reasonable way to understand a result or design. It is not proof of its cause or a claim about undocumented historical motives. |
| Hypothesis | A possible explanation that still needs a controlled test. |
| Proposal | A future change; it has not been implemented by this course. |

“FinBERT” means our fine-tuned `yiyanghkust/finbert-pretrain` unless the text explicitly says **ProsusAI**, the separate reference. The current annotation file has user error categories and labeled AI-assisted notes. Older Phase 3 prose describes the initially blank handoff; see [what remains unknown](evidence_gaps.md).

## Your route through the course

| Chapter | Question you will learn to answer | Read first |
|---|---|---|
| [00 — Project overview](00_project_overview.md) | What did we investigate, and what did we find? | Start here |
| [01 — Foundations](01_foundations_and_theory.md) | How does a sentence become numbers and a prediction? | 00 |
| [02 — Dataset](02_dataset_and_preprocessing.md) | How did we create a fair test? | 00; token basics from 01 help |
| [03 — Architecture](03_model_architecture.md) | What happens inside each model? | 01 |
| [04 — Training](04_baselines_and_training.md) | How were weights and settings chosen? | 02–03 |
| [05 — Evaluation](05_evaluation_and_validation.md) | What do accuracy, precision, recall and F1 actually count? | 02; start here early if metrics are unfamiliar |
| [06 — Calibration and robustness](06_calibration_and_robustness.md) | Can I trust confidence, and what happens when text changes? | 01 and 05 |
| [07 — Integrated Gradients](07_integrated_gradients.md) | How does a change in an input affect a prediction? | 01 and 03 |
| [08 — Attention](08_attention_mechanisms.md) | Which token information is mixed together? | 01 and 03 |
| [09 — Explanation experiments](09_explainability_experiments.md) | How did we compare highlighting methods fairly? | 05 and 07–08 |
| [10 — Results and discussion](10_results_and_discussion.md) | What can we honestly conclude? | 05–09 |
| [11 — Future research](11_improvements_and_future_research.md) | What experiment should come next, and why? | 04–10 |

For an **engineering conversation**, revisit 02–04 and 07. For a **research discussion**, revisit 05 and 08–11. For **model validation**, focus on 02, 05–06 and 10. These are revision routes; the sequential order is the easiest first read.

## Practice and reference shelf

- [60 interview questions and a 30-minute mock interview](interview_preparation.md)
- [Plain-English glossary and mathematical notation](technical_glossary.md)
- [Primary papers and official documentation](references.md)
- [Six-phase coverage matrix](coverage_matrix.md)
- [Questions the evidence cannot yet answer](evidence_gaps.md)
- [Historical reproduction commands](reproduction_commands.md)
- [Teaching and validation review](teaching_review.md)

## Eight exercises that do not rerun experiments

1. Use Chapter 05’s real confusion matrix to calculate negative precision and recall. Explain why their denominators differ.
2. Use a pencil to reproduce Chapter 01’s one-weight update. Explain which direction reduces the error.
3. Trace `fpb50-00018` through the saved Phase 4 record and figure. Explain which class a positive attribution supports.
4. Compute Chapter 06’s four-example ECE. Change the bins and explain why the answer changes.
5. Calculate Chapter 09’s deletion AUC from the plotted points. Explain what a lower curve means.
6. Explain why 2199 moved assignments do not mean 2199 duplicate sentences.
7. Compare the 687-case and 727-case analyses. Name one conclusion that persists and one question that remains open.
8. Propose a reviewed pair such as “profits rose” / “losses rose.” State the intended label before discussing how a model should behave.

**Check:** Which of these exercises changes model weights? **Answer:** None. They use paper arithmetic and read existing evidence.

## Progress checklist

- [ ] I can give the 30-second, 2-minute and 10-minute explanations in Chapter 00.
- [ ] I can explain precision and recall without memorizing a slogan.
- [ ] I can trace tokens → embeddings → attention → logits → probabilities.
- [ ] I can explain gradients before writing the IG equation.
- [ ] I know which population and averaging rule each headline number uses.
- [ ] I can explain the 40 numerical failures without hiding or exaggerating them.
- [ ] I can distinguish human categories from AI-assisted interpretation.
- [ ] I can explain BERT’s higher score without claiming a proven cause.
- [ ] I can defend a future experiment that might reject my own hypothesis.
- [ ] I can complete the 30-minute mock interview and say “we did not measure that” when appropriate.

## Where the evidence lives

[Evidence ledger](evidence_ledger.json) records checked numbers and source hashes. [Validation results](validation_results.json) reports file sizes, local links, real CLI parsing, numerical assertions and preservation checks. [The validator](validate.py) only reads earlier project files and writes documentation evidence here; it does not load or train models.

The course adds no deployment permission or redistribution rights. Local artifact links need the original checkout; a public code-only copy may omit datasets and weights. Read the [model card](../publication/MODEL_CARD.md) and [license review](../publication/LICENSE_REVIEW.md) before reuse.
