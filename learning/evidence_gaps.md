# What we still do not know

You can understand the project without pretending these questions are answered. They need new evidence, not a more confident explanation. An unchecked box means open work; it does not imply a hidden failed experiment.

## Why did a model behave this way?

- [ ] **Why did BERT have the higher mean F1?** We have competing explanations involving data, vocabulary and training. We did not isolate one cause.
- [ ] **Do these results transfer to new dates, companies or languages?** A new independent evaluation is needed.
- [ ] **Did upstream pretraining include similar test text?** Local duplicate checks cannot inspect every upstream corpus.
- [ ] **What were the exact annotator votes?** Subset membership gives agreement bands, not all three vote totals.
- [ ] **How often do quantity or entity-relation skills fail?** Twenty selected errors and narrow edits do not answer that population question.
- [ ] **Would more seeds or a larger equal-budget search change the comparison?** Only three final seeds and four neural candidates were used.
- [ ] **Would longer inputs help?** Fifteen truncated examples identify a question, not the effect of changing length.

**Check:** Can better prose resolve these gaps? **Answer:** It can explain what evidence is missing. It cannot replace the experiment.

## How reliable are confidence and explanations?

- [ ] **Would validation-fitted temperature scaling help on new data?** It was not fitted here.
- [ ] **Are all edited sentences grammatical and correctly labeled?** They need human semantic review; we did not invent new labels.
- [ ] **Would other seeds, heads, layers or references give the same ranking?** The main result uses one fixed model and one attention summary.
- [ ] **Why exactly did 40 IG integrations miss absolute tolerance?** Their residuals and composition are documented, but a numerical cause was not isolated.
- [ ] **Could a new numerical study resolve those cases?** Possibly; it must be separately versioned and preserve historical thresholds and records.
- [ ] **Would whole-word masking or natural edits give the same method ordering?** Not tested.
- [ ] **Are IG and attention practically equivalent?** An interval including zero does not establish equivalence; a prespecified acceptable margin and suitable test are missing.
- [ ] **Do highlights help analysts make better decisions?** No human-utility study was performed.

**Check:** Are the 40 records missing? **Answer:** No. They remain saved and explicitly flagged. What is missing is an established explanation of their numerical difficulty, not their existence.

## What remains before reproduction or release?

- [ ] Independently review the AI-assisted interpretation/comments. User error categories are complete; these notes are not independently verified assessments.
- [ ] Clear exact model/data rights and redistribution terms. Current metadata does not settle every historical licensing question.
- [ ] Have the owner choose a code license and maintainer details.
- [ ] Check that required historical packages and drivers can be obtained on a fresh machine.
- [ ] Account for broad Phase 4/5 source fingerprints when reproducing old contexts after new documentation is added. Do not rewrite old manifests to conceal a mismatch.
- [ ] Measure real serving latency, cost, review coverage and monitoring needs before claiming operational readiness.

**Check:** Does the annotation hash exception mean the whole audit failed? **Answer:** No. The authorized later edit is disclosed against its original seal. Hiding it or silently rewriting history would be the problem.

The [research roadmap](11_improvements_and_future_research.md) turns these questions into proposed controlled experiments. The [discussion](10_results_and_discussion.md) explains what recommendation is defensible now. This course does not authorize new training, threshold changes or artifact repair.
