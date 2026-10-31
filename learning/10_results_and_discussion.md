# 10 — What can we honestly conclude?

## Learning objectives

By the end, you should be able to explain the results in everyday language, separate a measured difference from its possible causes, and recommend a use that the evidence can support. Read [evaluation](05_evaluation_and_validation.md) and [explanation experiments](09_explainability_experiments.md) first.

Think of this chapter as a review meeting. Someone asks, “Which model should we trust, and why?” A good answer brings together several kinds of evidence rather than pointing to the highest number.

## Lesson: A higher score tells us what happened, not why

### Why it matters
We expected financial pretraining to help. General BERT nevertheless had the higher average macro-F1. How do we explain that without making up a story?

### Build the intuition
First describe the score difference. Then separate possible explanations from explanations that an experiment has actually tested. An unexpected result is still useful evidence.

### A concrete example
**Measured:** general BERT scored 0.8476 mean macro-F1; our fine-tuned FinBERT scored 0.8339. **Teaching analogy:** if two runners finish at different times, the stopwatch tells us the gap. It does not tell us whether shoes, training or weather caused it.

### Walk through it
1. Confirm both workflows used the same frozen test rows and label mapping.
2. Check that their settings were selected with validation data.
3. Read all three seed results, not just the best run.
4. State the observed gap.
5. List possible causes as questions for future experiments.

### Give the idea its technical name
An **observed difference** is a measured comparison. A **causal explanation** identifies what produced that difference. A **controlled experiment** changes the suspected cause while keeping relevant other conditions comparable.

### Add the mathematics
The rounded gap is 0.8476 − 0.8339 = **0.0137 macro-F1**, or 1.37 percentage points on the 0–1 scale. That subtraction is not a significance test. Seed SD describes variation among fits on the same dataset; it does not cover every source of uncertainty.

### Connect it to our project
The [saved comparison](../artifacts/phase2/comparison/comparison.json) and [training configuration](../configs/phase2.json) establish the observed result. BERT had SD 0.0034; FinBERT had SD 0.0093. Only two learning rates and two epoch budgets were searched, at seed 42. Different pretraining histories and tokenizers remain possible influences. The study did not isolate their separate effects.

### Alternatives and trade-offs
A wider equal-budget validation search would test whether tuning matters. Matched continued-pretraining experiments could test domain adaptation. More seeds would better describe training variation. Each answers a different question; none should use the already inspected test set to select its winner.

### Check your understanding
**Question:** Can we say financial pretraining made the model worse?

**Answer explained:** No. We can say this financial-model workflow had a lower measured mean under this protocol. A claim about pretraining itself needs a more controlled comparison.

### Deeper dive and common confusion
**Common confusion:** a plausible story is not a measured mechanism. “FinBERT overfits filings” is too strong: forgetting, overfitting and genre mismatch were not directly measured here. See the competing tests in [Chapter 11](11_improvements_and_future_research.md).


## Lesson: Reliability is more than one leaderboard number

### Why it matters
A reviewer sees good overall F1 but asks whether the model struggles on ambiguous news and whether its confidence can be trusted.

### Build the intuition
Think of a model as having several report cards. One grades its class choices. Another grades its confidence. A third checks which kinds of sentences are difficult. Good performance on one does not erase problems on another.

### A concrete example
**Measured:** BERT allagree macro-F1 is 0.9583, versus 0.7500 for non-allagree. In ordinary language, its performance is much stronger on sentences where the original annotators unanimously agreed. **Toy:** a model can choose the right class on 8 of 10 sentences yet claim 99% confidence on all 10; its confidence is too high.

### Walk through it
1. Read the overall F1 and class-specific recall.
2. Split evaluation by agreement membership and inspect class counts.
3. Compare confidence with correctness using calibration metrics.
4. Examine edited sentences and selected errors.
5. Keep each conclusion attached to its evaluation population.

### Give the idea its technical name
A **subgroup** is a stated subset of examples. **Calibration** asks whether stated probabilities match observed frequencies. A **robustness test** checks behavior under a specified change; it does not automatically establish performance on every real-world change.

### Add the mathematics
The rounded allagree gap is 0.9583 − 0.7500 = **0.2083**. This is a difference between two observed groups, not the effect of making people agree. Their class mixes differ: the 338 allagree rows contain 50/209/79 negative/neutral/positive examples; the 389 other rows contain 41/223/125.

### Connect it to our project
[Phase 3 summary](../artifacts/phase3/summary.json) contains the subgroup, calibration and edit results. FinBERT has lower mean ECE-10 (0.0923 versus 0.1064), while BERT has lower mean Brier and NLL. The metrics ask different questions. Numerical-edit flip rates are low, but doubling a currency amount is not a full test of understanding financial growth.

### Alternatives and trade-offs
Evaluate class-conditional results, review naturally occurring errors, and create human-checked contrast pairs such as “losses increased” versus “losses decreased.” These give more focused evidence than treating every stable prediction as correct.

### Check your understanding
**Question:** If a prediction never changes after changing a number, does the model understand numbers?

**Answer explained:** No. The edit may preserve the sentiment, the model may ignore the number, or the changed token may be invisible to its representation. We need validated expected labels and visibility checks to distinguish those cases.

### Deeper dive and common confusion
**Common confusion:** the 20 annotated errors are selected examples, not a random sample of every mistake. User categories include 7 entity-context cases and only 1 numerical-reasoning case. Their interpretation/comments are labeled AI-assisted notes, not independent human assessments. “Most errors are numerical” is not supported by that file.


## Lesson: Explanation results are useful even without a winning method

### Why it matters
A stakeholder asks whether IG is a better explanation than attention. Our experiment gives a qualified answer rather than a universal ranking.

### Build the intuition
Two highlighting methods can disagree because they measure different things. To learn more, we remove their highly ranked tokens and watch the original prediction. Even then, the answer depends on how we remove tokens and what we measure.

### A concrete example
**Measured:** mean per-sentence Spearman correlation is 0.2406 on the 687 numerically converged cases. IG deletion AUC is 0.6555, attention 0.6608 and random 0.7866. Lower AUC means the original predicted-class probability fell more over this masking experiment.

### Walk through it
1. Keep the fixed FinBERT seed-42 checkpoint and original target class.
2. Compare rankings on the same content tokens.
3. Mask increasing fractions and measure probability curves.
4. Compare methods within each sentence.
5. Read the uncertainty interval and the 40-case sensitivity analysis before making a claim.

### Give the idea its technical name
An **operational test** defines a concrete measurement for a broader idea. MASK deletion is an operational test of whether ranked tokens support a prediction under that intervention. It is not a complete definition of explanation correctness.

### Add the mathematics
IG minus attention mean AUC is about −0.0053. Its paired 95% interval is **[−0.0130, 0.0024]**. Negative values favor IG under this metric, positive values favor attention, and zero means equal mean AUC. Because the interval spans zero, the experiment does not clearly separate the methods.

### Connect it to our project
[Phase 5 summary](../artifacts/phase5/full/summary.json) and [Phase 6 sensitivity](../artifacts/phase6/audit-final/ig_sensitivity.json) show the result persists qualitatively when all 727 records are included with warnings: the interval still spans zero. The 40 flagged IG cases are not a random subset. PAD-versus-MASK reference comparisons also changed rankings substantially in the small Phase 4 sample.

### Alternatives and trade-offs
Whole-word deletion, natural contrast pairs, multiple references and model-randomization checks would broaden the evidence. Each introduces its own assumptions and cost. Agreement with a human highlight measures whether an explanation looks reasonable, which is not identical to whether it reflects model computation.

### Check your understanding
**Question:** Does an interval spanning zero prove IG and attention are equivalent?

**Answer explained:** No. It means this study does not clearly establish a directional difference. Showing practical equivalence would require an acceptable difference chosen before looking at the results and a suitable analysis.

### Deeper dive and common confusion
**Common confusion:** passing IG completeness checks numerical accounting along a chosen path. It does not certify a faithful causal story about real language. Replacing many words with MASK creates unusual input; that distribution shift limits the deletion interpretation.


## Lesson: Turn evidence into a bounded deployment recommendation

### Why it matters
An engineer asks whether the saved checkpoint is ready for automated financial decisions.

### Build the intuition
A successful research evaluation is one part of a deployment decision. We also need to know the real inputs, the cost of errors, who reviews uncertain cases, and whether the data and weights can be used lawfully.

### A concrete example
**Proposed use:** an analyst receives suggested sentiment labels and reviews the text. **Unsupported use:** automatically buy or sell an asset because a sentence was labeled positive. The project learned sentence sentiment, not future returns.

### Walk through it
1. State the intended task: short English financial-sentence sentiment.
2. State observed strengths and weak groups.
3. Restrict use to research or a monitored human-review pilot.
4. Check rights, new-data performance, latency and review capacity.
5. Set acceptance criteria before any wider deployment.

### Give the idea its technical name
**External validity** asks whether findings transfer to a different setting. **Operational validation** asks whether the complete service meets practical requirements. **Provenance** records where data, models and results came from.

### Add the mathematics
A simple **toy** service cost illustrates the missing evidence: if 100 missed negative sentences cost 5 review units each, their cost is 500 units. Another system with slightly lower accuracy could be preferable if it avoids those misses. The project did not estimate such real decision costs.

### Connect it to our project
The [model card](../publication/MODEL_CARD.md) and [final validation report](../artifacts/phase6/MODEL_VALIDATION_REPORT.md) support a cautious recommendation. [Phase 6 audit code](../src/finbert_explainability/phase6/audit.py) checked 30 manifests and 3025 entries; 3212 earlier files remained byte-identical. The authorized annotation edit remains a disclosed historical manifest exception. A clean audit establishes traceability, not production certification.

### Alternatives and trade-offs
For a future pilot, compare a CPU baseline, a single encoder and an abstaining workflow that sends uncertain cases to review. Measure latency, class-specific errors, confidence, coverage and reviewer workload. These serving features are proposals, not completed capabilities.

### Check your understanding
**Question:** Can a frozen test still be unsuitable for choosing the next improvement?

**Answer explained:** Yes. Its bytes can remain unchanged while repeated inspection influences research choices. Future improvement claims need a new independent holdout.

### Deeper dive and common confusion
**Common confusion:** training time is not serving latency. Unknown upstream overlap, only three training seeds, one explanation checkpoint, small subgroups and narrow perturbations all limit generalization. No serving-latency distribution or rights-cleared production system was established.


## Keep the report cards together

Read each row as a separate question; do not average these different metrics into one invented trust score.

| Dimension | BERT | FinBERT | Interpretation |
|---|---:|---:|---|
| Macro-F1 mean ± SD | 0.8476 ± 0.0034 | 0.8339 ± 0.0093 | Higher observed mean and lower observed seed variation for BERT |
| Accuracy | 0.8556 | 0.8469 | Same directional comparison, not independent evidence of causality |
| Allagree F1 | 0.9583 | 0.9553 | Strong performance on this easier/clearer observed subgroup |
| Non-allagree F1 | 0.7500 | 0.7262 | Major reliability limitation on ambiguous population |
| ECE-10 | 0.1064 | 0.0923 | Different ordering from F1; FinBERT SD is large |
| Brier / NLL | 0.2468 / 0.5115 | 0.2620 / 0.6022 | Full-distribution scores favor BERT on mean |
| Company flip | 0.0128 | 0.0192 | Low sensitivity under a narrow, unvalidated edit rule |
| Numerical flip | 0.0031 | 0.0000 | Not proof of reasoning or semantic robustness |
| Negation flip | 0.2255 | 0.1889 | No validated target labels; higher is not automatically better |

The independent lexical baseline has macro-F1 0.6713 and takes only a few seconds for its workflow. ProsusAI scores 0.8797 but remains a **non-independent reference**, because its prior training may overlap PhraseBank. The main explanation checkpoint has 623/727 correct predictions; that single-seed count is not the three-seed mean accuracy.

**Check:** Which result should lead the conclusion? **Answer:** The measured independent comparison, followed immediately by its scope and reliability limitations—not a claim that one model is universally best.

## Practice explaining it

1. **Why did BERT win?**

   Its measured mean was higher; the cause has not been isolated.

2. **Can lower ECE coexist with worse F1?**

   Yes. Confidence quality and class decisions are different questions.

3. **What does the selected error review establish?**

   It records categories in 20 chosen errors, not population error frequencies or proven mechanisms.

4. **What does the explanation comparison establish?**

   Modest ranking agreement and better mean masking suppression than random under this protocol; no clear IG-over-attention advantage.

5. **What is the defensible recommendation?**

   Research or a carefully monitored analyst-review pilot, with rights and new-data validation before consequential use.

## Sources and next lesson

[Phase 6 results](../PHASE6_RESULTS.md), [validation report](../artifacts/phase6/MODEL_VALIDATION_REPORT.md), [model card](../publication/MODEL_CARD.md), [remaining evidence gaps](evidence_gaps.md). Continue to the [research roadmap](11_improvements_and_future_research.md).
