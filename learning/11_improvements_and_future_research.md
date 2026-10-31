# 11 — Design better experiments before promising a better model

## Learning objectives

Explain nine competing reasons for BERT’s higher score, understand candidate improvements from simple examples, compare their costs and risks, and design a research plan that could reject your favorite idea. Read [training](04_baselines_and_training.md), [calibration](06_calibration_and_robustness.md) and [discussion](10_results_and_discussion.md) first.

**Everything proposed here is future work.** The course does not retrain, retune, calibrate or replace a saved model. The historical rationale is not assumed: priorities below are technical judgments based on the evidence.

## Lesson: Make the next improvement claim testable

### Why it matters
We have many ideas for improving FinBERT. How do we avoid trying them all and reporting only the one that looks good?

### Build the intuition
Decide what improvement means before looking at new results. Choose a comparison, a meaningful target and safeguards. Keep fresh examples out of development so the final check can still surprise you.

### A concrete example
**Toy proposal:** compare standard training with a lower learning rate using the same rows and seeds. Decide beforehand that an F1 gain is useful only if negative recall does not fall by an unacceptable amount. The actual acceptable margins must come from the intended use, not this toy example.

### Walk through it
1. Write a hypothesis that could be wrong.
2. Identify the single change being tested.
3. Choose training/validation data and matched compute budgets.
4. Lock success criteria.
5. Evaluate a selected workflow on a new independent holdout once.

### Give the idea its technical name
A **hypothesis** is a proposed explanation or prediction. An **ablation** changes or removes one component to learn its contribution. A **holdout** is data reserved for evaluation; it must not guide the choices it is meant to assess.

### Add the mathematics
Let M_new and M_control be the same metric on the same new examples. The improvement is Δ = M_new − M_control. In a **toy** result, 0.85 − 0.83 = 0.02. A positive difference alone does not show that it is stable or useful; compare paired uncertainty with a practical margin chosen beforehand.

### Connect it to our project
The current 727 test sentences have repeatedly informed analysis. Keep them as a historical benchmark, but use a new time/event-separated, deduplicated holdout for meaningful future claims. Start from [Phase 2 settings](../configs/phase2.json), [saved selection](../artifacts/phase2/finbert/selection.json) and [training code](../src/finbert_explainability/phase2/neural.py). No proposal in this chapter has been run for the course.

### Alternatives and trade-offs
A broad model sweep explores more options but can overfit the validation set and obscure causes. A narrow ablation is easier to interpret but may miss interactions. A staged program diagnoses first, tests a few changes second, and confirms independently last.

### Check your understanding
**Question:** Is a new random split of the already inspected sentences a new independent dataset?

**Answer explained:** No. The content has already influenced development. It can help internal diagnostics, but it cannot provide the same protection as newly collected, independently evaluated data.

### Deeper dive and common confusion
**Common confusion:** locking the final file is not enough if its content already shaped the hypothesis. Priority labels below are planning judgments, not predicted or measured gains.


## Lesson: Compare competing explanations for BERT’s advantage

### Why it matters
The observed gap is small enough that several mechanisms could matter. Choosing a favorite story too soon can lead to an expensive but uninformative intervention.

### Build the intuition
Treat each explanation as a suspect. Ask what it predicts, what evidence we already have, and what test would make it less believable. Several suspects may contribute at once.

### A concrete example
**Measured:** FinBERT seed SD is 0.0093 versus BERT 0.0034. **Hypothesis:** optimization is less stable for the financial checkpoint. **Counterpoint:** three runs are too few to identify a mechanism, and different validation-selected checkpoints can occur without harmful instability.

### Walk through it
1. State the mechanism in plain language.
2. Find existing supporting observations without turning them into proof.
3. Identify missing measurements.
4. Design a comparison that holds other changes fixed.
5. Specify what result would weaken the hypothesis.

### Give the idea its technical name
A **confounding difference** is another change that could explain the result. **Catastrophic forgetting** means new training damages useful knowledge acquired earlier. A **learning curve** tracks performance as training data or training time changes.

### Add the mathematics
In a **toy** paired study, adaptation helps by +0.02 on one new sample and −0.01 on another. Their average is +0.005, but the disagreement matters. Repeated datasets and paired seeds help distinguish a dependable effect from a setting-specific gain; no such adaptation result was measured here.

### Connect it to our project
The evidence motivating the hypothesis cards below is concrete: 3391 training rows; a four-candidate single-search-seed grid; FinBERT seed 123 selecting epoch 1 while the other two select epoch 4; different vocabularies; and large agreement-subgroup gaps. These observations motivate tests but do not determine why BERT scored higher.

### Alternatives and trade-offs
More seeds are cheap compared with pretraining from scratch but cannot isolate corpus causes. Controlled continued pretraining is more feasible than recreating entire historical model training, yet it tests a new adaptation intervention, not the full history of the two existing checkpoints.

### Check your understanding
**Question:** Would swapping FinBERT’s tokenizer for BERT’s isolate vocabulary effects?

**Answer explained:** No. Token IDs would point to different embedding rows, breaking the learned correspondence unless embeddings and training were handled appropriately. Begin with descriptive tokenization analysis on development data.

### Deeper dive and common confusion
**Common confusion:** “finance” is not one uniform text style. Reports, calls and short news differ. That makes genre mismatch plausible, not established. The detailed cards preserve nine competing hypotheses below.


## Lesson: Change the loss carefully: difficult does not always mean useful

### Why it matters
Negative examples are less common, and some labels are ambiguous. Should training pay extra attention to difficult or minority examples?

### Build the intuition
A training loss is the penalty used to update weights. We can increase the penalty for a minority class, reduce the influence of easy examples, or avoid pretending every label is perfectly certain. Each changes what the model is encouraged to learn.

### A concrete example
**Toy:** the correct class has probability 0.8. Ordinary cross-entropy is −ln(0.8) ≈ 0.2231. A class weight of 2 makes it 0.4463. Focal loss with exponent 2 multiplies by (1−0.8)² = 0.04, giving about 0.0089 without class weighting. An easy example receives much less pressure.

### Walk through it
1. Start with standard cross-entropy as the control.
2. Estimate any class weights from training labels only.
3. Tune one change using validation data.
4. Inspect class precision/recall and confidence, not only F1.
5. Check whether difficult examples were useful cases or noisy labels.

### Give the idea its technical name
**Class weighting** gives different labels different loss weights. **Focal loss** reduces easy-example contributions. **Label smoothing** replaces a perfectly certain training target with a slightly softer one. **Soft labels** are probability targets, ideally based on actual votes rather than invented uncertainty.

### Add the mathematics
For true class y, predicted probability p_y, class weight w_y and focal exponent γ, weighted loss is −w_y ln(p_y); focal loss is −w_y(1−p_y)^γ ln(p_y). Setting γ=0 gives weighted cross-entropy.

For K=3 classes and smoothing ε=0.1, q_k=(1−ε)1[k=y]+ε/K. The true-class target becomes 0.9333 and the other two become 0.0333 each. Here 1[k=y] is 1 for the true class and 0 otherwise. The soft-target loss is −Σ_k q_k ln(p_k). This is a proposed rule, not recovered annotator votes.

### Connect it to our project
The completed project used ordinary supervised loss, not these proposed alternatives. Agreement bands do not reveal the three class vote proportions. [Phase 3 subgroup results](../artifacts/phase3/summary.json) motivate careful testing; they do not justify discarding all non-unanimous rows.

### Alternatives and trade-offs
Class weighting is simple and cheap but may hurt confidence calibration. Focal loss may concentrate on mislabeled examples. Smoothing may reduce overconfidence but blur genuinely clear labels. True soft labels require additional annotation evidence.

### Check your understanding
**Question:** Can a 25–34% disagreement band directly become a three-class soft label?

**Answer explained:** No. It constrains overall agreement but does not say how the remaining votes divide between the other classes, nor give exact vote totals.

### Deeper dive and common confusion
**Common confusion:** improving minority recall can lower precision or accuracy. An intervention is not a success just because the metric it directly encourages improves. Evaluate classwise safeguards and probability quality on independent data.


## Lesson: Adapt knowledge, combine predictions, or reduce model cost

### Why it matters
If a bigger training search is insufficient, should we add domain text, combine models, or build a smaller model?

### Build the intuition
These solve different problems. Extra unlabeled reading may improve representations. Combining predictions may average away some mistakes. A smaller student may copy useful behavior at lower serving cost. None guarantees better sentiment decisions.

### A concrete example
**Toy ensemble:** two models assign positive probability 0.9 and 0.5. Their equally weighted average is 0.7. That is a new prediction distribution, not the average of their eventual F1 scores. **Toy distillation:** a teacher’s [0.1,0.2,0.7] target tells a student more than the hard label “positive” alone.

### Walk through it
1. Choose the bottleneck: language coverage, prediction quality or serving cost.
2. Pick the matching intervention.
3. Keep lawful training-side data separate from evaluation.
4. Include an unchanged-model control.
5. Measure the intended benefit and the possible loss of other skills.

### Give the idea its technical name
**Domain-adaptive pretraining (DAPT)** continues language-model training on domain text. **Task-adaptive pretraining (TAPT)** uses unlabeled text close to the task. Both are forms of continued pretraining. **Ensembling** combines model predictions. **Knowledge distillation** trains a student to imitate a teacher. **Multi-task learning** shares training across related labeled tasks.

### Add the mathematics
An equal ensemble uses p_ensemble,k = (1/S)Σ_s p_s,k, where S is the number of models and k is a class. The probabilities still sum to 1.

One proposed distillation loss combines ordinary label loss with T² KL(p_teacher^(T) || p_student^(T)). KL(q||p)=Σ_k q_k ln(q_k/p_k) is zero for identical distributions and penalizes mismatch. T is a temperature that softens probabilities; the mixture weight and T must be selected on development data. This formula explains a candidate method, not an implemented training path.

### Connect it to our project
The existing models were fine-tuned from saved upstream checkpoints; the project did not run DAPT, TAPT, distillation, ensembling or multi-task training. A financial-news adaptation experiment could test genre mismatch. A student or cascade would address deployment cost, which was not measured by the training-duration table.

### Alternatives and trade-offs
Adaptation costs GPU time and risks contamination or forgetting. Ensembles multiply inference/storage and can share the same errors. Distillation adds teacher inference and student training, and can copy teacher biases. Frozen sentence embeddings with a linear head are cheaper to fit but may miss subtle task-specific distinctions.

### Check your understanding
**Question:** Does an ensemble of three seeds necessarily score the mean seed F1?

**Answer explained:** No. Average the probability vectors, choose classes, then recompute F1. F1 is nonlinear, so averaging member scores answers a different question.

### Deeper dive and common confusion
**Common confusion:** an LLM’s fluent explanation is generated text, not proof of the computation behind its label. Any LLM comparison needs fixed prompts, label parsing, model versions, data-handling rules and contamination checks.


## Lesson: Prioritize evidence before expensive training

### Why it matters
An RTX 4070 laptop can support useful controlled experiments, but a long list of ideas can exhaust time without answering a clear question.

### Build the intuition
Start with inexpensive measurements that can rule out bad ideas. Secure an independent test first. Then spend compute where the existing evidence points, with a clear stopping rule.

### A concrete example
**Measured motivation:** 15 test inputs were truncated, so longer context is worth investigating on development examples. It does not follow that doubling the sequence length will improve the whole benchmark. **Toy cost:** dense attention pairs grow from 64²=4096 to 128²=16384, four times as many pairs per head.

### Walk through it
1. Establish new data provenance and label rules.
2. Run development-only diagnostics for tokenization, learning curves and calibration.
3. Test a small number of matched interventions.
4. Confirm selected workflows on a locked holdout.
5. Report failures and compute, not only favorable results.

### Give the idea its technical name
A **safeguard** is a requirement that must remain acceptable while optimizing another metric. A **cost–quality frontier** compares systems where improving quality tends to require more compute or latency. **p95 latency** is the response time below which 95% of measured requests fall.

### Add the mathematics
A **toy** review policy sends 30 of 100 inputs to a human, so automatic coverage is 70/100 = 70%. Its automatic error rate must be computed on those 70 accepted cases. A lower accepted-case error rate is not meaningful without also reporting coverage and reviewer workload.

### Connect it to our project
The roadmap below uses FinBERT non-allagree F1 0.7262, ECE-10 seed SD 0.0556, and explanation numerical/baseline issues as motivation. These are different reliability problems; one new training trick need not solve all of them. The [Phase 6 sensitivity](../artifacts/phase6/audit-final/ig_sensitivity.json) must remain a frozen historical control.

### Alternatives and trade-offs
P0 establishes valid evaluation. P1 favors directly motivated, relatively inexpensive work. P2 needs more engineering or annotation. P3 is expensive or weakly supported by current evidence. These are proposed priorities, not measured return-on-investment forecasts.

### Check your understanding
**Question:** Should we optimize IG–attention correlation to make explanations better?

**Answer explained:** Not by itself. Two explanations agreeing more could simply share the same mistake. Evaluate sensitivity, numerical reliability and behavior under justified interventions instead.

### Deeper dive and common confusion
**Common confusion:** selecting the intervention or equivalence margin after seeing a holdout result weakens the claim. Keep all new configurations/results in new locations; never overwrite this project’s evidence.


## Research notebook A: nine explanations to test

All cards are hypotheses, not findings. Read one at a time after Lesson 2. For each, ask what result would make you abandon it.

### A1. Small labeled dataset and finite-seed variation.

**Start with an example:** Imagine two training shuffles emphasizing different rare negative examples.

**Mechanism:** **Small labeled dataset and finite-seed variation.** With 3391 training rows, different shuffles/head initialization can yield different boundaries.

**Existing evidence:** FinBERT F1 SD 0.0093 exceeds BERT's 0.0034 across the three fits.

**What we still need:** Stable variance estimates across more seeds and datasets; data-sampling variation.

**Test step by step:** Prespecify more matched seeds and grouped training subsamples, keep search budget fixed, compare learning curves on development data and confirm selected workflows on a new holdout.

**Check:** Would more seeds alone test corpus mismatch? **Answer:** No. They characterize training variation while leaving the corpus difference intact.

### A2. Domain mismatch.

**Start with an example:** Compare “net proceeds pursuant to the offering” with “profits jump.” Both are financial language, but their style differs.

**Mechanism:** **Domain mismatch.** Financial communications pretraining may emphasize reporting styles unlike short news sentences.

**Existing evidence:** Model provenance describes reports/calls/analyst text; target is short sentence sentiment. This is plausibility, not a measured mismatch effect.

**What we still need:** Corpus distribution statistics and controlled adaptation comparisons.

**Test step by step:** Continue pretraining on lawful financial news versus equal-token original-style financial text, starting from the same checkpoint and tokenizer; equalize steps and evaluate both in-domain and external news.

**Check:** Does a financial label guarantee the same language distribution? **Answer:** No. Genre and task can differ within one domain.

### A3. Tokenizer/vocabulary effects.

**Start with an example:** Count how many pieces each tokenizer produces for the same company name before changing either model.

**Mechanism:** **Tokenizer/vocabulary effects.** Different subword fragmentation may affect names, quantities or rare financial terms.

**Existing evidence:** Vocabularies differ, 30522 versus 30873; length cap is shared.

**What we still need:** Per-category fragmentation, unknown-token and truncation analyses; causal separation from pretrained weights.

**Test step by step:** First measure tokenization on development text. Then compare controlled tokenization adaptations with appropriate embedding initialization; a naive tokenizer swap into old embeddings is invalid. Include unchanged-tokenizer controls.

**Check:** Is fragmentation automatically harmful? **Answer:** No. It is a diagnostic; downstream comparisons are needed.

### A4. Pretraining corpus/objective differences.

**Start with an example:** Train from one shared base on equal amounts of general and financial text.

**Mechanism:** **Pretraining corpus/objective differences.** General pretraining may provide broader syntax/semantic patterns; financial pretraining may encode different associations.

**Existing evidence:** Same downstream architecture does not imply same learned representations.

**What we still need:** Matched-corpus, matched-budget pretraining or credible upstream ablations.

**Test step by step:** At feasible scale, compare continued pretraining from one common base on matched-size general and financial corpora; keep objective, steps and tokenizer fixed. This tests adaptation, not the complete historical origin story.

**Check:** Would that reproduce both historical models? **Answer:** No. It tests a controlled adaptation effect, not their entire histories.

### A5. Limited hyperparameter search.

**Start with an example:** A four-point search might miss a useful lower learning rate.

**Mechanism:** **Limited hyperparameter search.** The two-by-two grid may be closer to BERT's optimum.

**Existing evidence:** Only two learning rates and two epoch budgets were searched at one seed.

**What we still need:** Wider multi-seed validation response surfaces.

**Test step by step:** Equal-compute random/Bayesian search over rate, warmup, decay and dropout, using development-only selection and a locked confirmation protocol. Report budget, failed trials and seed sensitivity.

**Check:** May we enlarge only FinBERT’s search and call the comparison equal-budget? **Answer:** No. Account for and match the resources or state the different question.

### A6. Fine-tuning instability.

**Start with an example:** A high first update can move many pretrained weights before the classifier becomes useful.

**Mechanism:** **Fine-tuning instability.** Early updates may move representations into different basins.

**Existing evidence:** FinBERT seed 123 selected epoch 1 while other final seeds selected epoch 4; larger observed SD.

**What we still need:** Gradient/representation trajectories and repeated controlled optimization changes.

**Test step by step:** Compare warmup, lower/layer-wise learning rates and frozen-layer schedules one at a time, recording validation curves and gradient norms. Do not infer instability solely from a selected epoch.

**Check:** Does selecting epoch 1 prove instability? **Answer:** No. It motivates examining curves and controlled optimization changes.

### A7. Catastrophic forgetting.

**Start with an example:** Check a separate language task before and after sentiment training.

**Mechanism:** **Catastrophic forgetting.** Supervised updates may erase useful pretrained representations.

**Existing evidence:** No direct forgetting measurement exists.

**What we still need:** Before/after performance on retained financial language tasks or representation probes.

**Test step by step:** Evaluate a separate, disjoint financial language probe before/after fine-tuning; compare lower-layer freezing or regularization at equal development budgets. Improvement plus retained probe performance would support, not uniquely prove, this mechanism.

**Check:** Is lower sentiment accuracy sufficient evidence of forgetting? **Answer:** No. We must measure loss of previously demonstrated capability.

### A8. Ambiguous labels and imbalance.

**Start with an example:** Two annotators can reasonably disagree about “revenue rose but margins fell.”

**Mechanism:** **Ambiguous labels and imbalance.** Domain-specific cues may conflict with majority labels or minority class boundaries.

**Existing evidence:** Large allagree/non-allagree gaps and class imbalance exist for both models.

**What we still need:** True vote distributions, category-stratified errors and balanced controlled training.

**Test step by step:** Acquire independently reviewed labels/votes; compare hard-label training, agreement-aware weighting and soft labels on identical examples. Report classwise calibration and minority recall, not just aggregate F1.

**Check:** Would deleting difficult labels solve the deployment task? **Answer:** It changes the population and may hide the hard cases users still encounter.

### A9. General language understanding.

**Start with an example:** Compare “A acquired B” with “B acquired A” while holding the names fixed.

**Mechanism:** **General language understanding.** Negation, entity relations and clause composition may matter more than financial vocabulary.

**Existing evidence:** Selected categories include entity context and implicit sentiment, but are not representative causal evidence.

**What we still need:** Human-validated challenge sets with matched lexical content and changed syntax/relations.

**Test step by step:** Build contrast pairs for entity roles, negation scope and conflicting clauses; balance financial terms and sentence length, blind annotators, compare fixed models before retraining.

**Check:** Do shared words guarantee shared meaning? **Answer:** No. Relations and syntax change who did what.

## Research notebook B: improvement cards

Each card states its mechanism, cost, risk and proper evaluation. These proposals build on Lessons 3–5; none is an achieved improvement.

### B1. New independent holdout / P0

**Concrete starting point:** Reserve later news by event before developing a new model. This is a proposal, not a completed experiment.

**Why it could help:** Makes improvement claims credible after repeated test inspection

**Cost and complexity:** Annotation/provenance effort high; model compute low

**What can go wrong:** Hidden overlap, inconsistent label policy

**Walkthrough for evaluation:** Deduplicate across all old data; lock collection/label rules and holdout access

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B2. Systematic equal-budget search / P1

**Concrete starting point:** Try different rates with the same total fit budget for both encoders. This is a proposal, not a completed experiment.

**Why it could help:** Finds model-specific optimizer settings missed by four trials

**Cost and complexity:** Several to many more fits; orchestration moderate

**What can go wrong:** Validation overfitting, unfair search budgets

**Walkthrough for evaluation:** Nested/grouped development splits or locked confirmation split; all trials logged

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B3. Warmup, layer-wise rates and regularization / P1

**Concrete starting point:** Begin updates gently; optionally update lower layers more slowly. This is a proposal, not a completed experiment.

**Why it could help:** Avoids disruptive initial updates; constrains small-data fitting

**Cost and complexity:** Modest extra fits; moderate parameter-group implementation

**What can go wrong:** Underfitting; many interacting knobs

**Walkthrough for evaluation:** One-factor ablations before combinations; paired seeds and fixed budgets

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B4. Class weighting / P2

**Concrete starting point:** Give a negative training example a larger loss weight. This is a proposal, not a completed experiment.

**Why it could help:** Increases loss pressure on minority labels

**Cost and complexity:** Near-zero per-step overhead; simple

**What can go wrong:** Reduced probability calibration; overemphasis on noisy minority cases

**Walkthrough for evaluation:** Per-class precision/recall, macro-F1 and calibration; weights estimated from train only

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B5. Focal loss / P2

**Concrete starting point:** Reduce the loss pressure from an already easy example. This is a proposal, not a completed experiment.

**Why it could help:** Downweights easy examples to emphasize difficult ones

**Cost and complexity:** Small overhead; custom loss moderate

**What can go wrong:** May focus on mislabeled/ambiguous examples

**Walkthrough for evaluation:** Tune gamma on validation; compare with weighting and standard CE at equal budgets

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B6. Label smoothing / P2

**Concrete starting point:** Use targets 0.9333/0.0333/0.0333 instead of 1/0/0. This is a proposal, not a completed experiment.

**Why it could help:** Discourages extreme one-hot confidence

**Cost and complexity:** Negligible compute; simple

**What can go wrong:** Blurs clear labels and can hurt probability interpretation

**Walkthrough for evaluation:** Validation NLL/Brier, F1 and subgroup calibration; no test-tuned smoothing

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B7. Soft-label training / P2 if votes acquired

**Concrete starting point:** Use actual votes such as 6/10 positive, 3/10 neutral and 1/10 negative. This is a proposal, not a completed experiment.

**Why it could help:** Learns annotator probability distributions instead of forcing a majority label

**Cost and complexity:** Annotation cost high; loss code simple

**What can go wrong:** Votes reflect annotator biases; current bands are insufficient

**Walkthrough for evaluation:** Obtain actual votes; score against held-out adjudicated and distributional labels

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B8. Agreement-aware handling / P1 diagnostic, P2 training

**Concrete starting point:** Inspect clear and disputed examples separately before inventing weights. This is a proposal, not a completed experiment.

**Why it could help:** Separates uncertainty due to ambiguous examples

**Cost and complexity:** Low diagnostic compute; weighting design moderate

**What can go wrong:** Discarding non-allagree changes target population and hides hard cases

**Walkthrough for evaluation:** Preserve all evaluation strata; test weighting versus no weighting and report coverage

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B9. Financial-news DAPT / P3

**Concrete starting point:** Continue masked-word learning on lawful financial news. This is a proposal, not a completed experiment.

**Why it could help:** Aligns representations with the target genre

**Cost and complexity:** Much more compute and corpus governance

**What can go wrong:** Contamination, forgetting, licensing and over-specialization

**Walkthrough for evaluation:** Equal-token controls, time separation and independent news holdout

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B10. TAPT on training-associated text / P2

**Concrete starting point:** Continue masked-word learning on training-side task text. This is a proposal, not a completed experiment.

**Why it could help:** Adapts MLM representations to task language

**Cost and complexity:** Additional pretraining; moderate

**What can go wrong:** Tiny corpus memorization; accidental use of held-out text

**Walkthrough for evaluation:** Use only authorized training-side unlabeled material; equal-step no-adaptation control

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B11. Continued pretraining more broadly / P3

**Concrete starting point:** Check whether more recent text changes both useful and harmful associations. This is a proposal, not a completed experiment.

**Why it could help:** Refreshes vocabulary usage or domain context

**Cost and complexity:** Potentially substantial GPU/storage cost

**What can go wrong:** Drift, forgetting and inability to isolate data versus compute

**Walkthrough for evaluation:** Declare corpus cutoff, budget and retention tests; treat DAPT/TAPT as specific forms, not separate magic gains

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B12. Financially constrained augmentation / P2

**Concrete starting point:** Paraphrase a sentence only when its financial meaning is preserved. This is a proposal, not a completed experiment.

**Why it could help:** Expands linguistic forms while preserving reviewed labels

**Cost and complexity:** Generation/review moderate to high

**What can go wrong:** Synthetic artifacts; invalid labels or duplicated event content

**Walkthrough for evaluation:** Human-check augmentation rules; split by original source; evaluate natural independent text

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B13. Active learning / P2

**Concrete starting point:** Ask reviewers to label a diverse set of uncertain new sentences. This is a proposal, not a completed experiment.

**Why it could help:** Allocates annotation to uncertain/diverse examples

**Cost and complexity:** Repeated inference and annotation cycles

**What can go wrong:** Sampling bias; uncertainty may select noise

**Walkthrough for evaluation:** Reserve a random untouched evaluation sample; compare equal annotation budgets

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B14. Knowledge distillation / P2 for deployment

**Concrete starting point:** Train a smaller student using a fixed teacher’s probabilities. This is a proposal, not a completed experiment.

**Why it could help:** Transfers predictive behavior to a smaller student

**Cost and complexity:** Teacher inference plus student training

**What can go wrong:** Copies teacher errors; may lose calibration/rare skills

**Walkthrough for evaluation:** Compare accuracy-cost frontier, calibration, subgroup recalls and measured latency

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B15. Ensembling / P2

**Concrete starting point:** Average probability vectors, then evaluate the resulting classifier. This is a proposal, not a completed experiment.

**Why it could help:** Reduces some model/seed-specific errors

**Cost and complexity:** Multiple models at inference; storage multiplies

**What can go wrong:** Correlated errors and costly explanations

**Walkthrough for evaluation:** Select ensemble weights on validation; evaluate mean probabilities as a new workflow

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B16. Multi-task learning / P3

**Concrete starting point:** Train entity relations alongside sentiment with separate ablations. This is a proposal, not a completed experiment.

**Why it could help:** Shares supervision for entities, relations or financial quantities

**Cost and complexity:** Additional labeled tasks; architecture moderate/high

**What can go wrong:** Negative transfer and dataset mismatch

**Walkthrough for evaluation:** Single-task and auxiliary-task ablations; matched compute and disjoint evaluation

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B17. Validation-only calibration / P1

**Concrete starting point:** Fit a temperature on validation logits, leaving predicted classes unchanged. This is a proposal, not a completed experiment.

**Why it could help:** Improves confidence for review/abstention

**Cost and complexity:** Very low compute; simple scalar temperature

**What can go wrong:** Small calibration set; transfer failure under shift

**Walkthrough for evaluation:** Separate validation calibration split/cross-fitting; NLL/Brier/ECE and coverage-risk on new holdout

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B18. Robustness-driven training / P2

**Concrete starting point:** Teach a reviewed invariance such as a harmless company rename. This is a proposal, not a completed experiment.

**Why it could help:** Enforces justified invariances or correct contrastive changes

**Cost and complexity:** Extra examples/forward passes; moderate

**What can go wrong:** Incorrect invariances harm semantics

**Walkthrough for evaluation:** Human-validated edits with known labels, unseen edit families and natural-text safeguards

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

### B19. Longer context where needed / P2 diagnostic

**Concrete starting point:** Compare 64 and 128 tokens only after checking where evidence was cut off. This is a proposal, not a completed experiment.

**Why it could help:** Retains truncated evidence

**Cost and complexity:** More activation/attention cost

**What can go wrong:** Extra irrelevant text; confounds context and batch/steps

**Walkthrough for evaluation:** Analyze truncated development cases first; compare lengths with matched effective batch and report both truncated/nontruncated strata

**Check:** What would make this proposal a defensible improvement? **Answer:** A matched development comparison followed by independent confirmation, with its stated risks measured—not merely a higher score after repeated test inspection.

## Research notebook C: choose a model for the constraint

A frozen embedding classifier turns each sentence into a fixed feature vector and fits a small head. A zero-shot LLM uses instructions without labeled examples in its prompt; few-shot adds examples. A cascade routes only selected inputs to a more expensive component. None has been benchmarked here beyond the completed workflows.

| Family | Accuracy potential and data needs | Compute/latency/cost | Explainability and calibration | Deployment complexity |
|---|---|---|---|---|
| TF-IDF/linear | Strong lexical baseline; benefits from enough labeled examples; observed ceiling here is not universal | Very low train/inference cost, CPU-friendly | Coefficients inspectable but correlated features complicate interpretation; calibration still needs validation | Simple artifact/version management |
| Modern encoders: RoBERTa, DeBERTa, ModernBERT | Plausible representation gains; requires task validation and comparable search | Depends on size/context/kernel; measure on target GPU/CPU | New hooks and tokenization require explanation validation; no automatic calibration guarantee | Moderate; checkpoint/tokenizer compatibility |
| Finance-specific encoders | Potential domain vocabulary/style benefit | Similar size can have similar runtime; upstream rights vary | Corpus contamination and domain mismatch need auditing | Moderate plus provenance/licensing review |
| Frozen sentence embeddings + classifier | Useful with few labels; limited adaptation to subtle financial distinctions | Embedding inference dominates, head fitting cheap; cached features help | Linear head is interpretable in embedding space, not directly in words; calibrate separately | Moderate; separate embedding and classifier versions |
| Zero/few-shot LLM | Can use instructions and examples with little supervised fitting; performance unknown here | Prompt-token cost, latency and service variability; local large models need memory | Verbal rationales are not faithful internal explanations; output probabilities often unavailable/comparable only with care | High: prompts, parsing, privacy, rate limits, version drift |
| Fine-tuned small LLM | Can specialize outputs and multi-task behavior | More sequence-generation overhead than a three-logit head; adapters may reduce training state | Label-token probabilities need normalization/protocol; explanations remain unvalidated | Higher than encoder classification; serving/quantization controls |
| Hybrid/ensemble | Combines complementary errors or routes difficult cases | Cost varies with cascade coverage; worst-case latency matters | Routing and abstention must be calibrated and audited | Highest orchestration complexity; monitor component and route drift |

**Check:** Which family has the best latency? **Answer:** We cannot rank unmeasured systems from architecture names alone. Measure on the intended hardware, input lengths and batch/concurrency settings. The table describes likely cost drivers, not benchmark timings.

## Research notebook D: a sequence of controlled experiments

Read these as an execution proposal. No training is authorized or performed by this learning material. Each item supplies a hypothesis, controls and acceptance criteria.

### P0: build an independent evaluation set

**Question to test:** The historical benchmark may look easier than later, more varied news.

**Steps:** Collect lawful, dated news with event IDs and multiple annotators. Define labels and adjudication rules before reviewing model outputs. Deduplicate against all old data; reserve a locked temporal/event-group holdout.

**Keep controlled:** Keep the same label policy and model versions. Do not use holdout labels for development.

**Acceptance rule:** Complete provenance, no detected cross-group overlap, and documented class/agreement support. These are data-quality requirements, not a model-score win.

**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.

### P1: compare equal-budget optimization

**Question to test:** A wider, multi-seed validation search may reduce FinBERT’s gap or variation.

**Steps:** Keep training rows, input length and optimizer-step accounting comparable. Give both encoders the same declared compute budget and prespecified seeds. Select on development data; confirm once on the new holdout.

**Keep controlled:** Match data access and resources; log every trial, including failures.

**Acceptance rule:** A prespecified useful macro-F1 improvement with paired uncertainty and no unacceptable minority-recall loss. Set the numeric margin from use-case needs before seeing results.

**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.

### P1: test calibration and a review policy

**Question to test:** A validation-fitted temperature may improve probability quality without changing class choices.

**Steps:** Reserve calibration data or use an appropriate development cross-fitting scheme. Fit temperature while keeping model weights fixed. Evaluate NLL, Brier, ECE and accepted-case error versus coverage on independent data.

**Keep controlled:** Use identical model logits and evaluation rows before/after fitting; do not tune on the holdout.

**Acceptance rule:** Improved held-out probability scores with uncertainty and an agreed review workload/error target. One favorable ECE binning alone is insufficient.

**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.

### P1: test quantities and relations directly

**Question to test:** Some errors may concern comparisons, entity roles or negation rather than missing financial vocabulary.

**Steps:** Create human-reviewed matched pairs: revenue up/down, losses up/down, percentages with clear denominators, swapped entity roles and conflicting clauses. Keep challenge-test labels out of training.

**Keep controlled:** Match vocabulary, length and amount format where feasible. Blind reviewers to model predictions.

**Acceptance rule:** Prespecified contrast accuracy/consistency with class-specific results, plus no unacceptable deterioration on natural independent news.

**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.

### P2: test one motivated intervention at a time

**Question to test:** One diagnosed bottleneck may respond to warmup, weighting, soft labels or news adaptation.

**Steps:** Use the previous diagnostics to choose an intervention. Start with one-factor comparisons; use a factorial design only if interactions matter and resources allow.

**Keep controlled:** Common initialization/tokenizer where feasible, equal steps and data access, paired seeds. Document unavoidable differences.

**Acceptance rule:** A result that replicates on a second independent slice while meeting calibration and compute safeguards. Report negative outcomes too.

**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.

### P2: broaden explanation checks

**Question to test:** The ordering of explanation methods may depend on reference and intervention choices.

**Steps:** Prespecify PAD/MASK or multiple-reference comparisons, whole-word masking and natural contrast tests on a new sample. Add parameter-randomization checks and more than one checkpoint seed.

**Keep controlled:** Fixed targets, sampling units, budgets and convergence thresholds. A numerical investigation gets a new version; historical records remain untouched.

**Acceptance rule:** Transparent effect/stability intervals across protocols. Do not choose a winning protocol afterward or silently redefine numerical success.

**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.

### P2: measure a deployable cost–quality trade-off

**Question to test:** A student or cascade may satisfy a latency target with acceptable error.

**Steps:** Measure warm and cold starts, input-length mixtures, batch size and concurrency. Evaluate abstention/review and a rollback plan alongside prediction performance.

**Keep controlled:** Fixed hardware, precision, workload and independent traffic-like data.

**Acceptance rule:** Prespecified p95 latency, memory, calibration and error-cost bounds. Passing them would support a later deployment decision, not automatically authorize publication or deployment.

**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.

**Check:** Why is collecting a holdout first priority even though it does not improve weights? **Answer:** Without an independent check, we cannot tell whether later improvements generalize beyond examples that shaped our choices.

## Practice explaining it

1. **What could weaken the domain-mismatch hypothesis?**

   Repeated matched financial-news adaptation failing to help while another controlled intervention explains the gap would weaken it; one negative result does not prove impossibility.

2. **Why not reconstruct soft labels from agreement bands?**

   The exact votes across all three classes are missing.

3. **Why can focal loss hurt?**

   Hard examples may be ambiguous or mislabeled, so emphasizing them can amplify noise.

4. **What is the first prerequisite for a meaningful improvement claim?**

   A clearly defined goal and an independent evaluation population that has not shaped the intervention.

5. **How do you defend an unexpected result?**

   State what was measured, show the uncertainty, separate competing mechanisms, and propose a test that could prove your preferred explanation wrong.

## Sources and next lesson

[Domain/task adaptation research](https://aclanthology.org/2020.acl-main.740/), [calibration research](https://proceedings.mlr.press/v70/guo17a.html), [architecture and attribution papers](references.md), [unanswered evidence checklist](evidence_gaps.md). Practice with [interview questions](interview_preparation.md).
