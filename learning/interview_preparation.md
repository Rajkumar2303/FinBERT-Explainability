# Explain it aloud: interview practice

Use this after the relevant lesson, not as a list of definitions to memorize. Read the question, answer without looking, then compare your reasoning. The short answer is a starting point; the deeper explanation adds an example or project evidence. Each follow-up includes an answer so you can check the boundary of the claim.

Examples explicitly called toy are invented for teaching. Project numbers are measured. For full equations and code, follow the chapter link at each group. [Chapter 00](00_project_overview.md) supplies the 30-second, 2-minute and 10-minute project explanations.

## Beginner fundamentals

Revisit the [worked lessons](01_foundations_and_theory.md) before practicing these questions.

### Q01. What does this project predict?

**Spoken answer:** It predicts whether an English financial sentence is negative, neutral or positive.

**Deeper explanation:** Think of “profits rose” as a possible positive example. The label is a sentiment judgment, not a forecast of tomorrow’s stock price. The model does not verify that the news is true.

**Follow-up:** Could it trade stocks? Not on this evidence; that would require an entirely different target, temporal evaluation and decision-cost study.

### Q02. Why is financial sentiment difficult?

**Spoken answer:** The same word can mean different things depending on what changed.

**Deeper explanation:** Compare “profits rose” with “losses rose.” The verb is identical, but the financial direction differs. Names, comparisons and negation can matter too; this does not prove which mechanism caused a particular saved error.

**Follow-up:** Does a growth word guarantee positivity? No; the variable, expectations and context determine the interpretation.

### Q03. What is a token?

**Spoken answer:** A token is one piece of text the model processes.

**Deeper explanation:** A piece may be a word, punctuation or part of a word. In a toy WordPiece split, a word could become a starting piece plus `##` continuation. The actual tokenizer determines the pieces, so word count and token count need not match.

**Follow-up:** Is token count word count? No, and model-specific vocabularies can produce different counts.

### Q04. What is an embedding?

**Spoken answer:** An embedding is a learned list of numbers for a token.

**Deeper explanation:** The token ID works like a row number in a lookup table. The row contains continuous values that the model can change during learning. Toy vectors [1,0] and [0,1] illustrate the lookup; real BERT vectors have 768 coordinates, without a guaranteed human meaning for each coordinate.

**Follow-up:** Does one coordinate mean profitability? There is no such guaranteed semantic interpretation.

### Q05. What is the difference between a logit and probability?

**Spoken answer:** A logit is a raw class score; softmax turns the scores into probabilities.

**Deeper explanation:** For toy logits [0,0,0], softmax gives one third to each class. Adding 10 to every score leaves those probabilities unchanged. Therefore probabilities alone do not recover the exact original logits needed by our IG target.

**Follow-up:** Does 0.99 probability mean 99% reliability? Only if calibration and the evaluation population support that interpretation.

## Project-specific implementation

Revisit the [worked lessons](02_dataset_and_preprocessing.md) before practicing these questions.

### Q06. What are the six phases?

**Spoken answer:** We froze data, compared models, checked reliability, computed IG, compared attention, and audited the evidence.

**Deeper explanation:** Each step answers the next question: Is the test fair? Which workflow predicts well? Where does it fail? What affects one score? Do explanation rankings behave usefully? Can every conclusion be traced to saved records?

**Follow-up:** Is Phase 6 another training run? No, it recomputes and audits saved outputs.

### Q07. Why did 2199 split assignments move?

**Spoken answer:** The algorithm rebuilt the whole split while keeping duplicate groups together.

**Deeper explanation:** Imagine repacking every item into balanced boxes rather than moving only one conflicting pair. The objective did not reward retaining old assignments. Of 2199 moved rows, 2143 were singletons and 56 were in duplicate groups.

**Follow-up:** Does that imply 2199 duplicates? No; pair counts and assignment changes are different quantities.

### Q08. Why explicitly load BertConfig?

**Spoken answer:** We explicitly used the verified BERT configuration so the software loaded the intended architecture.

**Deeper explanation:** A checkpoint name is not a sufficient interface guarantee. The financial configuration needed explicit handling in this Transformers environment. The classifier uses canonical labels, and IG checks the actual BERT class and embedding module before attaching its hook.

**Follow-up:** Would this work unchanged for DeBERTa? No; architecture-specific hooks must be validated.

### Q09. How are external labels mapped?

**Spoken answer:** We read the checkpoint’s label names before interpreting probability columns.

**Deeper explanation:** ProsusAI IDs 0/1/2 mean positive/negative/neutral; ours mean negative/neutral/positive. A toy highest value in column 0 would be positive for ProsusAI but negative under our mapping. We reorder both labels and probability columns.

**Follow-up:** Can correct argmax indices still produce wrong labels? Yes, if interpreted under the wrong mapping.

### Q10. Why choose FinBERT seed 42 for explanations?

**Spoken answer:** Seed 42 was fixed explicitly instead of chosen because its test results looked good.

**Deeper explanation:** Picking the prettiest heatmap or highest-scoring seed afterward would favor an attractive story. A fixed checkpoint avoids that choice. It is still one model, so the explanation result does not automatically describe other seeds.

**Follow-up:** Is it the average FinBERT? No, it is one realized checkpoint.

## Transformer architecture

Revisit the [worked lessons](08_attention_mechanisms.md) before practicing these questions.

### Q11. Explain Q, K and V.

**Spoken answer:** Queries ask what information to look for, keys help match it, and values supply what gets mixed.

**Deeper explanation:** For “losses increased,” the representation at “increased” may use information from “losses.” Learned projections produce query, key and value vectors; query–key scores become weights, and those weights mix values. Chapter 08 works through the numbers.

**Follow-up:** Does a high attention weight guarantee high effect? No; value content and downstream transformations matter.

### Q12. What are the attention dimensions here?

**Spoken answer:** For each layer, attention has batch, head, query-token and key-token dimensions.

**Deeper explanation:** With one 64-token input, a layer returns [1,12,64,64]. We take the CLS query row and average its 12 heads in the final layer for the main comparison. All-layer CLS rows are saved, but full query-by-key matrices are not saved by default.

**Follow-up:** Is 0.2406 based on all query rows? No; it uses the declared CLS summary.

### Q13. Why are positions necessary?

**Spoken answer:** Positions let the model distinguish the order of the same words.

**Deeper explanation:** “A acquired B” and “B acquired A” contain the same names but describe different roles. BERT adds learned position vectors. Our masking test leaves positions in place rather than shifting all later tokens as physical deletion would.

**Follow-up:** Does the model's 512-position capacity imply 512-token evaluation? No, the protocol caps inputs at 64.

### Q14. What do residual connections and LayerNorm do?

**Spoken answer:** Residual connections keep a direct route for existing information; LayerNorm rescales a token’s coordinates.

**Deeper explanation:** For a toy residual, input [1,2] plus an update [0.2,-0.1] gives [1.2,1.9]. LayerNorm then centers/scales coordinates using learned adjustments. These routes explain why attention weights alone are not the whole model calculation.

**Follow-up:** Is LayerNorm batch-dependent? Its normalization is across a token's hidden coordinates, unlike batch normalization.

### Q15. Encoder versus decoder: why BERT?

**Spoken answer:** BERT reads both sides of a complete sentence, which fits this classification task.

**Deeper explanation:** An encoder can use “not” and the following words together. A causal decoder reads only prior positions when generating the next token. Decoders can classify too, but that alternative was not measured here.

**Follow-up:** Did this prove encoders are best? No, it tested selected encoder workflows.

## Training and optimization

Revisit the [worked lessons](04_baselines_and_training.md) before practicing these questions.

### Q16. What was the hyperparameter search?

**Spoken answer:** We selected settings on validation data, then repeated the chosen setup with three seeds.

**Deeper explanation:** Each encoder searched learning rates 2e-5/3e-5 and budgets 3/4 epochs at seed 42. BERT selected 3e-5 and 3 epochs; FinBERT selected 2e-5 and 4. The baseline searched three C values times two n-gram choices.

**Follow-up:** Is equal search space equal optimality? No; either model's optimum may lie outside the grid.

### Q17. What is effective batch size?

**Spoken answer:** It is the number of examples contributing to one optimizer update.

**Deeper explanation:** One GPU processes 8 examples, accumulates another 8, then updates: 8 × 2 = 16. This saves activation memory. It need not be byte-identical to a physical batch of 16 because floating-point order and dropout can differ.

**Follow-up:** What if OOM occurred? Reduce microbatch size and increase accumulation to preserve the intended effective batch, documenting changes; none occurred here.

### Q18. Why AdamW and mixed precision?

**Spoken answer:** AdamW adjusts update sizes using recent gradients, and mixed precision reduces training memory and arithmetic cost.

**Deeper explanation:** Weight decay gently pulls weights toward zero separately from Adam’s adaptive update. The recorded run used decay 0.01, a linear learning-rate schedule, no warmup and gradient clipping at 1.0. FP16 loss scaling can skip overflowing updates; that is different from running out of GPU memory.

**Follow-up:** Is a loss-scaling overflow an OOM? No, it is a numerical event, not exhausted memory.

### Q19. Which epoch's model is evaluated?

**Spoken answer:** The checkpoint with the highest validation macro-F1 in each fit is evaluated.

**Deeper explanation:** A four-epoch budget does not mean epoch 4 is always exported. FinBERT seed 123 selected epoch 1; seeds 42 and 2026 selected epoch 4. All BERT final fits selected epoch 3. Test scores did not select the epoch.

**Follow-up:** Can you replace it with the best test epoch? No, that would use the test for selection.

### Q20. What does reproducibility require beyond a seed?

**Spoken answer:** A seed is only one part of repeatability.

**Deeper explanation:** Keep data, row order, model revision, code, packages and numerical settings fixed too. A hash lets us check that a saved file has not changed. The byte-identical seed-42 repeats demonstrate repeatability in this environment, not every possible GPU or library version.

**Follow-up:** Does baseline SD zero eliminate uncertainty? No, it removes only observed seed variation in those runs.

## Validation and leakage

Revisit the [worked lessons](05_evaluation_and_validation.md) before practicing these questions.

### Q21. Why macro-F1?

**Spoken answer:** It gives negative, neutral and positive sentiment equal influence in the final score.

**Deeper explanation:** Compute each class’s F1, then average the three. The baseline’s real negative recall is only 42/91 = 0.4615 despite accuracy 0.7442. Macro-F1 0.6713 makes that broader weakness harder to hide behind the neutral majority.

**Follow-up:** Is macro-F1 F1 of macro precision/recall? No.

### Q22. What does zero detected leakage mean?

**Spoken answer:** It means no pair found by our stated detector crosses the saved splits.

**Deeper explanation:** The detector normalizes text and looks for character-pattern similarity at cosine 0.90. It can miss a paraphrase with different words or two stories about one event. “No detected leakage” is a bounded claim, not proof of all possible independence.

**Follow-up:** Why connected components? To keep indirect duplicate chains together.

### Q23. Is fitting the duplicate detector on all text leakage?

**Spoken answer:** Grouping defines the split before model fitting; classifier features must still learn only from training rows.

**Deeper explanation:** Imagine using all candidate sentences to keep near copies in the same exam partition. That is different from letting the classifier learn its vocabulary statistics from the exam. The project uses separate TF-IDF fits for those two jobs.

**Follow-up:** Would test-informed feature engineering be acceptable? It would compromise future independent improvement claims.

### Q24. Why is ProsusAI non-independent?

**Spoken answer:** Its earlier sentiment training used PhraseBank, so our test may not be new to it.

**Deeper explanation:** Making a new local split cannot undo upstream exposure. We keep its 0.8797 macro-F1 as a reference, but do not use it to declare an independent winner.

**Follow-up:** Can its low ECE prove superior deployment reliability? Not without independent evaluation.

### Q25. SD versus confidence interval?

**Spoken answer:** Seed SD describes how fits varied; a confidence interval describes uncertainty for a stated comparison.

**Deeper explanation:** Toy seed scores 0.82/0.84/0.86 have mean 0.84 and sample SD 0.02. The explanation bootstrap instead holds a checkpoint fixed and repeatedly resamples sentences. These procedures vary different things.

**Follow-up:** Can non-overlapping SD bars prove significance? No.

## Calibration and robustness

Revisit the [worked lessons](06_calibration_and_robustness.md) before practicing these questions.

### Q26. How can BERT have better F1 but worse ECE?

**Spoken answer:** Choosing the right class and stating a realistic confidence are different skills.

**Deeper explanation:** A model can be right 80% of the time while claiming 99% confidence. BERT has better mean F1 here but higher ECE-10. Brier and NLL favor BERT on mean, so “best calibrated” needs a named metric and population.

**Follow-up:** Which model is “best calibrated”? The evidence needs metric, seed and population qualifications.

### Q27. What are ECE's weaknesses?

**Spoken answer:** ECE depends on how we group confidence values and can hide differences inside a group.

**Deeper explanation:** If one bin mixes underconfident and overconfident cases, their average gaps can cancel before the absolute value is taken. Chapter 06’s four-row example changes from 0.4 to zero when groups are merged. We report both 10 and 15 bins.

**Follow-up:** Is ECE a proper scoring rule? Not the same as Brier or log loss; don't use it as the only optimization target.

### Q28. Explain the Brier convention.

**Spoken answer:** We add squared probability errors across all three classes, then average over sentences.

**Deeper explanation:** Toy truth positive and probabilities [0.1,0.2,0.7] give 0.1² + 0.2² + (0.7−1)² = 0.14. We do not divide that sum by three. This convention ranges from 0 to 2; NLL for the same row is −ln(0.7) ≈ 0.3567.

**Follow-up:** Why clip NLL probabilities? To avoid numerical log zero; the clip is documented as 1e-15.

### Q29. Does zero numerical flip rate prove reasoning?

**Spoken answer:** No. A stable model might understand the edit, or it might ignore the number.

**Deeper explanation:** Our currency edits double selected amounts; many such changes do not determine a new sentiment label. We also check whether the edit changes represented tokens or TF-IDF features. A number beyond truncation cannot test the model’s response.

**Follow-up:** Why check visibility? An edit outside the represented input cannot test the model's response.

### Q30. How would you calibrate the model?

**Spoken answer:** I would fit a positive temperature using validation logits, then check it on new held-out data.

**Deeper explanation:** Dividing logits by a positive temperature changes confidence without changing their order. Temperature 2 softens a distribution more than temperature 1. The completed project did not fit temperature scaling, and the test must not choose it.

**Follow-up:** Could you optimize temperature on the frozen test? No, that would invalidate held-out calibration assessment.

## Integrated Gradients

Revisit the [worked lessons](07_integrated_gradients.md) before practicing these questions.

### Q31. Derive IG in one sentence.

**Spoken answer:** IG adds up how the target score changes along a path from a reference input to the real input.

**Deeper explanation:** For the toy score F(x)=3x and reference 0, moving to x=2 changes the score by 6. The gradient is always 3, so IG is (2−0)×3=6. A neural network needs many sampled gradients because its slope changes along the path.

**Follow-up:** Which scalar here? The original predicted-class logit.

### Q32. Why use LayerIntegratedGradients?

**Spoken answer:** Token IDs are lookup addresses, so we interpolate continuous embedding outputs instead.

**Deeper explanation:** Halfway between token ID 10 and 20 is not half of each word; it is another address. LayerIntegratedGradients works at `model.bert.embeddings` output, after position/type addition and LayerNorm. Its scores have shape batch × length × 768.

**Follow-up:** What is the attribution shape? Batch × sequence length × hidden width.

### Q33. What exactly is the PAD baseline?

**Spoken answer:** Content IDs become PAD, while CLS, SEP, actual padding and the original attention mask stay fixed.

**Deeper explanation:** The inserted PAD references at content positions remain attended. For toy [CLS,profits,rose,SEP], the reference is [CLS,PAD,PAD,SEP] with the same active positions. This artificial reference is not a neutral financial sentence.

**Follow-up:** Why not call it absence of information? PAD embeddings and preserved position/type structure still affect the network.

### Q34. What is completeness and what failed?

**Spoken answer:** The attribution total should match the input-minus-reference target logit.

**Deeper explanation:** A toy score difference of 2 with attribution total 1.9994 has absolute residual 0.0006. Our full-test run retained 40 cases above the fixed 0.001 absolute threshold after 200 steps. All were below the 0.01 relative threshold; both conditions were required.

**Follow-up:** Can you loosen the threshold now? Only as a separately declared sensitivity study, never to silently redefine historical success.

### Q35. Signed versus absolute attribution?

**Spoken answer:** A signed score says which direction a token contributes for the chosen class; its magnitude ranks strength.

**Deeper explanation:** Toy embedding contributions +2 and −1 sum to +1; the absolute signed sum is 1, whereas the sum of absolute dimensions is 3. We save these separately. A positive score can support the negative class when that is the target.

**Follow-up:** Can an inhibitory token rank high? Yes, which can make its deletion increase target confidence.

## Attention and faithfulness

Revisit the [worked lessons](09_explainability_experiments.md) before practicing these questions.

### Q36. Is attention an explanation?

**Spoken answer:** Attention tells us how representations are mixed, but its usefulness as an explanation needs testing.

**Deeper explanation:** A large weight can multiply a weak value vector, and residual routes carry other information. We compare one attention summary with IG and masking behavior. Low agreement cannot disprove attention, because IG is not ground truth either.

**Follow-up:** Did low IG agreement disprove attention? No, IG is not ground truth.

### Q37. Why average final-layer CLS heads?

**Spoken answer:** It is a simple, declared summary near the classifier’s final representation.

**Deeper explanation:** Averaging makes one score per token from 12 heads. It also hides differences between heads and earlier routes. That is a defensible technical rationale, not evidence that this choice was empirically best.

**Follow-up:** Could you pick the best-correlating head now? That would be exploratory selection requiring separate confirmation.

### Q38. What does Spearman 0.2406 mean?

**Spoken answer:** It is a modest average association between two token rankings, not 24% explanation accuracy.

**Deeper explanation:** Each of 687 converged sentences contributes one correlation between absolute IG signed sums and final-layer CLS head-mean attention. Toy ranks [1,2,3] and [1,3,2] give Spearman 0.5. We do not pool all tokens into one long list.

**Follow-up:** How are ties handled? Averaged ranks for Spearman and tau-b for Kendall.

### Q39. Why is special-token attention not automatically bad?

**Spoken answer:** CLS and SEP can contain information collected from other tokens in earlier layers.

**Deeper explanation:** Mean SEP attention mass is about 29.80%. That describes where weights go, not how much computation is wasted. We report special-token mass separately and exclude specials from content ranking, while retaining them in IG completeness.

**Follow-up:** How would you test the role? With carefully controlled interventions and alternative path analyses, not mass alone.

### Q40. What does deletion AUC measure?

**Spoken answer:** It measures how much target probability remains as more ranked tokens are masked.

**Deeper explanation:** Toy probabilities 0.9, 0.5 and 0.2 at fractions 0, 0.5 and 1 give trapezoid area 0.525. A lower curve means stronger suppression of the original target under this masking rule. It does not automatically mean a causal explanation of natural text.

**Follow-up:** Why not retarget after a class flip? That would change the question mid-curve.

## Results and unexpected outcomes

Revisit the [worked lessons](10_results_and_discussion.md) before practicing these questions.

### Q41. Why did general BERT beat FinBERT?

**Spoken answer:** BERT had the higher measured mean, but we did not isolate the cause.

**Deeper explanation:** The gap is 0.8476 − 0.8339 = 0.0137. Corpus style, tokenization, tuning and training variation are competing hypotheses. I would test them with matched controls and a new holdout rather than pick one as a retrospective fact.

**Follow-up:** Which hypothesis would you test first? Equal-budget development search and low-cost tokenization/learning-curve diagnostics before expensive pretraining.

### Q42. What is the biggest subgroup finding?

**Spoken answer:** Performance was much stronger on unanimously labeled sentences.

**Deeper explanation:** BERT allagree macro-F1 is 0.9583 versus 0.7500 for non-allagree. The groups contain 338 and 389 examples and different class mixes. This motivates ambiguity-aware validation; it does not show that agreement itself caused the difference.

**Follow-up:** Should you discard ambiguous examples? Not if deployment includes them; that changes coverage and the target population.

### Q43. Did IG beat attention?

**Spoken answer:** IG’s average deletion AUC was slightly lower, but the paired interval did not clearly separate it from attention.

**Deeper explanation:** The means are 0.6555 and 0.6608. The IG-minus-attention 95% interval is [−0.0130,0.0024], spanning zero. Both have lower mean AUC than random 0.7866 under this protocol.

**Follow-up:** Are they equivalent? That requires a prespecified equivalence margin and suitable analysis; it was not established.

### Q44. What changes when the 40 failures are included?

**Spoken answer:** The main qualitative conclusion persists, but the population and means change.

**Deeper explanation:** All-727 Spearman is 0.2465 versus 0.2406 for converged-only. All-727 IG/attention AUC is 0.6353/0.6372, and its paired interval still spans zero. The added records remain flagged as numerically unreliable; inclusion does not repair them.

**Follow-up:** Are all-record estimates fully reliable? No, they deliberately include flagged numerical integrations as sensitivity evidence.

### Q45. What did human annotation establish?

**Spoken answer:** It records user categories for 20 selected errors, not a census of model failures.

**Deeper explanation:** There are 7 entity-context, 4 financial-context, 4 implicit-sentiment, 2 negation and 1 each complex-sentence, numerical-reasoning and label-ambiguity categories. Interpretations/comments are AI-assisted notes. They are not independent human proof of a neural mechanism.

**Follow-up:** Can you claim most failures are numerical? No; the user category numerical reasoning appears once in this selected sample.

## Critical research challenges

Revisit the [worked lessons](10_results_and_discussion.md) before practicing these questions.

### Q46. What is the strongest validity threat?

**Spoken answer:** The main risk is claiming more generality or independence than the evidence provides.

**Deeper explanation:** A frozen file protects bytes, but repeated test inspection can influence future decisions. One benchmark, three training seeds and one main explanation checkpoint also limit scope. The strongest threat depends on the specific claim being made.

**Follow-up:** Does hashing solve this? It preserves bytes, not independence or representativeness.

### Q47. Why not count tokens as bootstrap samples?

**Spoken answer:** Tokens in one sentence share context, so treating them as independent would exaggerate the amount of evidence.

**Deeper explanation:** We compute a statistic per sentence and resample sentences together across methods. Even sentence resampling assumes enough independence between sentences; shared events may still create dependence. Random-mask repeats also do not create new sentences.

**Follow-up:** What alternative unit could help? Source-event clusters, if reliable metadata exists.

### Q48. What is wrong with MASK as a causal intervention?

**Spoken answer:** MASK-heavy sentences may be unlike ordinary financial news.

**Deeper explanation:** Replacing “profits rose” with “MASK MASK” changes the input distribution. A score drop shows behavior under that artificial operation. It does not isolate a real-world causal effect, and different removal rules may rank explanation methods differently.

**Follow-up:** Is infilling perfect? No, it introduces another model and uncertain semantic changes.

### Q49. Does convergence establish faithfulness?

**Spoken answer:** No. It checks whether the numerical attribution total matches its defined score difference.

**Deeper explanation:** A toy sum can be exactly correct for an artificial reference that is unhelpful to a human. Completeness, baseline choice and explanation usefulness are separate questions. We preserve failures and test reference sensitivity instead of treating one check as certification.

**Follow-up:** What extra checks help? Baseline/intervention sensitivity, randomization controls, multiple seeds and independent contrast sets.

### Q50. How do you defend a negative result?

**Spoken answer:** I describe the result, show its uncertainty and avoid inventing a reason.

**Deeper explanation:** “We did not clearly separate IG and attention” is useful evidence about this protocol. It is not a failure to tell a positive story. A next experiment should distinguish hypotheses rather than repeat comparisons until one is favorable.

**Follow-up:** What if an interviewer insists on a cause? Offer competing hypotheses and the evidence needed to discriminate them.

## Future improvements

Revisit the [worked lessons](11_improvements_and_future_research.md) before practicing these questions.

### Q51. What would you do next with one day of compute?

**Spoken answer:** I would first diagnose the development data and decide what question the compute should answer.

**Deeper explanation:** Inspect tokenization, learning curves and calibration by seed. If a valid new evaluation plan exists, run a small equal-budget optimization study. One day of compute cannot substitute for independent labels or lawful new data.

**Follow-up:** Why not train a larger LLM immediately? It changes too many variables and does not solve evaluation or annotation uncertainty.

### Q52. Would class weighting or focal loss help?

**Spoken answer:** They might help minority errors, but can also amplify noisy examples.

**Deeper explanation:** Weighting multiplies a class’s loss. Focal loss downweights easy cases; with p=0.8 and gamma=2 its multiplier is 0.04. Test against ordinary loss with validation-only tuning and classwise/calibration safeguards.

**Follow-up:** What metrics safeguard the result? Per-class precision/recall, macro-F1 and proper probability scores.

### Q53. Can agreement bands become soft labels?

**Spoken answer:** No. A band is not the full set of votes across three labels.

**Deeper explanation:** Knowing that disagreement is at least 25% does not tell us how much was negative versus neutral. Real soft-label training needs actual vote proportions or another justified target model, which the current records do not provide.

**Follow-up:** Is label smoothing the same? No, it applies a chosen synthetic target distribution rather than observed disagreement.

### Q54. Why might financial-news continued pretraining help?

**Spoken answer:** It could make the model more familiar with the style of short financial news.

**Deeper explanation:** That is a hypothesis about genre, not proof that the existing checkpoint lacks financial knowledge. Start from the same base, compare equal-token adaptation controls, and check new news plus retained skills. Audit rights and contamination first.

**Follow-up:** Would a gain prove the original corpus caused the gap? Not uniquely; adaptation also adds compute and data, which controls must address.

### Q55. What is your acceptance criterion for improvement?

**Spoken answer:** I would set a useful metric gain and acceptable side effects before looking at the new holdout.

**Deeper explanation:** A positive F1 difference alone may not justify worse negative recall or unacceptable latency. Use paired uncertainty and a practical margin chosen for the application. The current project did not define a universal deployment threshold.

**Follow-up:** Can you choose the margin from this test gap? That would be post-hoc and needs disclosure, not confirmatory framing.

## Deployment and system design

Revisit the [worked lessons](11_improvements_and_future_research.md) before practicing these questions.

### Q56. Would you deploy this now?

**Spoken answer:** I would limit it to research or a monitored analyst-review pilot, subject to rights and further validation.

**Deeper explanation:** The model predicts sentiment in short English sentences. It does not forecast returns or prove robustness on current news. A pilot still needs reviewed outcomes, independent evaluation and an agreed way to handle uncertain cases.

**Follow-up:** Which model would you shortlist? General BERT based on this benchmark, without treating the shortlist as final production approval.

### Q57. What would a serving system log?

**Spoken answer:** Record enough to reproduce a decision and monitor errors without collecting unnecessary sensitive text.

**Deeper explanation:** Proposed logs include model/tokenizer versions, input length, probabilities, review decisions and later outcomes. Compare error and coverage by relevant groups. This is a serving design proposal, not a system already built in the project.

**Follow-up:** Does this logging exist already? This is a proposal, not a completed serving implementation.

### Q58. How would you reduce latency?

**Spoken answer:** Measure the bottleneck first, then compare smaller models, batching or a review-aware cascade.

**Deeper explanation:** A smaller student can be cheaper but may lose rare-class skills. A cascade can send easy cases to a cheap model and difficult cases elsewhere. Measure p95 latency, memory, confidence and errors on the intended workload; training time is not serving latency.

**Follow-up:** Can saved IG be reused after quantization? Not as if it explained identical weights/computation; version the new analysis.

### Q59. What prevents a reproducibility package from being self-contained?

**Spoken answer:** Data rights, model rights, large weights and exact environment availability still matter.

**Deeper explanation:** A public code repository can link to lawful sources without bundling all data and checkpoints. Financial-model license details need care, and package pins do not guarantee every historical wheel/driver remains available. An audit documents these limits rather than erasing them.

**Follow-up:** Is a public model card enough to grant redistribution rights? Do not assume so, especially for the exact pinned revision.

### Q60. How would you explain the project to a validator rather than an engineer?

**Spoken answer:** Start with intended use, the evidence chain, uncertainty and remaining risks.

**Deeper explanation:** An engineer may ask how tensors flow; a validator asks whether the test is fair and the conclusion justified. I would show the same source-linked results, explain the non-independent reference and annotation exception, and state what needs a new holdout.

**Follow-up:** What remains the key blocker? A new independent holdout and use-case-specific risk/rights assessment, not another attractive heatmap.

## A 30-minute mock technical interview

Use a timer. Ask a friend to play the interviewer, or record yourself. A good answer explains the example first, then adds the formal name and the boundary of the evidence.

| Time | Interviewer prompt | What a strong answer demonstrates |
|---|---|---|
| 0–2 min | “Tell me what the project did.” | Give the two-minute story: task, fair comparison, main findings and scope. |
| 2–6 min | “Why can I trust your test split?” | Explain connected groups, 2199 rebuilt assignments, frozen hashes, and what the detector can miss. |
| 6–10 min | “Walk a sentence through BERT.” | Tokens, lookup vectors, positions, attention mixing, final representation and three logits; no unexplained acronym list. |
| 10–14 min | “Compute precision and recall from this matrix.” | Use Chapter 05’s negative counts: 42/51 and 42/91. Explain the two different questions. |
| 14–18 min | “Why is confidence not correctness?” | Work the 90%-confidence group example, then explain ECE, Brier and validation-only temperature fitting. |
| 18–23 min | “Derive IG and defend your baseline.” | Begin with changing an input and observing the score. Then show the path equation, logit target, attended PAD reference, completeness and limitations. |
| 23–27 min | “Did IG beat attention? Why did BERT beat FinBERT?” | Distinguish the measured AUC/F1 differences from uncertainty and untested causes. Mention the 40 flagged cases. |
| 27–30 min | “What would you do next, and would you deploy?” | Propose an independent holdout and controlled test, state safeguards, and bound the deployment recommendation. |

### Practice the difficult pause

If asked “Why exactly did this model fail?”, use three sentences:

1. “Here is what the saved output shows.”
2. “Here are plausible explanations, but this experiment does not distinguish them.”
3. “Here is the controlled test I would run next.”

**Check:** Is admitting missing evidence a weak answer? **Answer:** No. A strong technical answer separates knowledge from conjecture and proposes a way to reduce uncertainty.

### Self-review

- [ ] I used an example before an advanced definition.
- [ ] I stated the target class, population and metric when discussing an explanation result.
- [ ] I did not turn an AI-assisted note into a verified human conclusion.
- [ ] I did not call an interval spanning zero proof of equality.
- [ ] I did not present future changes as completed experiments.
- [ ] I could answer the follow-up without contradicting the original answer.

Use the [glossary](technical_glossary.md) for a term you cannot yet explain, then return to its linked full lesson.
