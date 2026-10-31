from course_writer import finish, lesson, tables

name='04_baselines_and_training.md'
ls=[lesson('Adjust the model without destroying what it already knows',
'''The pretrained model knows useful language patterns, but its new sentiment head can be wrong. How much should each training step change the weights?''',
'''Chapter 01 showed that a derivative tells us which small changes reduce loss. The learning rate controls how far we move. Think of adjusting a sensitive dial: tiny turns may take too long; large turns can overshoot.''',
'''**Invented scalar example:** weight 0.50 has loss gradient -2. With learning rate 0.01, a simple gradient step gives 0.52. With rate 0.5 it gives 1.50. Neither is automatically best; the effect depends on the loss surface.''',
'''1. Run a batch through the classifier.
2. Calculate cross-entropy loss from the true labels.
3. Backpropagate to obtain weight gradients.
4. Use the optimizer's rule to choose an update.
5. Repeat, evaluating validation performance at the configured points.''',
'''An **optimizer** is the rule for updating weights from gradients. **AdamW** keeps running summaries of gradient direction and squared size, then applies **weight decay**, a separate shrinkage of selected weights. A **learning-rate scheduler** changes the step-size setting over training.''',
r'''A simple update is $w_{new}=w-\eta g$, where w is the weight, eta is learning rate and g is its gradient. Thus $0.50-0.01(-2)=0.52$.

After that intuition, AdamW's summaries are $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ and $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$. Here t is the step, m tracks direction, v tracks squared gradient size, and the beta values control memory of earlier steps. Bias-corrected versions, called m-hat and v-hat, compensate for starting at zero. An approximate coordinate update is $(1-\eta_t\lambda)w_t-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Lambda is decay strength; epsilon prevents unstable division. This is a summary, not a replacement for the framework's parameter-group implementation.''',
'''[neural.py](../src/finbert_explainability/phase2/neural.py), through `arguments`, sets `adamw_torch`, weight decay 0.01 and a linear scheduler with warmup_steps 0. Saved Adam values are beta1 0.9, beta2 0.999, epsilon 1e-8. Gradient norm is clipped to 1.0: unusually large combined gradients are limited before updating.''',
'''Plain stochastic gradient descent uses a simpler update and may need more tuning. Warmup—gradually increasing the rate at the start—can help some fine-tuning procedures, but was not used here. Different rates for different layers may protect earlier representations, at the cost of more choices to validate.''',
'''Does weight_decay=0.01 mean every tensor is shrunk identically?''',
'''No. The Trainer manages parameter groups, and some types of parameters may be exempt from decay. The recorded global setting does not replace checking that implementation detail.''',
'''**Common confusion:** settings such as learning rate are not learned weights. They are **hyperparameters**, choices about how learning runs. A defensible technical reason for small fine-tuning rates is to adapt pretrained features gently; the record does not prove that any one rate is universally optimal.'''),
lesson('Fit training into GPU memory',
'''The laptop GPU has 8 GB of VRAM. What if the batch size we want for learning is larger than the batch that fits comfortably in memory?''',
'''Process smaller portions one after another, collect their gradients, and update only after enough portions have been seen. This saves temporary activation memory without requiring the optimizer to update after every small portion.''',
'''**Actual setting:** eight sentences per device, two accumulation steps, one GPU. The intended effective batch is 8×2×1 = 16 sentences per optimizer update.''',
'''1. Run eight sentences and backpropagate a correctly scaled loss.
2. Keep the accumulated parameter gradients.
3. Run the next eight and add their gradients.
4. Perform one optimizer update.
5. Clear gradients for the next accumulation group.''',
'''The small portion is a **microbatch**. **Gradient accumulation** combines microbatch gradients before updating. **Effective batch size** counts examples contributing to an update. An **epoch** is one pass through the training data.''',
'''For two equally sized portions, a combined mean gradient is `(g1+g2)/2`. Adding unscaled mean losses could accidentally double the update; the Trainer handles scaling. With 3391 examples there are 424 size-eight microbatches including a smaller last one, and about 212 updates per epoch.''',
'''[phase2.json](../configs/phase2.json) fixes per-device batch 8, accumulation 2 and max length 64. FP16 training uses selected 16-bit calculations to reduce resource use. **Loss scaling** enlarges gradients during computation to reduce tiny-value underflow, then rescales appropriately. Evaluation follows the FP32 path.''',
'''Reducing microbatch size can prevent out-of-memory errors but may reduce throughput. Increasing length can retain more sentence context but costs memory. Mixed precision can save resources but introduces numerical concerns. If OOM had occurred, the plan was to reduce microbatch size and raise accumulation while documenting the change; no OOM adjustment was needed.''',
'''Is an overflowing loss-scaled gradient the same as running out of memory?''',
'''No. Overflow is a number too large for the numerical format; it can lead to a skipped update. OOM means memory allocations could not be satisfied. The sanity check recorded early scaling overflow/skip events, not an OOM.''',
'''**Common confusion:** accumulation is not always bitwise identical to a physical batch of 16. Dropout draws, summation order and partial batches can differ. Also, FP16 training does not mean all saved weights or all evaluation operations are FP16.'''),
lesson('Choose settings without practicing on the exam',
'''We can try several learning rates and epoch budgets. If we pick the one with the best test score, how much of the apparent improvement comes from repeatedly looking at the exam answers?''',
'''Use validation examples to choose the recipe, then judge the chosen procedure on test examples. Keep the search small and explicit so we know how many chances each model had to look good.''',
'''**Actual neural grid:** learning rates 2e-5 and 3e-5, each with maximum epoch budgets 3 and 4. That creates four candidates per encoder. The baseline tries three C values and two n-gram choices, giving six candidates.''',
'''1. Fit a candidate using training rows.
2. Measure validation macro-F1.
3. Repeat for every configured candidate at search seed 42.
4. Select the largest validation score; ties keep the first configured candidate.
5. Start new final fits from the original pretrained checkpoint with the selected settings.
6. Within each fit, export the epoch checkpoint with the best validation macro-F1.''',
'''**Hyperparameter search** compares training recipes. **Checkpoint selection** chooses a saved state along one training run. A **training budget** is how far the run is allowed to proceed; it need not equal the epoch of the exported model.''',
'''Two rates × two budgets = four search runs. Three final seeds then give three new final fits per neural workflow. In the baseline, C is inverse regularization strength: larger C means weaker penalty on large coefficients. C=10 was selected; that does not mean the model has ten classes.''',
'''[Settings](../src/finbert_explainability/phase2/settings.py) defines the grid. [`choose_candidate`](../src/finbert_explainability/phase2/common.py) implements stable tie selection. [baseline.py](../src/finbert_explainability/phase2/baseline.py) fits the vectorizer on training text only and turns solver convergence warnings into errors. The neural Trainer evaluates/saves every epoch and loads the validation-best checkpoint.''',
'''Random or Bayesian search can cover more settings, but costs more and can overfit a small validation set. Successive halving stops weak candidates early, but early performance may misrepresent slow learners. Nested group cross-validation repeatedly separates selection from evaluation, at several times the compute cost.''',
'''If FinBERT has a four-epoch budget, must its exported checkpoint be from epoch four?''',
'''No. Seed 123 selected step 212, epoch 1. Seeds 42 and 2026 selected step 848, epoch 4. All BERT final fits selected step 636, epoch 3. Selection used validation, never test performance.''',
'''**Common confusion:** reusing validation for grid selection and best-epoch selection is allowed by this protocol, but many adaptive choices can still overfit validation. A wider search is not automatically more scientifically convincing unless its evaluation accounts for the extra selection.'''),
lesson('Repeat the experiment and keep its identity',
'''A favorable run might depend on a lucky classifier initialization or training order. How do we avoid presenting one lucky result as the whole experiment?''',
'''Train from the same original starting model several times with declared random seeds. Save the result of each fit and identify every input file and software choice used. That lets us distinguish training variation from accidental changes in data or code.''',
'''**Actual final seeds:** 42, 123 and 2026. BERT mean macro-F1 is 0.8476 with sample SD 0.0034. FinBERT mean is 0.8339 with SD 0.0093. Chapter 05 shows how sample SD is calculated.''',
'''1. Set model and data-order seeds before fitting.
2. Use the same saved split, tokenizer/model revision and protocol.
3. Save training arguments, validation history, selected checkpoint and predictions.
4. Evaluate each final model on the same frozen test.
5. Average per-run metrics and record sample SD.
6. Keep successful outputs instead of overwriting them on rerun.''',
'''**Repeatability** means repeating the work under the recorded conditions. **Reproducibility** also depends on access to code, data, versions and procedural details. A random seed alone is not a complete experiment identity.''',
'''If three baseline runs give the same score, their observed SD is zero. That does not remove uncertainty about future examples. Likewise, averaging three probability vectors produces an ensemble prediction; its F1 is not generally the average of the three individual F1 values.''',
'''Saved runs use deterministic settings, workers 0 and TF32 disabled. Repeated seed-42 CUDA fits produced byte-identical weights in the recorded environment. Training records and tokenizer files accompany each checkpoint. `save_only_model=True` and `save_total_limit=1` preserve selected model states, not a promise of exact optimizer-state continuation.''',
'''More seeds improve information about training variation but consume more compute. Recording full optimizer state helps exact continuation but increases storage. Testing across hardware can reveal portability limits; a single-environment byte-identical repeat does not guarantee identical outputs everywhere.''',
'''Why not choose the best test seed for the explanation study?''',
'''That would select the explanation model using the exam score. The project instead fixes FinBERT seed 42 explicitly, so it does not pretend the explanation checkpoint is representative because it won a test comparison.''',
'''**Common confusion:** “independent final fits” does not mean the models have independent errors. They share data and pretraining and may make similar mistakes. The baseline's lbfgs solver is ordinarily deterministic under this setup, explaining its zero observed score variation without claiming universal determinism.''')]
finish(name,'''# 04 — Teach the classifier, choose a recipe, keep the exam separate

**Before you begin:** Chapter 01 explains gradients and loss; Chapter 02 explains the three splits. Here we turn those ideas into the exact training procedure, without running training again.''',ls,'''## Read the actual configuration as a recipe

You now know the roles of optimizer, rate, batch, epoch, search and checkpoint. Use this table as a reference rather than a list to memorize.

'''+tables(name)[0]+'''

**Check:** Which setting determines how many times the model is allowed to see the training set? **Answer:** The epoch budget. Which checkpoint gets exported is a separate validation decision.

## Read the measured costs without mixing different kinds of time

'''+tables(name)[1]+'''

BERT's individual final fits took approximately 1.68/1.71/1.76 minutes; FinBERT's took 2.26/2.28/2.47 minutes for seeds 42/123/2026. Workflow time includes search, loading, checks and exports; it is not just GPU arithmetic. ProsusAI took 2.48 seconds for its inference/loading/export workflow and did not train locally.

Peak allocated CUDA memory was 2.079 GiB for each neural workflow; reserved peaks were 2.348 GiB for BERT and 2.357 GiB for FinBERT. Allocated counts tensor memory; reserved includes the allocator's cached reservation. Neither includes all display, driver or other-application memory.

**Check:** Can we compare ProsusAI's 2.48 seconds with 15.46 minutes and call it a faster training method? **Answer:** No. One is reference inference, the other includes training and search.

## Follow the implementation and reproduce only in a separate authorized run

[BERT training_record.json](../artifacts/phase2/bert/seed-42/training_record.json) and [FinBERT training_record.json](../artifacts/phase2/finbert/seed-42/training_record.json) record arguments, logs, selected settings and best checkpoints. Inputs are training/validation rows; outputs are models, tokenizer files, records and validation predictions. Test evaluation is a separate workflow.

The complete actual CLI inventory is in [reproduction_commands.md](reproduction_commands.md), copied from [publication/commands.json](../publication/commands.json). Those historical commands target existing outputs: do not paste them into this completed checkout. A fresh authorized reproduction needs new outputs and matching config paths, plus smoke checks. Reading this course does not require any training.

The recorded environment was rag-env, Python 3.11.15, PyTorch 2.12.1+cu126, CUDA 12.6 and RTX 4070 Laptop GPU. Existing PyTorch is preserved. From the project root, read help with `python -m finbert_explainability.phase2 --help` after activating rag-env and setting PYTHONPATH to src. Consult [PHASE2.md](../PHASE2.md) and [measured results](../PHASE2_RESULTS.md).

**Check:** Does a saved package list guarantee bitwise replay on every machine? **Answer:** No. Driver, hardware, kernel and library differences can matter. Pinned versions and hashes identify the historical run; they do not abolish numerical portability limits.''', [('Why use a small fine-tuning learning rate?','To adapt useful pretrained weights without unnecessarily disruptive updates; optimality still needs validation.'),('How is batch 16 achieved here?','Eight examples per microbatch, accumulated twice on one device.'),('Which examples choose hyperparameters?','Validation examples; training examples fit weights; test examples evaluate the chosen procedure.'),('Why save both epoch budget and selected checkpoint?','The validation-best state can come from an earlier epoch.'),('What does zero baseline seed SD establish?','Only that these completed fits produced identical scores; future-data uncertainty remains.')])

name='06_calibration_and_robustness.md'
ls=[lesson('Does “90% confident” mean what it sounds like?',
'''An analyst may rely more heavily on a prediction marked 90% confident. If such predictions are correct only 65% of the time, the model is overselling its reliability.''',
'''Do not judge a confidence statement using one sentence. Gather many predictions with similar confidence and compare their average confidence with how often they were right.''',
'''**Invented example:** a model makes ten predictions, each at confidence 0.90. Seven are correct. Its claimed confidence is 90%, while observed correctness is 70%. The gap is 20 percentage points.''',
'''1. For each example, take the largest predicted class probability as confidence.
2. Mark whether its predicted class matches the human label.
3. Put similar confidences into groups.
4. Within each group, average confidence and correctness.
5. Plot those averages and also show how many examples support each group.''',
'''**Calibration** means confidence statements match observed correctness frequencies in the relevant population. A **reliability diagram** compares those frequencies with confidence. A **confidence histogram** shows how many predictions fall in each confidence range. A **bin** is one such range.''',
'''In the example, mean confidence = 9/10 = 0.90 because ten values of 0.90 sum to 9. Observed accuracy = 7/10 = 0.70. The absolute gap is |0.90−0.70| = 0.20. In a reliability diagram, a perfectly matching bin lies on the line “accuracy = confidence.”''',
'''[calibration.py](../src/finbert_explainability/phase3/calibration.py) checks original probabilities with 10 and 15 equal-width bins and saves each bin's count. It does not fit a calibration correction. The top probability may be high on a wrong answer, as the selected error cases demonstrate.''',
'''One overall confidence average is easy to report but can hide opposite errors in different ranges. More bins give detail but may contain few examples. Equal-count bins can reduce sparse groups, but produce different ranges and still need uncertainty checks.''',
'''If one 90%-confidence prediction is wrong, does that alone prove poor calibration?''',
'''No. Even a well-calibrated collection of 90%-confidence predictions should include some errors. We compare many predictions, with enough data to estimate the frequency.''',
'''**Common confusion:** calibration is not accuracy. A model can be less accurate but more honest about uncertainty. Top-label calibration checks only the winning class's probability; class-specific or subgroup calibration can reveal additional problems.'''),
lesson('Summarize confidence mismatch with ECE',
'''A reliability diagram has several bins. How can we summarize their confidence gaps while allowing a bin with many examples to matter more than one with very few?''',
'''Calculate the gap inside each bin, then take a weighted average using the fraction of examples in that bin. The result describes average binned mismatch, not whether every individual prediction is reliable.''',
'''**Invented example:** confidences `[0.55,0.65,0.85,0.95]`; the first three predictions are right and the last is wrong. Group the first two in [0.50,0.75) and the last two in [0.75,1.00]. These are teaching bins, not the project's 10/15-bin settings.''',
'''1. First bin: confidence (0.55+0.65)/2 = 0.60; correctness 2/2 = 1.00; gap 0.40.
2. Second bin: confidence 0.90; correctness 1/2 = 0.50; gap 0.40.
3. Each bin holds half the examples, so the combined gap is 0.5×0.40 + 0.5×0.40 = 0.40.
4. Now imagine putting all four in one bin: mean confidence and accuracy are both 0.75, so that coarse calculation would hide the mismatch.''',
'''This weighted bin-gap summary is **Expected Calibration Error (ECE)**. The example shows why its value depends on how we create the bins.''',
r'''Let N be the total number of examples, n_b the count in bin b, a_b its accuracy and c_b its mean confidence. Then $ECE=\sum_b(n_b/N)|a_b-c_b|$. Empty bins have no weight. This formula says exactly what the walkthrough did: “fraction of examples × gap,” summed over bins.''',
'''The code uses intervals [lower, upper), with 1.0 included in the last bin. BERT ECE-10 is 0.1064 ± 0.0037; FinBERT is 0.0923 ± 0.0556, across three seeds. Lower means less mismatch under this calculation. FinBERT's large run-to-run variation matters.''',
'''ECE is readable but bin-sensitive. It can miss errors within a bin or in non-winning probabilities. That is why the project also reports Brier score and negative log-likelihood in the next lesson, rather than selecting a “best calibrated” model from one ECE number.''',
'''Why did the one-bin teaching calculation become zero?''',
'''Overconfidence in one region canceled underconfidence in another before the absolute gap was taken. That illustrates why zero coarse-bin ECE does not establish perfect calibration.''',
'''**Common confusion:** ECE 0.10 is not a statement that each prediction is wrong by ten percentage points. It is an average of bin-level gaps. Changing bins, data or subgroup can change the estimate.'''),
lesson('Score the entire probability answer, not just its largest number',
'''Two models can predict positive, but one assigns the correct positive label probability 0.70 and another 0.99. When the correct label is negative, those confidence choices should be penalized differently.''',
'''A probability prediction is a three-number answer. Compare it with the ideal answer for the known label. One score measures squared distance; another asks how much probability the model gave the actual answer.''',
'''**Invented example:** the label is positive and the model gives `[0.1,0.2,0.7]`. The ideal label vector is `[0,0,1]`: no probability on the other two classes, all probability on the answer.''',
'''1. Subtract ideal from predicted: `[0.1,0.2,-0.3]`.
2. Square and add: 0.01+0.04+0.09 = 0.14.
3. Separately, take minus log of the true-class probability: −log(0.7) ≈ 0.3567.
4. Average each score across examples, then summarize each seed's metric.''',
'''The squared-distance score is the **multiclass Brier score**. The logarithmic score is **negative log-likelihood (NLL)**, also called log loss for this classification setting. Lower is better for both. A **one-hot label vector** has one 1 at the known class and zeros elsewhere.''',
r'''For example i and class k, let p_ik be predicted probability and y_ik its one-hot target. $Brier=(1/N)\sum_i\sum_k(p_{ik}-y_{ik})^2$. The project sums across three classes, without dividing by three or two, giving range 0–2. NLL is $(1/N)\sum_i[-\log p_{i,true}]$. Probability 0.001 on the true class costs about 6.9078, much more than 0.3567.''',
'''BERT mean Brier/NLL are 0.2468/0.5115; FinBERT's are 0.2620/0.6022. BERT therefore has better mean values on these scores even though its ECE is higher. This is not a contradiction: different summaries reward different aspects of the probability distribution. The code clips true probabilities below 1e-15 for NLL to avoid log zero.''',
'''Hard class accuracy ignores probability magnitude. Brier is bounded and evaluates all class probabilities. NLL heavily penalizes a very confident wrong answer. Both reflect ability to separate classes as well as confidence quality, so neither is a pure isolated measurement of calibration alone.''',
'''Why must we say which Brier convention we use?''',
'''Some reporting conventions rescale it. A number divided by two or by the class count is not directly comparable with the project's unhalved sum. The convention changes scale, not the saved predictions.''',
'''**Common confusion:** the baseline has smaller ECE but worse Brier/NLL than BERT. Being less confident can make a weak model's confidence look closer to its lower accuracy; that does not make it the better classifier. Always read calibration alongside class performance.'''),
lesson('Adjust confidence without changing the learned classifier',
'''Suppose a model's class choices are useful but its probabilities are too extreme. Can we make the confidence more cautious without retraining every weight?''',
'''Divide all three logits by the same positive number before softmax. This changes how sharply probabilities concentrate on the winner while preserving the score order.''',
'''**Invented example:** logits `[0,1,2]` give probabilities about `[0.0900,0.2447,0.6652]`. Divide by 2 to get `[0,0.5,1]`; probabilities become about `[0.1863,0.3072,0.5065]`. Positive remains the winner, but confidence falls.''',
'''1. Keep the trained model fixed.
2. Obtain logits on a designated validation calibration set.
3. Find a positive scale that minimizes validation NLL.
4. Freeze that scale.
5. Evaluate the corrected probabilities on an independent holdout.''',
'''The scale is called **temperature**, and the method is **temperature scaling**. A temperature above one softens the distribution; below one sharpens it.''',
r'''For logits z_k and temperature T>0, $p_k(T)=\exp(z_k/T)/\sum_j\exp(z_j/T)$. T is fitted from validation labels. Because positive division preserves order, the winning class normally stays unchanged, so accuracy and F1 do not improve merely from this scalar adjustment.''',
'''This is a **proposal**, not an implemented experiment. Phase 3 did not fit a temperature. It recomputed frozen neural logits because Phase 2 prediction exports contained probabilities, not raw logits, and checked agreement with earlier predictions. It did not treat log probabilities as the model's original logits.''',
'''Temperature scaling is simple and low-cost but cannot fix every class-specific confidence problem. Isotonic calibration fits a more flexible monotonic mapping, but can overfit a small validation sample. More flexible methods need stronger data support and an evaluation protocol that accounts for their selection.''',
'''Why not fit the temperature on the frozen test set?''',
'''Then test labels would directly improve the reported confidence estimates. The result would no longer be an independent check of calibration. Use validation data for the fit and new held-out data for confirmation.''',
'''**Common confusion:** confidence correction does not recover omitted context or repair wrong class ordering. It also may not transfer to a different news distribution. A separate calibration partition or cross-fitting can reduce reuse of the same validation examples for too many decisions.'''),
lesson('Edit a sentence and decide what should change',
'''A model may rely on a company's name rather than the financial event. It may also ignore a meaningful number change. We need examples that separate those behaviors.''',
'''Change one part of a sentence and ask what a sensible reader would expect. Some edits should preserve sentiment; others should change it. That expectation must be reviewed, not assumed from the edit's name.''',
'''**Invented examples:** “Acme profit rose from EUR 10 to EUR 12 million” → replace Acme with Example Holdings. Or scale both currency amounts by two, giving EUR 20 to EUR 24 million. Those may preserve the favorable direction. But “losses increased” → “losses did not increase” need not simply become positive; it may merely remove bad news.''',
'''1. Check whether a supported edit rule applies.
2. Save the original text, edited text and exact change.
3. Flag uncertain grammar or meaning instead of inventing a new true label.
4. Check whether the model's actual tokens or TF-IDF features changed.
5. Compare classes and probabilities, using both all eligible edits and edits visible to the model.''',
'''A **perturbation** is a controlled input edit. **Invariance** means the answer should stay the same under a meaning-preserving edit. A **flip rate** counts how often the predicted class changes. **Visibility** means the edit changes the model representation, not merely the displayed string.''',
'''If 10 edits change the model's input and 2 change its prediction, visible flip rate is 2/10 = 0.20. If another 5 edited strings are invisible, using all 15 gives 2/15 ≈ 0.1333. Those denominators answer different questions. A flip is not necessarily an error because edited labels have not been established.''',
'''[perturbations.py](../src/finbert_explainability/phase3/perturbations.py) applies a configured name lexicon, doubles currency-prefixed amounts and inserts “not” after supported auxiliaries. [inference.py](../src/finbert_explainability/phase3/inference.py) checks visibility. Eligible counts are 104 company, 109 numerical and 375 negation edits. The baseline sees only 78/109 numerical edits; BERT sees all 109 and 374/375 negations.''',
'''Human-written, independently labeled contrast sets would directly test whether the new answer is correct. Automatic rules are cheap and repeatable but have limited coverage. Company identity can matter; dates, percentages and unmarked quantities are not scaled; negation scope can be unclear. Tests must not reward ignoring meaningful changes.''',
'''Does FinBERT's numerical visible flip rate of 0.0000 prove it understands financial growth?''',
'''No. It could be appropriately stable or simply insensitive to numbers. A labeled contrast set where some quantity changes must change the sentiment is needed to distinguish those possibilities.''',
'''**Common confusion:** high negation flip rate is not automatically good either. The rule skips existing negation, ambiguous monetary ranges and decimal-comma formats; spaced-decimal normalization is flagged. Every variant needs semantic/grammar review. Text after the 64-token cap can change while the model input does not. Confidence change in the new winning class also differs from change in the original class's probability.''')]
finish(name,'''# 06 — Ask whether confidence and sentence edits mean what we think

**Before you begin:** Chapter 05 explains accuracy; Chapter 01 explains probabilities and logits. We begin with ten predictions so that “calibration” means something concrete before introducing ECE.''',ls,'''## Read the actual probability scores

Read each row as a different question: ECE asks about binned confidence gaps, Brier asks about squared probability error, and NLL asks how much probability the true answer received. Each cell below is a three-seed mean and sample SD.

'''+tables(name)[0]+'''

BERT can have stronger class decisions and better mean Brier/NLL while having worse binned ECE. FinBERT's ECE-10 SD of 0.0556 makes a blanket “better calibrated” claim especially weak.

**Check:** Is a lower ECE sufficient to choose the baseline over BERT? **Answer:** No. The baseline's poorer classification and proper probability scores still matter. A deployment policy must consider several kinds of error.

## Read the edit results as sensitivity, not accuracy

'''+tables(name)[1]+'''

These are three-seed means over model-visible edits. Full SDs and all-eligible summaries remain in [summary.json](../artifacts/phase3/summary.json) and [PHASE3_RESULTS.md](../PHASE3_RESULTS.md). [phase3.json](../configs/phase3.json) records the exact lexicon and scale. Low flips under company/amount edits suggest stability under these narrow rules, not general semantic robustness. Higher negation flips do not show that changes were correct.

**Check:** What evidence would strengthen a numerical-reasoning claim? **Answer:** Reviewed sentence pairs that distinguish revenue, costs, losses, comparison periods and denominators, with known required answers and model-visible changes. That work is proposed, not completed.''', [('What does calibration check?','Whether many similar confidence claims match how often those predictions are right.'),('Why show bin counts?','A curve point supported by two examples is much less informative than one supported by hundreds.'),('What is the project Brier range?','Zero to two because squared errors are summed across classes without halving or class-count scaling.'),('Can temperature scaling fix a mistaken class order?','A positive common temperature preserves the order; it adjusts confidence rather than the chosen class.'),('Why distinguish visible edits?','A model cannot respond to a change that its tokenizer, truncation or vocabulary never represents.')])
