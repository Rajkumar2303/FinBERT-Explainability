# 04 — Teach the classifier, choose a recipe, keep the exam separate

**Before you begin:** Chapter 01 explains gradients and loss; Chapter 02 explains the three splits. Here we turn those ideas into the exact training procedure, without running training again.

## Learning objectives

- Explain a weight update, effective batch size and mixed precision.
- Separate validation search, checkpoint selection and final seed fits.
- Read the actual configuration, runtime records and reproduction commands.

## Lesson: Adjust the model without destroying what it already knows

### Why it matters
The pretrained model knows useful language patterns, but its new sentiment head can be wrong. How much should each training step change the weights?

### Build the intuition
Chapter 01 showed that a derivative tells us which small changes reduce loss. The learning rate controls how far we move. Think of adjusting a sensitive dial: tiny turns may take too long; large turns can overshoot.

### A concrete example
**Invented scalar example:** weight 0.50 has loss gradient -2. With learning rate 0.01, a simple gradient step gives 0.52. With rate 0.5 it gives 1.50. Neither is automatically best; the effect depends on the loss surface.

### Walk through it
1. Run a batch through the classifier.
2. Calculate cross-entropy loss from the true labels.
3. Backpropagate to obtain weight gradients.
4. Use the optimizer's rule to choose an update.
5. Repeat, evaluating validation performance at the configured points.

### Give the idea its technical name
An **optimizer** is the rule for updating weights from gradients. **AdamW** keeps running summaries of gradient direction and squared size, then applies **weight decay**, a separate shrinkage of selected weights. A **learning-rate scheduler** changes the step-size setting over training.

### Add the mathematics
A simple update is $w_{new}=w-\eta g$, where w is the weight, eta is learning rate and g is its gradient. Thus $0.50-0.01(-2)=0.52$.

After that intuition, AdamW's summaries are $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$ and $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$. Here t is the step, m tracks direction, v tracks squared gradient size, and the beta values control memory of earlier steps. Bias-corrected versions, called m-hat and v-hat, compensate for starting at zero. An approximate coordinate update is $(1-\eta_t\lambda)w_t-\eta_t\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Lambda is decay strength; epsilon prevents unstable division. This is a summary, not a replacement for the framework's parameter-group implementation.

### Connect it to our project
[neural.py](../src/finbert_explainability/phase2/neural.py), through `arguments`, sets `adamw_torch`, weight decay 0.01 and a linear scheduler with warmup_steps 0. Saved Adam values are beta1 0.9, beta2 0.999, epsilon 1e-8. Gradient norm is clipped to 1.0: unusually large combined gradients are limited before updating.

### Alternatives and trade-offs
Plain stochastic gradient descent uses a simpler update and may need more tuning. Warmup—gradually increasing the rate at the start—can help some fine-tuning procedures, but was not used here. Different rates for different layers may protect earlier representations, at the cost of more choices to validate.

### Check your understanding
**Question:** Does weight_decay=0.01 mean every tensor is shrunk identically?

**Answer explained:** No. The Trainer manages parameter groups, and some types of parameters may be exempt from decay. The recorded global setting does not replace checking that implementation detail.

### Deeper dive and common confusion
**Common confusion:** settings such as learning rate are not learned weights. They are **hyperparameters**, choices about how learning runs. A defensible technical reason for small fine-tuning rates is to adapt pretrained features gently; the record does not prove that any one rate is universally optimal.


## Lesson: Fit training into GPU memory

### Why it matters
The laptop GPU has 8 GB of VRAM. What if the batch size we want for learning is larger than the batch that fits comfortably in memory?

### Build the intuition
Process smaller portions one after another, collect their gradients, and update only after enough portions have been seen. This saves temporary activation memory without requiring the optimizer to update after every small portion.

### A concrete example
**Actual setting:** eight sentences per device, two accumulation steps, one GPU. The intended effective batch is 8×2×1 = 16 sentences per optimizer update.

### Walk through it
1. Run eight sentences and backpropagate a correctly scaled loss.
2. Keep the accumulated parameter gradients.
3. Run the next eight and add their gradients.
4. Perform one optimizer update.
5. Clear gradients for the next accumulation group.

### Give the idea its technical name
The small portion is a **microbatch**. **Gradient accumulation** combines microbatch gradients before updating. **Effective batch size** counts examples contributing to an update. An **epoch** is one pass through the training data.

### Add the mathematics
For two equally sized portions, a combined mean gradient is `(g1+g2)/2`. Adding unscaled mean losses could accidentally double the update; the Trainer handles scaling. With 3391 examples there are 424 size-eight microbatches including a smaller last one, and about 212 updates per epoch.

### Connect it to our project
[phase2.json](../configs/phase2.json) fixes per-device batch 8, accumulation 2 and max length 64. FP16 training uses selected 16-bit calculations to reduce resource use. **Loss scaling** enlarges gradients during computation to reduce tiny-value underflow, then rescales appropriately. Evaluation follows the FP32 path.

### Alternatives and trade-offs
Reducing microbatch size can prevent out-of-memory errors but may reduce throughput. Increasing length can retain more sentence context but costs memory. Mixed precision can save resources but introduces numerical concerns. If OOM had occurred, the plan was to reduce microbatch size and raise accumulation while documenting the change; no OOM adjustment was needed.

### Check your understanding
**Question:** Is an overflowing loss-scaled gradient the same as running out of memory?

**Answer explained:** No. Overflow is a number too large for the numerical format; it can lead to a skipped update. OOM means memory allocations could not be satisfied. The sanity check recorded early scaling overflow/skip events, not an OOM.

### Deeper dive and common confusion
**Common confusion:** accumulation is not always bitwise identical to a physical batch of 16. Dropout draws, summation order and partial batches can differ. Also, FP16 training does not mean all saved weights or all evaluation operations are FP16.


## Lesson: Choose settings without practicing on the exam

### Why it matters
We can try several learning rates and epoch budgets. If we pick the one with the best test score, how much of the apparent improvement comes from repeatedly looking at the exam answers?

### Build the intuition
Use validation examples to choose the recipe, then judge the chosen procedure on test examples. Keep the search small and explicit so we know how many chances each model had to look good.

### A concrete example
**Actual neural grid:** learning rates 2e-5 and 3e-5, each with maximum epoch budgets 3 and 4. That creates four candidates per encoder. The baseline tries three C values and two n-gram choices, giving six candidates.

### Walk through it
1. Fit a candidate using training rows.
2. Measure validation macro-F1.
3. Repeat for every configured candidate at search seed 42.
4. Select the largest validation score; ties keep the first configured candidate.
5. Start new final fits from the original pretrained checkpoint with the selected settings.
6. Within each fit, export the epoch checkpoint with the best validation macro-F1.

### Give the idea its technical name
**Hyperparameter search** compares training recipes. **Checkpoint selection** chooses a saved state along one training run. A **training budget** is how far the run is allowed to proceed; it need not equal the epoch of the exported model.

### Add the mathematics
Two rates × two budgets = four search runs. Three final seeds then give three new final fits per neural workflow. In the baseline, C is inverse regularization strength: larger C means weaker penalty on large coefficients. C=10 was selected; that does not mean the model has ten classes.

### Connect it to our project
[Settings](../src/finbert_explainability/phase2/settings.py) defines the grid. [`choose_candidate`](../src/finbert_explainability/phase2/common.py) implements stable tie selection. [baseline.py](../src/finbert_explainability/phase2/baseline.py) fits the vectorizer on training text only and turns solver convergence warnings into errors. The neural Trainer evaluates/saves every epoch and loads the validation-best checkpoint.

### Alternatives and trade-offs
Random or Bayesian search can cover more settings, but costs more and can overfit a small validation set. Successive halving stops weak candidates early, but early performance may misrepresent slow learners. Nested group cross-validation repeatedly separates selection from evaluation, at several times the compute cost.

### Check your understanding
**Question:** If FinBERT has a four-epoch budget, must its exported checkpoint be from epoch four?

**Answer explained:** No. Seed 123 selected step 212, epoch 1. Seeds 42 and 2026 selected step 848, epoch 4. All BERT final fits selected step 636, epoch 3. Selection used validation, never test performance.

### Deeper dive and common confusion
**Common confusion:** reusing validation for grid selection and best-epoch selection is allowed by this protocol, but many adaptive choices can still overfit validation. A wider search is not automatically more scientifically convincing unless its evaluation accounts for the extra selection.


## Lesson: Repeat the experiment and keep its identity

### Why it matters
A favorable run might depend on a lucky classifier initialization or training order. How do we avoid presenting one lucky result as the whole experiment?

### Build the intuition
Train from the same original starting model several times with declared random seeds. Save the result of each fit and identify every input file and software choice used. That lets us distinguish training variation from accidental changes in data or code.

### A concrete example
**Actual final seeds:** 42, 123 and 2026. BERT mean macro-F1 is 0.8476 with sample SD 0.0034. FinBERT mean is 0.8339 with SD 0.0093. Chapter 05 shows how sample SD is calculated.

### Walk through it
1. Set model and data-order seeds before fitting.
2. Use the same saved split, tokenizer/model revision and protocol.
3. Save training arguments, validation history, selected checkpoint and predictions.
4. Evaluate each final model on the same frozen test.
5. Average per-run metrics and record sample SD.
6. Keep successful outputs instead of overwriting them on rerun.

### Give the idea its technical name
**Repeatability** means repeating the work under the recorded conditions. **Reproducibility** also depends on access to code, data, versions and procedural details. A random seed alone is not a complete experiment identity.

### Add the mathematics
If three baseline runs give the same score, their observed SD is zero. That does not remove uncertainty about future examples. Likewise, averaging three probability vectors produces an ensemble prediction; its F1 is not generally the average of the three individual F1 values.

### Connect it to our project
Saved runs use deterministic settings, workers 0 and TF32 disabled. Repeated seed-42 CUDA fits produced byte-identical weights in the recorded environment. Training records and tokenizer files accompany each checkpoint. `save_only_model=True` and `save_total_limit=1` preserve selected model states, not a promise of exact optimizer-state continuation.

### Alternatives and trade-offs
More seeds improve information about training variation but consume more compute. Recording full optimizer state helps exact continuation but increases storage. Testing across hardware can reveal portability limits; a single-environment byte-identical repeat does not guarantee identical outputs everywhere.

### Check your understanding
**Question:** Why not choose the best test seed for the explanation study?

**Answer explained:** That would select the explanation model using the exam score. The project instead fixes FinBERT seed 42 explicitly, so it does not pretend the explanation checkpoint is representative because it won a test comparison.

### Deeper dive and common confusion
**Common confusion:** “independent final fits” does not mean the models have independent errors. They share data and pretraining and may make similar mistakes. The baseline's lbfgs solver is ordinarily deterministic under this setup, explaining its zero observed score variation without claiming universal determinism.


## Read the actual configuration as a recipe

You now know the roles of optimizer, rate, batch, epoch, search and checkpoint. Use this table as a reference rather than a list to memorize.

| Setting | Actual value |
|---|---|
| Final seeds / search seed | 42, 123, 2026 / 42 |
| Neural grid | Learning rate {2e-5, 3e-5} × maximum epochs {3, 4} |
| Sequence length | 64 including special tokens |
| Per-device batch / accumulation | 8 / 2; effective 16 on one GPU |
| Precision | FP16 CUDA training; FP32 evaluation path |
| Optimizer / scheduler | `adamw_torch` / linear; warmup_steps 0 |
| Adam settings | beta1 0.9, beta2 0.999, epsilon 1e-8 |
| Weight decay / gradient clipping | 0.01 / max_grad_norm 1.0 |
| Validation and saving | Every epoch; best validation macro-F1 loaded at end |
| Checkpoint retention | save_total_limit 1; save_only_model true |
| Determinism | Fixed model/data seeds, full_determinism, workers 0, TF32 disabled |
| TF-IDF grid | C {0.1, 1, 10} × word n-grams {(1,1), (1,2)} |
| TF-IDF and LR | max_features 20000, sublinear_tf true, lbfgs, max_iter 2000 |

**Check:** Which setting determines how many times the model is allowed to see the training set? **Answer:** The epoch budget. Which checkpoint gets exported is a separate validation decision.

## Read the measured costs without mixing different kinds of time

| Workflow | Validation-selected setting | Validation F1 | Full search/final workflow | Three final fits |
|---|---|---:|---:|---:|
| Baseline | C=10, unigrams | 0.7203 | 3.61 s | Included in workflow |
| BERT | 3e-5, max 3 epochs | 0.8380 | 13.43 min | 5.15 min |
| FinBERT | 2e-5, max 4 epochs | 0.8221 | 15.46 min | 7.01 min |

BERT's individual final fits took approximately 1.68/1.71/1.76 minutes; FinBERT's took 2.26/2.28/2.47 minutes for seeds 42/123/2026. Workflow time includes search, loading, checks and exports; it is not just GPU arithmetic. ProsusAI took 2.48 seconds for its inference/loading/export workflow and did not train locally.

Peak allocated CUDA memory was 2.079 GiB for each neural workflow; reserved peaks were 2.348 GiB for BERT and 2.357 GiB for FinBERT. Allocated counts tensor memory; reserved includes the allocator's cached reservation. Neither includes all display, driver or other-application memory.

**Check:** Can we compare ProsusAI's 2.48 seconds with 15.46 minutes and call it a faster training method? **Answer:** No. One is reference inference, the other includes training and search.

## Follow the implementation and reproduce only in a separate authorized run

[BERT training_record.json](../artifacts/phase2/bert/seed-42/training_record.json) and [FinBERT training_record.json](../artifacts/phase2/finbert/seed-42/training_record.json) record arguments, logs, selected settings and best checkpoints. Inputs are training/validation rows; outputs are models, tokenizer files, records and validation predictions. Test evaluation is a separate workflow.

The complete actual CLI inventory is in [reproduction_commands.md](reproduction_commands.md), copied from [publication/commands.json](../publication/commands.json). Those historical commands target existing outputs: do not paste them into this completed checkout. A fresh authorized reproduction needs new outputs and matching config paths, plus smoke checks. Reading this course does not require any training.

The recorded environment was rag-env, Python 3.11.15, PyTorch 2.12.1+cu126, CUDA 12.6 and RTX 4070 Laptop GPU. Existing PyTorch is preserved. From the project root, read help with `python -m finbert_explainability.phase2 --help` after activating rag-env and setting PYTHONPATH to src. Consult [PHASE2.md](../PHASE2.md) and [measured results](../PHASE2_RESULTS.md).

**Check:** Does a saved package list guarantee bitwise replay on every machine? **Answer:** No. Driver, hardware, kernel and library differences can matter. Pinned versions and hashes identify the historical run; they do not abolish numerical portability limits.

## Practice explaining it

1. **Why use a small fine-tuning learning rate?**

   To adapt useful pretrained weights without unnecessarily disruptive updates; optimality still needs validation.

2. **How is batch 16 achieved here?**

   Eight examples per microbatch, accumulated twice on one device.

3. **Which examples choose hyperparameters?**

   Validation examples; training examples fit weights; test examples evaluate the chosen procedure.

4. **Why save both epoch budget and selected checkpoint?**

   The validation-best state can come from an earlier epoch.

5. **What does zero baseline seed SD establish?**

   Only that these completed fits produced identical scores; future-data uncertainty remains.

## Sources and next lesson

[AdamW paper](https://arxiv.org/abs/1711.05101), [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html), [Phase 2 experiment documentation](../PHASE2.md), [reference catalog](references.md). Next: [evaluation](05_evaluation_and_validation.md).
