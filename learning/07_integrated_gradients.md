# 07 — Follow a prediction from a reference input to the real input

**Before you begin:** Chapter 01's single-weight example explains gradients. Chapter 03 explains embeddings and logits. We now keep the model fixed and ask how its chosen score responds to input representations.

## Learning objectives

- Build IG from local sensitivity and a reference-to-input path.
- Explain the exact layer, target, baseline and completeness checks.
- Read signed token contributions while preserving numerical and reference limitations.

## Lesson: Start with “what changes the prediction?”

### Why it matters
FinBERT predicts positive for a growth sentence. Which parts of its input helped produce that score? A plausible-looking highlight is not enough; we need to specify the calculation behind it.

### Build the intuition
Start small: change one continuous input feature a little and observe the change in the chosen output score. A gradient captures that local sensitivity. It tells us about a small change at the current point, not every possible rewrite of the sentence.

### A concrete example
**Invented calculation:** let a toy positive-class score be F(x)=2x. At x=3 the score is 6. Increasing x by 0.01 raises the score to 6.02. The score changes by 0.02 for a 0.01 feature change, a sensitivity of 2.

### Walk through it
1. Choose one output to explain, such as the original positive-class score.
2. Identify continuous inputs to that output.
3. Differentiate the score with respect to those inputs.
4. Interpret each derivative as a local rate of change.
5. Ask whether a single local derivative is enough for the contrast we care about.

### Give the idea its technical name
A **derivative** is the rate of change with respect to one variable; a **gradient** collects derivatives over many variables. **Feature attribution** assigns contributions to input features for a specified output and comparison. Gradients are one ingredient, not automatically a complete attribution.

### Add the mathematics
A finite-change estimate is `[F(x+h)−F(x)]/h`, where h is a small increment. The derivative is its limiting value as h approaches zero. Here `(6.02−6)/0.01=2`. The gradient used for explanation concerns the target score, unlike training gradients that reduce a loss by changing model weights.

### Connect it to our project
`target_forward` in [core.py](../src/finbert_explainability/phase4/core.py) returns the original predicted-class **logit**, the raw class score explained in Chapter 01. The model stays in eval mode, so dropout is disabled. No weights are retrained.

### Alternatives and trade-offs
One gradient is cheap, but a score can become locally flat after a feature has already influenced it strongly. Comparing outputs after fully replacing a word is another option, but requires choosing a replacement and can change the sentence drastically. IG builds a reference-to-input comparison instead.

### Check your understanding
**Question:** For a sentence incorrectly predicted negative, what does explaining the original predicted class explain?

**Answer explained:** It explains the negative score that produced the model's wrong answer. It does not explain the human positive or neutral label unless we explicitly choose that other target in a separate analysis.

### Deeper dive and common confusion
**Common confusion:** a positive attribution means support for the chosen score. If the target class is negative, a positive attribution supports the negative prediction. “Positive score contribution” and “positive financial sentiment” are different uses of the word positive.


## Lesson: Add up sensitivity along a path

### Why it matters
A local derivative can miss what changed on the way from a reference input to the actual input. How can we account for the whole transition?

### Build the intuition
Imagine walking from a starting point to an endpoint. Measure the slope at many points along the walk. Multiply the average slope for each feature by how far that feature moved. That produces a contribution for the chosen reference-to-input journey.

### A concrete example
**Invented linear example:** F(x1,x2)=2x1−x2+3. Reference is `(0,0)`; actual input is `(2,1)`. The output moves from 3 to 6. The first feature contributes +4 and the second -1, adding to the output change +3.

### Walk through it
1. Choose reference x-prime and input x.
2. Move along the straight line between them.
3. Measure the target score's gradient at points along that line.
4. Average each coordinate's gradient over the path.
5. Multiply by that coordinate's endpoint difference.
6. Sum contributions to compare with the total score change.

### Give the idea its technical name
This method is **Integrated Gradients (IG)**. “Integrated” means accumulating the gradient along the path. Its **completeness** property says that, for the exact integral under the required regularity conditions, all contributions add to the input score minus the reference score.

### Add the mathematics
For coordinate j,

$$IG_j=(x_j-x'_j)\int_0^1 \frac{\partial F(x'+\alpha(x-x'))}{\partial x_j}\,d\alpha.$$

F is the target score; x is the input representation; x-prime is the reference; j selects one coordinate; alpha moves from 0 at the reference to 1 at the input. The integral averages the derivative along that journey. In the linear example, derivatives are always 2 and -1, so `(2−0)×2=4` and `(1−0)×(−1)=−1`.

The chain rule says the rate of change along the path is the sum of these coordinate rates times their movements. Integrating gives $\sum_j IG_j=F(x)-F(x')$. Here 4−1 = 6−3 = 3.

### Connect it to our project
The implementation uses Captum `LayerIntegratedGradients` with `multiply_by_inputs=True`, which includes the reference-to-input difference. It approximates the integral with Gauss–Legendre quadrature: a weighted set of gradient measurements rather than infinitely many points. It starts at 50 steps, then tries 100 or 200 when needed.

### Alternatives and trade-offs
Gradient × input is cheaper but uses one local derivative and an implicit comparison to zero. Occlusion uses actual replacement forward passes, but may need many of them and has a different replacement question. Neither is automatically a fair substitute for this path-based contrast.

### Check your understanding
**Question:** Should the contributions sum to the input logit itself?

**Answer explained:** No. They sum to the difference from the reference logit. In the toy example the reference already scores 3, so contributions account for the additional 3, not the entire final score of 6.

### Deeper dive and common confusion
**Deeper interaction example:** if F=x1×x2 and the path goes from `(0,0)` to `(2,3)`, the gradients along the path are `3×alpha` and `2×alpha`. Their average values are 1.5 and 1, so contributions are 2×1.5=3 and 3×1=3. They share an interaction. This does not establish two independent real-world causes of equal strength.


## Lesson: Choose a reference that the implementation really uses

### Why it matters
What does “remove the content” mean to a BERT model? Removing words, using PAD, using MASK and switching off attention positions are different operations. We must choose one and record it.

### Build the intuition
Keep the sentence's structural positions fixed, but replace the ordinary content token IDs with a chosen reference ID. Then compare the continuous embedding outputs for original and reference inputs. We are not sliding through integer vocabulary IDs.

### A concrete example
**Illustrative sequence:** original `[CLS] profit rose [SEP] [PAD]`; reference `[CLS] [PAD] [PAD] [SEP] [PAD]`. The mask stays `[1,1,1,1,0]`. The two replacement PAD tokens are still attended; only the original last padding position remains unavailable.

### Walk through it
1. Tokenize at length 64 and retain the inserted-special-token mask.
2. Keep CLS, SEP and original padding positions unchanged.
3. Replace actual content positions with PAD, or with MASK in the sensitivity comparison.
4. Keep attention mask, token types, positions and sequence length fixed.
5. Obtain the two complete embedding-module outputs.
6. Interpolate those continuous outputs for LayerIG.

### Give the idea its technical name
The **baseline** or **reference** defines the explanatory contrast. **LayerIntegratedGradients** attributes a selected layer's continuous representation. Here the layer is `model.bert.embeddings` **output**, after token, position and type vectors have been combined and normalized.

### Add the mathematics
In a toy embedding coordinate, original value 0.8 and reference 0.2 give midpoint 0.5. That is meaningful continuous arithmetic. Averaging vocabulary IDs 100 and 200 to obtain 150 is not meaningful: it chooses a different arbitrary vocabulary entry. Actual attribution tensors have shape `[1,64,768]`.

### Connect it to our project
`embedding_layer` verifies `BertForSequenceClassification` and the module structure. `reference_ids` changes attended non-inserted-special positions. `explain` uses `attribute_to_layer_input=False`. [phase4.json](../configs/phase4.json) fixes FinBERT seed 42, selected before inspecting test performance for explanation purposes.

### Alternatives and trade-offs
PAD and MASK are reproducible references, but neither is a neutral financial sentence. Multiple references can expose sensitivity but need a justified distribution. Masking positions out or deleting tokens changes attention availability or position, defining a different experiment.

### Check your understanding
**Question:** If CLS receives zero IG because its endpoint embedding is unchanged, does the model ignore CLS?

**Answer explained:** No. The endpoint difference for that position is zero, so its path attribution can be zero. Later contextual CLS states still gather information and feed the classifier. Attribution under one contrast is not a universal importance statement.

### Deeper dive and common confusion
**Common confusion:** reference construction excludes tokenizer-marked inserted specials, while ranking also excludes all special token IDs. An unknown or literal special ID in an ordinary content position can therefore be changed in the reference but omitted from ranking. It must still remain in the completeness sum. The path after LayerNorm is not separate lexical-versus-position attribution, and intermediate points need not correspond to natural sentences.


## Lesson: Check the numerical answer before interpreting it

### Why it matters
We approximate an integral with finitely many gradient evaluations. How can we notice when that approximation is not accurate enough?

### Build the intuition
Compare two ways of calculating the same total: add all attributions, and separately subtract reference score from input score. Their mismatch is a useful numerical check.

### A concrete example
**Invented check:** target input score is 6, reference score is 3, and summed numerical attribution is 2.9994. The expected change is 3. The signed mismatch is -0.0006; absolute mismatch is 0.0006.

### Walk through it
1. Sum every attribution coordinate, including special/padding positions.
2. Calculate input logit minus reference logit.
3. Subtract to obtain the signed residual.
4. Check its absolute size and its size relative to the score difference.
5. If either check fails, try the next declared integration step count.
6. Retain all attempts, including any final failure.

### Give the idea its technical name
A **residual** is the mismatch between expected and computed quantities. **Convergence delta** is Captum's associated completeness check. Here “converged” means the record passed the project's two declared numerical tolerances, not that a human explanation has been validated.

### Add the mathematics
Let S be the sum of attributions and D the input-minus-reference logit. Signed residual is r=S−D; absolute residual is |r|; relative residual is $|r|/\max(|D|,10^{-6})$. For the toy numbers, `0.0006/3=0.0002`, or 0.02%. The tiny denominator floor avoids unstable division when D is near zero. Both absolute ≤0.001 and relative ≤0.01 must pass.

### Connect it to our project
The first 10 PAD and 4 MASK records all eventually passed. Ten escalated beyond 50 steps. Maximum absolute residual was 0.000757853; maximum relative was 0.000795425. Wall time was 107.14 seconds, attribution time 44.21 seconds and throughput 0.317 records/second. Full-test work later found 40 failures, discussed in Chapter 09.

### Alternatives and trade-offs
More integration steps can improve approximation but cost time and do not guarantee every record will pass. A looser threshold would accept more results, but changing it after seeing outcomes would redefine success. A separate numerical sensitivity study can explore alternatives while preserving the historical rules.

### Check your understanding
**Question:** Can a perfectly converged explanation still mislead us?

**Answer explained:** Yes. It can accurately calculate a poorly justified reference contrast or an artificial path. Convergence checks the numerical calculation, not whether it represents human reasoning or real-world causes.

### Deeper dive and common confusion
**Common confusion:** a small relative residual can accompany a larger absolute error when the score difference is large. That is why both checks are used. Eval mode disables dropout; it does not disable the derivatives Captum needs. FP32 eager execution is used, with explicit checks against prior inference rather than a promise of universal bitwise cross-device identity.


## Lesson: Turn hundreds of feature contributions into readable token scores

### Why it matters
A token has 768 attribution coordinates. A reader needs a compact score, but careless aggregation can change the meaning or hide cancellation.

### Build the intuition
Add the signed contributions first to see the token's net support for the target. If you want to rank strength regardless of direction, take the absolute value afterward. Keep the original coordinates' absolute-sum measure separately.

### A concrete example
**Invented token coordinates:** +2 and -1 sum to +1. Absolute net contribution is 1; sum of absolute coordinates is 3. For two WordPieces contributing +0.8 and -0.3, the word's net score is +0.5, while summed piece magnitudes are 1.1.

### Walk through it
1. Sum embedding dimensions for each token's signed contribution.
2. Save its absolute net contribution as a separate field.
3. Save sum of absolute embedding-coordinate contributions too.
4. Exclude padding and special tokens from content ranking, not from completeness.
5. For an optional word display, merge adjacent `##` pieces and sum signed scores.
6. Retain the original WordPiece records and disclose the display rule.

### Give the idea its technical name
A **signed contribution** supports or opposes the selected target relative to the reference. **Absolute contribution** in this ranking means the absolute value of that signed token sum. **Word aggregation** is a presentation calculation, not a replacement for token-level evidence.

### Add the mathematics
For token i, signed score s_i is the sum over its embedding-coordinate IG values. Ranking uses |s_i|. For a merged word, sum the member s_i values first; taking an absolute value before summing gives a different number. The toy examples above show exactly where cancellation enters.

### Connect it to our project
`aggregate_words` in [core.py](../src/finbert_explainability/phase4/core.py) joins only adjacent continuations and leaves punctuation separate. The initial selection is the first five correct and five incorrect test IDs in lexical order. Four examples—the first two in each group—also use MASK references. On `fpb50-00018`, the actual positive prediction has probability 0.9936 and “double” contributes about +2.783 to its positive logit relative to PAD.

### Alternatives and trade-offs
Word displays are easier to read; token displays are faithful to the actual model units. Absolute ranking exposes strong inhibitory as well as supportive tokens. Positive-only ranking answers a narrower question and may stop before all tokens are covered; it cannot be compared as if it has identical deletion coverage.

### Check your understanding
**Question:** Why can removing a highly ranked absolute-IG token increase confidence?

**Answer explained:** Its signed score may be negative: it opposed the chosen target. Removing opposing evidence can strengthen that target. Absolute magnitude deliberately discards this direction.

### Deeper dive and common confusion
**Common confusion:** baseline sensitivity is not automatically a numerical error. On four examples, PAD/MASK absolute-ranking Spearman values were 0.1980, 0.4007, 0.6443 and 0.6202; top-five Jaccard ranged from 0.25 to 0.6667. Both references converged. Spearman compares orderings; Jaccard is shared top-five positions divided by positions in their union. Chapter 09 calculates rank agreement step by step.


## Walk through the actual function call

Input: a saved model/tokenizer, one row with its sentence and label, and [phase4.json](../configs/phase4.json). `embedding_layer` verifies architecture; `reference_ids` constructs the reference; `target_forward` fixes the score; `explain` computes attributions, checks shape/finite values, tries the declared steps and returns the record. `completeness` sums with double-precision accounting. The layer computation itself uses the declared FP32 protocol.

Output: row ID, original sentence, token IDs/strings, original/reference logits, class probabilities, target, signed and absolute scores, all convergence attempts, baseline details, word display and runtime. [Pipeline](../src/finbert_explainability/phase4/pipeline.py) and [artifact checks](../src/finbert_explainability/phase4/artifacts.py) verify earlier hashes, labels and predictions; records use sealed `payload.record` envelopes. Internal integration batch size is 4. Peak allocated/reserved CUDA memory was 0.567/0.641 GiB.

See [the growth example figure](../artifacts/phase4/initial/figures/fpb50-00018.png) and [Phase 4 report](../PHASE4_RESULTS.md). This example shows sensitivity to a growth-related word, not universal numerical reasoning.

**Check:** Would a new model checkpoint be explained by the old records? **Answer:** No. The explanation is tied to the weights, tokenizer, input, target and baseline. A different model requires a separately identified analysis.

## Compare methods after understanding the question each asks

**Occlusion** replaces a token/span and measures the output difference. **LIME** samples edited inputs and fits a simple local model to approximate the complex model nearby. **SHAP** methods allocate credit using combinations of present/absent features, with different practical approximations. **DeepLIFT** propagates changes in activations relative to a reference. Each needs its own rules for missing features and interactions.

| Method | Mechanism | Main cost and limitation |
|---|---|---|
| Gradient × input | One local derivative times input | Cheap; local saturation and implicit zero reference |
| IG | Integrate reference-to-input derivatives | Many backward passes; reference/path dependence |
| DeepLIFT | Propagate activation differences from reference | Efficient in supported architectures; nonlinear/module rules need care |
| Occlusion | Replace a token/span and measure output change | Many forward passes; replacement-induced shift and interactions |
| LIME | Fit a local surrogate on perturbed inputs | Model-agnostic; sampling/kernel/perturbation dependence |
| SHAP-family methods | Allocate contributions using coalition concepts | Exact enumeration exponential; approximations and missing-feature definition matter |

For a tiny three-feature SHAP teaching problem, there are 2³=8 possible present/absent combinations. That number grows quickly with features; practical methods approximate. LIME's sampled neighborhood and kernel weights decide what “nearby” means. DeepLIFT's handling of nonlinear modules must be compatible with the network. None was evaluated as a competitor here.

**Check:** Does changing the attribution algorithm remove the need for a baseline or perturbation assumption? **Answer:** Usually not. Even methods without an explicit PAD input make assumptions about what absence or local change means.

## Practice explaining it

1. **What scalar does IG explain here?**

   The original predicted-class logit, kept fixed along the path.

2. **Why not interpolate token IDs?**

   Their numerical order is arbitrary; interpolation belongs in continuous embeddings.

3. **What remains fixed in the PAD reference?**

   Inserted specials, original padding positions, attention availability, positions, length and token types.

4. **What does completeness verify?**

   That the numerical contributions match the input-minus-reference score change within the declared tolerance.

5. **Why keep signed and absolute scores?**

   Direction and magnitude answer different questions, and cancellation must remain visible.

## Sources and next lesson

[IG paper](https://arxiv.org/abs/1703.01365), [Captum LayerIG](https://captum.ai/api/layer.html#layer-integrated-gradients), [Phase 4 report](../PHASE4_RESULTS.md), [alternative-method references](references.md). Continue to [attention](08_attention_mechanisms.md).
