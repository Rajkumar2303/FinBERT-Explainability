from course_writer import finish, lesson, tables, graph

name='07_integrated_gradients.md'
ls=[lesson('Start with “what changes the prediction?”',
'''FinBERT predicts positive for a growth sentence. Which parts of its input helped produce that score? A plausible-looking highlight is not enough; we need to specify the calculation behind it.''',
'''Start small: change one continuous input feature a little and observe the change in the chosen output score. A gradient captures that local sensitivity. It tells us about a small change at the current point, not every possible rewrite of the sentence.''',
'''**Invented calculation:** let a toy positive-class score be F(x)=2x. At x=3 the score is 6. Increasing x by 0.01 raises the score to 6.02. The score changes by 0.02 for a 0.01 feature change, a sensitivity of 2.''',
'''1. Choose one output to explain, such as the original positive-class score.
2. Identify continuous inputs to that output.
3. Differentiate the score with respect to those inputs.
4. Interpret each derivative as a local rate of change.
5. Ask whether a single local derivative is enough for the contrast we care about.''',
'''A **derivative** is the rate of change with respect to one variable; a **gradient** collects derivatives over many variables. **Feature attribution** assigns contributions to input features for a specified output and comparison. Gradients are one ingredient, not automatically a complete attribution.''',
r'''A finite-change estimate is `[F(x+h)−F(x)]/h`, where h is a small increment. The derivative is its limiting value as h approaches zero. Here `(6.02−6)/0.01=2`. The gradient used for explanation concerns the target score, unlike training gradients that reduce a loss by changing model weights.''',
'''`target_forward` in [core.py](../src/finbert_explainability/phase4/core.py) returns the original predicted-class **logit**, the raw class score explained in Chapter 01. The model stays in eval mode, so dropout is disabled. No weights are retrained.''',
'''One gradient is cheap, but a score can become locally flat after a feature has already influenced it strongly. Comparing outputs after fully replacing a word is another option, but requires choosing a replacement and can change the sentence drastically. IG builds a reference-to-input comparison instead.''',
'''For a sentence incorrectly predicted negative, what does explaining the original predicted class explain?''',
'''It explains the negative score that produced the model's wrong answer. It does not explain the human positive or neutral label unless we explicitly choose that other target in a separate analysis.''',
'''**Common confusion:** a positive attribution means support for the chosen score. If the target class is negative, a positive attribution supports the negative prediction. “Positive score contribution” and “positive financial sentiment” are different uses of the word positive.'''),
lesson('Add up sensitivity along a path',
'''A local derivative can miss what changed on the way from a reference input to the actual input. How can we account for the whole transition?''',
'''Imagine walking from a starting point to an endpoint. Measure the slope at many points along the walk. Multiply the average slope for each feature by how far that feature moved. That produces a contribution for the chosen reference-to-input journey.''',
'''**Invented linear example:** F(x1,x2)=2x1−x2+3. Reference is `(0,0)`; actual input is `(2,1)`. The output moves from 3 to 6. The first feature contributes +4 and the second -1, adding to the output change +3.''',
'''1. Choose reference x-prime and input x.
2. Move along the straight line between them.
3. Measure the target score's gradient at points along that line.
4. Average each coordinate's gradient over the path.
5. Multiply by that coordinate's endpoint difference.
6. Sum contributions to compare with the total score change.''',
'''This method is **Integrated Gradients (IG)**. “Integrated” means accumulating the gradient along the path. Its **completeness** property says that, for the exact integral under the required regularity conditions, all contributions add to the input score minus the reference score.''',
r'''For coordinate j,

$$IG_j=(x_j-x'_j)\int_0^1 \frac{\partial F(x'+\alpha(x-x'))}{\partial x_j}\,d\alpha.$$

F is the target score; x is the input representation; x-prime is the reference; j selects one coordinate; alpha moves from 0 at the reference to 1 at the input. The integral averages the derivative along that journey. In the linear example, derivatives are always 2 and -1, so `(2−0)×2=4` and `(1−0)×(−1)=−1`.

The chain rule says the rate of change along the path is the sum of these coordinate rates times their movements. Integrating gives $\sum_j IG_j=F(x)-F(x')$. Here 4−1 = 6−3 = 3.''',
'''The implementation uses Captum `LayerIntegratedGradients` with `multiply_by_inputs=True`, which includes the reference-to-input difference. It approximates the integral with Gauss–Legendre quadrature: a weighted set of gradient measurements rather than infinitely many points. It starts at 50 steps, then tries 100 or 200 when needed.''',
'''Gradient × input is cheaper but uses one local derivative and an implicit comparison to zero. Occlusion uses actual replacement forward passes, but may need many of them and has a different replacement question. Neither is automatically a fair substitute for this path-based contrast.''',
'''Should the contributions sum to the input logit itself?''',
'''No. They sum to the difference from the reference logit. In the toy example the reference already scores 3, so contributions account for the additional 3, not the entire final score of 6.''',
'''**Deeper interaction example:** if F=x1×x2 and the path goes from `(0,0)` to `(2,3)`, the gradients along the path are `3×alpha` and `2×alpha`. Their average values are 1.5 and 1, so contributions are 2×1.5=3 and 3×1=3. They share an interaction. This does not establish two independent real-world causes of equal strength.'''),
lesson('Choose a reference that the implementation really uses',
'''What does “remove the content” mean to a BERT model? Removing words, using PAD, using MASK and switching off attention positions are different operations. We must choose one and record it.''',
'''Keep the sentence's structural positions fixed, but replace the ordinary content token IDs with a chosen reference ID. Then compare the continuous embedding outputs for original and reference inputs. We are not sliding through integer vocabulary IDs.''',
'''**Illustrative sequence:** original `[CLS] profit rose [SEP] [PAD]`; reference `[CLS] [PAD] [PAD] [SEP] [PAD]`. The mask stays `[1,1,1,1,0]`. The two replacement PAD tokens are still attended; only the original last padding position remains unavailable.''',
'''1. Tokenize at length 64 and retain the inserted-special-token mask.
2. Keep CLS, SEP and original padding positions unchanged.
3. Replace actual content positions with PAD, or with MASK in the sensitivity comparison.
4. Keep attention mask, token types, positions and sequence length fixed.
5. Obtain the two complete embedding-module outputs.
6. Interpolate those continuous outputs for LayerIG.''',
'''The **baseline** or **reference** defines the explanatory contrast. **LayerIntegratedGradients** attributes a selected layer's continuous representation. Here the layer is `model.bert.embeddings` **output**, after token, position and type vectors have been combined and normalized.''',
'''In a toy embedding coordinate, original value 0.8 and reference 0.2 give midpoint 0.5. That is meaningful continuous arithmetic. Averaging vocabulary IDs 100 and 200 to obtain 150 is not meaningful: it chooses a different arbitrary vocabulary entry. Actual attribution tensors have shape `[1,64,768]`.''',
'''`embedding_layer` verifies `BertForSequenceClassification` and the module structure. `reference_ids` changes attended non-inserted-special positions. `explain` uses `attribute_to_layer_input=False`. [phase4.json](../configs/phase4.json) fixes FinBERT seed 42, selected before inspecting test performance for explanation purposes.''',
'''PAD and MASK are reproducible references, but neither is a neutral financial sentence. Multiple references can expose sensitivity but need a justified distribution. Masking positions out or deleting tokens changes attention availability or position, defining a different experiment.''',
'''If CLS receives zero IG because its endpoint embedding is unchanged, does the model ignore CLS?''',
'''No. The endpoint difference for that position is zero, so its path attribution can be zero. Later contextual CLS states still gather information and feed the classifier. Attribution under one contrast is not a universal importance statement.''',
'''**Common confusion:** reference construction excludes tokenizer-marked inserted specials, while ranking also excludes all special token IDs. An unknown or literal special ID in an ordinary content position can therefore be changed in the reference but omitted from ranking. It must still remain in the completeness sum. The path after LayerNorm is not separate lexical-versus-position attribution, and intermediate points need not correspond to natural sentences.'''),
lesson('Check the numerical answer before interpreting it',
'''We approximate an integral with finitely many gradient evaluations. How can we notice when that approximation is not accurate enough?''',
'''Compare two ways of calculating the same total: add all attributions, and separately subtract reference score from input score. Their mismatch is a useful numerical check.''',
'''**Invented check:** target input score is 6, reference score is 3, and summed numerical attribution is 2.9994. The expected change is 3. The signed mismatch is -0.0006; absolute mismatch is 0.0006.''',
'''1. Sum every attribution coordinate, including special/padding positions.
2. Calculate input logit minus reference logit.
3. Subtract to obtain the signed residual.
4. Check its absolute size and its size relative to the score difference.
5. If either check fails, try the next declared integration step count.
6. Retain all attempts, including any final failure.''',
'''A **residual** is the mismatch between expected and computed quantities. **Convergence delta** is Captum's associated completeness check. Here “converged” means the record passed the project's two declared numerical tolerances, not that a human explanation has been validated.''',
r'''Let S be the sum of attributions and D the input-minus-reference logit. Signed residual is r=S−D; absolute residual is |r|; relative residual is $|r|/\max(|D|,10^{-6})$. For the toy numbers, `0.0006/3=0.0002`, or 0.02%. The tiny denominator floor avoids unstable division when D is near zero. Both absolute ≤0.001 and relative ≤0.01 must pass.''',
'''The first 10 PAD and 4 MASK records all eventually passed. Ten escalated beyond 50 steps. Maximum absolute residual was 0.000757853; maximum relative was 0.000795425. Wall time was 107.14 seconds, attribution time 44.21 seconds and throughput 0.317 records/second. Full-test work later found 40 failures, discussed in Chapter 09.''',
'''More integration steps can improve approximation but cost time and do not guarantee every record will pass. A looser threshold would accept more results, but changing it after seeing outcomes would redefine success. A separate numerical sensitivity study can explore alternatives while preserving the historical rules.''',
'''Can a perfectly converged explanation still mislead us?''',
'''Yes. It can accurately calculate a poorly justified reference contrast or an artificial path. Convergence checks the numerical calculation, not whether it represents human reasoning or real-world causes.''',
'''**Common confusion:** a small relative residual can accompany a larger absolute error when the score difference is large. That is why both checks are used. Eval mode disables dropout; it does not disable the derivatives Captum needs. FP32 eager execution is used, with explicit checks against prior inference rather than a promise of universal bitwise cross-device identity.'''),
lesson('Turn hundreds of feature contributions into readable token scores',
'''A token has 768 attribution coordinates. A reader needs a compact score, but careless aggregation can change the meaning or hide cancellation.''',
'''Add the signed contributions first to see the token's net support for the target. If you want to rank strength regardless of direction, take the absolute value afterward. Keep the original coordinates' absolute-sum measure separately.''',
'''**Invented token coordinates:** +2 and -1 sum to +1. Absolute net contribution is 1; sum of absolute coordinates is 3. For two WordPieces contributing +0.8 and -0.3, the word's net score is +0.5, while summed piece magnitudes are 1.1.''',
'''1. Sum embedding dimensions for each token's signed contribution.
2. Save its absolute net contribution as a separate field.
3. Save sum of absolute embedding-coordinate contributions too.
4. Exclude padding and special tokens from content ranking, not from completeness.
5. For an optional word display, merge adjacent `##` pieces and sum signed scores.
6. Retain the original WordPiece records and disclose the display rule.''',
'''A **signed contribution** supports or opposes the selected target relative to the reference. **Absolute contribution** in this ranking means the absolute value of that signed token sum. **Word aggregation** is a presentation calculation, not a replacement for token-level evidence.''',
'''For token i, signed score s_i is the sum over its embedding-coordinate IG values. Ranking uses |s_i|. For a merged word, sum the member s_i values first; taking an absolute value before summing gives a different number. The toy examples above show exactly where cancellation enters.''',
'''`aggregate_words` in [core.py](../src/finbert_explainability/phase4/core.py) joins only adjacent continuations and leaves punctuation separate. The initial selection is the first five correct and five incorrect test IDs in lexical order. Four examples—the first two in each group—also use MASK references. On `fpb50-00018`, the actual positive prediction has probability 0.9936 and “double” contributes about +2.783 to its positive logit relative to PAD.''',
'''Word displays are easier to read; token displays are faithful to the actual model units. Absolute ranking exposes strong inhibitory as well as supportive tokens. Positive-only ranking answers a narrower question and may stop before all tokens are covered; it cannot be compared as if it has identical deletion coverage.''',
'''Why can removing a highly ranked absolute-IG token increase confidence?''',
'''Its signed score may be negative: it opposed the chosen target. Removing opposing evidence can strengthen that target. Absolute magnitude deliberately discards this direction.''',
'''**Common confusion:** baseline sensitivity is not automatically a numerical error. On four examples, PAD/MASK absolute-ranking Spearman values were 0.1980, 0.4007, 0.6443 and 0.6202; top-five Jaccard ranged from 0.25 to 0.6667. Both references converged. Spearman compares orderings; Jaccard is shared top-five positions divided by positions in their union. Chapter 09 calculates rank agreement step by step.''')]
finish(name,'''# 07 — Follow a prediction from a reference input to the real input

**Before you begin:** Chapter 01's single-weight example explains gradients. Chapter 03 explains embeddings and logits. We now keep the model fixed and ask how its chosen score responds to input representations.''',ls,'''## Walk through the actual function call

Input: a saved model/tokenizer, one row with its sentence and label, and [phase4.json](../configs/phase4.json). `embedding_layer` verifies architecture; `reference_ids` constructs the reference; `target_forward` fixes the score; `explain` computes attributions, checks shape/finite values, tries the declared steps and returns the record. `completeness` sums with double-precision accounting. The layer computation itself uses the declared FP32 protocol.

Output: row ID, original sentence, token IDs/strings, original/reference logits, class probabilities, target, signed and absolute scores, all convergence attempts, baseline details, word display and runtime. [Pipeline](../src/finbert_explainability/phase4/pipeline.py) and [artifact checks](../src/finbert_explainability/phase4/artifacts.py) verify earlier hashes, labels and predictions; records use sealed `payload.record` envelopes. Internal integration batch size is 4. Peak allocated/reserved CUDA memory was 0.567/0.641 GiB.

See [the growth example figure](../artifacts/phase4/initial/figures/fpb50-00018.png) and [Phase 4 report](../PHASE4_RESULTS.md). This example shows sensitivity to a growth-related word, not universal numerical reasoning.

**Check:** Would a new model checkpoint be explained by the old records? **Answer:** No. The explanation is tied to the weights, tokenizer, input, target and baseline. A different model requires a separately identified analysis.

## Compare methods after understanding the question each asks

**Occlusion** replaces a token/span and measures the output difference. **LIME** samples edited inputs and fits a simple local model to approximate the complex model nearby. **SHAP** methods allocate credit using combinations of present/absent features, with different practical approximations. **DeepLIFT** propagates changes in activations relative to a reference. Each needs its own rules for missing features and interactions.

'''+tables(name)[0]+'''

For a tiny three-feature SHAP teaching problem, there are 2³=8 possible present/absent combinations. That number grows quickly with features; practical methods approximate. LIME's sampled neighborhood and kernel weights decide what “nearby” means. DeepLIFT's handling of nonlinear modules must be compatible with the network. None was evaluated as a competitor here.

**Check:** Does changing the attribution algorithm remove the need for a baseline or perturbation assumption? **Answer:** Usually not. Even methods without an explicit PAD input make assumptions about what absence or local change means.''', [('What scalar does IG explain here?','The original predicted-class logit, kept fixed along the path.'),('Why not interpolate token IDs?','Their numerical order is arbitrary; interpolation belongs in continuous embeddings.'),('What remains fixed in the PAD reference?','Inserted specials, original padding positions, attention availability, positions, length and token types.'),('What does completeness verify?','That the numerical contributions match the input-minus-reference score change within the declared tolerance.'),('Why keep signed and absolute scores?','Direction and magnitude answer different questions, and cancellation must remain visible.')])

name='08_attention_mechanisms.md'
ls=[lesson('Let “increased” read what increased',
'''A model reading “Acme's losses increased” needs to connect “increased” to “losses.” Merely assigning a permanent positive meaning to “increased” would fail.''',
'''At one processing step, each token can read a weighted mixture of information from other positions. Different learned views can make different connections. We do not assume a particular trained head always means “find the subject”; we use that as intuition.''',
'''**Invented sentence-level illustration:** while updating “increased,” use weights 0.1 for Acme, 0.6 for losses and 0.3 for increased itself. The weights sum to one. This is not an actual attention row from the saved checkpoint.''',
'''1. Give each position a representation from the preceding layer.
2. Calculate compatibility between the reading position and every allowed position.
3. Convert compatibility scores into nonnegative weights summing to one.
4. Mix the information supplied by other positions using those weights.
5. Repeat using several separately learned views, then combine their outputs.''',
'''A **query** is the learned representation used to decide what a position seeks. A **key** is the learned representation other positions offer for compatibility. A **value** is the information actually mixed. One such set of projections is an attention **head**. Multiple heads form **multi-head attention**.''',
r'''Suppose toy compatibility scores are `[1,0]`. Softmax gives weights about `[0.731,0.269]`. If values are `[2,0]` and `[0,4]`, the weighted result is `[1.462,1.076]`.

Now write the same process compactly. For input matrix H, let Q=HW_Q, K=HW_K and V=HW_V. The W matrices are learned transformations. Then $A=softmax(QK^T/\sqrt{d_k}+M)$ and O=AV. A holds the weights, O the mixed outputs, d_k is head width, and M adds a large negative value where reading is forbidden, such as padding keys. Softmax runs across keys for each query.''',
'''The saved model has 12 heads of width 64, so score scaling divides by sqrt(64)=8. Each layer's attention array has shape `[batch,12,query positions,key positions]`. [inference.py](../src/finbert_explainability/phase5/inference.py) checks those dimensions rather than assuming an arbitrary returned array has the right axes.''',
'''A single head is simpler but compresses all compatibility patterns into one view. Multiple heads can represent different relationships, yet averaging them hides differences. An RNN carries state sequentially rather than building this pairwise score grid; both are design choices, not proof that either has human-like understanding.''',
'''Does an attention row tell us which sentiment class is being explained?''',
'''Not by itself. The extracted attention summary is not conditioned on a target class, whereas IG explicitly targets one class logit. This difference matters when comparing their rankings.''',
'''**Common confusion:** query/key compatibility is not the same as contribution to the final decision. The value vector, output projection and later network operations still determine what the mixture does. Dividing by square-root head width helps control score scale, rather than giving a semantic meaning to the weights.'''),
lesson('Choose the exact attention view we will compare',
'''A BERT run produces attention for many layers, heads and query positions. “We used attention” leaves too much unspecified to reproduce an explanation comparison.''',
'''Choose one clearly defined view: how the final-layer CLS position reads other positions, averaged over its 12 heads. CLS feeds the sentence classifier, so this is a reasonable diagnostic location. It is not the only possible view.''',
'''**Invented two-head example:** on three positions, head 1 gives `[0.2,0.7,0.1]` and head 2 `[0.4,0.2,0.4]`. Their mean is `[0.3,0.45,0.25]`. The mean shows an overall tendency but hides how strongly the heads disagree.''',
'''1. Request attention with the model in eval mode.
2. Verify each of the 12 layer arrays and their 12 heads.
3. Find the actual CLS query position.
4. Save its rows for all layers/heads.
5. Select the last layer and average its heads.
6. Check finite values, row sums and negligible padding mass.''',
'''**CLS-to-token attention** means the CLS query's weight on each key position. **Head averaging** is the ordinary mean across heads. It is not an average over layers, all query positions or independent trained models.''',
r'''If $A^{(l,h)}_{i,j}$ is the weight from query i to key j in layer l and head h, our score is $a_j=(1/12)\sum_{h=1}^{12} A^{(12,h)}_{CLS,j}$. The superscripts select layer/head; j selects a token position. In the toy example, the second position's score is `(0.7+0.2)/2=0.45`.''',
'''`attention_and_deletion` uses eager attention with `output_attentions=True`, compares logits/probabilities with the frozen IG record and saves all-layer/all-head raw CLS rows on active positions. Complete query-key matrices are optional and disabled in [phase5.json](../configs/phase5.json) to bound disk use.''',
'''Analyzing each head can expose differences but creates many comparisons. Choosing the head that looks best after seeing test results risks cherry-picking. Earlier layers or full-query matrices answer different questions and cost additional storage. The final-layer choice is technically defensible, but no local ablation proves it optimal.''',
'''If a head emphasizes one token and another head ignores it, can the average hide that disagreement?''',
'''Yes. The average is a summary, not a claim that all heads behave alike. That is one reason the raw rows were retained.''',
'''**Common confusion:** the final CLS state passes through residual and feed-forward transformations, a pooler and the classifier. The classifier does not simply read the attention-weight row as its three-class answer. The chosen summary is close to the output in layer order, but not the entire output computation.'''),
lesson('Why special-token attention is not automatically wasted',
'''Almost 30% of the measured attention mass goes to SEP. It is tempting to say the model is looking at a meaningless separator. What information might be stored there by the final layer?''',
'''A token begins with an embedding, then repeatedly exchanges context. By the last layer, SEP can carry information from the sentence. Also, a large mixing weight has little effect if its value is uninformative, while a smaller weight can mix a powerful value.''',
'''**Invented counterexample:** weight 0.9 multiplies value `[0,0]`, contributing nothing. Weight 0.1 multiplies `[10,0]`, contributing `[1,0]`. Looking only at weights would miss this difference.''',
'''1. Before removing any positions, classify them as CLS, SEP, punctuation, content, other special or padding.
2. Sum attention within each category.
3. Verify categories account for the entire row.
4. Inspect head variation and contextual roles.
5. Avoid turning mass alone into a causal judgment.''',
'''**Attention mass** for a category is the sum of its weights. **Faithfulness** asks whether an explanation reflects the model behavior it claims to describe. **Plausibility** asks whether it looks sensible to people. Neither follows from a weight's size alone.''',
'''If SEP has weights 0.2 and 0.4 in a two-head toy example, mean SEP mass is 0.3. Category masses sum to one because they partition the positions. Renormalizing after excluding special tokens would hide how much weight originally went there, so the project measures mass before filtering.''',
'''Across 727 examples, mean CLS mass is 0.1046, SEP 0.2980, punctuation 0.0704 and content 0.5270. Other specials have approximately 0.0000469; padding is zero. [Token-type plot](../artifacts/phase5/full/figures/attention_token_types.png) and [head plot](../artifacts/phase5/full/figures/attention_heads.png) show the saved evidence.''',
'''**Attention rollout** combines attention across layers, usually including a residual identity route before multiplying matrices. **Attention flow** treats routes as a graph. Gradient-weighted attention adds sensitivity of a selected output. These can answer richer questions but make additional assumptions and do not automatically recover a causal explanation.''',
'''Can unchanged CLS embedding endpoints yield zero IG while CLS has nonzero attention?''',
'''Yes. IG under that reference gives zero movement to the initial CLS embedding. Attention describes later contextual mixing. These are different objects, so the results do not contradict each other.''',
'''**Common confusion:** low agreement with IG does not settle the attention debate because IG is not ground truth. Jain and Wallace challenge attention explanations with empirical controls; Wiegreffe and Pinter stress explanatory definitions and additional tests. This project does not reproduce all their alternative-attention or adversarial-training experiments. Read the evidence as a narrower comparison, not a universal theorem.''')]
finish(name,'''# 08 — Watch how tokens exchange information

**Before you begin:** Chapter 01 introduced weighted mixing and softmax. This chapter begins with a short sentence, then gives the Q/K/V names and matrix shapes. Chapter 07 is useful when we compare attention with IG, but not needed for the first lesson.''',ls,'''## Read the implementation as inputs and outputs

Input: the same saved sentence, tokenizer IDs, attention mask, checkpoint and IG target record. The function verifies token/mask equality before inference. Output: raw CLS rows, final head-mean scores, per-head category masses and later deletion results. Shape and normalization checks catch errors such as mixing up query and key axes. NumPy indexing can move axes, so category mass is explicitly checked to have one entry per head and to sum correctly.

The technical rationale for this summary is simplicity and proximity to the classifier's CLS representation. The historical record does not establish it as an optimal explanation layer or aggregation. Inspect [the actual inference code](../src/finbert_explainability/phase5/inference.py) and [token categorization](../src/finbert_explainability/phase5/analysis.py).

**Check:** Why retain all-layer CLS rows if the reported summary uses only the last layer? **Answer:** They preserve evidence for separately declared follow-up analyses without pretending those analyses were already part of the reported result.

## Read the literature with the right question in mind

[Jain and Wallace](https://aclanthology.org/N19-1357/) ask whether attention behaves like a dependable explanation under their diagnostics. [Wiegreffe and Pinter](https://aclanthology.org/D19-1002/) argue that the answer depends on what explanation means and on appropriate controls. [Attention flow/rollout](https://aclanthology.org/2020.acl-main.385/) offers ways to follow mixing through layers. None supplies a license to call every heatmap either faithful or useless.

**Check:** What extra evidence would be stronger than visual plausibility? **Answer:** Declared interventions, appropriate controls, uncertainty and stability across settings. The next chapter performs a limited version of that evaluation.''', [('What do values do?','They supply the information being mixed; queries and keys determine the mixing weights.'),('Which attention is used here?','Final-layer CLS-query attention averaged across 12 heads.'),('Why are columns not required to sum to one?','Normalization is across keys separately for each query row.'),('Why can SEP matter?','Its final contextual vector can carry sentence information.'),('What does a low IG correlation establish?','Limited rank agreement under the chosen definitions, not which method is universally faithful.')])

name='09_explainability_experiments.md'
ls=[lesson('Compare two short ranked lists before calculating correlation',
'''IG and attention both assign scores to positions, but their units differ. Rather than comparing a logit contribution directly with an attention weight, ask whether they put the same positions near the top.''',
'''Ignore the raw score scale for a moment and compare order. Two methods agree perfectly if they order all tokens the same way; swapping a few positions weakens agreement.''',
'''**Invented three-token example:** IG ranks `profit, rose, sharply`; attention ranks `profit, sharply, rose`. In the fixed token order `[profit,rose,sharply]`, their ranks are `[1,2,3]` and `[1,3,2]`.''',
'''1. Check both methods refer to identical token IDs and positions.
2. Apply the same inclusion rule to both.
3. Replace scores by ranks.
4. Compare rank differences or pairwise ordering.
5. Calculate one coefficient per sentence, then summarize sentences.''',
'''**Spearman correlation** measures association between ranks. **Kendall tau-b** compares concordant token pairs—ordered the same way—with discordant pairs, accounting for ties. A tie receives an averaged rank in Spearman.''',
r'''With no ties, $\rho=1-6\sum_i d_i^2/[n(n^2-1)]$, where n is token count and d_i is the difference between the two ranks of token i. Here differences are `[0,-1,1]`, squared total 2, so $\rho=1-12/24=0.5$. For tied values use rank correlation directly, not this simplified formula. Constant vectors have no meaningful rank correlation.''',
'''[`correlations`](../src/finbert_explainability/phase5/analysis.py) requires aligned finite vectors and at least three eligible tokens. It compares absolute signed-sum IG with final-layer head-mean CLS attention. On 687 converged sentences, mean Spearman is 0.2406, median 0.2681, SD 0.2632 and bootstrap mean interval [0.2214,0.2607]. Signed-IG mean Spearman is 0.1008; mean absolute Kendall is 0.1724.''',
'''Pearson correlation of raw scores asks about linear numerical association, which is different from ranking. Top-k overlap focuses only on the most emphasized positions, ignoring the rest. None makes one method the truth against which the other is automatically judged.''',
'''Does mean Spearman 0.2406 mean the classifier is correct 24.06% of the time?''',
'''No. It describes average agreement between two token orderings. Classification accuracy is a separate calculation using sentiment labels.''',
'''**Common confusion:** this is the mean of per-sentence coefficients, not a pooled correlation of every token in the test set. Pooling would give long sentences more weight and mix different score contexts. Padding, all special IDs and pure Unicode punctuation are excluded from both methods; currency/math symbols remain. Ordinary punctuation was retained in Phase 4 displays, so that display filter is not identical to this comparison filter.'''),
lesson('Hide important tokens and read the curve',
'''If a ranking identifies influential input, what happens when we replace its highest-ranked tokens? We need a repeatable test, and we must be clear about what its score means.''',
'''Start with the original prediction. Replace more and more ranked tokens with MASK while tracking the probability of the original predicted class. A curve that falls earlier has less area below it. Under this test, that means stronger average suppression of the original prediction.''',
'''**Invented curve:** mask fractions `[0,0.5,1]` and original-class probabilities `[0.9,0.5,0.2]`. Picture a line descending from 0.9 to 0.5 to 0.2. Its area is smaller than a curve that stays near 0.9 until the end.''',
'''1. Fix the original predicted class and original ranking.
2. For each requested fraction, select the required number of top-ranked eligible positions.
3. Replace those IDs with MASK without changing positions or the original attention mask.
4. Run the fixed model and record all logits/probabilities and any class flip.
5. Plot original-class probability against the actual masked fraction.
6. Use trapezoids to calculate area under the curve.''',
'''**Deletion AUC** here means the area under that probability curve, even though tokens are replaced rather than physically removed. **Trapezoidal integration** approximates each segment's area by width times the average of its two endpoint heights.''',
r'''For adjacent points (x_j,p_j) and (x_next,p_next), area is `(x_next−x_j)×(p_j+p_next)/2`. The toy curve gives `0.5×(0.9+0.5)/2 + 0.5×(0.5+0.2)/2 = 0.525`. x is the fraction masked; p is the probability of the fixed original class.

For n eligible tokens and fraction f, requested count is k=ceil(f×n), except zero stays zero. With n=7, both 5% and 10% request one token, so actual fraction is 1/7. Duplicate actual points are counted once when integrating.''',
'''Budgets are 0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75 and 1.00. [`deletion_plan` and `auc_metrics`](../src/finbert_explainability/phase5/analysis.py) implement the counts and AUC; [inference.py](../src/finbert_explainability/phase5/inference.py) creates/evaluates the masked inputs. Primary mean AUC is IG 0.6555, attention 0.6608, random 0.7866. Lower means stronger suppression in this specific test.''',
'''Removing words physically shifts positions. Whole-word masking avoids deleting only part of a WordPiece word but changes the budget unit. Language-model infilling can create more natural text but introduces another model and uncertain meaning changes. MASK replacement controls positions while creating artificial inputs.''',
'''Why keep tracking the original class after the new prediction flips?''',
'''Otherwise the curve would change its target halfway through. Tracking whichever class is currently largest would answer a different question and could hide suppression of the original answer.''',
'''**Common confusion:** MASK introduces distribution shift—inputs unlike ordinary financial sentences. It does not prove real-world causality. Negative probability/logit drops are retained because removing inhibitory evidence can raise confidence. Positive-only IG stops when positive scores run out; its mean coverage is 0.6628. Partial-area curves are not ranked against full-coverage AUCs.'''),
lesson('Use random rankings and uncertainty to avoid a lucky comparison',
'''IG's mean AUC is slightly lower than attention's. Is that a clear difference, or could the observed set of sentences support either ordering? Random rankings also help show whether any token order would have worked similarly.''',
'''Compare methods on the same sentence, then repeatedly imagine a new test made by drawing sentences from the observed set with replacement. Keep each sentence's method scores together. Look at how much the mean difference varies.''',
'''**Invented bootstrap example:** four sentence-level IG-minus-attention AUC differences are `[-0.02,0.01,-0.01,0.00]`, mean -0.005. Draw four IDs with replacement. `[1,1,3,4]` gives mean -0.0125; `[2,2,4,4]` gives +0.005. Repeated draws show how the mean depends on which sentences are represented.''',
'''1. Generate five fixed random token orders for each sentence and average their AUCs within that sentence.
2. Compute IG-minus-attention and method-minus-random differences on each sentence.
3. Sample sentence IDs with replacement, retaining each paired difference.
4. Calculate the mean difference for each draw.
5. Repeat 2000 times and take the middle 95% by percentile endpoints.
6. Report the mean difference, interval and assumptions together.''',
'''A **paired comparison** keeps methods matched on the same example. The **bootstrap sampling unit** is the object drawn each time—here a sentence. A **percentile confidence interval** uses quantiles of the repeated statistic. Chapter 05 introduced this distinction from seed SD.''',
r'''For N sentence differences d_i, mean difference is $\bar d=(1/N)\sum_i d_i$. Each bootstrap draw makes a new length-N sample of these differences with replacement. The 2.5th and 97.5th percentiles give the reported interval. The toy draws above teach the mechanics; they are not a reliable confidence analysis from only four examples.''',
'''Random seeds are 11,23,37,53,71; a SHA256 combination with row ID produces repeatable per-row orderings. Bootstrap seed is 2042. IG-minus-attention mean is -0.0053 with interval [-0.0130,0.0024]. It includes zero, so this experiment does not clearly separate them. IG-minus-random is -0.1311; attention-minus-random -0.1258, with both intervals below zero.''',
'''More random orders reduce sensitivity to one arbitrary ordering but do not multiply the number of independent sentences. Event-group resampling could account for shared news stories if trustworthy metadata existed. An equivalence test would require a meaningful tolerance chosen in advance; an interval crossing zero does not supply that tolerance.''',
'''Why not treat all tokens, heads and five random orders as separate bootstrap examples?''',
'''They share the same sentence and model output. Pretending they are independent would exaggerate how much evidence we have and can make intervals too narrow.''',
'''**Common confusion:** the CI concerns the mean statistic under these sampling assumptions, not a range containing 95% of individual token effects. Curve intervals are pointwise, and there is no correction for many comparisons. Similar stories can remain dependent. One fixed checkpoint also leaves variation across model seeds outside this interval.'''),
lesson('Keep the failed calculations visible',
'''Forty IG records did not pass the fixed numerical rule. If we exclude them, are we silently changing the kind of sentences represented by the conclusion?''',
'''Report the reliable subset as the primary analysis, but also show what happens when every record is included with failures clearly marked. Compare the excluded group's makeup. Do not make the numerical rule easier after seeing the results.''',
'''**Measured composition:** the 40 flagged examples contain 11 negative, zero neutral and 29 positive labels. Thirty-one are allagree, and the model classified 36 correctly. They are not simply the hardest classification errors.''',
'''1. Keep all 727 records and their final convergence attempts.
2. Apply the unchanged absolute and relative tolerances.
3. Analyze the 687 passing records as primary.
4. Recalculate summary statistics for all 727, flagging unreliable IG.
5. Compare class, agreement, correctness, length and truncation counts.
6. Report what stays the same and what changes without claiming the groups are independent.''',
'''An **exclusion sensitivity analysis** asks how conclusions change when a declared exclusion is varied in reporting. **Non-random exclusion** means the removed examples differ systematically from those retained. It can change the population described by an average.''',
'''The flagged group is 40/727, about 5.50% of the test. Its mean eligible-token count is 28.6 versus 24.7 for passing records. Adding it changes mean Spearman from 0.2406 to 0.2465, a shift of approximately 0.0059. The groups overlap, so their intervals cannot be treated as independent comparisons.''',
'''All 40 fail absolute tolerance 0.001 after 200 steps, although every relative residual is below 1%. Maximum absolute residual is 0.015395. All-record IG/attention/random AUCs are 0.6353/0.6372/0.7778; paired IG-minus-attention is -0.0019 with interval [-0.0097,0.0061], still including zero. [Phase 6 sensitivity](../artifacts/phase6/audit-final/ig_sensitivity.json) preserves exact values.''',
'''Discarding failures without reporting composition would hide a selection effect. Treating them as fully reliable would ignore numerical evidence. A separately versioned experiment could investigate integration settings; it must preserve old thresholds and records rather than rewrite history.''',
'''Can we say the exclusion has absolutely no material effect?''',
'''No. Means and composition do change, and no practical-materiality margin was chosen in advance. We can say the IG-versus-attention comparison remains unresolved in both summaries, while noting the numerical and population differences.''',
'''**Common confusion:** the unreliable-only subset's paired result favors attention in direction, but it uses selected, numerically unreliable IG records. It cannot establish a method winner. Why those integrations fail numerically has not been isolated by this composition analysis.''')]
finish(name,'''# 09 — Test explanation rankings without assuming one is right

**Before you begin:** Chapters 07 and 08 define the two score types. Chapter 05 introduces uncertainty. Here we calculate a rank correlation, draw a deletion curve and walk through resampling before interpreting the real results.''',ls,'''## Follow the experiment from one sentence to a conclusion

'''+graph(name)+'''

The primary record set contains 80 negative, 432 neutral and 175 positive labels, with 307 allagree rows. The full set has 91/432/204 and 338 allagree. The excluded group has 2 truncated inputs; the full set has 15. There are no constant/too-short correlation failures among these 727 records, but the code returns undefined rather than zero if such a case appears.

'''+tables(name)[0]+'''

**Check:** Which row gives the primary numerical claim? **Answer:** The converged row. The all-record row is a sensitivity result that deliberately contains unreliable IG estimates.

## Inspect more than one deletion summary

Any-budget flips are 0.4309 for IG, 0.4178 attention and 0.4684 random. Final-budget flip fraction is 0.3916 for all methods because all complete rankings eventually mask the same positions. Mean logit-drop AUC is 2.0677/1.9840/1.2460. Probability and logit changes have different units; larger logit-drop area and smaller remaining-probability area point toward stronger suppression. Mean within-sentence random-order AUC SD is 0.0991.

The result “both beat random in mean probability AUC” does not mean both win on every diagnostic or sentence. Absolute IG can rank opposing evidence highly. The signed-IG comparison and positive-only partial curves help expose that distinction.

**Check:** Does random having more any-budget flips contradict its higher probability AUC? **Answer:** No. A flip asks whether another class overtakes the original at any tested point. AUC summarizes its probability across the full path. They can rank methods differently.

## Runtime, files and what was not tested

The run processed 727 rows, reusing 10 Phase 4 records. Final integration-step counts were 151 at 50, 400 at 100 and 176 at 200. Wall time was 951.44 seconds; attention/deletion time 80.15 seconds; recorded IG time 830.38 seconds includes reused work and is not simply extra wall time. Peak allocated/reserved GPU memory was 0.567/0.641 GiB. Full summary: [JSON](../artifacts/phase5/full/summary.json). Implementation: [statistics.py](../src/finbert_explainability/phase5/statistics.py), [config](../configs/phase5.json). Figures: [paired AUC](../artifacts/phase5/full/figures/paired_auc.png), [deletion curves](../artifacts/phase5/full/figures/deletion_curves.png).

Future checks could retain only selected evidence (**sufficiency**), remove it and measure the lost evidence (**comprehensiveness**), add evidence back (**insertion**), or randomize learned weights to see whether explanations depend on training. Human-highlight overlap checks plausibility, not automatically the actual computation. Whole-word masking, natural contrast pairs and multiple model seeds would broaden the evidence. None was performed merely by being proposed here.

**Check:** What is the narrow conclusion we can defend? **Answer:** Under this fixed checkpoint, baseline, token filter and MASK protocol, the rankings have modest agreement; both suppress more than random on mean AUC, and IG versus attention is not clearly separated.''', [('How is Spearman computed in the project?','One rank correlation per sentence, then a summary across sentences.'),('Why integrate actual mask fractions?','Rounding token counts changes the realized fraction, especially in short sentences.'),('Why average random orderings within each sentence first?','The sentence remains the evidence unit; random repeats reduce ordering noise.'),('What does an interval including zero mean here?','The paired experiment does not clearly identify a directional difference, not that methods are proven equivalent.'),('Why report the 40 cases separately?','Their numerical failure and different class mix make silent exclusion or unqualified inclusion misleading.')])
