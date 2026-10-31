# 08 — Watch how tokens exchange information

**Before you begin:** Chapter 01 introduced weighted mixing and softmax. This chapter begins with a short sentence, then gives the Q/K/V names and matrix shapes. Chapter 07 is useful when we compare attention with IG, but not needed for the first lesson.

## Learning objectives

- Explain attention through token relationships before Q, K and V.
- Trace the exact final-layer CLS summary used here.
- Discuss special-token mass and alternatives without treating weights as automatic causal explanations.

## Lesson: Let “increased” read what increased

### Why it matters
A model reading “Acme's losses increased” needs to connect “increased” to “losses.” Merely assigning a permanent positive meaning to “increased” would fail.

### Build the intuition
At one processing step, each token can read a weighted mixture of information from other positions. Different learned views can make different connections. We do not assume a particular trained head always means “find the subject”; we use that as intuition.

### A concrete example
**Invented sentence-level illustration:** while updating “increased,” use weights 0.1 for Acme, 0.6 for losses and 0.3 for increased itself. The weights sum to one. This is not an actual attention row from the saved checkpoint.

### Walk through it
1. Give each position a representation from the preceding layer.
2. Calculate compatibility between the reading position and every allowed position.
3. Convert compatibility scores into nonnegative weights summing to one.
4. Mix the information supplied by other positions using those weights.
5. Repeat using several separately learned views, then combine their outputs.

### Give the idea its technical name
A **query** is the learned representation used to decide what a position seeks. A **key** is the learned representation other positions offer for compatibility. A **value** is the information actually mixed. One such set of projections is an attention **head**. Multiple heads form **multi-head attention**.

### Add the mathematics
Suppose toy compatibility scores are `[1,0]`. Softmax gives weights about `[0.731,0.269]`. If values are `[2,0]` and `[0,4]`, the weighted result is `[1.462,1.076]`.

Now write the same process compactly. For input matrix H, let Q=HW_Q, K=HW_K and V=HW_V. The W matrices are learned transformations. Then $A=softmax(QK^T/\sqrt{d_k}+M)$ and O=AV. A holds the weights, O the mixed outputs, d_k is head width, and M adds a large negative value where reading is forbidden, such as padding keys. Softmax runs across keys for each query.

### Connect it to our project
The saved model has 12 heads of width 64, so score scaling divides by sqrt(64)=8. Each layer's attention array has shape `[batch,12,query positions,key positions]`. [inference.py](../src/finbert_explainability/phase5/inference.py) checks those dimensions rather than assuming an arbitrary returned array has the right axes.

### Alternatives and trade-offs
A single head is simpler but compresses all compatibility patterns into one view. Multiple heads can represent different relationships, yet averaging them hides differences. An RNN carries state sequentially rather than building this pairwise score grid; both are design choices, not proof that either has human-like understanding.

### Check your understanding
**Question:** Does an attention row tell us which sentiment class is being explained?

**Answer explained:** Not by itself. The extracted attention summary is not conditioned on a target class, whereas IG explicitly targets one class logit. This difference matters when comparing their rankings.

### Deeper dive and common confusion
**Common confusion:** query/key compatibility is not the same as contribution to the final decision. The value vector, output projection and later network operations still determine what the mixture does. Dividing by square-root head width helps control score scale, rather than giving a semantic meaning to the weights.


## Lesson: Choose the exact attention view we will compare

### Why it matters
A BERT run produces attention for many layers, heads and query positions. “We used attention” leaves too much unspecified to reproduce an explanation comparison.

### Build the intuition
Choose one clearly defined view: how the final-layer CLS position reads other positions, averaged over its 12 heads. CLS feeds the sentence classifier, so this is a reasonable diagnostic location. It is not the only possible view.

### A concrete example
**Invented two-head example:** on three positions, head 1 gives `[0.2,0.7,0.1]` and head 2 `[0.4,0.2,0.4]`. Their mean is `[0.3,0.45,0.25]`. The mean shows an overall tendency but hides how strongly the heads disagree.

### Walk through it
1. Request attention with the model in eval mode.
2. Verify each of the 12 layer arrays and their 12 heads.
3. Find the actual CLS query position.
4. Save its rows for all layers/heads.
5. Select the last layer and average its heads.
6. Check finite values, row sums and negligible padding mass.

### Give the idea its technical name
**CLS-to-token attention** means the CLS query's weight on each key position. **Head averaging** is the ordinary mean across heads. It is not an average over layers, all query positions or independent trained models.

### Add the mathematics
If $A^{(l,h)}_{i,j}$ is the weight from query i to key j in layer l and head h, our score is $a_j=(1/12)\sum_{h=1}^{12} A^{(12,h)}_{CLS,j}$. The superscripts select layer/head; j selects a token position. In the toy example, the second position's score is `(0.7+0.2)/2=0.45`.

### Connect it to our project
`attention_and_deletion` uses eager attention with `output_attentions=True`, compares logits/probabilities with the frozen IG record and saves all-layer/all-head raw CLS rows on active positions. Complete query-key matrices are optional and disabled in [phase5.json](../configs/phase5.json) to bound disk use.

### Alternatives and trade-offs
Analyzing each head can expose differences but creates many comparisons. Choosing the head that looks best after seeing test results risks cherry-picking. Earlier layers or full-query matrices answer different questions and cost additional storage. The final-layer choice is technically defensible, but no local ablation proves it optimal.

### Check your understanding
**Question:** If a head emphasizes one token and another head ignores it, can the average hide that disagreement?

**Answer explained:** Yes. The average is a summary, not a claim that all heads behave alike. That is one reason the raw rows were retained.

### Deeper dive and common confusion
**Common confusion:** the final CLS state passes through residual and feed-forward transformations, a pooler and the classifier. The classifier does not simply read the attention-weight row as its three-class answer. The chosen summary is close to the output in layer order, but not the entire output computation.


## Lesson: Why special-token attention is not automatically wasted

### Why it matters
Almost 30% of the measured attention mass goes to SEP. It is tempting to say the model is looking at a meaningless separator. What information might be stored there by the final layer?

### Build the intuition
A token begins with an embedding, then repeatedly exchanges context. By the last layer, SEP can carry information from the sentence. Also, a large mixing weight has little effect if its value is uninformative, while a smaller weight can mix a powerful value.

### A concrete example
**Invented counterexample:** weight 0.9 multiplies value `[0,0]`, contributing nothing. Weight 0.1 multiplies `[10,0]`, contributing `[1,0]`. Looking only at weights would miss this difference.

### Walk through it
1. Before removing any positions, classify them as CLS, SEP, punctuation, content, other special or padding.
2. Sum attention within each category.
3. Verify categories account for the entire row.
4. Inspect head variation and contextual roles.
5. Avoid turning mass alone into a causal judgment.

### Give the idea its technical name
**Attention mass** for a category is the sum of its weights. **Faithfulness** asks whether an explanation reflects the model behavior it claims to describe. **Plausibility** asks whether it looks sensible to people. Neither follows from a weight's size alone.

### Add the mathematics
If SEP has weights 0.2 and 0.4 in a two-head toy example, mean SEP mass is 0.3. Category masses sum to one because they partition the positions. Renormalizing after excluding special tokens would hide how much weight originally went there, so the project measures mass before filtering.

### Connect it to our project
Across 727 examples, mean CLS mass is 0.1046, SEP 0.2980, punctuation 0.0704 and content 0.5270. Other specials have approximately 0.0000469; padding is zero. [Token-type plot](../artifacts/phase5/full/figures/attention_token_types.png) and [head plot](../artifacts/phase5/full/figures/attention_heads.png) show the saved evidence.

### Alternatives and trade-offs
**Attention rollout** combines attention across layers, usually including a residual identity route before multiplying matrices. **Attention flow** treats routes as a graph. Gradient-weighted attention adds sensitivity of a selected output. These can answer richer questions but make additional assumptions and do not automatically recover a causal explanation.

### Check your understanding
**Question:** Can unchanged CLS embedding endpoints yield zero IG while CLS has nonzero attention?

**Answer explained:** Yes. IG under that reference gives zero movement to the initial CLS embedding. Attention describes later contextual mixing. These are different objects, so the results do not contradict each other.

### Deeper dive and common confusion
**Common confusion:** low agreement with IG does not settle the attention debate because IG is not ground truth. Jain and Wallace challenge attention explanations with empirical controls; Wiegreffe and Pinter stress explanatory definitions and additional tests. This project does not reproduce all their alternative-attention or adversarial-training experiments. Read the evidence as a narrower comparison, not a universal theorem.


## Read the implementation as inputs and outputs

Input: the same saved sentence, tokenizer IDs, attention mask, checkpoint and IG target record. The function verifies token/mask equality before inference. Output: raw CLS rows, final head-mean scores, per-head category masses and later deletion results. Shape and normalization checks catch errors such as mixing up query and key axes. NumPy indexing can move axes, so category mass is explicitly checked to have one entry per head and to sum correctly.

The technical rationale for this summary is simplicity and proximity to the classifier's CLS representation. The historical record does not establish it as an optimal explanation layer or aggregation. Inspect [the actual inference code](../src/finbert_explainability/phase5/inference.py) and [token categorization](../src/finbert_explainability/phase5/analysis.py).

**Check:** Why retain all-layer CLS rows if the reported summary uses only the last layer? **Answer:** They preserve evidence for separately declared follow-up analyses without pretending those analyses were already part of the reported result.

## Read the literature with the right question in mind

[Jain and Wallace](https://aclanthology.org/N19-1357/) ask whether attention behaves like a dependable explanation under their diagnostics. [Wiegreffe and Pinter](https://aclanthology.org/D19-1002/) argue that the answer depends on what explanation means and on appropriate controls. [Attention flow/rollout](https://aclanthology.org/2020.acl-main.385/) offers ways to follow mixing through layers. None supplies a license to call every heatmap either faithful or useless.

**Check:** What extra evidence would be stronger than visual plausibility? **Answer:** Declared interventions, appropriate controls, uncertainty and stability across settings. The next chapter performs a limited version of that evaluation.

## Practice explaining it

1. **What do values do?**

   They supply the information being mixed; queries and keys determine the mixing weights.

2. **Which attention is used here?**

   Final-layer CLS-query attention averaged across 12 heads.

3. **Why are columns not required to sum to one?**

   Normalization is across keys separately for each query row.

4. **Why can SEP matter?**

   Its final contextual vector can carry sentence information.

5. **What does a low IG correlation establish?**

   Limited rank agreement under the chosen definitions, not which method is universally faithful.

## Sources and next lesson

[Attention flow/rollout paper](https://aclanthology.org/2020.acl-main.385/), [Phase 5 report](../PHASE5_RESULTS.md), [literature catalog](references.md). Continue to [experiments](09_explainability_experiments.md).
