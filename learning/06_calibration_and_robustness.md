# 06 — Ask whether confidence and sentence edits mean what we think

**Before you begin:** Chapter 05 explains accuracy; Chapter 01 explains probabilities and logits. We begin with ten predictions so that “calibration” means something concrete before introducing ECE.

## Learning objectives

- Distinguish confidence from correctness using groups of predictions.
- Calculate ECE, Brier score and NLL in small examples.
- Explain what calibration and text-edit tests do—and do not—establish.

## Lesson: Does “90% confident” mean what it sounds like?

### Why it matters
An analyst may rely more heavily on a prediction marked 90% confident. If such predictions are correct only 65% of the time, the model is overselling its reliability.

### Build the intuition
Do not judge a confidence statement using one sentence. Gather many predictions with similar confidence and compare their average confidence with how often they were right.

### A concrete example
**Invented example:** a model makes ten predictions, each at confidence 0.90. Seven are correct. Its claimed confidence is 90%, while observed correctness is 70%. The gap is 20 percentage points.

### Walk through it
1. For each example, take the largest predicted class probability as confidence.
2. Mark whether its predicted class matches the human label.
3. Put similar confidences into groups.
4. Within each group, average confidence and correctness.
5. Plot those averages and also show how many examples support each group.

### Give the idea its technical name
**Calibration** means confidence statements match observed correctness frequencies in the relevant population. A **reliability diagram** compares those frequencies with confidence. A **confidence histogram** shows how many predictions fall in each confidence range. A **bin** is one such range.

### Add the mathematics
In the example, mean confidence = 9/10 = 0.90 because ten values of 0.90 sum to 9. Observed accuracy = 7/10 = 0.70. The absolute gap is |0.90−0.70| = 0.20. In a reliability diagram, a perfectly matching bin lies on the line “accuracy = confidence.”

### Connect it to our project
[calibration.py](../src/finbert_explainability/phase3/calibration.py) checks original probabilities with 10 and 15 equal-width bins and saves each bin's count. It does not fit a calibration correction. The top probability may be high on a wrong answer, as the selected error cases demonstrate.

### Alternatives and trade-offs
One overall confidence average is easy to report but can hide opposite errors in different ranges. More bins give detail but may contain few examples. Equal-count bins can reduce sparse groups, but produce different ranges and still need uncertainty checks.

### Check your understanding
**Question:** If one 90%-confidence prediction is wrong, does that alone prove poor calibration?

**Answer explained:** No. Even a well-calibrated collection of 90%-confidence predictions should include some errors. We compare many predictions, with enough data to estimate the frequency.

### Deeper dive and common confusion
**Common confusion:** calibration is not accuracy. A model can be less accurate but more honest about uncertainty. Top-label calibration checks only the winning class's probability; class-specific or subgroup calibration can reveal additional problems.


## Lesson: Summarize confidence mismatch with ECE

### Why it matters
A reliability diagram has several bins. How can we summarize their confidence gaps while allowing a bin with many examples to matter more than one with very few?

### Build the intuition
Calculate the gap inside each bin, then take a weighted average using the fraction of examples in that bin. The result describes average binned mismatch, not whether every individual prediction is reliable.

### A concrete example
**Invented example:** confidences `[0.55,0.65,0.85,0.95]`; the first three predictions are right and the last is wrong. Group the first two in [0.50,0.75) and the last two in [0.75,1.00]. These are teaching bins, not the project's 10/15-bin settings.

### Walk through it
1. First bin: confidence (0.55+0.65)/2 = 0.60; correctness 2/2 = 1.00; gap 0.40.
2. Second bin: confidence 0.90; correctness 1/2 = 0.50; gap 0.40.
3. Each bin holds half the examples, so the combined gap is 0.5×0.40 + 0.5×0.40 = 0.40.
4. Now imagine putting all four in one bin: mean confidence and accuracy are both 0.75, so that coarse calculation would hide the mismatch.

### Give the idea its technical name
This weighted bin-gap summary is **Expected Calibration Error (ECE)**. The example shows why its value depends on how we create the bins.

### Add the mathematics
Let N be the total number of examples, n_b the count in bin b, a_b its accuracy and c_b its mean confidence. Then $ECE=\sum_b(n_b/N)|a_b-c_b|$. Empty bins have no weight. This formula says exactly what the walkthrough did: “fraction of examples × gap,” summed over bins.

### Connect it to our project
The code uses intervals [lower, upper), with 1.0 included in the last bin. BERT ECE-10 is 0.1064 ± 0.0037; FinBERT is 0.0923 ± 0.0556, across three seeds. Lower means less mismatch under this calculation. FinBERT's large run-to-run variation matters.

### Alternatives and trade-offs
ECE is readable but bin-sensitive. It can miss errors within a bin or in non-winning probabilities. That is why the project also reports Brier score and negative log-likelihood in the next lesson, rather than selecting a “best calibrated” model from one ECE number.

### Check your understanding
**Question:** Why did the one-bin teaching calculation become zero?

**Answer explained:** Overconfidence in one region canceled underconfidence in another before the absolute gap was taken. That illustrates why zero coarse-bin ECE does not establish perfect calibration.

### Deeper dive and common confusion
**Common confusion:** ECE 0.10 is not a statement that each prediction is wrong by ten percentage points. It is an average of bin-level gaps. Changing bins, data or subgroup can change the estimate.


## Lesson: Score the entire probability answer, not just its largest number

### Why it matters
Two models can predict positive, but one assigns the correct positive label probability 0.70 and another 0.99. When the correct label is negative, those confidence choices should be penalized differently.

### Build the intuition
A probability prediction is a three-number answer. Compare it with the ideal answer for the known label. One score measures squared distance; another asks how much probability the model gave the actual answer.

### A concrete example
**Invented example:** the label is positive and the model gives `[0.1,0.2,0.7]`. The ideal label vector is `[0,0,1]`: no probability on the other two classes, all probability on the answer.

### Walk through it
1. Subtract ideal from predicted: `[0.1,0.2,-0.3]`.
2. Square and add: 0.01+0.04+0.09 = 0.14.
3. Separately, take minus log of the true-class probability: −log(0.7) ≈ 0.3567.
4. Average each score across examples, then summarize each seed's metric.

### Give the idea its technical name
The squared-distance score is the **multiclass Brier score**. The logarithmic score is **negative log-likelihood (NLL)**, also called log loss for this classification setting. Lower is better for both. A **one-hot label vector** has one 1 at the known class and zeros elsewhere.

### Add the mathematics
For example i and class k, let p_ik be predicted probability and y_ik its one-hot target. $Brier=(1/N)\sum_i\sum_k(p_{ik}-y_{ik})^2$. The project sums across three classes, without dividing by three or two, giving range 0–2. NLL is $(1/N)\sum_i[-\log p_{i,true}]$. Probability 0.001 on the true class costs about 6.9078, much more than 0.3567.

### Connect it to our project
BERT mean Brier/NLL are 0.2468/0.5115; FinBERT's are 0.2620/0.6022. BERT therefore has better mean values on these scores even though its ECE is higher. This is not a contradiction: different summaries reward different aspects of the probability distribution. The code clips true probabilities below 1e-15 for NLL to avoid log zero.

### Alternatives and trade-offs
Hard class accuracy ignores probability magnitude. Brier is bounded and evaluates all class probabilities. NLL heavily penalizes a very confident wrong answer. Both reflect ability to separate classes as well as confidence quality, so neither is a pure isolated measurement of calibration alone.

### Check your understanding
**Question:** Why must we say which Brier convention we use?

**Answer explained:** Some reporting conventions rescale it. A number divided by two or by the class count is not directly comparable with the project's unhalved sum. The convention changes scale, not the saved predictions.

### Deeper dive and common confusion
**Common confusion:** the baseline has smaller ECE but worse Brier/NLL than BERT. Being less confident can make a weak model's confidence look closer to its lower accuracy; that does not make it the better classifier. Always read calibration alongside class performance.


## Lesson: Adjust confidence without changing the learned classifier

### Why it matters
Suppose a model's class choices are useful but its probabilities are too extreme. Can we make the confidence more cautious without retraining every weight?

### Build the intuition
Divide all three logits by the same positive number before softmax. This changes how sharply probabilities concentrate on the winner while preserving the score order.

### A concrete example
**Invented example:** logits `[0,1,2]` give probabilities about `[0.0900,0.2447,0.6652]`. Divide by 2 to get `[0,0.5,1]`; probabilities become about `[0.1863,0.3072,0.5065]`. Positive remains the winner, but confidence falls.

### Walk through it
1. Keep the trained model fixed.
2. Obtain logits on a designated validation calibration set.
3. Find a positive scale that minimizes validation NLL.
4. Freeze that scale.
5. Evaluate the corrected probabilities on an independent holdout.

### Give the idea its technical name
The scale is called **temperature**, and the method is **temperature scaling**. A temperature above one softens the distribution; below one sharpens it.

### Add the mathematics
For logits z_k and temperature T>0, $p_k(T)=\exp(z_k/T)/\sum_j\exp(z_j/T)$. T is fitted from validation labels. Because positive division preserves order, the winning class normally stays unchanged, so accuracy and F1 do not improve merely from this scalar adjustment.

### Connect it to our project
This is a **proposal**, not an implemented experiment. Phase 3 did not fit a temperature. It recomputed frozen neural logits because Phase 2 prediction exports contained probabilities, not raw logits, and checked agreement with earlier predictions. It did not treat log probabilities as the model's original logits.

### Alternatives and trade-offs
Temperature scaling is simple and low-cost but cannot fix every class-specific confidence problem. Isotonic calibration fits a more flexible monotonic mapping, but can overfit a small validation sample. More flexible methods need stronger data support and an evaluation protocol that accounts for their selection.

### Check your understanding
**Question:** Why not fit the temperature on the frozen test set?

**Answer explained:** Then test labels would directly improve the reported confidence estimates. The result would no longer be an independent check of calibration. Use validation data for the fit and new held-out data for confirmation.

### Deeper dive and common confusion
**Common confusion:** confidence correction does not recover omitted context or repair wrong class ordering. It also may not transfer to a different news distribution. A separate calibration partition or cross-fitting can reduce reuse of the same validation examples for too many decisions.


## Lesson: Edit a sentence and decide what should change

### Why it matters
A model may rely on a company's name rather than the financial event. It may also ignore a meaningful number change. We need examples that separate those behaviors.

### Build the intuition
Change one part of a sentence and ask what a sensible reader would expect. Some edits should preserve sentiment; others should change it. That expectation must be reviewed, not assumed from the edit's name.

### A concrete example
**Invented examples:** “Acme profit rose from EUR 10 to EUR 12 million” → replace Acme with Example Holdings. Or scale both currency amounts by two, giving EUR 20 to EUR 24 million. Those may preserve the favorable direction. But “losses increased” → “losses did not increase” need not simply become positive; it may merely remove bad news.

### Walk through it
1. Check whether a supported edit rule applies.
2. Save the original text, edited text and exact change.
3. Flag uncertain grammar or meaning instead of inventing a new true label.
4. Check whether the model's actual tokens or TF-IDF features changed.
5. Compare classes and probabilities, using both all eligible edits and edits visible to the model.

### Give the idea its technical name
A **perturbation** is a controlled input edit. **Invariance** means the answer should stay the same under a meaning-preserving edit. A **flip rate** counts how often the predicted class changes. **Visibility** means the edit changes the model representation, not merely the displayed string.

### Add the mathematics
If 10 edits change the model's input and 2 change its prediction, visible flip rate is 2/10 = 0.20. If another 5 edited strings are invisible, using all 15 gives 2/15 ≈ 0.1333. Those denominators answer different questions. A flip is not necessarily an error because edited labels have not been established.

### Connect it to our project
[perturbations.py](../src/finbert_explainability/phase3/perturbations.py) applies a configured name lexicon, doubles currency-prefixed amounts and inserts “not” after supported auxiliaries. [inference.py](../src/finbert_explainability/phase3/inference.py) checks visibility. Eligible counts are 104 company, 109 numerical and 375 negation edits. The baseline sees only 78/109 numerical edits; BERT sees all 109 and 374/375 negations.

### Alternatives and trade-offs
Human-written, independently labeled contrast sets would directly test whether the new answer is correct. Automatic rules are cheap and repeatable but have limited coverage. Company identity can matter; dates, percentages and unmarked quantities are not scaled; negation scope can be unclear. Tests must not reward ignoring meaningful changes.

### Check your understanding
**Question:** Does FinBERT's numerical visible flip rate of 0.0000 prove it understands financial growth?

**Answer explained:** No. It could be appropriately stable or simply insensitive to numbers. A labeled contrast set where some quantity changes must change the sentiment is needed to distinguish those possibilities.

### Deeper dive and common confusion
**Common confusion:** high negation flip rate is not automatically good either. The rule skips existing negation, ambiguous monetary ranges and decimal-comma formats; spaced-decimal normalization is flagged. Every variant needs semantic/grammar review. Text after the 64-token cap can change while the model input does not. Confidence change in the new winning class also differs from change in the original class's probability.


## Read the actual probability scores

Read each row as a different question: ECE asks about binned confidence gaps, Brier asks about squared probability error, and NLL asks how much probability the true answer received. Each cell below is a three-seed mean and sample SD.

| Model | ECE-10 mean ± SD | ECE-15 mean ± SD | Brier mean ± SD | NLL mean ± SD |
|---|---:|---:|---:|---:|
| Baseline | 0.0701 ± 0 | 0.0730 ± 0 | 0.3674 ± 0 | 0.6320 ± 0 |
| BERT | 0.1064 ± 0.0037 | 0.1088 ± 0.0028 | 0.2468 ± 0.0010 | 0.5115 ± 0.0039 |
| FinBERT | 0.0923 ± 0.0556 | 0.0944 ± 0.0570 | 0.2620 ± 0.0221 | 0.6022 ± 0.1669 |

BERT can have stronger class decisions and better mean Brier/NLL while having worse binned ECE. FinBERT's ECE-10 SD of 0.0556 makes a blanket “better calibrated” claim especially weak.

**Check:** Is a lower ECE sufficient to choose the baseline over BERT? **Answer:** No. The baseline's poorer classification and proper probability scores still matter. A deployment policy must consider several kinds of error.

## Read the edit results as sensitivity, not accuracy

| Model | Company visible flip | Numerical visible flip | Negation visible flip |
|---|---:|---:|---:|
| Baseline | 0.2019 | 0.0769 | 0.0640 |
| BERT | 0.0128 | 0.0031 | 0.2255 |
| FinBERT | 0.0192 | 0.0000 | 0.1889 |

These are three-seed means over model-visible edits. Full SDs and all-eligible summaries remain in [summary.json](../artifacts/phase3/summary.json) and [PHASE3_RESULTS.md](../PHASE3_RESULTS.md). [phase3.json](../configs/phase3.json) records the exact lexicon and scale. Low flips under company/amount edits suggest stability under these narrow rules, not general semantic robustness. Higher negation flips do not show that changes were correct.

**Check:** What evidence would strengthen a numerical-reasoning claim? **Answer:** Reviewed sentence pairs that distinguish revenue, costs, losses, comparison periods and denominators, with known required answers and model-visible changes. That work is proposed, not completed.

## Practice explaining it

1. **What does calibration check?**

   Whether many similar confidence claims match how often those predictions are right.

2. **Why show bin counts?**

   A curve point supported by two examples is much less informative than one supported by hundreds.

3. **What is the project Brier range?**

   Zero to two because squared errors are summed across classes without halving or class-count scaling.

4. **Can temperature scaling fix a mistaken class order?**

   A positive common temperature preserves the order; it adjusts confidence rather than the chosen class.

5. **Why distinguish visible edits?**

   A model cannot respond to a change that its tokenizer, truncation or vocabulary never represents.

## Sources and next lesson

[Guo et al. on calibration](https://proceedings.mlr.press/v70/guo17a.html), [Phase 3 code](../src/finbert_explainability/phase3/calibration.py), [reference catalog](references.md). Next: [Integrated Gradients](07_integrated_gradients.md).
