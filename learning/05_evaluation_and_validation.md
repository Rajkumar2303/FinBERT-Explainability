# 05 — Count the mistakes, then name the metrics

**Before you begin:** you only need to know that every sentence has a human label and a model prediction. We will calculate the main metrics from one real table. Later lessons add repeated-run variation and uncertainty; you can pause there on a first reading.

## Learning objectives

- Calculate accuracy, precision, recall and all three class F1 scores from one real matrix.
- Explain macro averaging, seed variation and resampling uncertainty.
- Interpret subgroup results and selected errors without overstating them.

## Lesson: Read the prediction table before calculating a score

### Why it matters
An analyst asks, “How well did the model do?” A single percentage cannot tell you which mistakes it made. First we need a count of the answers.

### Build the intuition
Put the human labels down the side and the model's answers across the top. A correct answer lands where the row and column names match. Everything else is a mistake.

### A concrete example
**Actual project data: the TF-IDF baseline on all 727 test sentences.** Read a row as the human-assigned label, and a column as the model's answer.

| Human label / model answer | Negative | Neutral | Positive | Row total |
|---|---:|---:|---:|---:|
| Negative | 42 | 37 | 12 | 91 |
| Neutral | 4 | 390 | 38 | 432 |
| Positive | 5 | 90 | 109 | 204 |
| Column total | 51 | 517 | 159 | 727 |

For example, 37 genuinely negative examples were labeled neutral by the model. We will use this same table throughout the chapter.

### Walk through it
1. Follow the negative row: 42 correct, 37 called neutral, 12 called positive.
2. Follow the negative column: the model called 51 sentences negative, but 9 came from other human labels.
3. Add the matching cells: 42 + 390 + 109 = 541 correct answers.
4. Add all cells: 727 examples.

### Give the idea its technical name
This count table is a **confusion matrix**. “Confusion” simply refers to one class being mistaken for another. The dataset's reference label is sometimes called the **gold label**; that term does not imply that people can never disagree or be wrong.

### Add the mathematics
**Accuracy** is correct answers divided by all answers: 541 / 727 = 0.7442, or about 74.42%. If we use symbols, $A=\text{correct}/N$, where $A$ is accuracy and $N$ is the number of evaluated examples.

### Connect it to our project
The matrix is saved in [comparison.json](../artifacts/phase2/comparison/comparison.json). [metrics.py](../src/finbert_explainability/phase2/metrics.py) uses rows=true and columns=predicted. Keeping that convention explicit avoids reading the wrong denominator.

### Alternatives and trade-offs
Accuracy is easy to explain. But a model that always says neutral would get 432 / 727, about 59.42%, because neutral is common. We therefore need scores that look separately at negative, neutral and positive predictions.

### Check your understanding
**Question:** What does the cell 90 in the positive row and neutral column mean?

**Answer explained:** Ninety sentences labeled positive by the dataset were predicted neutral. It does not mean 90 neutral sentences were called positive; that would be the opposite cell, which contains 38.

### Deeper dive and common confusion
**Common confusion:** a mean matrix across three runs can contain decimals. Those are averages of integer counts, not fractions of a sentence in a single run. Do not calculate a mean-run F1 by assuming the averaged matrix is one actual run; F1 is a nonlinear calculation, explained below.


## Lesson: When the model raises a negative flag, how often is it right?

### Why it matters
Suppose an analyst reviews every sentence the model calls negative. Too many false alarms waste review time. We want to know the quality of that queue.

### Build the intuition
Start with the model's negative answers—not all real negative sentences. Ask how many of those answers were correct.

### A concrete example
In the actual matrix, the negative column contains 42 + 4 + 5 = 51 answers. Forty-two are correct. Nine are false alarms: 4 neutral and 5 positive examples.

### Walk through it
1. Choose the negative prediction column.
2. Count all 51 entries in that column.
3. Keep the 42 whose human label is also negative.
4. Divide 42 by 51.

### Give the idea its technical name
This is **precision** for the negative class. A **true positive** for this calculation means a correctly predicted negative example. A **false positive** means a negative prediction that should have been another class. “Positive” in these error-count names means “the class currently being checked,” not necessarily positive financial sentiment.

### Add the mathematics
Let $TP$ be correct negative flags and $FP$ be false negative-class flags. Then $P=TP/(TP+FP)=42/(42+9)=0.8235$. Here $P$ is precision. Read this as: about 82.35% of the negative flags were right.

### Connect it to our project
The baseline's saved negative precision is 0.8235. That sounds strong, but it does not yet tell us how many negative sentences never entered the queue.

### Alternatives and trade-offs
Precision matters when false alarms are costly. A very cautious model can improve precision by making fewer flags. That may be appropriate for a small review team, but dangerous if missed negative news matters more. We need the next lesson before choosing a policy.

### Check your understanding
**Question:** If a model flags only one sentence and gets it right, is its precision 100%?

**Answer explained:** Yes, for that tiny queue. It could still miss almost every other negative sentence. High precision alone says nothing about how complete the queue is.

### Deeper dive and common confusion
**Common confusion:** high precision is not “high overall accuracy.” Each uses a different starting set. Also, if there are no predictions for a class, the raw fraction has a zero denominator. Software must declare its convention; inspect the metric implementation rather than silently dropping that class.


## Lesson: How many real negative sentences did the model find?

### Why it matters
Now imagine that missing bad news is the main concern. A precise queue is not enough if most negative sentences were never flagged.

### Build the intuition
Start with all sentences that people labeled negative. Ask how many the model successfully found. This reverses the direction of the previous question.

### A concrete example
The actual negative row has 91 sentences. The model found 42. It missed 37 + 12 = 49 by calling them neutral or positive.

### Walk through it
1. Choose the negative human-label row.
2. Count all 91 sentences in that row.
3. Keep the 42 correctly predicted negative.
4. Divide 42 by 91.

### Give the idea its technical name
This is **recall** for the negative class. The 49 missed examples are **false negatives** for the class we are checking. Recall answers, “Of the negative sentences that were there, what fraction did we find?”

### Add the mathematics
Let $FN$ be the missed negative examples and $R$ be recall. Then $R=TP/(TP+FN)=42/(42+49)=0.4615$. The model found about 46.15% of the real negative examples and missed more than half.

### Connect it to our project
The baseline has negative precision 0.8235 but recall 0.4615. In everyday language: its negative flags were usually sensible, but it failed to flag many negative sentences. That is why the 0.7442 accuracy figure is not the whole story.

### Alternatives and trade-offs
Recall matters when missing a case is costly. Calling every sentence negative would find all 91 negatives, giving recall 1. But precision would fall to 91/727, about 0.1252. Whether to favor precision or recall depends on the intended use, not which number looks nicer.

### Check your understanding
**Question:** Which denominator belongs to negative recall: 51 or 91?

**Answer explained:** Ninety-one. Recall starts with the true negative sentences, which form a row. Fifty-one belongs to precision because it counts the model's negative answers, which form a column.

### Deeper dive and common confusion
**Common confusion:** “false negative” is an error type, not the sentiment label. For positive-sentiment recall, a positive sentence called neutral is also a false negative—because the positive class was missed.


## Lesson: Combine precision and recall, then give all three classes a voice

### Why it matters
We want a model that both finds negative sentences and avoids too many false flags. We also do not want the many neutral examples to dominate the headline score.

### Build the intuition
First balance precision and recall within each class. Then average the three class scores equally. This gives negative and positive sentiment a voice even though neutral examples are more numerous.

### A concrete example
**Actual counts:** negative has 42 correct, 9 false flags and 49 misses. Neutral has 390 correct, 127 false flags and 42 misses. Positive has 109 correct, 50 false flags and 95 misses.

### Walk through it
1. Calculate negative F1: $84/(84+9+49)=0.5915$.
2. Calculate neutral F1: $780/(780+127+42)=0.8219$.
3. Calculate positive F1: $218/(218+50+95)=0.6006$.
4. Add the three scores and divide by three: approximately 0.6713.

### Give the idea its technical name
**F1** is a score that balances precision and recall using their harmonic mean, a type of average that is pulled down by a low value. **Macro-F1** means calculate F1 for each class first, then take an ordinary equal-weight average.

### Add the mathematics
For precision $P$ and recall $R$, $F1=2PR/(P+R)$. The count version is $2TP/(2TP+FP+FN)$. Both give the same answer. For the baseline's negative class, the ordinary average of 0.8235 and 0.4615 would be 0.6425, while F1 is 0.5915: poor recall matters.

With $K$ classes, $F1_{macro}=(F1_1+...+F1_K)/K$. Here $K=3$. Using unrounded saved class scores gives 0.6713393560603845; the displayed value is 0.6713.

### Connect it to our project
Macro-F1 is the project's main classification score. The same calculation is applied to each saved model run before taking the average across runs. [Phase 3's pipeline](../src/finbert_explainability/phase3/pipeline.py) recalculates the metrics from saved predictions.

### Alternatives and trade-offs
**Weighted F1** gives each class a weight equal to its share of examples: here 91/727, 432/727 and 204/727. It lets neutral influence the answer more. **Micro-F1** combines error counts before calculating F1; for this single-label, three-class task with all classes included, it equals accuracy. Macro-F1 is useful for equal class importance, but a real application may need explicit costs for different mistakes.

### Check your understanding
**Question:** Can we average the three precisions and recalls first, then calculate F1 and call that macro-F1?

**Answer explained:** No. The required order is F1 within each class, then average those F1 values. Changing the order changes the calculation because division and averaging do not generally commute.

### Deeper dive and common confusion
**Common confusion:** F1 has no separate term for true negatives—the examples outside a chosen class that were correctly left outside it. It also does not encode financial costs. A missed negative sentence and an incorrect negative flag can have different consequences even if a summary score treats them symmetrically.


## Lesson: How much should we trust a difference between runs?

### Why it matters
BERT's mean is higher than FinBERT's. Could a different training shuffle or a different sample of test sentences change the gap? These are two different sources of variation.

### Build the intuition
Imagine repeating an exam with three independently trained students, then imagining different collections of exam questions. Changing the student resembles changing the training seed. Changing the questions resembles resampling the evaluation examples. One measure does not cover both.

### A concrete example
**Invented arithmetic example:** three run scores are 0.80, 0.82 and 0.84. Their average is 0.82. The deviations are -0.02, 0 and +0.02.

**Invented resampling example:** four sentences have paired method differences `[0.02, -0.01, 0.03, 0.00]`. Their mean is 0.01. A new draw of indices `[1,1,3,4]` gives differences `[0.02,0.02,0.03,0]` and mean 0.0175. Another draw can give a lower mean.

### Walk through it
1. For seed variation, calculate the mean across trained runs.
2. Square each run's distance from that mean, add, divide by one fewer than the number of runs, then take the square root.
3. For resampling uncertainty, draw the same number of test rows with replacement: a row may be drawn more than once.
4. Recalculate the statistic for that draw. Repeat many times.
5. For paired model comparisons, keep both models' predictions for each drawn row together.

### Give the idea its technical name
The first calculation is **sample standard deviation**, or SD. The resampling procedure is the **bootstrap**. A **confidence interval** summarizes uncertainty from a specified statistical procedure. It is not simply the range of seed scores.

### Add the mathematics
For $S$ run scores $m_s$, let $m_{avg}$ be their mean. Then $s=\sqrt{\sum_{s=1}^{S}(m_s-m_{avg})^2/(S-1)}$. In the toy example, $s=\sqrt{(0.0004+0+0.0004)/2}=0.02$.

A percentile bootstrap takes, for example, the 2.5th and 97.5th percentiles of repeated estimates for a 95% interval. The four-row toy example explains drawing; it is far too small to establish a reliable real-world interval.

### Connect it to our project
The real results are BERT 0.8476 ± 0.0034 and FinBERT 0.8339 ± 0.0093 macro-F1, with sample SD across seeds 42, 123 and 2026. These SDs are not a significance test. Phase 5 separately uses 2000 sentence-bootstrap draws, seed 2042, for explanation statistics. It does not bootstrap tokens as independent examples.

### Alternatives and trade-offs
A paired bootstrap of classification F1 would recompute the entire F1 on each sampled set; there is no useful per-sentence F1 to average. McNemar's test is another option for paired correct/wrong outcomes, but targets accuracy rather than macro-F1. If sentences share events, resampling whole event groups can be more appropriate, provided the grouping is trustworthy.

### Check your understanding
**Question:** Can non-overlapping seed SD bars prove BERT is better in general?

**Answer explained:** No. They describe variation among these three fits on this fixed test. They do not directly account for different test samples, future news or all possible training conditions.

### Deeper dive and common confusion
**Common confusion:** a 95% interval is not a promise that 95% of individual sentences behave like its mean. It concerns the statistic under the resampling assumptions. Also, an interval containing zero does not prove two methods are equivalent. Equivalence needs a meaningful difference limit chosen before looking at the result.


## Lesson: Find out who the average score leaves behind

### Why it matters
An average may look good even if the model struggles on sentences people find ambiguous. We should check those sentences rather than assuming one score applies equally to every kind of example.

### Build the intuition
Separate sentences that every annotator agreed about from the rest. Compare their scores, but also count how many examples of each sentiment are in each group. Different mixtures can affect a score.

### A concrete example
**Measured data:** allagree has 338 test sentences: 50 negative, 209 neutral, 79 positive. Non-allagree has 389: 41 negative, 223 neutral, 125 positive. BERT macro-F1 is 0.9583 on the first group and 0.7500 on the second.

### Walk through it
1. Use the saved unanimous-agreement membership, not a new guess about difficulty.
2. Select each group's predictions by stable row ID.
3. Recalculate its confusion matrix and F1.
4. Report group sizes and class counts alongside the score.
5. Inspect errors while keeping their selection method visible.

### Give the idea its technical name
A **subgroup evaluation** checks a specified subset of the data. An **observational comparison** means we observed existing groups; we did not randomly change agreement while holding everything else fixed.

### Add the mathematics
The displayed BERT gap is 0.9583 − 0.7500 = 0.2083. That is a large observed score difference, but it is not a measured causal effect of disagreement. Only 41 negative examples are in the second group, so a few changed predictions can matter.

### Connect it to our project
BERT subgroup SDs are 0.0094 and 0.0142. FinBERT means are 0.9553 and 0.7262. Counts at or below 50 are flagged. [inputs.py](../src/finbert_explainability/phase3/inputs.py) checks alignment; [errors.py](../src/finbert_explainability/phase3/errors.py) creates a selected 20-error review file.

### Alternatives and trade-offs
Looking only at allagree gives cleaner labels but a different, easier population. Randomly sampling errors would estimate their overall mix better than selecting extreme errors. The project's confidence/disagreement-focused sample is useful for finding instructive cases, not estimating how common each cause is.

### Check your understanding
**Question:** Fourteen of 20 reviewed errors have at least 25% annotator disagreement. Does that prove disagreement causes most errors?

**Answer explained:** No. These 20 were deliberately selected for confident errors and model disagreements. They are not a random sample of all errors, and disagreement may travel with other difficult sentence features.

### Deeper dive and common confusion
**Common confusion:** user-supplied error categories and AI-assisted explanatory notes are different evidence. All 20 categories are complete; the interpretations/comments contain labeled AI-assisted drafts, not independent human-confirmed causes. Exact annotator vote counts are unavailable. “At least 25%” must not become “strictly more than 25%.”


## Read the complete saved comparison

Now the columns should have concrete meanings: precision asks about a prediction queue, recall asks about finding the true examples, and macro-F1 gives all three classes equal weight. “±” below is sample SD across three runs, not a confidence interval.

| Workflow | Macro-F1 mean ± sample SD | Accuracy mean ± sample SD | Negative / neutral / positive F1 means |
|---|---:|---:|---|
| Baseline | 0.6713 ± 0.0000 | 0.7442 ± 0.0000 | 0.5915 / 0.8219 / 0.6006 |
| BERT | 0.8476 ± 0.0034 | 0.8556 ± 0.0014 | 0.8670 / 0.8839 / 0.7919 |
| FinBERT | 0.8339 ± 0.0093 | 0.8469 ± 0.0097 | 0.8390 / 0.8799 / 0.7828 |

| Workflow / class | Precision | Recall |
|---|---:|---:|
| Baseline negative | 0.8235 ± 0.0000 | 0.4615 ± 0.0000 |
| Baseline neutral | 0.7544 ± 0.0000 | 0.9028 ± 0.0000 |
| Baseline positive | 0.6855 ± 0.0000 | 0.5343 ± 0.0000 |
| BERT negative | 0.8625 ± 0.0135 | 0.8718 ± 0.0168 |
| BERT neutral | 0.8926 ± 0.0143 | 0.8758 ± 0.0171 |
| BERT positive | 0.7797 ± 0.0200 | 0.8056 ± 0.0283 |
| FinBERT negative | 0.8290 ± 0.0132 | 0.8498 ± 0.0277 |
| FinBERT neutral | 0.8922 ± 0.0105 | 0.8681 ± 0.0083 |
| FinBERT positive | 0.7659 ± 0.0194 | 0.8007 ± 0.0075 |

Recompute F1 within each run before averaging. The mean of three F1 scores need not equal F1 calculated from the mean precision and mean recall.

The current user categories in [human_annotation.csv](../artifacts/phase3/human_annotation.csv) are entity context 7, financial context 4, implicit sentiment 4, negation 2, and complex sentence, numerical reasoning and label ambiguity 1 each. The selection includes 20 confidence-at-least-0.9 errors and 10 paired-seed BERT/FinBERT disagreements. These counts describe the reviewed sample only.

The ProsusAI reference has F1 0.8797, but its earlier PhraseBank training can overlap this test. It is therefore kept out of the independent ranking. [Chapter 03](03_model_architecture.md) explains that distinction.

**Check:** Why not declare the largest number in this chapter the best deployable model? **Answer:** We must check whether evaluation was independent, what mistakes matter, how confidence behaves and whether performance transfers to the intended use. No one number supplies those answers.

## Deeper research choices, after the basics

If you compare more than one model pair or many groups, you have more chances to find an apparently interesting result by luck. Choose important comparisons in advance and label later discoveries as exploratory. Report the difference itself and its uncertainty, not just whether a test crosses a threshold.

A production use may prefer a table of error costs, a curve showing precision as the review threshold changes, or a curve showing remaining error as uncertain cases are sent to humans. Choose these policies on validation data. Future confirmation needs a new independent holdout because this test has already shaped our research questions.

**Check:** What would a paired classification bootstrap keep together? **Answer:** A sampled sentence's human label and both models' predictions. Splitting those apart would discard the fact that both models answered the same question.


## Practice explaining it

1. **Explain precision without using a formula.**

   Of the sentences the model called negative, how many really were negative?

2. **Explain recall without using a formula.**

   Of all the real negative sentences, how many did the model find?

3. **Why is the baseline negative recall lower than its precision?**

   It found only 42 of 91 true negatives, even though 42 of its 51 negative predictions were right.

4. **What does macro-F1 do?**

   Calculate an F1 for each of negative, neutral and positive, then average the three equally.

5. **Why does seed SD not prove superiority?**

   It covers variation among a few trained runs on one fixed test, not every relevant source of uncertainty.

## Sources and next lesson

[Saved classification comparison](../artifacts/phase2/comparison/comparison.json), [Phase 3 results](../PHASE3_RESULTS.md), [Phase 6 annotation interpretation](../PHASE6_RESULTS.md), [statistical resources](references.md). Continue to [calibration and robustness](06_calibration_and_robustness.md).
