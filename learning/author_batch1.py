from course_writer import finish, lesson, tables, graph

name='00_project_overview.md'
lessons=[lesson('What are we asking the computer to do?',
'''Imagine an analyst receiving hundreds of financial headlines. Which ones sound favorable, unfavorable or mainly factual? Reading everything takes time. A model might help sort the queue, but we need to know when its suggestions are dependable.''',
'''Our model reads one sentence and chooses one of three labels: positive, negative or neutral. Think of it as a sorting assistant. It is not being asked to predict tomorrow's share price.''',
'''**Invented teaching examples:**

| Sentence | Plausible sentiment in this simplified example |
|---|---|
| “Operating profit increased from EUR 10 million to EUR 12 million.” | Positive |
| “Operating losses increased from EUR 10 million to EUR 12 million.” | Negative |
| “The company will publish its report on Tuesday.” | Neutral |

The first two sentences contain the same numbers and the word “increased.” What increased changes the meaning.''',
'''1. Read the sentence, including the financial quantity being discussed.
2. Ask whether the statement sounds favorable, unfavorable or mainly factual.
3. Let the model make the same three-way choice.
4. Compare its choice with the dataset's human-assigned label.
5. Inspect mistakes rather than assuming the model understands the financial relationship.''',
'''This is **financial sentiment classification**: assigning a financial-language example to a sentiment category. A **label** is the category attached to an example. Here, labels come from human judgments.''',
'''The computer stores labels as numbers: negative = 0, neutral = 1, positive = 2. These are names in numerical form, not a scale where positive is “twice neutral.” A model might return probabilities `[0.10, 0.15, 0.75]` in that order. It chooses positive because 0.75 is largest. We will explain how it produces those probabilities in [Chapter 01](01_foundations_and_theory.md).''',
'''The project uses Financial PhraseBank, a collection of labeled financial sentences. [data.py](../src/finbert_explainability/data.py) defines the label order. The final evaluation uses the same 727 test sentences for all independently trained workflows.''',
'''A word-list rule is cheap, but “increased” shows why individual words are not enough. A language model can use surrounding words, but costs more and can still learn shortcuts. A human-review queue is a plausible use; autonomous trading is outside the evaluated task.''',
'''If the model labels “profits increased” correctly, have we shown that it can predict profitable trades?''',
'''No. We tested agreement with sentiment labels. Trading would require different targets, future-market data and a separate evaluation of decisions and costs.''',
'''**Common confusion:** a correct label does not prove the model used the right reasoning. It might react to “increased” even when the subject is losses. Later chapters test this possibility with controlled edits. Those tests also have limits: a changed sentence needs its own reviewed label before we can call the new prediction correct or wrong.'''),
lesson('How do six phases turn a model into evidence?',
'''Suppose a model gets an excellent score because a nearly identical headline appeared during training. The score looks impressive, but it may tell us little about genuinely new sentences. We need a process that can catch that problem.''',
'''Treat the project like a practical exam. Training examples are the practice questions. Validation examples help choose a study strategy. Test examples are the final exam. Explanations are then another claim to examine, not a decorative chart added after the score.''',
'''**Invented teaching example:** “Acme profit rose 10%” appears in practice, while “Acme's profit rose 10 %” appears in the exam. A punctuation difference should not make us call them completely independent evidence.''',
'''1. **Phase 1:** find similar sentences, keep related groups together, and save the assignments.
2. **Phase 2:** train the models, choose settings using validation examples, then evaluate on test examples.
3. **Phase 3:** ask where predictions fail and whether confidence is trustworthy.
4. **Phase 4:** assign contributions to input tokens for a fixed saved model.
5. **Phase 5:** compare those contributions with attention and test what happens when selected tokens are replaced.
6. **Phase 6:** recalculate results from saved records, inspect exclusions, and write a bounded validation report.''',
'''A **split** is a division of examples into training, validation and test groups. **Data leakage** occurs when the evaluation benefits from information it should not have had. An **artifact** is a saved output such as a prediction file or checkpoint. A **checkpoint** is a saved model state.''',
'''Our saved split has 3391 training + 728 validation + 727 test rows = 4846 total. The intended proportions are 70/15/15; indivisible duplicate groups and integer row counts make exact percentages impractical. Nothing was dropped to make the totals look neat.''',
'''The code path is [Phase 1 pipeline](../src/finbert_explainability/pipeline.py) → [Phase 2 workflows](../src/finbert_explainability/phase2/workflows.py) → [Phase 3 analysis](../src/finbert_explainability/phase3/pipeline.py) → [IG core](../src/finbert_explainability/phase4/core.py) → [attention/deletion](../src/finbert_explainability/phase5/inference.py) → [final audit](../src/finbert_explainability/phase6/audit.py). Each stage reads earlier evidence and saves new outputs.''',
'''A single random split is simpler, but can separate duplicate headlines. Keeping entire news events or time periods apart could test stronger forms of independence, but needs reliable event or date information. The present lexical detector checks text similarity; it cannot prove that all meaning-level overlap is gone.''',
'''Can we choose the settings that give the highest test score and still call that test an untouched exam?''',
'''No. Once test results guide the choice, they are helping us practice. We need separate examples to assess the resulting choice fairly.''',
'''The project saves file hashes: fingerprints used to detect changed bytes. A hash tells us a file stayed the same; it does not tell us the labels are correct. Similarly, the test can remain byte-identical while repeated inspection influences future research ideas. That is why future improvement claims need a new independent holdout—a set reserved for confirmation, not development.'''),
lesson('What did we actually learn?',
'''The project name might make you expect FinBERT to win. What should we do when general BERT gets the higher score? Report the result, then separate what happened from our guesses about why.''',
'''We asked several different questions. How often does a model classify well across all three classes? Does it know when it is likely to be wrong? Do two explanation methods highlight similar tokens? Does hiding their highlighted tokens weaken the prediction? A good answer to one question does not automatically answer the others.''',
'''**Measured finding:** general BERT's mean macro-F1 is 0.8476; FinBERT's is 0.8339. Macro-F1 is a score that gives each of the three classes equal importance; [Chapter 05](05_evaluation_and_validation.md) calculates it from actual counts. For now, read higher as better on that particular classification score.''',
'''1. Compare models on the same saved examples.
2. Repeat training with three fixed random starting sequences, called seeds.
3. Report the average and the variation across runs.
4. Inspect agreement groups, confidence and mistakes.
5. Compare explanation methods without assuming either is the correct answer.''',
'''A **measured finding** is something the saved results show. An **interpretation** is a reasoned reading of those results. A **hypothesis** is a possible explanation that still needs a test. A **proposal** describes work we have not done.''',
'''Subtract the displayed scores: 0.8476 − 0.8339 = 0.0137, or about 1.37 percentage points. This tells us the size and direction of the displayed difference. It does not identify its cause or prove the same winner on future data.''',
'''The table below preserves the measured headlines. Each row has a later lesson that explains its calculation. The separate ProsusAI model scored 0.8797 F1, but its earlier training included PhraseBank; some local test examples may not be new to it. We therefore do not treat it as an independent winner.''',
'''We could report only the best score. That is shorter, but hides confidence errors, difficult examples and explanation failures. We could demand a full production trial before reporting anything, but that answers a broader question. This project supports research conclusions and a cautious human-review recommendation, not production certification.''',
'''Does “BERT scored higher” justify saying “financial pretraining made FinBERT worse”?''',
'''No. The workflows differ in more than one way, including their starting weights and vocabulary. A small search and only three seeds also leave uncertainty. Chapter 11 designs tests of competing explanations instead of presenting one as proven.''',
'''**Common confusion:** a colorful token explanation is not automatically a faithful account of the prediction. Here “faithful” means it reflects the model behavior it claims to describe. Numerical accuracy, human plausibility and causal evidence are separate questions. We build those distinctions gradually in Chapters 07–09.''')]
notebook='''## Keep the whole project in view

'''+graph(name)+'''

### The results, translated before the technical detail

| Everyday reading | Saved result | Where we unpack it |
|---|---|---|
| General BERT performed better on the main class-balanced score | TF-IDF 0.6713; BERT 0.8476; FinBERT 0.8339 | [05](05_evaluation_and_validation.md) |
| Examples people agreed about were much easier for BERT | Allagree 0.9583; non-allagree 0.7500 macro-F1 | [05](05_evaluation_and_validation.md) |
| The two token rankings agreed somewhat, but far from perfectly | Mean Spearman 0.2406 over 687 sentences | [09](09_explainability_experiments.md) |
| Both selected rankings weakened the original prediction more than random masking on average | IG / attention / random probability AUC 0.6555 / 0.6608 / 0.7866 | [09](09_explainability_experiments.md) |
| We could not clearly separate IG from attention in that comparison | Paired difference 95% interval [-0.0130, 0.0024] includes zero | [09](09_explainability_experiments.md) |
| Some IG calculations did not meet the fixed numerical accuracy rule | 40 of 727 retained and flagged | [07](07_integrated_gradients.md), [09](09_explainability_experiments.md) |

**Check:** Is 0.2406 an accuracy of 24.06%? **Answer:** No. It measures agreement between token rankings, not correctness of sentiment predictions. Different columns answer different questions.

### Explain the project at different lengths

**30 seconds.** I studied how well language models classify financial sentences and how much we can trust their explanations. I prevented detected duplicate sentences from crossing data splits, compared a simple model with general and financial BERT, and checked confidence and errors. General BERT had the higher average class-balanced score. For one fixed FinBERT model, I compared gradient-based token contributions with attention. Neither explanation method clearly won the paired comparison, and I kept all numerical failures visible.

**Two minutes.** Start with “profits rose” versus “losses rose”: financial sentiment needs context. I used human-labeled financial sentences, grouped detected duplicates, and froze training, validation and test assignments. Models learned only from training rows; validation rows selected settings. Three final runs per supervised workflow helped show how much training randomness mattered.

General BERT's mean macro-F1 was 0.8476, versus 0.8339 for financial BERT and 0.6713 for TF-IDF. I do not know the isolated cause of the BERT–FinBERT gap. The ProsusAI checkpoint was kept separate because its original training may overlap this test. Models did much better on unanimous human labels. Their confidence and reactions to edited sentences also needed qualification.

I then fixed FinBERT seed 42, rather than choosing an attractive test run. Integrated Gradients assigns token contributions relative to a reference input. Changing the reference changed rankings. I compared those rankings with attention and replaced selected tokens with MASK. Both suppressed predictions more than random ordering on average, but the uncertainty interval for IG versus attention included zero. Forty calculations missed the unchanged numerical tolerance and remained in a separate sensitivity analysis. The result is a reproducible research study, not proof that a model or heatmap is safe for autonomous financial decisions.

**Ten-minute route.** Spend 1 minute on the sentence examples; 2 on data and duplicate grouping; 2 on model choices and validation-only selection; 1 on the confusion matrix and agreement groups; 1 on confidence; 2 on IG, attention and masking; 1 on the uncertain findings and next independent evaluation. Use [the timed 30-minute interview](interview_preparation.md) for the longer version.

**Check:** Which sentence should follow “BERT won”? **Answer:** “Under this benchmark and evaluation procedure; the reason and wider generalization need further testing.” That is a boundary on the evidence, not an excuse for it.

### Find the saved work

The outputs include saved data splits, prediction files, model checkpoints, search records, error-review CSVs, explanation records, figures, a five-page report and publication-preparation files. The [coverage matrix](coverage_matrix.md) maps all six phases to chapters. The [comparison JSON](../artifacts/phase2/comparison/comparison.json), [Phase 3 report](../PHASE3_RESULTS.md) and [sensitivity JSON](../artifacts/phase6/audit-final/ig_sensitivity.json) support the figures above.

**Check:** Do we need to retrain to read these results? **Answer:** No. This course uses completed records. Reproduction is a separate, explicitly authorized activity with fresh outputs.
'''
finish(name,'''# 00 — Start with a financial sentence

**Before you begin:** no project-specific knowledge is needed. By the end, you should be able to explain what the project asked, how it tested the answers and what it still cannot claim. Leave unfamiliar metric names in the results table for their linked lessons; you do not need their formulas yet.''',lessons,notebook,[('What is the model sorting?','English financial sentences into negative, neutral and positive sentiment.'),('Why do “profit rose” and “loss rose” matter?','They show why a word can have different financial implications depending on its context.'),('What are validation examples for?','Choosing settings before the final evaluation.'),('What did the BERT comparison show?','A higher observed mean for general BERT, without a proven causal explanation.'),('Why keep failed explanations?','They reveal limits of the numerical procedure and can change which examples the conclusion describes.')])

name='05_evaluation_and_validation.md'
matrix='''**Actual project data: the TF-IDF baseline on all 727 test sentences.** Read a row as the human-assigned label, and a column as the model's answer.

| Human label / model answer | Negative | Neutral | Positive | Row total |
|---|---:|---:|---:|---:|
| Negative | 42 | 37 | 12 | 91 |
| Neutral | 4 | 390 | 38 | 432 |
| Positive | 5 | 90 | 109 | 204 |
| Column total | 51 | 517 | 159 | 727 |

For example, 37 genuinely negative examples were labeled neutral by the model. We will use this same table throughout the chapter.'''
lessons=[lesson('Read the prediction table before calculating a score',
'''An analyst asks, “How well did the model do?” A single percentage cannot tell you which mistakes it made. First we need a count of the answers.''',
'''Put the human labels down the side and the model's answers across the top. A correct answer lands where the row and column names match. Everything else is a mistake.''',matrix,
'''1. Follow the negative row: 42 correct, 37 called neutral, 12 called positive.
2. Follow the negative column: the model called 51 sentences negative, but 9 came from other human labels.
3. Add the matching cells: 42 + 390 + 109 = 541 correct answers.
4. Add all cells: 727 examples.''',
'''This count table is a **confusion matrix**. “Confusion” simply refers to one class being mistaken for another. The dataset's reference label is sometimes called the **gold label**; that term does not imply that people can never disagree or be wrong.''',
r'''**Accuracy** is correct answers divided by all answers: 541 / 727 = 0.7442, or about 74.42%. If we use symbols, $A=\text{correct}/N$, where $A$ is accuracy and $N$ is the number of evaluated examples.''',
'''The matrix is saved in [comparison.json](../artifacts/phase2/comparison/comparison.json). [metrics.py](../src/finbert_explainability/phase2/metrics.py) uses rows=true and columns=predicted. Keeping that convention explicit avoids reading the wrong denominator.''',
'''Accuracy is easy to explain. But a model that always says neutral would get 432 / 727, about 59.42%, because neutral is common. We therefore need scores that look separately at negative, neutral and positive predictions.''',
'''What does the cell 90 in the positive row and neutral column mean?''',
'''Ninety sentences labeled positive by the dataset were predicted neutral. It does not mean 90 neutral sentences were called positive; that would be the opposite cell, which contains 38.''',
'''**Common confusion:** a mean matrix across three runs can contain decimals. Those are averages of integer counts, not fractions of a sentence in a single run. Do not calculate a mean-run F1 by assuming the averaged matrix is one actual run; F1 is a nonlinear calculation, explained below.'''),
lesson('When the model raises a negative flag, how often is it right?',
'''Suppose an analyst reviews every sentence the model calls negative. Too many false alarms waste review time. We want to know the quality of that queue.''',
'''Start with the model's negative answers—not all real negative sentences. Ask how many of those answers were correct.''',
'''In the actual matrix, the negative column contains 42 + 4 + 5 = 51 answers. Forty-two are correct. Nine are false alarms: 4 neutral and 5 positive examples.''',
'''1. Choose the negative prediction column.
2. Count all 51 entries in that column.
3. Keep the 42 whose human label is also negative.
4. Divide 42 by 51.''',
'''This is **precision** for the negative class. A **true positive** for this calculation means a correctly predicted negative example. A **false positive** means a negative prediction that should have been another class. “Positive” in these error-count names means “the class currently being checked,” not necessarily positive financial sentiment.''',
'''Let $TP$ be correct negative flags and $FP$ be false negative-class flags. Then $P=TP/(TP+FP)=42/(42+9)=0.8235$. Here $P$ is precision. Read this as: about 82.35% of the negative flags were right.''',
'''The baseline's saved negative precision is 0.8235. That sounds strong, but it does not yet tell us how many negative sentences never entered the queue.''',
'''Precision matters when false alarms are costly. A very cautious model can improve precision by making fewer flags. That may be appropriate for a small review team, but dangerous if missed negative news matters more. We need the next lesson before choosing a policy.''',
'''If a model flags only one sentence and gets it right, is its precision 100%?''',
'''Yes, for that tiny queue. It could still miss almost every other negative sentence. High precision alone says nothing about how complete the queue is.''',
'''**Common confusion:** high precision is not “high overall accuracy.” Each uses a different starting set. Also, if there are no predictions for a class, the raw fraction has a zero denominator. Software must declare its convention; inspect the metric implementation rather than silently dropping that class.'''),
lesson('How many real negative sentences did the model find?',
'''Now imagine that missing bad news is the main concern. A precise queue is not enough if most negative sentences were never flagged.''',
'''Start with all sentences that people labeled negative. Ask how many the model successfully found. This reverses the direction of the previous question.''',
'''The actual negative row has 91 sentences. The model found 42. It missed 37 + 12 = 49 by calling them neutral or positive.''',
'''1. Choose the negative human-label row.
2. Count all 91 sentences in that row.
3. Keep the 42 correctly predicted negative.
4. Divide 42 by 91.''',
'''This is **recall** for the negative class. The 49 missed examples are **false negatives** for the class we are checking. Recall answers, “Of the negative sentences that were there, what fraction did we find?”''',
'''Let $FN$ be the missed negative examples and $R$ be recall. Then $R=TP/(TP+FN)=42/(42+49)=0.4615$. The model found about 46.15% of the real negative examples and missed more than half.''',
'''The baseline has negative precision 0.8235 but recall 0.4615. In everyday language: its negative flags were usually sensible, but it failed to flag many negative sentences. That is why the 0.7442 accuracy figure is not the whole story.''',
'''Recall matters when missing a case is costly. Calling every sentence negative would find all 91 negatives, giving recall 1. But precision would fall to 91/727, about 0.1252. Whether to favor precision or recall depends on the intended use, not which number looks nicer.''',
'''Which denominator belongs to negative recall: 51 or 91?''',
'''Ninety-one. Recall starts with the true negative sentences, which form a row. Fifty-one belongs to precision because it counts the model's negative answers, which form a column.''',
'''**Common confusion:** “false negative” is an error type, not the sentiment label. For positive-sentiment recall, a positive sentence called neutral is also a false negative—because the positive class was missed.'''),
lesson('Combine precision and recall, then give all three classes a voice',
'''We want a model that both finds negative sentences and avoids too many false flags. We also do not want the many neutral examples to dominate the headline score.''',
'''First balance precision and recall within each class. Then average the three class scores equally. This gives negative and positive sentiment a voice even though neutral examples are more numerous.''',
'''**Actual counts:** negative has 42 correct, 9 false flags and 49 misses. Neutral has 390 correct, 127 false flags and 42 misses. Positive has 109 correct, 50 false flags and 95 misses.''',
'''1. Calculate negative F1: $84/(84+9+49)=0.5915$.
2. Calculate neutral F1: $780/(780+127+42)=0.8219$.
3. Calculate positive F1: $218/(218+50+95)=0.6006$.
4. Add the three scores and divide by three: approximately 0.6713.''',
'''**F1** is a score that balances precision and recall using their harmonic mean, a type of average that is pulled down by a low value. **Macro-F1** means calculate F1 for each class first, then take an ordinary equal-weight average.''',
r'''For precision $P$ and recall $R$, $F1=2PR/(P+R)$. The count version is $2TP/(2TP+FP+FN)$. Both give the same answer. For the baseline's negative class, the ordinary average of 0.8235 and 0.4615 would be 0.6425, while F1 is 0.5915: poor recall matters.

With $K$ classes, $F1_{macro}=(F1_1+...+F1_K)/K$. Here $K=3$. Using unrounded saved class scores gives 0.6713393560603845; the displayed value is 0.6713.''',
'''Macro-F1 is the project's main classification score. The same calculation is applied to each saved model run before taking the average across runs. [Phase 3's pipeline](../src/finbert_explainability/phase3/pipeline.py) recalculates the metrics from saved predictions.''',
'''**Weighted F1** gives each class a weight equal to its share of examples: here 91/727, 432/727 and 204/727. It lets neutral influence the answer more. **Micro-F1** combines error counts before calculating F1; for this single-label, three-class task with all classes included, it equals accuracy. Macro-F1 is useful for equal class importance, but a real application may need explicit costs for different mistakes.''',
'''Can we average the three precisions and recalls first, then calculate F1 and call that macro-F1?''',
'''No. The required order is F1 within each class, then average those F1 values. Changing the order changes the calculation because division and averaging do not generally commute.''',
'''**Common confusion:** F1 has no separate term for true negatives—the examples outside a chosen class that were correctly left outside it. It also does not encode financial costs. A missed negative and a false negative flag can have different consequences even if a summary score treats them symmetrically.'''),
lesson('How much should we trust a difference between runs?',
'''BERT's mean is higher than FinBERT's. Could a different training shuffle or a different sample of test sentences change the gap? These are two different sources of variation.''',
'''Imagine repeating an exam with three independently trained students, then imagining different collections of exam questions. Changing the student resembles changing the training seed. Changing the questions resembles resampling the evaluation examples. One measure does not cover both.''',
'''**Invented arithmetic example:** three run scores are 0.80, 0.82 and 0.84. Their average is 0.82. The deviations are -0.02, 0 and +0.02.

**Invented resampling example:** four sentences have paired method differences `[0.02, -0.01, 0.03, 0.00]`. Their mean is 0.01. A new draw of indices `[1,1,3,4]` gives differences `[0.02,0.02,0.03,0]` and mean 0.0175. Another draw can give a lower mean.''',
'''1. For seed variation, calculate the mean across trained runs.
2. Square each run's distance from that mean, add, divide by one fewer than the number of runs, then take the square root.
3. For resampling uncertainty, draw the same number of test rows with replacement: a row may be drawn more than once.
4. Recalculate the statistic for that draw. Repeat many times.
5. For paired model comparisons, keep both models' predictions for each drawn row together.''',
'''The first calculation is **sample standard deviation**, or SD. The resampling procedure is the **bootstrap**. A **confidence interval** summarizes uncertainty from a specified statistical procedure. It is not simply the range of seed scores.''',
r'''For $S$ run scores $m_s$, let $m_{avg}$ be their mean. Then $s=\sqrt{\sum_{s=1}^{S}(m_s-m_{avg})^2/(S-1)}$. In the toy example, $s=\sqrt{(0.0004+0+0.0004)/2}=0.02$.

A percentile bootstrap takes, for example, the 2.5th and 97.5th percentiles of repeated estimates for a 95% interval. The four-row toy example explains drawing; it is far too small to establish a reliable real-world interval.''',
'''The real results are BERT 0.8476 ± 0.0034 and FinBERT 0.8339 ± 0.0093 macro-F1, with sample SD across seeds 42, 123 and 2026. These SDs are not a significance test. Phase 5 separately uses 2000 sentence-bootstrap draws, seed 2042, for explanation statistics. It does not bootstrap tokens as independent examples.''',
'''A paired bootstrap of classification F1 would recompute the entire F1 on each sampled set; there is no useful per-sentence F1 to average. McNemar's test is another option for paired correct/wrong outcomes, but targets accuracy rather than macro-F1. If sentences share events, resampling whole event groups can be more appropriate, provided the grouping is trustworthy.''',
'''Can non-overlapping seed SD bars prove BERT is better in general?''',
'''No. They describe variation among these three fits on this fixed test. They do not directly account for different test samples, future news or all possible training conditions.''',
'''**Common confusion:** a 95% interval is not a promise that 95% of individual sentences behave like its mean. It concerns the statistic under the resampling assumptions. Also, an interval containing zero does not prove two methods are equivalent. Equivalence needs a meaningful difference limit chosen before looking at the result.'''),
lesson('Find out who the average score leaves behind',
'''An average may look good even if the model struggles on sentences people find ambiguous. We should check those sentences rather than assuming one score applies equally to every kind of example.''',
'''Separate sentences that every annotator agreed about from the rest. Compare their scores, but also count how many examples of each sentiment are in each group. Different mixtures can affect a score.''',
'''**Measured data:** allagree has 338 test sentences: 50 negative, 209 neutral, 79 positive. Non-allagree has 389: 41 negative, 223 neutral, 125 positive. BERT macro-F1 is 0.9583 on the first group and 0.7500 on the second.''',
'''1. Use the saved unanimous-agreement membership, not a new guess about difficulty.
2. Select each group's predictions by stable row ID.
3. Recalculate its confusion matrix and F1.
4. Report group sizes and class counts alongside the score.
5. Inspect errors while keeping their selection method visible.''',
'''A **subgroup evaluation** checks a specified subset of the data. An **observational comparison** means we observed existing groups; we did not randomly change agreement while holding everything else fixed.''',
'''The displayed BERT gap is 0.9583 − 0.7500 = 0.2083. That is a large observed score difference, but it is not a measured causal effect of disagreement. Only 41 negative examples are in the second group, so a few changed predictions can matter.''',
'''BERT subgroup SDs are 0.0094 and 0.0142. FinBERT means are 0.9553 and 0.7262. Counts at or below 50 are flagged. [inputs.py](../src/finbert_explainability/phase3/inputs.py) checks alignment; [errors.py](../src/finbert_explainability/phase3/errors.py) creates a selected 20-error review file.''',
'''Looking only at allagree gives cleaner labels but a different, easier population. Randomly sampling errors would estimate their overall mix better than selecting extreme errors. The project's confidence/disagreement-focused sample is useful for finding instructive cases, not estimating how common each cause is.''',
'''Fourteen of 20 reviewed errors have at least 25% annotator disagreement. Does that prove disagreement causes most errors?''',
'''No. These 20 were deliberately selected for confident errors and model disagreements. They are not a random sample of all errors, and disagreement may travel with other difficult sentence features.''',
'''**Common confusion:** user-supplied error categories and AI-assisted explanatory notes are different evidence. All 20 categories are complete; the interpretations/comments contain labeled AI-assisted drafts, not independent human-confirmed causes. Exact annotator vote counts are unavailable. “At least 25%” must not become “strictly more than 25%.”''')]
notebook='''## Read the complete saved comparison

Now the columns should have concrete meanings: precision asks about a prediction queue, recall asks about finding the true examples, and macro-F1 gives all three classes equal weight. “±” below is sample SD across three runs, not a confidence interval.

'''+tables(name)[1]+'\n\n'+tables(name)[2]+'''

Recompute F1 within each run before averaging. The mean of three F1 scores need not equal F1 calculated from the mean precision and mean recall.

The current user categories in [human_annotation.csv](../artifacts/phase3/human_annotation.csv) are entity context 7, financial context 4, implicit sentiment 4, negation 2, and complex sentence, numerical reasoning and label ambiguity 1 each. The selection includes 20 confidence-at-least-0.9 errors and 10 paired-seed BERT/FinBERT disagreements. These counts describe the reviewed sample only.

The ProsusAI reference has F1 0.8797, but its earlier PhraseBank training can overlap this test. It is therefore kept out of the independent ranking. [Chapter 03](03_model_architecture.md) explains that distinction.

**Check:** Why not declare the largest number in this chapter the best deployable model? **Answer:** We must check whether evaluation was independent, what mistakes matter, how confidence behaves and whether performance transfers to the intended use. No one number supplies those answers.

## Deeper research choices, after the basics

If you compare more than one model pair or many groups, you have more chances to find an apparently interesting result by luck. Choose important comparisons in advance and label later discoveries as exploratory. Report the difference itself and its uncertainty, not just whether a test crosses a threshold.

A production use may prefer a table of error costs, a curve showing precision as the review threshold changes, or a curve showing remaining error as uncertain cases are sent to humans. Choose these policies on validation data. Future confirmation needs a new independent holdout because this test has already shaped our research questions.

**Check:** What would a paired classification bootstrap keep together? **Answer:** A sampled sentence's human label and both models' predictions. Splitting those apart would discard the fact that both models answered the same question.
'''
finish(name,'''# 05 — Count the mistakes, then name the metrics

**Before you begin:** you only need to know that every sentence has a human label and a model prediction. We will calculate the main metrics from one real table. Later lessons add repeated-run variation and uncertainty; you can pause there on a first reading.''',lessons,notebook,[('Explain precision without using a formula.','Of the sentences the model called negative, how many really were negative?'),('Explain recall without using a formula.','Of all the real negative sentences, how many did the model find?'),('Why is the baseline negative recall lower than its precision?','It found only 42 of 91 true negatives, even though 42 of its 51 negative predictions were right.'),('What does macro-F1 do?','Calculate an F1 for each of negative, neutral and positive, then average the three equally.'),('Why does seed SD not prove superiority?','It covers variation among a few trained runs on one fixed test, not every relevant source of uncertainty.')])
