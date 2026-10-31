"""Rewrite navigation and revision aids in the course's teaching voice."""
import re
from course_writer import OLD, write, tables

write('README.md', '''# Learn the FinBERT Explainability project

Start with a sentence: **“The company’s losses fell.”** Why might a model call this positive? How would we know whether it is usually right? And if it highlights “losses,” what does that actually tell us?

This course builds the answers from small examples to the completed implementation. You need basic Python and some familiarity with training a classifier. You do **not** need to arrive knowing calculus, attention matrices or attribution methods. The foundations explain those before the later chapters use them.

## How a lesson works

Each main lesson begins with a project question, explains the idea in everyday language, and works through an example. Only then does it name the technical concept and introduce equations. The last part connects the idea to saved results, compares alternatives, checks your understanding and adds advanced caveats.

On your first pass, stop at each **Check your understanding** and answer aloud. On a second pass, follow the source links and work the arithmetic yourself. The research notebook in Chapter 11 is a reference library of future ideas; read a few cards at a time rather than trying to absorb the whole chapter in one sitting.

## Keep these kinds of claims separate

| Label | How to read it |
|---|---|
| Implementation fact | The source, configuration or saved model shows that we did this. |
| Measured finding | A saved experiment produced this value under its stated protocol. |
| Teaching example / toy | Small invented numbers or sentences used to explain an idea. They are not experiment results. |
| Interpretation | A reasonable way to understand a result or design. It is not proof of its cause or a claim about undocumented historical motives. |
| Hypothesis | A possible explanation that still needs a controlled test. |
| Proposal | A future change; it has not been implemented by this course. |

“FinBERT” means our fine-tuned `yiyanghkust/finbert-pretrain` unless the text explicitly says **ProsusAI**, the separate reference. The current annotation file has user error categories and labeled AI-assisted notes. Older Phase 3 prose describes the initially blank handoff; see [what remains unknown](evidence_gaps.md).

## Your route through the course

| Chapter | Question you will learn to answer | Read first |
|---|---|---|
| [00 — Project overview](00_project_overview.md) | What did we investigate, and what did we find? | Start here |
| [01 — Foundations](01_foundations_and_theory.md) | How does a sentence become numbers and a prediction? | 00 |
| [02 — Dataset](02_dataset_and_preprocessing.md) | How did we create a fair test? | 00; token basics from 01 help |
| [03 — Architecture](03_model_architecture.md) | What happens inside each model? | 01 |
| [04 — Training](04_baselines_and_training.md) | How were weights and settings chosen? | 02–03 |
| [05 — Evaluation](05_evaluation_and_validation.md) | What do accuracy, precision, recall and F1 actually count? | 02; start here early if metrics are unfamiliar |
| [06 — Calibration and robustness](06_calibration_and_robustness.md) | Can I trust confidence, and what happens when text changes? | 01 and 05 |
| [07 — Integrated Gradients](07_integrated_gradients.md) | How does a change in an input affect a prediction? | 01 and 03 |
| [08 — Attention](08_attention_mechanisms.md) | Which token information is mixed together? | 01 and 03 |
| [09 — Explanation experiments](09_explainability_experiments.md) | How did we compare highlighting methods fairly? | 05 and 07–08 |
| [10 — Results and discussion](10_results_and_discussion.md) | What can we honestly conclude? | 05–09 |
| [11 — Future research](11_improvements_and_future_research.md) | What experiment should come next, and why? | 04–10 |

For an **engineering conversation**, revisit 02–04 and 07. For a **research discussion**, revisit 05 and 08–11. For **model validation**, focus on 02, 05–06 and 10. These are revision routes; the sequential order is the easiest first read.

## Practice and reference shelf

- [60 interview questions and a 30-minute mock interview](interview_preparation.md)
- [Plain-English glossary and mathematical notation](technical_glossary.md)
- [Primary papers and official documentation](references.md)
- [Six-phase coverage matrix](coverage_matrix.md)
- [Questions the evidence cannot yet answer](evidence_gaps.md)
- [Historical reproduction commands](reproduction_commands.md)
- [Teaching and validation review](teaching_review.md)

## Eight exercises that do not rerun experiments

1. Use Chapter 05’s real confusion matrix to calculate negative precision and recall. Explain why their denominators differ.
2. Use a pencil to reproduce Chapter 01’s one-weight update. Explain which direction reduces the error.
3. Trace `fpb50-00018` through the saved Phase 4 record and figure. Explain which class a positive attribution supports.
4. Compute Chapter 06’s four-example ECE. Change the bins and explain why the answer changes.
5. Calculate Chapter 09’s deletion AUC from the plotted points. Explain what a lower curve means.
6. Explain why 2199 moved assignments do not mean 2199 duplicate sentences.
7. Compare the 687-case and 727-case analyses. Name one conclusion that persists and one question that remains open.
8. Propose a reviewed pair such as “profits rose” / “losses rose.” State the intended label before discussing how a model should behave.

**Check:** Which of these exercises changes model weights? **Answer:** None. They use paper arithmetic and read existing evidence.

## Progress checklist

- [ ] I can give the 30-second, 2-minute and 10-minute explanations in Chapter 00.
- [ ] I can explain precision and recall without memorizing a slogan.
- [ ] I can trace tokens → embeddings → attention → logits → probabilities.
- [ ] I can explain gradients before writing the IG equation.
- [ ] I know which population and averaging rule each headline number uses.
- [ ] I can explain the 40 numerical failures without hiding or exaggerating them.
- [ ] I can distinguish human categories from AI-assisted interpretation.
- [ ] I can explain BERT’s higher score without claiming a proven cause.
- [ ] I can defend a future experiment that might reject my own hypothesis.
- [ ] I can complete the 30-minute mock interview and say “we did not measure that” when appropriate.

## Where the evidence lives

[Evidence ledger](evidence_ledger.json) records checked numbers and source hashes. [Validation results](validation_results.json) reports file sizes, local links, real CLI parsing, numerical assertions and preservation checks. [The validator](validate.py) only reads earlier project files and writes documentation evidence here; it does not load or train models.

The course adds no deployment permission or redistribution rights. Local artifact links need the original checkout; a public code-only copy may omit datasets and weights. Read the [model card](../publication/MODEL_CARD.md) and [license review](../publication/LICENSE_REVIEW.md) before reuse.
''')

# Preserve the scope of all sixty questions, but replace their explanations.
answers=[
('It predicts whether an English financial sentence is negative, neutral or positive.','Think of “profits rose” as a possible positive example. The label is a sentiment judgment, not a forecast of tomorrow’s stock price. The model does not verify that the news is true.'),
('The same word can mean different things depending on what changed.','Compare “profits rose” with “losses rose.” The verb is identical, but the financial direction differs. Names, comparisons and negation can matter too; this does not prove which mechanism caused a particular saved error.'),
('A token is one piece of text the model processes.','A piece may be a word, punctuation or part of a word. In a toy WordPiece split, a word could become a starting piece plus `##` continuation. The actual tokenizer determines the pieces, so word count and token count need not match.'),
('An embedding is a learned list of numbers for a token.','The token ID works like a row number in a lookup table. The row contains continuous values that the model can change during learning. Toy vectors [1,0] and [0,1] illustrate the lookup; real BERT vectors have 768 coordinates, without a guaranteed human meaning for each coordinate.'),
('A logit is a raw class score; softmax turns the scores into probabilities.','For toy logits [0,0,0], softmax gives one third to each class. Adding 10 to every score leaves those probabilities unchanged. Therefore probabilities alone do not recover the exact original logits needed by our IG target.'),
('We froze data, compared models, checked reliability, computed IG, compared attention, and audited the evidence.','Each step answers the next question: Is the test fair? Which workflow predicts well? Where does it fail? What affects one score? Do explanation rankings behave usefully? Can every conclusion be traced to saved records?'),
('The algorithm rebuilt the whole split while keeping duplicate groups together.','Imagine repacking every item into balanced boxes rather than moving only one conflicting pair. The objective did not reward retaining old assignments. Of 2199 moved rows, 2143 were singletons and 56 were in duplicate groups.'),
('We explicitly used the verified BERT configuration so the software loaded the intended architecture.','A checkpoint name is not a sufficient interface guarantee. The financial configuration needed explicit handling in this Transformers environment. The classifier uses canonical labels, and IG checks the actual BERT class and embedding module before attaching its hook.'),
('We read the checkpoint’s label names before interpreting probability columns.','ProsusAI IDs 0/1/2 mean positive/negative/neutral; ours mean negative/neutral/positive. A toy highest value in column 0 would be positive for ProsusAI but negative under our mapping. We reorder both labels and probability columns.'),
('Seed 42 was fixed explicitly instead of chosen because its test results looked good.','Picking the prettiest heatmap or highest-scoring seed afterward would favor an attractive story. A fixed checkpoint avoids that choice. It is still one model, so the explanation result does not automatically describe other seeds.'),
('Queries ask what information to look for, keys help match it, and values supply what gets mixed.','For “losses increased,” the representation at “increased” may use information from “losses.” Learned projections produce query, key and value vectors; query–key scores become weights, and those weights mix values. Chapter 08 works through the numbers.'),
('For each layer, attention has batch, head, query-token and key-token dimensions.','With one 64-token input, a layer returns [1,12,64,64]. We take the CLS query row and average its 12 heads in the final layer for the main comparison. All-layer CLS rows are saved, but full query-by-key matrices are not saved by default.'),
('Positions let the model distinguish the order of the same words.','“A acquired B” and “B acquired A” contain the same names but describe different roles. BERT adds learned position vectors. Our masking test leaves positions in place rather than shifting all later tokens as physical deletion would.'),
('Residual connections keep a direct route for existing information; LayerNorm rescales a token’s coordinates.','For a toy residual, input [1,2] plus an update [0.2,-0.1] gives [1.2,1.9]. LayerNorm then centers/scales coordinates using learned adjustments. These routes explain why attention weights alone are not the whole model calculation.'),
('BERT reads both sides of a complete sentence, which fits this classification task.','An encoder can use “not” and the following words together. A causal decoder reads only prior positions when generating the next token. Decoders can classify too, but that alternative was not measured here.'),
('We selected settings on validation data, then repeated the chosen setup with three seeds.','Each encoder searched learning rates 2e-5/3e-5 and budgets 3/4 epochs at seed 42. BERT selected 3e-5 and 3 epochs; FinBERT selected 2e-5 and 4. The baseline searched three C values times two n-gram choices.'),
('It is the number of examples contributing to one optimizer update.','One GPU processes 8 examples, accumulates another 8, then updates: 8 × 2 = 16. This saves activation memory. It need not be byte-identical to a physical batch of 16 because floating-point order and dropout can differ.'),
('AdamW adjusts update sizes using recent gradients, and mixed precision reduces training memory and arithmetic cost.','Weight decay gently pulls weights toward zero separately from Adam’s adaptive update. The recorded run used decay 0.01, a linear learning-rate schedule, no warmup and gradient clipping at 1.0. FP16 loss scaling can skip overflowing updates; that is different from running out of GPU memory.'),
('The checkpoint with the highest validation macro-F1 in each fit is evaluated.','A four-epoch budget does not mean epoch 4 is always exported. FinBERT seed 123 selected epoch 1; seeds 42 and 2026 selected epoch 4. All BERT final fits selected epoch 3. Test scores did not select the epoch.'),
('A seed is only one part of repeatability.','Keep data, row order, model revision, code, packages and numerical settings fixed too. A hash lets us check that a saved file has not changed. The byte-identical seed-42 repeats demonstrate repeatability in this environment, not every possible GPU or library version.'),
('It gives negative, neutral and positive sentiment equal influence in the final score.','Compute each class’s F1, then average the three. The baseline’s real negative recall is only 42/91 = 0.4615 despite accuracy 0.7442. Macro-F1 0.6713 makes that broader weakness harder to hide behind the neutral majority.'),
('It means no pair found by our stated detector crosses the saved splits.','The detector normalizes text and looks for character-pattern similarity at cosine 0.90. It can miss a paraphrase with different words or two stories about one event. “No detected leakage” is a bounded claim, not proof of all possible independence.'),
('Grouping defines the split before model fitting; classifier features must still learn only from training rows.','Imagine using all candidate sentences to keep near copies in the same exam partition. That is different from letting the classifier learn its vocabulary statistics from the exam. The project uses separate TF-IDF fits for those two jobs.'),
('Its earlier sentiment training used PhraseBank, so our test may not be new to it.','Making a new local split cannot undo upstream exposure. We keep its 0.8797 macro-F1 as a reference, but do not use it to declare an independent winner.'),
('Seed SD describes how fits varied; a confidence interval describes uncertainty for a stated comparison.','Toy seed scores 0.82/0.84/0.86 have mean 0.84 and sample SD 0.02. The explanation bootstrap instead holds a checkpoint fixed and repeatedly resamples sentences. These procedures vary different things.'),
('Choosing the right class and stating a realistic confidence are different skills.','A model can be right 80% of the time while claiming 99% confidence. BERT has better mean F1 here but higher ECE-10. Brier and NLL favor BERT on mean, so “best calibrated” needs a named metric and population.'),
('ECE depends on how we group confidence values and can hide differences inside a group.','If one bin mixes underconfident and overconfident cases, their average gaps can cancel before the absolute value is taken. Chapter 06’s four-row example changes from 0.4 to zero when groups are merged. We report both 10 and 15 bins.'),
('We add squared probability errors across all three classes, then average over sentences.','Toy truth positive and probabilities [0.1,0.2,0.7] give 0.1² + 0.2² + (0.7−1)² = 0.14. We do not divide that sum by three. This convention ranges from 0 to 2; NLL for the same row is −ln(0.7) ≈ 0.3567.'),
('No. A stable model might understand the edit, or it might ignore the number.','Our currency edits double selected amounts; many such changes do not determine a new sentiment label. We also check whether the edit changes represented tokens or TF-IDF features. A number beyond truncation cannot test the model’s response.'),
('I would fit a positive temperature using validation logits, then check it on new held-out data.','Dividing logits by a positive temperature changes confidence without changing their order. Temperature 2 softens a distribution more than temperature 1. The completed project did not fit temperature scaling, and the test must not choose it.'),
('IG adds up how the target score changes along a path from a reference input to the real input.','For the toy score F(x)=3x and reference 0, moving to x=2 changes the score by 6. The gradient is always 3, so IG is (2−0)×3=6. A neural network needs many sampled gradients because its slope changes along the path.'),
('Token IDs are lookup addresses, so we interpolate continuous embedding outputs instead.','Halfway between token ID 10 and 20 is not half of each word; it is another address. LayerIntegratedGradients works at `model.bert.embeddings` output, after position/type addition and LayerNorm. Its scores have shape batch × length × 768.'),
('Content IDs become PAD, while CLS, SEP, actual padding and the original attention mask stay fixed.','The inserted PAD references at content positions remain attended. For toy [CLS,profits,rose,SEP], the reference is [CLS,PAD,PAD,SEP] with the same active positions. This artificial reference is not a neutral financial sentence.'),
('The attribution total should match the input-minus-reference target logit.','A toy score difference of 2 with attribution total 1.9994 has absolute residual 0.0006. Our full-test run retained 40 cases above the fixed 0.001 absolute threshold after 200 steps. All were below the 0.01 relative threshold; both conditions were required.'),
('A signed score says which direction a token contributes for the chosen class; its magnitude ranks strength.','Toy embedding contributions +2 and −1 sum to +1; the absolute signed sum is 1, whereas the sum of absolute dimensions is 3. We save these separately. A positive score can support the negative class when that is the target.'),
('Attention tells us how representations are mixed, but its usefulness as an explanation needs testing.','A large weight can multiply a weak value vector, and residual routes carry other information. We compare one attention summary with IG and masking behavior. Low agreement cannot disprove attention, because IG is not ground truth either.'),
('It is a simple, declared summary near the classifier’s final representation.','Averaging makes one score per token from 12 heads. It also hides differences between heads and earlier routes. That is a defensible technical rationale, not evidence that this choice was empirically best.'),
('It is a modest average association between two token rankings, not 24% explanation accuracy.','Each of 687 converged sentences contributes one correlation between absolute IG signed sums and final-layer CLS head-mean attention. Toy ranks [1,2,3] and [1,3,2] give Spearman 0.5. We do not pool all tokens into one long list.'),
('CLS and SEP can contain information collected from other tokens in earlier layers.','Mean SEP attention mass is about 29.80%. That describes where weights go, not how much computation is wasted. We report special-token mass separately and exclude specials from content ranking, while retaining them in IG completeness.'),
('It measures how much target probability remains as more ranked tokens are masked.','Toy probabilities 0.9, 0.5 and 0.2 at fractions 0, 0.5 and 1 give trapezoid area 0.525. A lower curve means stronger suppression of the original target under this masking rule. It does not automatically mean a causal explanation of natural text.'),
('BERT had the higher measured mean, but we did not isolate the cause.','The gap is 0.8476 − 0.8339 = 0.0137. Corpus style, tokenization, tuning and training variation are competing hypotheses. I would test them with matched controls and a new holdout rather than pick one as a retrospective fact.'),
('Performance was much stronger on unanimously labeled sentences.','BERT allagree macro-F1 is 0.9583 versus 0.7500 for non-allagree. The groups contain 338 and 389 examples and different class mixes. This motivates ambiguity-aware validation; it does not show that agreement itself caused the difference.'),
('IG’s average deletion AUC was slightly lower, but the paired interval did not clearly separate it from attention.','The means are 0.6555 and 0.6608. The IG-minus-attention 95% interval is [−0.0130,0.0024], spanning zero. Both have lower mean AUC than random 0.7866 under this protocol.'),
('The main qualitative conclusion persists, but the population and means change.','All-727 Spearman is 0.2465 versus 0.2406 for converged-only. All-727 IG/attention AUC is 0.6353/0.6372, and its paired interval still spans zero. The added records remain flagged as numerically unreliable; inclusion does not repair them.'),
('It records user categories for 20 selected errors, not a census of model failures.','There are 7 entity-context, 4 financial-context, 4 implicit-sentiment, 2 negation and 1 each complex-sentence, numerical-reasoning and label-ambiguity categories. Interpretations/comments are AI-assisted notes. They are not independent human proof of a neural mechanism.'),
('The main risk is claiming more generality or independence than the evidence provides.','A frozen file protects bytes, but repeated test inspection can influence future decisions. One benchmark, three training seeds and one main explanation checkpoint also limit scope. The strongest threat depends on the specific claim being made.'),
('Tokens in one sentence share context, so treating them as independent would exaggerate the amount of evidence.','We compute a statistic per sentence and resample sentences together across methods. Even sentence resampling assumes enough independence between sentences; shared events may still create dependence. Random-mask repeats also do not create new sentences.'),
('MASK-heavy sentences may be unlike ordinary financial news.','Replacing “profits rose” with “MASK MASK” changes the input distribution. A score drop shows behavior under that artificial operation. It does not isolate a real-world causal effect, and different removal rules may rank explanation methods differently.'),
('No. It checks whether the numerical attribution total matches its defined score difference.','A toy sum can be exactly correct for an artificial reference that is unhelpful to a human. Completeness, baseline choice and explanation usefulness are separate questions. We preserve failures and test reference sensitivity instead of treating one check as certification.'),
('I describe the result, show its uncertainty and avoid inventing a reason.','“We did not clearly separate IG and attention” is useful evidence about this protocol. It is not a failure to tell a positive story. A next experiment should distinguish hypotheses rather than repeat comparisons until one is favorable.'),
('I would first diagnose the development data and decide what question the compute should answer.','Inspect tokenization, learning curves and calibration by seed. If a valid new evaluation plan exists, run a small equal-budget optimization study. One day of compute cannot substitute for independent labels or lawful new data.'),
('They might help minority errors, but can also amplify noisy examples.','Weighting multiplies a class’s loss. Focal loss downweights easy cases; with p=0.8 and gamma=2 its multiplier is 0.04. Test against ordinary loss with validation-only tuning and classwise/calibration safeguards.'),
('No. A band is not the full set of votes across three labels.','Knowing that disagreement is at least 25% does not tell us how much was negative versus neutral. Real soft-label training needs actual vote proportions or another justified target model, which the current records do not provide.'),
('It could make the model more familiar with the style of short financial news.','That is a hypothesis about genre, not proof that the existing checkpoint lacks financial knowledge. Start from the same base, compare equal-token adaptation controls, and check new news plus retained skills. Audit rights and contamination first.'),
('I would set a useful metric gain and acceptable side effects before looking at the new holdout.','A positive F1 difference alone may not justify worse negative recall or unacceptable latency. Use paired uncertainty and a practical margin chosen for the application. The current project did not define a universal deployment threshold.'),
('I would limit it to research or a monitored analyst-review pilot, subject to rights and further validation.','The model predicts sentiment in short English sentences. It does not forecast returns or prove robustness on current news. A pilot still needs reviewed outcomes, independent evaluation and an agreed way to handle uncertain cases.'),
('Record enough to reproduce a decision and monitor errors without collecting unnecessary sensitive text.','Proposed logs include model/tokenizer versions, input length, probabilities, review decisions and later outcomes. Compare error and coverage by relevant groups. This is a serving design proposal, not a system already built in the project.'),
('Measure the bottleneck first, then compare smaller models, batching or a review-aware cascade.','A smaller student can be cheaper but may lose rare-class skills. A cascade can send easy cases to a cheap model and difficult cases elsewhere. Measure p95 latency, memory, confidence and errors on the intended workload; training time is not serving latency.'),
('Data rights, model rights, large weights and exact environment availability still matter.','A public code repository can link to lawful sources without bundling all data and checkpoints. Financial-model license details need care, and package pins do not guarantee every historical wheel/driver remains available. An audit documents these limits rather than erasing them.'),
('Start with intended use, the evidence chain, uncertainty and remaining risks.','An engineer may ask how tensors flow; a validator asks whether the test is fair and the conclusion justified. I would show the same source-linked results, explain the non-independent reference and annotation exception, and state what needs a new holdout.')]
assert len(answers)==60
old=OLD['interview_preparation.md']
blocks=re.split(r'(?=^### Q\d+\.)',old,flags=re.M)[1:]
groups=re.findall(r'^## (?!30)(.+)$',old,re.M)[:12]
chapter_for_group=['01_foundations_and_theory.md','02_dataset_and_preprocessing.md','08_attention_mechanisms.md','04_baselines_and_training.md','05_evaluation_and_validation.md','06_calibration_and_robustness.md','07_integrated_gradients.md','09_explainability_experiments.md','10_results_and_discussion.md','10_results_and_discussion.md','11_improvements_and_future_research.md','11_improvements_and_future_research.md']
out=['''# Explain it aloud: interview practice

Use this after the relevant lesson, not as a list of definitions to memorize. Read the question, answer without looking, then compare your reasoning. The short answer is a starting point; the deeper explanation adds an example or project evidence. Each follow-up includes an answer so you can check the boundary of the claim.

Examples explicitly called toy are invented for teaching. Project numbers are measured. For full equations and code, follow the chapter link at each group. [Chapter 00](00_project_overview.md) supplies the 30-second, 2-minute and 10-minute project explanations.''']
for i,(block,(spoken,deeper)) in enumerate(zip(blocks,answers)):
    if i%5==0:
        out.append('## '+groups[i//5]+'\n\nRevisit the [worked lessons]('+chapter_for_group[i//5]+') before practicing these questions.')
    heading=block.splitlines()[0]
    follow=block.split('**Follow-up:**',1)[1].split('\n\n')[0].strip()
    out.append(f'{heading}\n\n**Spoken answer:** {spoken}\n\n**Deeper explanation:** {deeper}\n\n**Follow-up:** {follow}')
out.append('''## A 30-minute mock technical interview

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
''')
write('interview_preparation.md','\n\n'.join(out))

# Bibliographic records remain unchanged; add a teaching route instead of a dense opening.
ref=OLD['references.md']
write('references.md','''# Reading the research behind the project

Start with a question, not a bibliography. Read the chapter’s worked example first, then use these sources to go deeper. For attention, read Vaswani and then Devlin. For calibration, start with Guo. For IG, read Sundararajan, Taly and Yan after Chapter 07. Read both Jain/Wallace and Wiegreffe/Pinter before making a broad claim about attention as explanation. For future adaptation experiments, start with Gururangan and colleagues.

**Check:** Does citing an IG paper show that our heatmap is faithful? **Answer:** No. A paper motivates a method; our chosen layer, reference, numerical implementation and evaluation still need scrutiny.

## Verification and version limits

Primary pages and official documentation below were checked during earlier guide authoring on **30 October 2026**. This rewrite retains that record, not a claim of a fresh live check. “Verified” means identity and relevant location were checked, not independent replication. Initial preprint years are used unless a conference is stated.

Online docs may differ from the saved environment (Transformers 5.13.0, sklearn 1.9.0, SciPy 1.17.1, PyTorch 2.12.1+cu126). Local source and saved arguments govern what ran.

## Verified academic sources'''+ref.split('## Verified academic sources',1)[1])

write('coverage_matrix.md','''# Connect the six phases to what you can explain

Use this as a revision checklist. Start with the question, open the chapter, then follow its code and evidence links. A phase is covered when you can explain both its calculation and its limitation.

'''+tables('coverage_matrix.md')[0]+'''

## Check the connections

**Question:** Why does Phase 1 appear in both the dataset and evaluation chapters? **Answer:** The split is an engineering artifact, but its integrity determines what later scores mean.

**Question:** Why study Phase 6 after explanation experiments? **Answer:** The audit checks that reported claims match preserved evidence, including the cases that did not pass numerical checks. It cannot turn a limited experiment into a universal result.

For a full rehearsal, use the [60 interview questions](interview_preparation.md). For missing evidence, use the [open-question checklist](evidence_gaps.md). The [teaching review](teaching_review.md) describes coverage of the worked examples and the [validation output](validation_results.json) records automated checks.
''')

write('evidence_gaps.md','''# What we still do not know

You can understand the project without pretending these questions are answered. They need new evidence, not a more confident explanation. An unchecked box means open work; it does not imply a hidden failed experiment.

## Why did a model behave this way?

- [ ] **Why did BERT have the higher mean F1?** We have competing explanations involving data, vocabulary and training. We did not isolate one cause.
- [ ] **Do these results transfer to new dates, companies or languages?** A new independent evaluation is needed.
- [ ] **Did upstream pretraining include similar test text?** Local duplicate checks cannot inspect every upstream corpus.
- [ ] **What were the exact annotator votes?** Subset membership gives agreement bands, not all three vote totals.
- [ ] **How often do quantity or entity-relation skills fail?** Twenty selected errors and narrow edits do not answer that population question.
- [ ] **Would more seeds or a larger equal-budget search change the comparison?** Only three final seeds and four neural candidates were used.
- [ ] **Would longer inputs help?** Fifteen truncated examples identify a question, not the effect of changing length.

**Check:** Can better prose resolve these gaps? **Answer:** It can explain what evidence is missing. It cannot replace the experiment.

## How reliable are confidence and explanations?

- [ ] **Would validation-fitted temperature scaling help on new data?** It was not fitted here.
- [ ] **Are all edited sentences grammatical and correctly labeled?** They need human semantic review; we did not invent new labels.
- [ ] **Would other seeds, heads, layers or references give the same ranking?** The main result uses one fixed model and one attention summary.
- [ ] **Why exactly did 40 IG integrations miss absolute tolerance?** Their residuals and composition are documented, but a numerical cause was not isolated.
- [ ] **Could a new numerical study resolve those cases?** Possibly; it must be separately versioned and preserve historical thresholds and records.
- [ ] **Would whole-word masking or natural edits give the same method ordering?** Not tested.
- [ ] **Are IG and attention practically equivalent?** An interval including zero does not establish equivalence; a prespecified acceptable margin and suitable test are missing.
- [ ] **Do highlights help analysts make better decisions?** No human-utility study was performed.

**Check:** Are the 40 records missing? **Answer:** No. They remain saved and explicitly flagged. What is missing is an established explanation of their numerical difficulty, not their existence.

## What remains before reproduction or release?

- [ ] Independently review the AI-assisted interpretation/comments. User error categories are complete; these notes are not independently verified assessments.
- [ ] Clear exact model/data rights and redistribution terms. Current metadata does not settle every historical licensing question.
- [ ] Have the owner choose a code license and maintainer details.
- [ ] Check that required historical packages and drivers can be obtained on a fresh machine.
- [ ] Account for broad Phase 4/5 source fingerprints when reproducing old contexts after new documentation is added. Do not rewrite old manifests to conceal a mismatch.
- [ ] Measure real serving latency, cost, review coverage and monitoring needs before claiming operational readiness.

**Check:** Does the annotation hash exception mean the whole audit failed? **Answer:** No. The authorized later edit is disclosed against its original seal. Hiding it or silently rewriting history would be the problem.

The [research roadmap](11_improvements_and_future_research.md) turns these questions into proposed controlled experiments. The [discussion](10_results_and_discussion.md) explains what recommendation is defensible now. This course does not authorize new training, threshold changes or artifact repair.
''')
