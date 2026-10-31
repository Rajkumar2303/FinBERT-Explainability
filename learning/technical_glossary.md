# A glossary you can use while learning

Look up a term when it interrupts your reading. Read its plain-English meaning and example, then follow the chapter link for the full intuition → walkthrough → mathematics lesson. This is a lookup aid, not a prerequisite list to memorize. Toy arithmetic is illustrative; explicitly named project counts are measured.

**Check:** Do you need to understand every term before Chapter 00? **Answer:** No. The chapters introduce their prerequisites in order. Return here when you want a quick reminder.

## Data, task and provenance

### Sentiment classification

Choose a sentiment label for a sentence. “Losses fell” can be favorable even though it contains “losses.”

**Worked lesson:** [00](00_project_overview.md)

### Gold label

The dataset’s reference answer, not unquestionable truth. Here a reference negative label is stored as 0.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Majority agreement

Enough annotators chose the same label to pass a stated threshold. A threshold does not reveal every vote.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Allagree

Every annotator agreed in the source subset. The frozen test contains 338 such rows; agreement does not guarantee that every possible reader would agree.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Disagreement band

A range inferred from which agreement subsets include a row. “At least 25%” includes 25%; it does not mean strictly more than 25%.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Stratification

Try to keep class proportions similar in each split. If neutral is common overall, it remains common in the test.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Class imbalance

Some labels occur more often than others. The test has 432 neutral rows but only 91 negative rows.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Normalization

Apply stated text-cleaning rules so equivalent spellings can match. The duplicate detector standardizes Unicode forms, letter case and whitespace.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### NFKC

A Unicode rule that converts some compatibility forms to a common form. For example, full-width Latin characters can become ordinary Latin characters; it is not a semantic paraphrase detector.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Character n-gram

A short consecutive sequence of characters. In a toy word “profit,” “pro” is a three-character fragment.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### TF-IDF

Give a term weight based on its frequency in a document and how uncommon it is across fitted documents. A word in almost every sentence is less distinctive than a rare one.

**Worked lesson:** [03](03_model_architecture.md)

### Document frequency

Count documents containing a term. If “profit” appears five times in one sentence-document, that document contributes one to this count.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Cosine similarity

Compare vector direction after accounting for vector length. Identical nonzero vectors have similarity 1; Chapter 02 calculates a small example.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Connected component

A group joined by direct or indirect links. If A matches B and B matches C, all three stay together even if A does not directly match C.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Union-find

A way to track groups as links join them. Joining A–B and then B–C lets the code quickly report one shared group.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Leakage

Information from evaluation improperly influences learning or selection. Fitting the classifier’s word statistics on test sentences is one example.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Benchmark contamination

A model may have seen evaluation content in earlier training. A fresh local split does not undo ProsusAI’s prior PhraseBank exposure.

**Worked lesson:** [03](03_model_architecture.md)

### Frozen split

Saved row assignments that later work must preserve. A row assigned to test is not moved to improve a result.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### SHA256

A file fingerprint used to detect byte changes. Editing one character normally changes the fingerprint; a matching hash does not prove scientific correctness.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### Manifest

A saved inventory of files and their fingerprints. It lets an audit check whether a result’s inputs still match the recorded versions.

**Worked lesson:** [10](10_results_and_discussion.md)

### Row alignment

Make sure predictions and labels refer to the same sample IDs. Swapping two prediction rows can make correct outputs look wrong.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Provenance

The recorded history of where evidence came from. A model revision, training seed and checkpoint path help explain which model produced a prediction.

**Worked lesson:** [10](10_results_and_discussion.md)

### Independent holdout

Evaluation examples that did not guide the choices being tested. New, separately collected news is stronger confirmation than reshuffling already inspected rows.

**Worked lesson:** [11](11_improvements_and_future_research.md)

## Representations and architectures

### Tokenization

Break text into the pieces the model accepts. A company name may become several pieces rather than one word.

**Worked lesson:** [01](01_foundations_and_theory.md)

### WordPiece

A vocabulary-based method that can split words into smaller pieces. A continuation is displayed with `##`; the actual saved tokenizer decides the split.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Token ID

The integer address of a vocabulary entry. ID 20 does not mean twice as much sentiment as ID 10.

**Worked lesson:** [07](07_integrated_gradients.md)

### Vocabulary

The set of pieces a tokenizer knows and their IDs. The two encoders here have different vocabulary sizes.

**Worked lesson:** [03](03_model_architecture.md)

### Embedding

A learned vector looked up by token ID. A toy embedding [0.2,−0.5] is a list of features, not a literal sentiment label.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Contextual representation

A token’s vector after information from the sentence has changed it. “Losses” can be represented differently in “losses rose” and “losses fell.”

**Worked lesson:** [01](01_foundations_and_theory.md)

### CLS

A special classification token placed at the start. Its final representation feeds the BERT classification route; it is not an ordinary word.

**Worked lesson:** [03](03_model_architecture.md)

### SEP

A special separator token. It marks an input boundary and can collect contextual information inside the encoder.

**Worked lesson:** [08](08_attention_mechanisms.md)

### PAD

A token used to fill unused sequence positions. Actual padding is masked out; content positions replaced with PAD in our IG reference remain attended.

**Worked lesson:** [07](07_integrated_gradients.md)

### MASK

A special token used for masked-word learning and our deletion intervention. Many MASK tokens make an unusual sentence, which limits interpretation.

**Worked lesson:** [09](09_explainability_experiments.md)

### Attention mask

A rule identifying positions the model may use. Here original padding is inactive, while replacement tokens at active content positions stay active.

**Worked lesson:** [03](03_model_architecture.md)

### Token type

An ID distinguishing segments when BERT receives sentence pairs. For a single sentence, segment IDs normally all identify the first segment.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Positional embedding

A learned vector that marks position. It helps distinguish “A bought B” from “B bought A.”

**Worked lesson:** [01](01_foundations_and_theory.md)

### Self-attention

Each token mixes information from tokens in the same sequence. “Increased” can use “losses” to help represent what increased.

**Worked lesson:** [08](08_attention_mechanisms.md)

### Query/key/value

Three learned views of token vectors: what to match, what can be matched, and what information to mix. Chapter 08 introduces them after a sentence example.

**Worked lesson:** [08](08_attention_mechanisms.md)

### Attention head

One separate attention calculation inside a layer. This BERT uses 12 heads, each with 64-dimensional query/key/value vectors.

**Worked lesson:** [08](08_attention_mechanisms.md)

### Residual connection

Add a transformation’s output to its original input. Toy input 2 plus update 0.3 gives 2.3, preserving a direct route for the input.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Layer normalization

Center and scale a token’s hidden coordinates, then apply learned scale and shift. It works within the token’s features, not by mixing different batch examples.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Feed-forward network

A learned transformation applied separately at each token after attention. In this BERT it expands 768 features to 3072 and maps back.

**Worked lesson:** [03](03_model_architecture.md)

### GELU

A smooth nonlinear function used inside the feed-forward network. It changes values in a way that cannot be represented by one straight-line transformation alone.

**Worked lesson:** [03](03_model_architecture.md)

### Encoder

A model that builds representations of an input sequence. BERT can use words on both sides of a token when classifying a complete sentence.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Causal decoder

A text-generation model that restricts a position to earlier/current allowed information. It predicts next tokens without reading future tokens.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Pooler

BERT’s dense transformation and tanh applied to final CLS features. It prepares the representation used by this classifier; it is not an average of all words.

**Worked lesson:** [03](03_model_architecture.md)

### Classifier head

The final layers that turn a representation into class scores. The last linear layer maps 768 features to three sentiment logits.

**Worked lesson:** [03](03_model_architecture.md)

### MLM

Masked language modeling: predict hidden token content from context. It teaches language patterns before the sentiment labels are used for fine-tuning.

**Worked lesson:** [01](01_foundations_and_theory.md)

### NSP

Next-sentence prediction, an original BERT pretraining objective about sentence pairs. The course distinguishes the original method from unverified assumptions about every financial checkpoint’s history.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Domain adaptation

Adjust a model toward a particular kind of data. Financial-news text may differ from general web text or long filings.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### DAPT / TAPT

Continue language-model training on domain text or on text close to a task, respectively. Both are proposed future interventions here, not additional completed runs.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### Distillation

Train a smaller student to imitate a teacher’s predictions. A teacher distribution [0.1,0.2,0.7] carries more information than only “positive,” but may also carry its mistakes.

**Worked lesson:** [11](11_improvements_and_future_research.md)

## Training and computation

### Parameter

A learned number in the model. A weight connecting a hidden feature to a class score is updated during training.

**Worked lesson:** [03](03_model_architecture.md)

### Hyperparameter

A setting selected around training rather than learned by ordinary weight updates. Learning rate and epoch budget are examples.

**Worked lesson:** [04](04_baselines_and_training.md)

### Logit

A raw class score before softmax. Scores [0,0,0] become equal class probabilities.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Softmax

Turn logits into positive probabilities summing to 1 by exponentiating and normalizing. Adding the same constant to all logits leaves the probabilities unchanged.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Cross-entropy

A loss that penalizes low probability for the reference label. For a correct-class probability 0.8, hard-label loss is −ln(0.8), about 0.2231.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Gradient

How an output changes for a small change in an input or weight. For F(x)=3x, increasing x by 0.01 increases F by 0.03; the slope is 3.

**Worked lesson:** [07](07_integrated_gradients.md)

### Backpropagation

Work backward through the calculation to find gradients using the chain rule. Chapter 01 calculates a one-weight example before discussing networks.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Learning rate

The scale of a weight update. With gradient −2 and rate 0.1, simple gradient descent adds 0.2 to the weight.

**Worked lesson:** [04](04_baselines_and_training.md)

### AdamW

An optimizer that adjusts updates using recent gradient information and applies weight decay separately. It is the recorded optimizer, not a proven best choice for this task.

**Worked lesson:** [04](04_baselines_and_training.md)

### Weight decay

A training rule that gently shrinks weights. It discourages some overly large values but can also underfit if too strong.

**Worked lesson:** [04](04_baselines_and_training.md)

### Regularization

A method intended to reduce fitting noise rather than useful patterns. Weight decay and dropout are examples; their strength still needs validation.

**Worked lesson:** [04](04_baselines_and_training.md)

### Dropout

Randomly omit some activations during training. `model.eval()` disables dropout for our explanation runs so repeated inference does not introduce those random omissions.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Epoch

One pass through the training data. A four-epoch budget can still export the epoch-1 checkpoint if validation favored it.

**Worked lesson:** [04](04_baselines_and_training.md)

### Microbatch

The examples processed in one forward/backward pass. Here the per-device training microbatch is 8.

**Worked lesson:** [04](04_baselines_and_training.md)

### Gradient accumulation

Add gradients from several microbatches before updating weights. Two batches of 8 give an intended effective batch of 16 on one device.

**Worked lesson:** [04](04_baselines_and_training.md)

### Effective batch

The examples contributing to one optimizer update. It is 8 × 2 × 1 = 16 for this project’s one-GPU setup.

**Worked lesson:** [04](04_baselines_and_training.md)

### Scheduler

A rule for changing learning rate during training. The recorded linear schedule decreases it over the run.

**Worked lesson:** [04](04_baselines_and_training.md)

### Warmup

Start with smaller learning rates and increase toward the configured rate. The completed configuration used zero warmup; adding it is a proposal.

**Worked lesson:** [04](04_baselines_and_training.md)

### Gradient clipping

Limit the size of a gradient before updating. The saved maximum norm is 1.0, which constrains unusually large updates.

**Worked lesson:** [04](04_baselines_and_training.md)

### FP16 / FP32

Floating-point formats using 16 or 32 bits. FP16 helps training resources; IG was computed in FP32 for its numerical analysis.

**Worked lesson:** [04](04_baselines_and_training.md)

### Loss scaling

Temporarily multiply loss to keep small FP16 gradients representable, then undo the scale. Overflow can cause a skipped update without exhausting memory.

**Worked lesson:** [04](04_baselines_and_training.md)

### OOM

Out of memory: a requested allocation does not fit available memory. No GPU OOM was recorded in the completed training runs.

**Worked lesson:** [04](04_baselines_and_training.md)

### Allocated/reserved VRAM

Allocated memory is currently used by tensors; reserved memory includes blocks kept by the allocator for reuse. These are different peak measurements.

**Worked lesson:** [03](03_model_architecture.md)

### Checkpoint selection

Choose which saved weights to evaluate using the stated selection rule. Here best validation macro-F1 decides, not best test performance.

**Worked lesson:** [04](04_baselines_and_training.md)

### Random seed

A starting state for pseudorandom operations. Fixed seeds help repeat runs, but code, data, hardware and numerical settings also matter.

**Worked lesson:** [04](04_baselines_and_training.md)

### Overfitting

Learning details that help the fitted examples but do not generalize well. Repeatedly choosing ideas from test results can overfit research decisions too.

**Worked lesson:** [01](01_foundations_and_theory.md)

### Catastrophic forgetting

New training damages a capability previously learned. It is a hypothesis here, not a measured explanation for FinBERT’s lower mean score.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### Focal loss

Reduce easy-example loss contributions so harder cases have more influence. For probability 0.8 and exponent 2, the multiplier is 0.04; noisy labels may also receive extra emphasis.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### Label smoothing

Use slightly softened targets instead of a fully certain one-hot label. With three classes and smoothing 0.1, the true target is 0.9333 and others 0.0333 each.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### Ensemble

Combine model predictions. Averaging probabilities produces a new classifier whose F1 must be recomputed; it is not the average of member F1 scores.

**Worked lesson:** [11](11_improvements_and_future_research.md)

## Evaluation, uncertainty and robustness

### Confusion matrix

A count table of reference labels versus predicted labels. In Chapter 05, the baseline correctly predicts 42 of 91 negative sentences.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Support

The number of reference examples for a class. Negative test support is 91, not the number predicted negative.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Precision / recall

Precision asks how many predictions of a class were right; recall asks how many actual examples of that class were found. Negative baseline precision is 42/51; recall is 42/91.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### F1

A score balancing precision and recall through their harmonic mean. With precision 1 and recall 0.5, F1 is 2×1×0.5/(1+0.5)=0.6667.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Accuracy

The fraction of all predictions that match reference labels. The baseline gets 541/727 right, about 0.7442.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Macro / micro / weighted

Macro averages class scores equally; weighted uses class support; micro pools counts first. In single-label multiclass classification over all classes, micro-F1 equals accuracy.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Sample SD

A measure of spread among observed values using n−1 in the variance denominator. Toy scores 0.82/0.84/0.86 have sample SD 0.02.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Confidence interval

An interval from a stated uncertainty procedure for a stated quantity. It is not the range of training seed scores or a guarantee about a new domain.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### Bootstrap

Repeatedly sample observed units with replacement and recompute a statistic. A toy sample [A,B,C] can become [A,A,C]; duplicates are expected.

**Worked lesson:** [09](09_explainability_experiments.md)

### Paired comparison

Compare methods on the same units. Keep each sentence’s IG and attention scores together when resampling.

**Worked lesson:** [09](09_explainability_experiments.md)

### Estimand

The exact quantity we want to estimate. “Mean per-sentence IG-minus-attention AUC” is more precise than “which explanation is best.”

**Worked lesson:** [11](11_improvements_and_future_research.md)

### Equivalence margin

A difference small enough to count as practically unimportant, chosen before testing. An interval containing zero alone does not establish equivalence.

**Worked lesson:** [10](10_results_and_discussion.md)

### Calibration

Whether confidence matches observed correctness over many examples. Among many predictions at 90% confidence, about 90% should be correct for that confidence claim to fit.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Reliability diagram

A plot comparing average confidence and correctness within confidence groups. A 0.9-confidence group with 0.7 accuracy is overconfident by 0.2.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### ECE

Average the absolute confidence–accuracy gap of each bin, weighted by its example count. Chapter 06 shows why changing bin boundaries can change the result.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Brier score

Add squared errors between the full probability vector and the one-hot reference, then average rows. [0.1,0.2,0.7] with positive truth gives 0.14 under our unhalved convention.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### NLL

Negative log-likelihood: penalize low probability for the reference class using −ln(p). Correct-class probability 0.7 gives about 0.3567.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Temperature scaling

Divide logits by a fitted positive scalar to adjust confidence. It must be fitted on validation data; it was proposed but not performed here.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Perturbation

A deliberate input change used to test behavior. Replacing a company name is one rule; its expected meaning must be reviewed.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Invariance

A desired unchanged prediction under a change that should preserve the target meaning. A harmless rename may qualify; negation generally needs separate reasoning.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Visibility

Whether an edit changes the features the model actually receives. A changed word beyond the 64-token cutoff is not visible to that encoder input.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Truncation

Cut an input at its maximum allowed length. Fifteen Phase 5 inputs were flagged as truncated; this alone does not measure the benefit of longer context.

**Worked lesson:** [09](09_explainability_experiments.md)

### Flip rate

The fraction of edited cases whose predicted class changes. Without reviewed edited labels, a flip is not automatically an error or an improvement.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### Distribution shift

Inputs differ from the population on which a result was established. A sentence full of MASK tokens differs from ordinary news.

**Worked lesson:** [09](09_explainability_experiments.md)

### Abstention

A policy that sends a case to review instead of automatically accepting a label. It is a proposed serving feature, not a completed deployment capability.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### Coverage-risk curve

Show error among automatically handled cases as the accepted fraction changes. Low error at 10% coverage is different from low error at 90% coverage.

**Worked lesson:** [11](11_improvements_and_future_research.md)

## Explainability and symbols

### Feature attribution

Assign contributions to input features for a specified output comparison. Our token scores concern a chosen class logit relative to a reference.

**Worked lesson:** [07](07_integrated_gradients.md)

### Baseline/reference $x'$

The starting representation for an explanation. Here content becomes attended PAD; that is an artificial reference, not an empty or neutral sentence.

**Worked lesson:** [07](07_integrated_gradients.md)

### Integrated Gradients

Average gradients along the reference-to-input path and multiply by each feature’s change. For F(x)=3x from 0 to 2, the attribution is 6.

**Worked lesson:** [07](07_integrated_gradients.md)

### Completeness

In exact IG, all feature contributions sum to the input-minus-reference target score. Numerical approximation can leave a residual.

**Worked lesson:** [07](07_integrated_gradients.md)

### Convergence delta

Captum’s numerical completeness difference. A small value checks the approximation, not whether the explanation is useful or causal.

**Worked lesson:** [07](07_integrated_gradients.md)

### Absolute/relative residual

Absolute residual is the size of the accounting error; relative residual divides by the score difference with a small denominator floor. Our rule requires both to pass.

**Worked lesson:** [07](07_integrated_gradients.md)

### Gauss–Legendre quadrature

Approximate an integral using weighted evaluations at chosen points. Captum uses these nodes for the 50/100/200-step IG schedule.

**Worked lesson:** [07](07_integrated_gradients.md)

### Signed contribution

A positive or negative contribution to the chosen target score. Positive attribution can support negative sentiment if negative is the target class.

**Worked lesson:** [07](07_integrated_gradients.md)

### Absolute signed sum

Sum embedding-coordinate contributions for a token, then take the absolute value. Contributions +2 and −1 yield magnitude 1, not 3.

**Worked lesson:** [07](07_integrated_gradients.md)

### Word aggregation

Combine adjacent continuation pieces into a word display by adding signed scores. Preserve the original pieces because grouping can hide cancellation.

**Worked lesson:** [07](07_integrated_gradients.md)

### Jaccard

The fraction of the combined set shared by two sets. If top-token sets {a,b} and {b,c} overlap at b, Jaccard is 1/3.

**Worked lesson:** [07](07_integrated_gradients.md)

### Spearman $\rho$

Compare the orderings rather than raw score magnitudes. Rankings [1,2,3] and [1,3,2] give correlation 0.5; ties need averaged ranks.

**Worked lesson:** [09](09_explainability_experiments.md)

### Kendall $\tau_b$

Compare how often token pairs appear in the same versus opposite order, with a correction for ties. It complements Spearman rather than measuring accuracy.

**Worked lesson:** [09](09_explainability_experiments.md)

### Deletion AUC

The area under target probability versus actual masked fraction. A lower area means stronger average suppression under that masking rule, not necessarily more class flips.

**Worked lesson:** [09](09_explainability_experiments.md)

### Faithfulness

How well an explanation reflects the model behavior it claims to describe. Specify a test; a plausible highlight by itself is not proof.

**Worked lesson:** [08](08_attention_mechanisms.md)

### Plausibility

How sensible an explanation looks to a person. A highlight can look persuasive without identifying what drove the model.

**Worked lesson:** [10](10_results_and_discussion.md)

### Attention rollout

Combine attention across layers to approximate information routes, usually including residual assumptions. It is a richer approximation, not a complete causal trace.

**Worked lesson:** [08](08_attention_mechanisms.md)

### LIME / SHAP / DeepLIFT

LIME fits a simple local model to changed-input predictions; SHAP allocates contributions using feature coalitions; DeepLIFT propagates differences from a reference. Chapter 07 explains their assumptions and costs.

**Worked lesson:** [07](07_integrated_gradients.md)

### $x,t_i,s_i$

Read these symbols as follows: input representation, token ID at position i, and segment ID in Chapter 01; context defines $s_i$ as signed score in Chapter 07. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [01](01_foundations_and_theory.md)

### $N,K,B,L,d,d_k$

Read these symbols as follows: samples, classes, batch size, sequence length, hidden width and head width; here K=3,d=768,d_k=64. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [03](03_model_architecture.md)

### $E,P,T,H$

Read these symbols as follows: token, position, type embedding tables and hidden-state matrix. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [01](01_foundations_and_theory.md)

### $Q,K,V,A,M$

Read these symbols as follows: queries, keys, values, attention weights and additive mask; matrix K differs from scalar class count K. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [08](08_attention_mechanisms.md)

### $W,b,\theta$

Read these symbols as follows: weight matrix, bias and full parameter collection. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [03](03_model_architecture.md)

### $z,p,y,c$

Read these symbols as follows: logits, class probabilities, gold label and chosen target class. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [07](07_integrated_gradients.md)

### $\eta,\lambda,\beta_1,\beta_2$

Read these symbols as follows: learning rate, decay coefficient and Adam moment factors. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [04](04_baselines_and_training.md)

### $\epsilon$

Read these symbols as follows: small stabilizer or smoothing strength depending on formula; never assume these uses are identical. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### $\alpha,F,IG,r$

Read these symbols as follows: path coordinate, scalar target function, attribution and completeness residual. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [07](07_integrated_gradients.md)

### $f,k,x_j$ in deletion

Read these symbols as follows: nominal fraction, actual masked count and actual fraction at curve point j. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [09](09_explainability_experiments.md)

### $\bar m,s,\Delta,\delta$

Read these symbols as follows: mean, sample SD, effect/difference and prespecified practical margin. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### $TP,FP,FN,TN$

Read these symbols as follows: true positive, false positive, false negative, true negative relative to a chosen one-versus-rest class. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [05](05_evaluation_and_validation.md)

### $C_{s,c},T_{s,c},a_c$

Read these symbols as follows: current split class count, target count and proposed component's count in split optimization. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### $\arg\max,\|\cdot\|,\mathbb R,\top$

Read these symbols as follows: index of largest value, vector norm, real numbers and transpose. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [01](01_foundations_and_theory.md)

### $\mathcal L,g_t,m_t,v_t$

Read these symbols as follows: loss, current gradient, Adam first moment and second moment; hats denote bias correction. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [04](04_baselines_and_training.md)

### $df_t,f_s,n_c$

Read these symbols as follows: document frequency of term t, desired split fraction and gold support of class c. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [02](02_dataset_and_preprocessing.md)

### $1[\cdot]$

Read these symbols as follows: indicator equal to one when its condition holds, zero otherwise; correctness is an indicator. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### $\gamma,w_y,q_k$

Read these symbols as follows: focal-loss exponent, gold-class loss weight and soft-target probability. The linked lesson defines each symbol before using its equation.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### KL divergence

A directional measure of difference between probability distributions. It is zero when they are identical and is used in proposed teacher–student training.

**Worked lesson:** [11](11_improvements_and_future_research.md)

### T in calibration

The positive number dividing logits. It is not the token-type embedding table also sometimes named T; read the formula’s local definition.

**Worked lesson:** [06](06_calibration_and_robustness.md)

### p95 latency

The time below which 95% of requests finish under a stated workload. Training runtime does not measure it.

**Worked lesson:** [11](11_improvements_and_future_research.md)

## Check that you can use a definition

**Question:** Explain recall without using the word “recovered.” **Answer:** Among the sentences whose reference label is negative, ask how many the model correctly called negative. The baseline found 42 of 91, so recall is 42/91.

**Question:** Explain a gradient without starting with “partial derivative.” **Answer:** It tells us how a score changes when we make a small change to an input or weight. The derivative is the formal mathematical version of that idea.

**Question:** Does one letter always mean the same thing? **Answer:** No. K can mean a class count in one formula and a key matrix in another. Use the local definition and tensor shape rather than guessing.

See [interview practice](interview_preparation.md) to use these terms in explanations and [references](references.md) for primary sources.
