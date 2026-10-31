from course_writer import finish, lesson, tables, graph

name='01_foundations_and_theory.md'
ls=[lesson('Turn words into numbers without losing the words',
'''Python can store “profits increased” as text, but a neural network calculates with numbers. How can we give it numbers without pretending that an alphabetical word number is a measurement?''',
'''Use an ID like a library shelf number: it tells us where to look, not what the book means. At that location, store a list of learned numerical features. The model works with the list, not with arithmetic on the shelf number.''',
'''**Invented teaching vocabulary:** `profits → ID 7 → [0.2, 0.8]`; `losses → ID 9 → [0.7, -0.1]`. These two-coordinate vectors are made up. Real project vectors have 768 coordinates. The coordinates do not have guaranteed names such as “profitability.”''',
'''1. Split the text into vocabulary pieces.
2. Look up an integer ID for each piece.
3. Use the ID to retrieve a row from a numerical table.
4. Pass those rows to the rest of the network.
5. During training, adjust table entries when that helps the task.''',
'''The pieces are **tokens**. Splitting text into them is **tokenization**. The numerical table is an **embedding matrix**, and one retrieved row is a token **embedding**. BERT uses **WordPiece**, which can divide an unfamiliar word into smaller pieces. A `##` prefix marks a continuation of the preceding word.''',
'''If there are V vocabulary entries and d features per entry, the table has V × d numbers. In our toy table, retrieving ID 7 gives `[0.2,0.8]`. We do not average IDs 7 and 9 to obtain the meaning of ID 8. Later, IG will interpolate continuous embedding coordinates instead.''',
'''The general BERT vocabulary has 30522 entries; the financial BERT vocabulary has 30873. Both have width 768. [tokenize in neural.py](../src/finbert_explainability/phase2/neural.py) converts sentences using the saved model's tokenizer. WordPiece records are preserved in explanations.''',
'''Whole-word vocabularies are easy to read but handle unseen words poorly. Character models avoid unknown words but create longer sequences. Subwords share useful pieces across words, at the cost of explanations sometimes splitting a word across several positions.''',
'''If a financial word becomes three pieces, do we have three separate original words?''',
'''No. We have three model input units representing one word. Keep their positions for accurate computation; combine them carefully only when presenting a word-level explanation.''',
'''**Common confusion:** the same integer can mean different tokens in different vocabularies. Always load a checkpoint with its own saved tokenizer. Also, a geometric distance between learned vectors can be useful, but it is not a guaranteed human meaning scale.'''),
lesson('Let a word use its neighbors',
'''“Profit increased” and “loss increased” should not trigger the same decision merely because both contain “increased.” How does the model let the subject change the interpretation?''',
'''Each token begins with a basic vector, then repeatedly mixes in information from other positions. The final representation of “increased” can therefore depend on whether it accompanies profit or loss.''',
'''**Invented attention example:** while processing “losses have increased,” a token might mix 70% of a vector from “losses” and 30% from “increased.” With toy value vectors `[1,0]` and `[0,2]`, the mixture is `[0.7,0.6]`. These weights are an illustration, not a saved model observation.''',
'''1. Give each position its token vector and information about its position.
2. Calculate how strongly a position should read other positions.
3. Form a weighted mixture of their vectors.
4. Keep some of the old information and add the new mixture.
5. Apply another learned transformation. Repeat through the encoder layers.''',
'''This mixing is **self-attention**: positions read other positions in the same input. A **contextual representation** is the updated vector that now depends on the surrounding text. A **Transformer encoder** stacks these attention-and-transformation layers. [Chapter 08](08_attention_mechanisms.md) builds Q, K and V only after a fuller sentence example.''',
'''For mixture weights 0.7 and 0.3, the first coordinate is 0.7 × 1 + 0.3 × 0 = 0.7. The second is 0.7 × 0 + 0.3 × 2 = 0.6. This small calculation is the core of attention's weighted mixing. Many heads repeat the operation with different learned views of the vectors.''',
'''The saved BERT models have 12 layers and 12 attention heads per layer. The sequence length is capped at 64, including special tokens. [Saved config](../artifacts/phase2/finbert/seed-42/model/config.json) records the dimensions. No RNN-versus-Transformer experiment was performed here.''',
'''An **RNN** processes a sequence by carrying a state from one step to the next. An **LSTM** is an RNN designed to control what that state remembers. They model order, but their sequential work is less parallel. Transformers offer pretrained contextual features and parallel processing; they also use more memory as sequences grow.''',
'''Does an attention weight of 0.7 mean a token has a 70% chance of being positive sentiment?''',
'''No. It is a mixing weight for representations, not a class probability. The network still has several transformations before it makes a sentiment decision.''',
'''**Common confusion:** “bidirectional” means an encoder can use words on both sides. A causal decoder, commonly used for text generation, restricts each position to earlier positions. It can still classify, but has a different interface and cost. An encoder-decoder model combines a representation stage with a generation stage. These alternatives were not benchmarked here.'''),
lesson('Preserve order and keep the calculations well behaved',
'''If token vectors alone were enough, how would the model distinguish “A acquired B” from “B acquired A”? And how would a deep stack avoid losing every useful feature during repeated updates?''',
'''Add position information to say where a token is. Keep a shortcut carrying the previous vector while learning a change to it. Regularly rescale the coordinates so one unusually large scale does not dominate the next calculation.''',
'''**Invented example:** token vector `[2,0]` plus position vector `[0,1]` becomes `[2,1]`. Suppose a transformation suggests a change `[0.3,-0.2]`; adding it to the old vector gives `[2.3,0.8]`. This is simpler than requiring the transformation to rebuild everything from scratch.''',
'''1. Add token, position and segment vectors. A segment says which part of a paired input a token belongs to; single sentences usually use segment zero.
2. Normalize the resulting coordinates.
3. During training, randomly hide some coordinates to discourage dependence on one feature.
4. After attention, add the old vector back to the transformed vector.
5. Pass each position through a learned nonlinear transformation, then another shortcut and normalization.''',
'''A **positional embedding** represents location. The shortcut is a **residual connection**. **Layer normalization** rescales a token's feature coordinates using their mean and variance. **Dropout** is the training-time random hiding. A **feed-forward network** transforms each position's vector with learned weights and a nonlinear function.''',
r'''For toy vector `[1,3]`, the coordinate mean is 2 and variance is 1. Subtracting 2 and dividing by the square root of 1 gives `[-1,1]`. Real LayerNorm adds a small positive epsilon to the variance to avoid division by zero, and learns a scale and offset.

For coordinate $h_j$, it computes $\gamma_j(h_j-\mu)/\sqrt{\sigma^2+\epsilon}+\beta_j$. Here j selects a coordinate, mu is the coordinate mean, sigma-squared is variance, gamma is learned scale and beta is learned offset.''',
'''The embedding module adds token, position and token-type vectors, applies LayerNorm and then dropout. Dropout probabilities are 0.1 in the saved configs. The feed-forward block expands 768 features to 3072, applies GELU—a smooth nonlinear transformation—and projects back to 768. BERT can represent 512 positions, but this experiment used 64.''',
'''Other architectures use different position encodings or put normalization before rather than after a block. Those can change training behavior. A simpler shallow model needs fewer stabilizing components, but lacks the same stack of learned contextual transformations.''',
'''Does `model.eval()` prevent us from calculating derivatives?''',
'''No. It turns off training behaviors such as dropout. Derivative recording is controlled separately. IG needs derivatives while the model remains in evaluation mode.''',
'''**Common confusion:** layer normalization is not batch normalization. It works across one token's coordinates rather than using statistics across examples. In row-vector notation, a BERT block can be written $U=LN(H+Dropout(MHA(H)))$, then $H'=LN(U+Dropout(GELU(UW_1+b_1)W_2+b_2))$. H is the input matrix; U is the intermediate matrix; W and b are learned weights and biases; MHA is multi-head attention. This equation is a compact summary of the steps, not a new unexplained mechanism.'''),
lesson('Learn language first, then learn this task',
'''We have 3391 labeled training sentences. Is that enough to teach a large network English, financial terminology and sentiment from nothing? Starting with a model that has already learned language patterns is more practical.''',
'''Imagine two analysts preparing for the same classification exercise. One has read broad English text; another has read financial communications. Both still need examples of the exact three-label task. Financial reading may help, but it does not guarantee a better score.''',
'''**Invented training tasks:** general language practice might ask for the missing word in “The company reported a [MASK].” Task-specific practice gives “Operating profit increased” together with the label positive. The first supplies a word-learning signal; the second directly teaches sentiment.''',
'''1. Begin with weights learned in a large earlier language-learning stage.
2. Add or initialize the output layer needed for three sentiment classes.
3. Show labeled training sentences.
4. Adjust the output layer and encoder weights to reduce sentiment mistakes.
5. Use separate validation sentences to choose training settings.''',
'''The earlier stage is **pretraining**. Pretraining on financial language is **domain pretraining**. Adapting the model using labeled sentiment examples is **task fine-tuning**. Original BERT used **masked language modeling (MLM)** to recover selected words and **next sentence prediction (NSP)** to distinguish actual adjacent sentence pairs from sampled alternatives.''',
'''The data volumes serve different purposes: many unlabeled sentences can teach word relationships; labeled sentences teach our decision rule. This project does not add a new numerical pretraining comparison. Its actual downstream scores are 0.8476 for general BERT and 0.8339 for financial BERT, with the metric explained in Chapter 05.''',
'''General BERT starts from `google-bert/bert-base-uncased`; financial BERT starts from `yiyanghkust/finbert-pretrain`. The project did not redo upstream pretraining. Both received the same local supervised data protocol. FinBERT's earlier financial reading includes communications such as reports and calls, not necessarily the same style as short news sentences.''',
'''Training from scratch gives full control but usually needs far more data and compute. Freezing an encoder and fitting only a small classifier is cheaper but allows less adaptation. Continued financial-news pretraining is a future proposal, not work completed here.''',
'''Does a finance-pretrained model already know the exact sentiment label policy?''',
'''Not necessarily. Recovering missing financial words is different from applying our annotators' sentiment categories. That is why the financial pretraining model still needs supervised fine-tuning.''',
'''**Common confusion:** our fine-tuned financial model is not the ProsusAI checkpoint. ProsusAI already had sentiment training involving PhraseBank, creating a different evaluation-independence concern. [Chapter 03](03_model_architecture.md) explains the four workflows. Original BERT did not replace every selected MLM token with MASK; some were kept or replaced differently. That detail does not make arbitrary MASK-heavy deletion sentences natural.'''),
lesson('Turn scores into probabilities, then learn from mistakes',
'''The model gives the positive class a high score on a negative sentence. How do we quantify that mistake, and how can training change the weights in the right direction?''',
'''First turn the three scores into numbers that sum to one. Then charge the model more when it assigns very little probability to the correct answer. Finally, work backward through the calculation to see which small weight changes would reduce that charge.''',
'''**Invented calculation:** scores `[0,1,2]` for negative/neutral/positive become approximately `[0.0900,0.2447,0.6652]`. If the label is positive, the model is doing reasonably well. If the label is negative, it is confidently favoring the wrong direction.''',
'''1. Exponentiate the scores: approximately `[1,2.7183,7.3891]`.
2. Divide each by their total, 11.1074.
3. Take minus the natural logarithm of the probability assigned to the correct label.
4. Calculate how this loss changes when each score changes.
5. Pass those sensitivities backward through the layers and update the weights.''',
'''The raw scores are **logits**. Their exponential normalization is **softmax**. The penalty is **cross-entropy loss**. A **gradient** is a list of derivatives: how sensitive a quantity is to small changes in several inputs or weights. **Backpropagation** applies the chain rule backward through the network to calculate these derivatives efficiently.''',
r'''For score $z_k$ of class k, $p_k=\exp(z_k)/\sum_j\exp(z_j)$. The denominator sums over all three classes. For the true class y, loss is $L=-\log p_y$. In the toy example it is about 0.4076 if y is positive, but 2.4076 if y is negative.

A tiny backpropagation example makes the chain concrete. Let a toy input feature be x=2, a weight w=0.5, score z=wx=1 and desired score t=2. Use toy loss $L=(z-t)^2/2=0.5$. Then the loss changes by z−t = −1 per unit change in z, while z changes by x=2 per unit change in w. Multiply: derivative with respect to w is −2. With learning rate 0.1, update w to 0.5−0.1(−2)=0.7. The new score is 1.4 and loss is 0.18. This squared-loss illustration teaches the chain rule; the actual classifier uses cross-entropy.''',
'''[neural.py](../src/finbert_explainability/phase2/neural.py) uses the three-class BERT classifier and Hugging Face Trainer for optimization. [Chapter 04](04_baselines_and_training.md) explains its AdamW settings. IG later uses gradients of a prediction score with respect to embeddings, not training-loss gradients used to update model weights.''',
'''Accuracy counts only right versus wrong and does not provide the same smooth training signal. Cross-entropy distinguishes “wrong with 0.49 probability for the answer” from “wrong with 0.001.” Other losses can emphasize certain classes, but may also emphasize noisy labels; Chapter 11 treats them as proposals.''',
'''If we add 10 to every logit, do the probabilities change?''',
'''No. The common exponential factor cancels when we divide by the total. This is why log probabilities cannot uniquely recover the original raw logits.''',
'''**Common confusion:** learning the training examples very well is not the same as doing well on new examples. **Overfitting** means learning patterns that do not transfer. **Generalization** means useful behavior on new relevant examples. **Leakage** gives the process information it should not have had, such as test examples influencing the feature vocabulary. The ordinary cross-entropy logit derivative is p_k−y_k, where y_k is 1 for the true class and 0 otherwise. That connects the toy chain-rule lesson to the actual classification loss.''')]
finish(name,'''# 01 — Build a prediction one understandable step at a time

**Before you begin:** know the three labels from Chapter 00. You do not need calculus beyond the idea of asking, “If I change this number a little, what happens?” We build that idea here before using it in IG.''',ls,'''## Put the pieces together

Text → vocabulary pieces → numerical vectors → context mixing → three scores → probabilities. During training, a loss tells us how to adjust weights. During evaluation, weights stay fixed.

Our pretrained Transformer workflows scored 0.8476 and 0.8339 macro-F1, versus 0.6713 for the lexical baseline. This compares complete workflows; it does not isolate attention, vocabulary or pretraining as the cause. The technical rationale for using pretrained Transformers is that contextual features and mature tooling are useful with limited labeled data, not that this project proved every alternative worse.

For longer sequences, the attention score grid has one entry for every pair of positions. Moving from 64 to 128 positions changes 4096 pairs to 16384: four times as many. Other computation also matters, so whole-program runtime need not increase exactly fourfold.

**Check:** Why not conclude that the bigger vocabulary caused the score difference? **Answer:** Vocabulary, weights, training history and optimization differ together. Counting entries is not a controlled vocabulary experiment.''', [('What is the difference between an ID and an embedding?','An ID selects a table row; the embedding is the continuous vector stored there.'),('Why does context matter?','The financial meaning of a word changes with its subject and surrounding clauses.'),('What does pretraining learn?','Reusable language patterns, not necessarily the exact downstream label rule.'),('How does backpropagation work?','Multiply local sensitivities along the calculation, working backward to weights.'),('Why can a high-confidence answer be wrong?','Softmax produces relative probabilities, but they may not match real correctness frequencies; Chapter 06 explains calibration.')])

name='02_dataset_and_preprocessing.md'
ls=[lesson('What do the human labels tell us—and not tell us?',
'''Two readers may disagree about whether a contract announcement is good news or simply factual. If we train and test on such sentences, we should know that the disagreement exists.''',
'''PhraseBank includes versions with different levels of agreement. Choosing a larger version gives us more examples, including harder-to-agree-on sentences. Choosing only unanimous examples gives cleaner agreement but changes what we are testing.''',
'''**Invented vote example:** if 6 of 8 readers choose positive, agreement is 75%. If all 8 choose it, agreement is 100%. The real project's source does not provide each sentence's full vote counts; this toy example only explains a proportion.''',
'''1. Load the 50agree collection.
2. Read each sentence and its assigned sentiment.
3. Read the allagree collection separately.
4. Match sentence-plus-label pairs after a declared text cleanup.
5. Store whether each main-dataset row appears in the unanimous collection.''',
'''**Annotation** is the act of assigning a human label. **Agreement** describes how much annotators agreed. The four configurations are `sentences_50agree`, `sentences_66agree`, `sentences_75agree` and `sentences_allagree`. They overlap; they are not four independent datasets. The archive describes strict greater-than threshold membership for the first three.''',
'''For the teaching votes, agreement = 6/8 = 0.75; disagreement = 2/8 = 0.25. Because threshold membership gives only an interval, a row's disagreement band cannot tell us whether it had 6/8 votes or some other voting pattern. Exactly 75% also differs from strictly greater than 75%.''',
'''[data.py](../src/finbert_explainability/data.py), through `read_subset` and `annotate_membership`, parses the pinned archive and adds stable IDs and `is_allagree`. There are 4846 rows and 2264 allagree members. The archive is decoded as Latin-1; exported project files use their recorded encoding. We do not infer exact votes.''',
'''Using allagree only reduces ambiguity but loses examples and excludes much of the intended language. Keeping 50agree preserves coverage and allows agreement-group evaluation. Actual per-class vote proportions would support richer training targets, but they are not available here.''',
'''Does non-allagree mean the label is definitely wrong?''',
'''No. It means at least some annotators disagreed. The majority label may still be reasonable, and disagreement can reflect subtle or incomplete context.''',
'''**Common confusion:** membership is matched using both normalized text and label. Text alone could confuse cases with different labels. Normalization here applies Unicode NFKC, case folding and whitespace cleanup. NFKC standardizes certain Unicode forms; case folding ignores case differences. Original model text is retained rather than replaced with the matching key.'''),
lesson('Build a fair practice set and exam',
'''If almost every example is neutral, a random small test could contain too few negative cases to assess them. We want all splits to retain a similar sentiment mix.''',
'''Divide the examples while keeping roughly the same fraction of each class in training, validation and test. This makes class coverage more predictable. It does not make the classes equally common.''',
'''**Actual source counts:** 604 negative, 2879 neutral and 1363 positive. Neutral is the largest group. The final test has 91 negative, 432 neutral and 204 positive.''',
'''1. Reserve about 70% for training.
2. Divide the remaining 30% approximately equally into validation and test.
3. Preserve class proportions during the initial split.
4. Apply the duplicate-group rule from the next lesson.
5. Save the resulting assignments instead of recreating them casually later.''',
'''**Stratification** means preserving the representation of chosen categories during a split. **Class imbalance** means the classes occur at unequal frequencies. A **seed** fixes the starting state of pseudorandom choices so the same declared procedure can be repeated.''',
'''The intended row targets are 4846×0.70 = 3392.2 and 4846×0.15 = 726.9 for each smaller split. Rows cannot be fractional, and duplicate groups are indivisible. Final counts are 3391/728/727. Their sum remains 4846.''',
'''[`stratified_split`](../src/finbert_explainability/splitting.py) makes two sklearn splits. The final grouped splitter optimizes class counts. [phase1.json](../configs/phase1.json) records fractions and the seed; [statistics.json](../artifacts/phase1/statistics.json) records actual balances. Allagree membership was not a stratification variable.''',
'''A class-by-agreement split could explicitly balance both properties, but would define a different benchmark. Company-separated or chronological splits ask stronger generalization questions, but require trustworthy company/event/date metadata. Never silently replace the saved test with a newly preferred design.''',
'''Does stratification give us the same number of negative and neutral examples?''',
'''No. It preserves their approximate proportions. Neutral remains common; macro-F1 later prevents that majority from entirely driving the main score.''',
'''**Common confusion:** a 70/15/15 target is a design goal, not a reason to drop examples or split a duplicate group. Store row IDs and assignments so later phases evaluate the exact same sentences.'''),
lesson('Keep look-alike headlines on the same side of the split',
'''“Acme sales rose 5%” in training and “Acme sales rose 5 %” in test would give the model an easy exam question. How can we detect that without pretending to understand every sentence's meaning?''',
'''First look for exactly matching cleaned text. Then compare small character fragments to catch close wording. Join detected pairs into groups, and keep every member of a group in one split.''',
'''**Invented grouping example:** A resembles B, and B resembles C. Even if A and C are not similar enough directly, putting A in training and C in test risks carrying the same story across the boundary through B.''',
'''1. Clean matching text with the fixed normalization.
2. Find exact matches.
3. Represent each sentence by character fragments of lengths 3, 4 and 5 within word boundaries.
4. Weight fragments by how distinctive they are and compare the vectors.
5. Add an edge between pairs whose similarity is at least 0.90.
6. Find all connected groups and assign whole groups to a split.''',
'''A **character n-gram** is n adjacent characters. **TF-IDF** weights a fragment using its frequency in a sentence and how uncommon it is across the collection. **Cosine similarity** compares the direction of two vectors rather than their raw length. A **connected component** contains sentences linked by a chain of detected pairs.''',
r'''For vectors a and b, cosine is $(a\cdot b)/(\|a\|\|b\|)$. The dot means multiply corresponding coordinates and add; the norm bars mean vector length. Toy vectors `[1,0]` and `[0.95,0.31225]` have approximately unit length and cosine 0.95, so they pass 0.90.

One standard smoothed inverse-frequency weight is $\log((1+N)/(1+df))+1$: N is the number of fitted sentences, and df is how many contain the fragment. Rare fragments receive larger weights. This duplicate detector is separate from the train-only word TF-IDF classifier.''',
'''[`detect_duplicates`](../src/finbert_explainability/leakage.py) uses `char_wb`, n-grams (3,5), brute-force cosine neighbors and query batches of 256. `connected_groups` merges pair memberships. The saved report has 8 exact pairs, 49 near pairs and 4789 components. Cross-split detected pairs fell from 25 initially to zero finally.''',
'''Word matching is simpler but more brittle to small spelling changes. Semantic embeddings could catch paraphrases with little shared spelling, but can also merge sentences with different financial implications. Grouping related text conservatively protects the evaluation without claiming their labels are interchangeable.''',
'''Does zero detected crossings prove no possible leakage?''',
'''No. It means none under this exact detector. Paraphrases, shared events or exposure during a model's earlier pretraining can remain.''',
'''**Common confusion:** the detector fits its character statistics to the corpus to construct groups before the split is frozen. The downstream classifier's TF-IDF is fitted only on training text. These are different uses. Grouping before splitting is a disclosed construction rule; it is not a license to fit predictive features on the test.'''),
lesson('Why 2199 assignments moved—and why we save hashes',
'''A report says 2199 rows changed split. That sounds like thousands of duplicates. The actual pair report is much smaller. Which number describes what?''',
'''The algorithm did not move only the troublesome pairs. Once grouping was needed, it rebuilt the entire seating plan. Even a person sitting alone can get a different seat when every table is reassigned.''',
'''**Measured finding:** 2199 assignments moved: 2143 singletons and 56 rows in duplicate groups. No row was lost. The objective balances class counts, not the number of rows moved from the initial plan.''',
'''1. Gather rows into components.
2. Shuffle group order with a fixed seed, then consider larger groups first.
3. Try adding a group to each split and measure the change in class-count error.
4. Choose the smallest cost, using deterministic tie behavior.
5. Repeat for every group, including singletons.
6. Save assignments and fingerprints of the files.''',
'''This is **greedy group-stratified splitting**: make one locally best placement at a time while keeping groups intact. A **hash** is a file fingerprint. A **manifest** records such fingerprints for a set of outputs.''',
r'''For split s and class c, call its current count C, its target T, and the incoming group's class count a. The contribution to the placement cost is $((C+a-T)^2-(C-T)^2)/\max(T,1)$, summed across classes. If C=8, T=10 and a=1, the cost is `(1−4)/10 = −0.3`: adding the row improves this part of the balance. There is no term penalizing a changed old assignment.''',
'''[`grouped_stratified_split`](../src/finbert_explainability/splitting.py) implements that cost. [Phase 2 audit](../artifacts/phase2/audit/audit.json) confirms deterministic replay and matching saved assignments. The frozen test SHA256 is `97ad15089bd04e0d61726a6218bfcc2bdb18a4bcc1f5e8a299e1da41607e36a9`. The source archive and dataset revision are pinned in the config.''',
'''A minimal-movement repair could retain more initial assignments, but that was not this algorithm's objective. Saving only a seed is smaller than saving all assignments, but code, library versions or input order can change a replay. Saved IDs plus hashes make the historical evidence much easier to check.''',
'''Does a matching hash prove the sentence's label is correct?''',
'''No. It proves the bytes match the referenced file. A wrong label can be preserved perfectly. Integrity and scientific validity are different questions.''',
'''**Common confusion:** deterministic means the declared procedure repeats under its conditions, not that every future software environment must behave identically. Allagree train/validation/test counts are 1592/334/338, with class counts 210/970/412, 43/212/79 and 50/209/79. Report these rather than assuming agreement was explicitly balanced.''')]
finish(name,'''# 02 — Make the exam fair before judging the model

**Before you begin:** Chapter 00 explains training, validation and test. Here we build the dataset decisions from the questions a skeptical reviewer would ask: who labeled the sentences, how similar are they, and did the same examples stay in the same splits?''',ls,'''## Follow the saved data through the code

'''+graph(name)+'\n\n'+tables(name)[0]+'''

`load_source` verifies the original ZIP hash. `read_subset` turns its text into a Hugging Face Dataset with the declared class labels. `annotate_membership` adds IDs based on original source indices and unanimous membership. The pipeline writes JSONL splits, a saved dataset, statistics, initial assignments and a leakage report.

See [metadata](../artifacts/phase1/metadata.json), [initial assignments](../artifacts/phase1/initial_assignments.json) and [leakage report](../artifacts/phase1/leakage_report.json). The detector can miss semantic paraphrases and related events; it can also group wording that differs in an important number or negation. Grouping is a precaution, not permission to relabel or delete those sentences.

**Check:** Which would test future-news performance better: another random split of the same collection or a properly collected later-time holdout? **Answer:** The later-time holdout addresses the future-use question more directly, though time shifts can make it harder. It needs new provenance and label review; this saved test stays unchanged.''', [('Why use 50agree?','It retains more examples and ambiguity; using only unanimous examples changes the target population.'),('Why keep connected components together?','Pairwise similarity can form a chain of related sentences across splits.'),('What moved 2199 rows?','A full grouped reallocation, not the discovery of 2199 duplicates.'),('What does cosine 0.90 mean?','A threshold on a particular lexical vector similarity, not 90% identical meaning.'),('Why save row IDs and hashes?','IDs align records across phases, and hashes detect changed bytes.')])

name='03_model_architecture.md'
ls=[lesson('Understand the inexpensive baseline first',
'''Before paying for a Transformer, ask whether word patterns already solve much of the task. That gives us a useful reference for the added complexity.''',
'''Turn a sentence into a list of word-feature weights, then add learned evidence for each class. The model can learn that “profit” often supports one decision, without building a contextual vector for every word.''',
'''**Invented toy calculation:** use features `[profit, loss]`. For vector `[1,0]`, a positive-class rule with weights `[2,-1]` and bias 0.2 gives score 2.2. For `[0,1]`, it gives -0.8. Real TF-IDF values are weighted and normalized, not just these binary indicators.''',
'''1. Fit a vocabulary and word-frequency statistics on training sentences only.
2. Convert each sentence into a sparse vector: mostly zeros for absent features.
3. Multiply features by class-specific weights and add a bias.
4. Convert the three scores into class probabilities.
5. Evaluate validation candidates, then keep the selected settings.''',
'''**TF-IDF** means term frequency–inverse document frequency. It emphasizes words that occur in the sentence but are less ubiquitous in the fitted collection. **Logistic regression** is a learned linear scoring model followed by a probability conversion. “Regression” in the name does not mean it predicts a continuous financial quantity here.''',
'''For feature values x_j, class weights w_j and bias b, score z is the sum of w_j×x_j plus b. In the toy example, 2×1 + (−1)×0 + 0.2 = 2.2. A **bigram** is a pair of adjacent words, such as “not improved”; it can add a little word-order information that unigrams alone lack.''',
'''[baseline.py](../src/finbert_explainability/phase2/baseline.py) builds a sklearn Pipeline with sublinear TF-IDF, at most 20000 features and an lbfgs logistic-regression solver. Sublinear means repeated term counts grow logarithmically rather than linearly. Selected settings were C=10 and unigrams; Chapter 04 explains C and the search.''',
'''The baseline trains and predicts cheaply, and its feature weights can be inspected. It struggles to represent distant negation, entity roles and new combinations of familiar words. Adding bigrams helps some local phrases but increases feature count and cannot represent every context.''',
'''Does a positive coefficient for “increased” prove it is always good financial news?''',
'''No. It is a learned association within this feature representation. The model may miss whether costs or profits increased, especially when important context is not represented jointly.''',
'''**Common confusion:** the duplicate detector also uses TF-IDF, but over character fragments for grouping. This classifier uses word features fitted only on training data. Same tool family, different role and fitting boundary.'''),
lesson('Trace one sentence through BERT',
'''What does the code actually do between receiving a sentence and returning three probabilities? Understanding the shapes makes the model less mysterious and helps catch implementation errors.''',
'''BERT first represents all text pieces, then lets them exchange information through 12 layers. A dedicated starting position gathers a sentence representation, and a small final layer turns that representation into three scores.''',
'''**Illustrative sequence, not a verified tokenizer output:** `[CLS] profits increased [SEP] [PAD] ...`. CLS marks the classification position; SEP marks the end of the sentence segment; PAD fills unused positions. An attention mask says which positions should be available to other tokens.''',
'''1. Tokenize and truncate/pad to 64 positions.
2. Look up token vectors and add position/segment information.
3. Run 12 layers of attention, residual additions, normalization and feed-forward transformations from Chapter 01.
4. Take the final CLS representation and apply the BERT pooler, a learned dense transformation followed by tanh, which squashes values between -1 and 1.
5. Apply training-time dropout and a three-output linear classifier.
6. Apply softmax when probabilities are needed.''',
'''A **tensor shape** lists array dimensions. The **classifier head** is the final task-specific layer. The **pooler** transforms the final CLS vector; it is not an average of every token. The encoder **hidden size** is the number of features at each position.''',
'''For B sentences, input IDs have shape `[B,64]`; embedding and layer outputs have `[B,64,768]`; pooled CLS vectors have `[B,768]`; logits have `[B,3]`. The three-class head has 768×3 + 3 = 2307 parameters: one weight per input/output pair plus three biases. Twelve heads divide width 768 into 64 features per head.''',
'''[neural.py](../src/finbert_explainability/phase2/neural.py) uses `BertConfig` and `BertForSequenceClassification`. The [general config](../artifacts/phase2/bert/seed-42/model/config.json) and [financial config](../artifacts/phase2/finbert/seed-42/model/config.json) verify 12 layers, 12 heads, width 768 and feed-forward width 3072. Explicit configuration avoids guessing from incomplete upstream metadata.''',
'''Mean pooling over token states is a possible alternative sentence representation, but not this classifier's saved behavior. DistilBERT uses a smaller architecture; other encoders use different internal modules. Explanation hooks must be checked for each architecture instead of blindly searching for `model.bert.embeddings`.''',
'''If B=8, how many sentiment logits are produced?''',
'''Eight rows of three scores, or 24 values. The 64 input positions do not each receive a sentiment label in this sentence-classification task.''',
'''**Common confusion:** masking padding keys does not force every internal padding vector to zero. Also, the 512-position architectural capacity does not mean the project evaluated 512-token sentences. The actual cap remained 64.'''),
lesson('Keep the three BERT identities separate',
'''The word “FinBERT” can refer to different checkpoints. If we confuse them, we may accidentally claim that an already-trained model passed an exam containing its old practice questions.''',
'''Compare the local general and financial starting models fairly: train both on the same local training rows. Treat the externally sentiment-trained ProsusAI model as a separate reference because its earlier exposure is different.''',
'''**Invented label-order example:** a reference model outputs `[0.70,0.20,0.10]` in positive/negative/neutral order. Our order is negative/neutral/positive, so we must rearrange it to `[0.20,0.10,0.70]`. Otherwise we could call a positive prediction negative.''',
'''1. Identify the exact upstream checkpoint and pinned revision.
2. Load its own tokenizer and config.
3. Inspect the ID-to-label mapping.
4. Reorder probability columns and prediction labels consistently.
5. Record whether the test can be independent of the checkpoint's original training.''',
'''A **model revision** identifies a particular saved version. A **label mapping** connects output indices to sentiment names. **Benchmark contamination** means evaluation examples may have been seen earlier by the model or development process.''',
'''The saved general classifier contains 109484547 parameters; financial BERT contains 109754115. The difference is 269568 = (30873−30522)×768, explained by the vocabulary embedding rows. Around 110 million FP32 weights require roughly four bytes each before format overhead; training also stores gradients, optimizer state and temporary computations.''',
'''General BERT is `google-bert/bert-base-uncased`. Our financial model is fine-tuned from `yiyanghkust/finbert-pretrain`, an earlier financial language model. `ProsusAI/finbert` already had sentiment training involving PhraseBank. [comparison.json](../artifacts/phase2/comparison/comparison.json) stores its original mapping: 0 positive, 1 negative, 2 neutral. Local trained heads use 0 negative, 1 neutral, 2 positive.''',
'''The local pair makes a more interpretable comparison than changing model scale radically. It still does not isolate financial pretraining alone: vocabulary and learned starting weights also differ. The external reference is informative if clearly labeled, but cannot be called the independent benchmark winner.''',
'''Does making a new random split of PhraseBank remove ProsusAI's possible earlier exposure?''',
'''No. A new local split cannot make an upstream model forget examples. A separate independently collected evaluation set would be needed to address that concern.''',
'''**Common confusion:** parameter count is not VRAM usage. Training requires additional state; CUDA also reserves memory beyond current tensor allocations. Peak allocated memory was 2.079 GiB for each neural training workflow, with reserved memory somewhat higher. These counters exclude other applications and some driver/display memory.''')]
finish(name,'''# 03 — Know which model is answering the question

**Before you begin:** Chapter 01 explains tokens, vectors, attention and softmax. Here we connect those pieces to the actual saved model interfaces. Start with the simple baseline, then follow BERT's array shapes.''',ls,'''## Compare the concrete workflows

'''+tables(name)[0]+'''

The parameter counts above come from saved safetensors headers, without running models. FinBERT's largest weight file is about 418.7 MiB. The TF-IDF feature cap is not a claim that every run necessarily used 20000 features.

**Check:** Is a model with the same parameter count guaranteed to have the same accuracy? **Answer:** No. The learned values, vocabulary, data and training procedure matter, not just the number of stored values.

## Explore alternatives after understanding this model

RoBERTa changes the BERT pretraining recipe; DeBERTa separates content and position information in its attention design; ModernBERT updates encoder design and efficiency, including long-context use; DistilBERT transfers knowledge into a smaller encoder. These are possible next comparisons, not models evaluated in this project.

'''+tables(name)[1]+'''

For a new architecture, first check the tokenizer, model output shape, label mapping and explanation layer. Then compare at a declared compute or latency budget. Published gains elsewhere do not predict gains here. [Chapter 11](11_improvements_and_future_research.md) explains how to test these alternatives fairly.

**Check:** Why might a long-context model not help this task automatically? **Answer:** The study uses short, length-64 inputs. Extra context capacity only helps if relevant omitted information exists and the new procedure learns to use it.''', [('What does logistic regression calculate?','A weighted sum of input features plus a bias for each class, then class probabilities.'),('What is CLS used for?','Its final contextual vector supplies the sentence representation used by this classifier.'),('Why is ProsusAI separate?','Its earlier sentiment training can include the local test examples.'),('Why inspect id2label?','Output index zero does not mean the same sentiment in every checkpoint.'),('Why is architecture verification important for IG?','Different models expose different continuous layers; choosing the wrong hook changes the question or breaks the computation.')])
