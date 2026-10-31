# 01 — Build a prediction one understandable step at a time

**Before you begin:** know the three labels from Chapter 00. You do not need calculus beyond the idea of asking, “If I change this number a little, what happens?” We build that idea here before using it in IG.

## Learning objectives

- Turn words into token IDs and continuous vectors.
- Explain contextual mixing, positions and the BERT block.
- Calculate probabilities, loss and a small gradient update before using advanced notation.

## Lesson: Turn words into numbers without losing the words

### Why it matters
Python can store “profits increased” as text, but a neural network calculates with numbers. How can we give it numbers without pretending that an alphabetical word number is a measurement?

### Build the intuition
Use an ID like a library shelf number: it tells us where to look, not what the book means. At that location, store a list of learned numerical features. The model works with the list, not with arithmetic on the shelf number.

### A concrete example
**Invented teaching vocabulary:** `profits → ID 7 → [0.2, 0.8]`; `losses → ID 9 → [0.7, -0.1]`. These two-coordinate vectors are made up. Real project vectors have 768 coordinates. The coordinates do not have guaranteed names such as “profitability.”

### Walk through it
1. Split the text into vocabulary pieces.
2. Look up an integer ID for each piece.
3. Use the ID to retrieve a row from a numerical table.
4. Pass those rows to the rest of the network.
5. During training, adjust table entries when that helps the task.

### Give the idea its technical name
The pieces are **tokens**. Splitting text into them is **tokenization**. The numerical table is an **embedding matrix**, and one retrieved row is a token **embedding**. BERT uses **WordPiece**, which can divide an unfamiliar word into smaller pieces. A `##` prefix marks a continuation of the preceding word.

### Add the mathematics
If there are V vocabulary entries and d features per entry, the table has V × d numbers. In our toy table, retrieving ID 7 gives `[0.2,0.8]`. We do not average IDs 7 and 9 to obtain the meaning of ID 8. Later, IG will interpolate continuous embedding coordinates instead.

### Connect it to our project
The general BERT vocabulary has 30522 entries; the financial BERT vocabulary has 30873. Both have width 768. [tokenize in neural.py](../src/finbert_explainability/phase2/neural.py) converts sentences using the saved model's tokenizer. WordPiece records are preserved in explanations.

### Alternatives and trade-offs
Whole-word vocabularies are easy to read but handle unseen words poorly. Character models avoid unknown words but create longer sequences. Subwords share useful pieces across words, at the cost of explanations sometimes splitting a word across several positions.

### Check your understanding
**Question:** If a financial word becomes three pieces, do we have three separate original words?

**Answer explained:** No. We have three model input units representing one word. Keep their positions for accurate computation; combine them carefully only when presenting a word-level explanation.

### Deeper dive and common confusion
**Common confusion:** the same integer can mean different tokens in different vocabularies. Always load a checkpoint with its own saved tokenizer. Also, a geometric distance between learned vectors can be useful, but it is not a guaranteed human meaning scale.


## Lesson: Let a word use its neighbors

### Why it matters
“Profit increased” and “loss increased” should not trigger the same decision merely because both contain “increased.” How does the model let the subject change the interpretation?

### Build the intuition
Each token begins with a basic vector, then repeatedly mixes in information from other positions. The final representation of “increased” can therefore depend on whether it accompanies profit or loss.

### A concrete example
**Invented attention example:** while processing “losses have increased,” a token might mix 70% of a vector from “losses” and 30% from “increased.” With toy value vectors `[1,0]` and `[0,2]`, the mixture is `[0.7,0.6]`. These weights are an illustration, not a saved model observation.

### Walk through it
1. Give each position its token vector and information about its position.
2. Calculate how strongly a position should read other positions.
3. Form a weighted mixture of their vectors.
4. Keep some of the old information and add the new mixture.
5. Apply another learned transformation. Repeat through the encoder layers.

### Give the idea its technical name
This mixing is **self-attention**: positions read other positions in the same input. A **contextual representation** is the updated vector that now depends on the surrounding text. A **Transformer encoder** stacks these attention-and-transformation layers. [Chapter 08](08_attention_mechanisms.md) builds Q, K and V only after a fuller sentence example.

### Add the mathematics
For mixture weights 0.7 and 0.3, the first coordinate is 0.7 × 1 + 0.3 × 0 = 0.7. The second is 0.7 × 0 + 0.3 × 2 = 0.6. This small calculation is the core of attention's weighted mixing. Many heads repeat the operation with different learned views of the vectors.

### Connect it to our project
The saved BERT models have 12 layers and 12 attention heads per layer. The sequence length is capped at 64, including special tokens. [Saved config](../artifacts/phase2/finbert/seed-42/model/config.json) records the dimensions. No RNN-versus-Transformer experiment was performed here.

### Alternatives and trade-offs
An **RNN** processes a sequence by carrying a state from one step to the next. An **LSTM** is an RNN designed to control what that state remembers. They model order, but their sequential work is less parallel. Transformers offer pretrained contextual features and parallel processing; they also use more memory as sequences grow.

### Check your understanding
**Question:** Does an attention weight of 0.7 mean a token has a 70% chance of being positive sentiment?

**Answer explained:** No. It is a mixing weight for representations, not a class probability. The network still has several transformations before it makes a sentiment decision.

### Deeper dive and common confusion
**Common confusion:** “bidirectional” means an encoder can use words on both sides. A causal decoder, commonly used for text generation, restricts each position to earlier positions. It can still classify, but has a different interface and cost. An encoder-decoder model combines a representation stage with a generation stage. These alternatives were not benchmarked here.


## Lesson: Preserve order and keep the calculations well behaved

### Why it matters
If token vectors alone were enough, how would the model distinguish “A acquired B” from “B acquired A”? And how would a deep stack avoid losing every useful feature during repeated updates?

### Build the intuition
Add position information to say where a token is. Keep a shortcut carrying the previous vector while learning a change to it. Regularly rescale the coordinates so one unusually large scale does not dominate the next calculation.

### A concrete example
**Invented example:** token vector `[2,0]` plus position vector `[0,1]` becomes `[2,1]`. Suppose a transformation suggests a change `[0.3,-0.2]`; adding it to the old vector gives `[2.3,0.8]`. This is simpler than requiring the transformation to rebuild everything from scratch.

### Walk through it
1. Add token, position and segment vectors. A segment says which part of a paired input a token belongs to; single sentences usually use segment zero.
2. Normalize the resulting coordinates.
3. During training, randomly hide some coordinates to discourage dependence on one feature.
4. After attention, add the old vector back to the transformed vector.
5. Pass each position through a learned nonlinear transformation, then another shortcut and normalization.

### Give the idea its technical name
A **positional embedding** represents location. The shortcut is a **residual connection**. **Layer normalization** rescales a token's feature coordinates using their mean and variance. **Dropout** is the training-time random hiding. A **feed-forward network** transforms each position's vector with learned weights and a nonlinear function.

### Add the mathematics
For toy vector `[1,3]`, the coordinate mean is 2 and variance is 1. Subtracting 2 and dividing by the square root of 1 gives `[-1,1]`. Real LayerNorm adds a small positive epsilon to the variance to avoid division by zero, and learns a scale and offset.

For coordinate $h_j$, it computes $\gamma_j(h_j-\mu)/\sqrt{\sigma^2+\epsilon}+\beta_j$. Here j selects a coordinate, mu is the coordinate mean, sigma-squared is variance, gamma is learned scale and beta is learned offset.

### Connect it to our project
The embedding module adds token, position and token-type vectors, applies LayerNorm and then dropout. Dropout probabilities are 0.1 in the saved configs. The feed-forward block expands 768 features to 3072, applies GELU—a smooth nonlinear transformation—and projects back to 768. BERT can represent 512 positions, but this experiment used 64.

### Alternatives and trade-offs
Other architectures use different position encodings or put normalization before rather than after a block. Those can change training behavior. A simpler shallow model needs fewer stabilizing components, but lacks the same stack of learned contextual transformations.

### Check your understanding
**Question:** Does `model.eval()` prevent us from calculating derivatives?

**Answer explained:** No. It turns off training behaviors such as dropout. Derivative recording is controlled separately. IG needs derivatives while the model remains in evaluation mode.

### Deeper dive and common confusion
**Common confusion:** layer normalization is not batch normalization. It works across one token's coordinates rather than using statistics across examples. In row-vector notation, a BERT block can be written $U=LN(H+Dropout(MHA(H)))$, then $H'=LN(U+Dropout(GELU(UW_1+b_1)W_2+b_2))$. H is the input matrix; U is the intermediate matrix; W and b are learned weights and biases; MHA is multi-head attention. This equation is a compact summary of the steps, not a new unexplained mechanism.


## Lesson: Learn language first, then learn this task

### Why it matters
We have 3391 labeled training sentences. Is that enough to teach a large network English, financial terminology and sentiment from nothing? Starting with a model that has already learned language patterns is more practical.

### Build the intuition
Imagine two analysts preparing for the same classification exercise. One has read broad English text; another has read financial communications. Both still need examples of the exact three-label task. Financial reading may help, but it does not guarantee a better score.

### A concrete example
**Invented training tasks:** general language practice might ask for the missing word in “The company reported a [MASK].” Task-specific practice gives “Operating profit increased” together with the label positive. The first supplies a word-learning signal; the second directly teaches sentiment.

### Walk through it
1. Begin with weights learned in a large earlier language-learning stage.
2. Add or initialize the output layer needed for three sentiment classes.
3. Show labeled training sentences.
4. Adjust the output layer and encoder weights to reduce sentiment mistakes.
5. Use separate validation sentences to choose training settings.

### Give the idea its technical name
The earlier stage is **pretraining**. Pretraining on financial language is **domain pretraining**. Adapting the model using labeled sentiment examples is **task fine-tuning**. Original BERT used **masked language modeling (MLM)** to recover selected words and **next sentence prediction (NSP)** to distinguish actual adjacent sentence pairs from sampled alternatives.

### Add the mathematics
The data volumes serve different purposes: many unlabeled sentences can teach word relationships; labeled sentences teach our decision rule. This project does not add a new numerical pretraining comparison. Its actual downstream scores are 0.8476 for general BERT and 0.8339 for financial BERT, with the metric explained in Chapter 05.

### Connect it to our project
General BERT starts from `google-bert/bert-base-uncased`; financial BERT starts from `yiyanghkust/finbert-pretrain`. The project did not redo upstream pretraining. Both received the same local supervised data protocol. FinBERT's earlier financial reading includes communications such as reports and calls, not necessarily the same style as short news sentences.

### Alternatives and trade-offs
Training from scratch gives full control but usually needs far more data and compute. Freezing an encoder and fitting only a small classifier is cheaper but allows less adaptation. Continued financial-news pretraining is a future proposal, not work completed here.

### Check your understanding
**Question:** Does a finance-pretrained model already know the exact sentiment label policy?

**Answer explained:** Not necessarily. Recovering missing financial words is different from applying our annotators' sentiment categories. That is why the financial pretraining model still needs supervised fine-tuning.

### Deeper dive and common confusion
**Common confusion:** our fine-tuned financial model is not the ProsusAI checkpoint. ProsusAI already had sentiment training involving PhraseBank, creating a different evaluation-independence concern. [Chapter 03](03_model_architecture.md) explains the four workflows. Original BERT did not replace every selected MLM token with MASK; some were kept or replaced differently. That detail does not make arbitrary MASK-heavy deletion sentences natural.


## Lesson: Turn scores into probabilities, then learn from mistakes

### Why it matters
The model gives the positive class a high score on a negative sentence. How do we quantify that mistake, and how can training change the weights in the right direction?

### Build the intuition
First turn the three scores into numbers that sum to one. Then charge the model more when it assigns very little probability to the correct answer. Finally, work backward through the calculation to see which small weight changes would reduce that charge.

### A concrete example
**Invented calculation:** scores `[0,1,2]` for negative/neutral/positive become approximately `[0.0900,0.2447,0.6652]`. If the label is positive, the model is doing reasonably well. If the label is negative, it is confidently favoring the wrong direction.

### Walk through it
1. Exponentiate the scores: approximately `[1,2.7183,7.3891]`.
2. Divide each by their total, 11.1074.
3. Take minus the natural logarithm of the probability assigned to the correct label.
4. Calculate how this loss changes when each score changes.
5. Pass those sensitivities backward through the layers and update the weights.

### Give the idea its technical name
The raw scores are **logits**. Their exponential normalization is **softmax**. The penalty is **cross-entropy loss**. A **gradient** is a list of derivatives: how sensitive a quantity is to small changes in several inputs or weights. **Backpropagation** applies the chain rule backward through the network to calculate these derivatives efficiently.

### Add the mathematics
For score $z_k$ of class k, $p_k=\exp(z_k)/\sum_j\exp(z_j)$. The denominator sums over all three classes. For the true class y, loss is $L=-\log p_y$. In the toy example it is about 0.4076 if y is positive, but 2.4076 if y is negative.

A tiny backpropagation example makes the chain concrete. Let a toy input feature be x=2, a weight w=0.5, score z=wx=1 and desired score t=2. Use toy loss $L=(z-t)^2/2=0.5$. Then the loss changes by z−t = −1 per unit change in z, while z changes by x=2 per unit change in w. Multiply: derivative with respect to w is −2. With learning rate 0.1, update w to 0.5−0.1(−2)=0.7. The new score is 1.4 and loss is 0.18. This squared-loss illustration teaches the chain rule; the actual classifier uses cross-entropy.

### Connect it to our project
[neural.py](../src/finbert_explainability/phase2/neural.py) uses the three-class BERT classifier and Hugging Face Trainer for optimization. [Chapter 04](04_baselines_and_training.md) explains its AdamW settings. IG later uses gradients of a prediction score with respect to embeddings, not training-loss gradients used to update model weights.

### Alternatives and trade-offs
Accuracy counts only right versus wrong and does not provide the same smooth training signal. Cross-entropy distinguishes “wrong with 0.49 probability for the answer” from “wrong with 0.001.” Other losses can emphasize certain classes, but may also emphasize noisy labels; Chapter 11 treats them as proposals.

### Check your understanding
**Question:** If we add 10 to every logit, do the probabilities change?

**Answer explained:** No. The common exponential factor cancels when we divide by the total. This is why log probabilities cannot uniquely recover the original raw logits.

### Deeper dive and common confusion
**Common confusion:** learning the training examples very well is not the same as doing well on new examples. **Overfitting** means learning patterns that do not transfer. **Generalization** means useful behavior on new relevant examples. **Leakage** gives the process information it should not have had, such as test examples influencing the feature vocabulary. The ordinary cross-entropy logit derivative is p_k−y_k, where y_k is 1 for the true class and 0 otherwise. That connects the toy chain-rule lesson to the actual classification loss.


## Put the pieces together

Text → vocabulary pieces → numerical vectors → context mixing → three scores → probabilities. During training, a loss tells us how to adjust weights. During evaluation, weights stay fixed.

Our pretrained Transformer workflows scored 0.8476 and 0.8339 macro-F1, versus 0.6713 for the lexical baseline. This compares complete workflows; it does not isolate attention, vocabulary or pretraining as the cause. The technical rationale for using pretrained Transformers is that contextual features and mature tooling are useful with limited labeled data, not that this project proved every alternative worse.

For longer sequences, the attention score grid has one entry for every pair of positions. Moving from 64 to 128 positions changes 4096 pairs to 16384: four times as many. Other computation also matters, so whole-program runtime need not increase exactly fourfold.

**Check:** Why not conclude that the bigger vocabulary caused the score difference? **Answer:** Vocabulary, weights, training history and optimization differ together. Counting entries is not a controlled vocabulary experiment.

## Practice explaining it

1. **What is the difference between an ID and an embedding?**

   An ID selects a table row; the embedding is the continuous vector stored there.

2. **Why does context matter?**

   The financial meaning of a word changes with its subject and surrounding clauses.

3. **What does pretraining learn?**

   Reusable language patterns, not necessarily the exact downstream label rule.

4. **How does backpropagation work?**

   Multiply local sensitivities along the calculation, working backward to weights.

5. **Why can a high-confidence answer be wrong?**

   Softmax produces relative probabilities, but they may not match real correctness frequencies; Chapter 06 explains calibration.

## Sources and next lesson

[BERT](https://aclanthology.org/N19-1423/) and [Transformer](https://arxiv.org/abs/1706.03762) are the foundational sources; the equations here explain the repository's BERT usage rather than claiming every Transformer uses the same normalization. See [official documentation and reference status](references.md), then [dataset](02_dataset_and_preprocessing.md).
