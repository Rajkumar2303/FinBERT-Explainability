# 03 — Know which model is answering the question

**Before you begin:** Chapter 01 explains tokens, vectors, attention and softmax. Here we connect those pieces to the actual saved model interfaces. Start with the simple baseline, then follow BERT's array shapes.

## Learning objectives

- Trace a sentence through the lexical baseline and BERT classifier.
- Check tensor shapes, special tokens and class mappings.
- Explain why the financial encoder and ProsusAI reference have different evaluation roles.

## Lesson: Understand the inexpensive baseline first

### Why it matters
Before paying for a Transformer, ask whether word patterns already solve much of the task. That gives us a useful reference for the added complexity.

### Build the intuition
Turn a sentence into a list of word-feature weights, then add learned evidence for each class. The model can learn that “profit” often supports one decision, without building a contextual vector for every word.

### A concrete example
**Invented toy calculation:** use features `[profit, loss]`. For vector `[1,0]`, a positive-class rule with weights `[2,-1]` and bias 0.2 gives score 2.2. For `[0,1]`, it gives -0.8. Real TF-IDF values are weighted and normalized, not just these binary indicators.

### Walk through it
1. Fit a vocabulary and word-frequency statistics on training sentences only.
2. Convert each sentence into a sparse vector: mostly zeros for absent features.
3. Multiply features by class-specific weights and add a bias.
4. Convert the three scores into class probabilities.
5. Evaluate validation candidates, then keep the selected settings.

### Give the idea its technical name
**TF-IDF** means term frequency–inverse document frequency. It emphasizes words that occur in the sentence but are less ubiquitous in the fitted collection. **Logistic regression** is a learned linear scoring model followed by a probability conversion. “Regression” in the name does not mean it predicts a continuous financial quantity here.

### Add the mathematics
For feature values x_j, class weights w_j and bias b, score z is the sum of w_j×x_j plus b. In the toy example, 2×1 + (−1)×0 + 0.2 = 2.2. A **bigram** is a pair of adjacent words, such as “not improved”; it can add a little word-order information that unigrams alone lack.

### Connect it to our project
[baseline.py](../src/finbert_explainability/phase2/baseline.py) builds a sklearn Pipeline with sublinear TF-IDF, at most 20000 features and an lbfgs logistic-regression solver. Sublinear means repeated term counts grow logarithmically rather than linearly. Selected settings were C=10 and unigrams; Chapter 04 explains C and the search.

### Alternatives and trade-offs
The baseline trains and predicts cheaply, and its feature weights can be inspected. It struggles to represent distant negation, entity roles and new combinations of familiar words. Adding bigrams helps some local phrases but increases feature count and cannot represent every context.

### Check your understanding
**Question:** Does a positive coefficient for “increased” prove it is always good financial news?

**Answer explained:** No. It is a learned association within this feature representation. The model may miss whether costs or profits increased, especially when important context is not represented jointly.

### Deeper dive and common confusion
**Common confusion:** the duplicate detector also uses TF-IDF, but over character fragments for grouping. This classifier uses word features fitted only on training data. Same tool family, different role and fitting boundary.


## Lesson: Trace one sentence through BERT

### Why it matters
What does the code actually do between receiving a sentence and returning three probabilities? Understanding the shapes makes the model less mysterious and helps catch implementation errors.

### Build the intuition
BERT first represents all text pieces, then lets them exchange information through 12 layers. A dedicated starting position gathers a sentence representation, and a small final layer turns that representation into three scores.

### A concrete example
**Illustrative sequence, not a verified tokenizer output:** `[CLS] profits increased [SEP] [PAD] ...`. CLS marks the classification position; SEP marks the end of the sentence segment; PAD fills unused positions. An attention mask says which positions should be available to other tokens.

### Walk through it
1. Tokenize and truncate/pad to 64 positions.
2. Look up token vectors and add position/segment information.
3. Run 12 layers of attention, residual additions, normalization and feed-forward transformations from Chapter 01.
4. Take the final CLS representation and apply the BERT pooler, a learned dense transformation followed by tanh, which squashes values between -1 and 1.
5. Apply training-time dropout and a three-output linear classifier.
6. Apply softmax when probabilities are needed.

### Give the idea its technical name
A **tensor shape** lists array dimensions. The **classifier head** is the final task-specific layer. The **pooler** transforms the final CLS vector; it is not an average of every token. The encoder **hidden size** is the number of features at each position.

### Add the mathematics
For B sentences, input IDs have shape `[B,64]`; embedding and layer outputs have `[B,64,768]`; pooled CLS vectors have `[B,768]`; logits have `[B,3]`. The three-class head has 768×3 + 3 = 2307 parameters: one weight per input/output pair plus three biases. Twelve heads divide width 768 into 64 features per head.

### Connect it to our project
[neural.py](../src/finbert_explainability/phase2/neural.py) uses `BertConfig` and `BertForSequenceClassification`. The [general config](../artifacts/phase2/bert/seed-42/model/config.json) and [financial config](../artifacts/phase2/finbert/seed-42/model/config.json) verify 12 layers, 12 heads, width 768 and feed-forward width 3072. Explicit configuration avoids guessing from incomplete upstream metadata.

### Alternatives and trade-offs
Mean pooling over token states is a possible alternative sentence representation, but not this classifier's saved behavior. DistilBERT uses a smaller architecture; other encoders use different internal modules. Explanation hooks must be checked for each architecture instead of blindly searching for `model.bert.embeddings`.

### Check your understanding
**Question:** If B=8, how many sentiment logits are produced?

**Answer explained:** Eight rows of three scores, or 24 values. The 64 input positions do not each receive a sentiment label in this sentence-classification task.

### Deeper dive and common confusion
**Common confusion:** masking padding keys does not force every internal padding vector to zero. Also, the 512-position architectural capacity does not mean the project evaluated 512-token sentences. The actual cap remained 64.


## Lesson: Keep the three BERT identities separate

### Why it matters
The word “FinBERT” can refer to different checkpoints. If we confuse them, we may accidentally claim that an already-trained model passed an exam containing its old practice questions.

### Build the intuition
Compare the local general and financial starting models fairly: train both on the same local training rows. Treat the externally sentiment-trained ProsusAI model as a separate reference because its earlier exposure is different.

### A concrete example
**Invented label-order example:** a reference model outputs `[0.70,0.20,0.10]` in positive/negative/neutral order. Our order is negative/neutral/positive, so we must rearrange it to `[0.20,0.10,0.70]`. Otherwise we could call a positive prediction negative.

### Walk through it
1. Identify the exact upstream checkpoint and pinned revision.
2. Load its own tokenizer and config.
3. Inspect the ID-to-label mapping.
4. Reorder probability columns and prediction labels consistently.
5. Record whether the test can be independent of the checkpoint's original training.

### Give the idea its technical name
A **model revision** identifies a particular saved version. A **label mapping** connects output indices to sentiment names. **Benchmark contamination** means evaluation examples may have been seen earlier by the model or development process.

### Add the mathematics
The saved general classifier contains 109484547 parameters; financial BERT contains 109754115. The difference is 269568 = (30873−30522)×768, explained by the vocabulary embedding rows. Around 110 million FP32 weights require roughly four bytes each before format overhead; training also stores gradients, optimizer state and temporary computations.

### Connect it to our project
General BERT is `google-bert/bert-base-uncased`. Our financial model is fine-tuned from `yiyanghkust/finbert-pretrain`, an earlier financial language model. `ProsusAI/finbert` already had sentiment training involving PhraseBank. [comparison.json](../artifacts/phase2/comparison/comparison.json) stores its original mapping: 0 positive, 1 negative, 2 neutral. Local trained heads use 0 negative, 1 neutral, 2 positive.

### Alternatives and trade-offs
The local pair makes a more interpretable comparison than changing model scale radically. It still does not isolate financial pretraining alone: vocabulary and learned starting weights also differ. The external reference is informative if clearly labeled, but cannot be called the independent benchmark winner.

### Check your understanding
**Question:** Does making a new random split of PhraseBank remove ProsusAI's possible earlier exposure?

**Answer explained:** No. A new local split cannot make an upstream model forget examples. A separate independently collected evaluation set would be needed to address that concern.

### Deeper dive and common confusion
**Common confusion:** parameter count is not VRAM usage. Training requires additional state; CUDA also reserves memory beyond current tensor allocations. Peak allocated memory was 2.079 GiB for each neural training workflow, with reserved memory somewhat higher. These counters exclude other applications and some driver/display memory.


## Compare the concrete workflows

| Model | Vocabulary | Saved classifier parameters | Test role |
|---|---:|---:|---|
| General BERT | 30522 | 109484547 | Independent supervised workflow |
| Financial BERT | 30873 | 109754115 | Independent supervised workflow; seed 42 explanation model |
| TF-IDF | At most 20000 configured features | Depends on fitted vocabulary | Independent inexpensive baseline |
| ProsusAI | Separate saved tokenizer | Not re-estimated here | Non-independent fixed reference |

The parameter counts above come from saved safetensors headers, without running models. FinBERT's largest weight file is about 418.7 MiB. The TF-IDF feature cap is not a claim that every run necessarily used 20000 features.

**Check:** Is a model with the same parameter count guaranteed to have the same accuracy? **Answer:** No. The learned values, vocabulary, data and training procedure matter, not just the number of stored values.

## Explore alternatives after understanding this model

RoBERTa changes the BERT pretraining recipe; DeBERTa separates content and position information in its attention design; ModernBERT updates encoder design and efficiency, including long-context use; DistilBERT transfers knowledge into a smaller encoder. These are possible next comparisons, not models evaluated in this project.

| Architecture | Relevant change | Potential benefit | Main trade-off |
|---|---|---|---|
| RoBERTa | Revised BERT pretraining recipe | Strong contextual baseline | Different tokenizer/corpus; cannot assume financial gain |
| DeBERTa | Disentangled content/position attention | More expressive position/content modeling | Different extraction hooks and resource profile |
| ModernBERT | Modern encoder design and long-context efficiency | Efficient batched/longer-context processing | Length-64 task may not exploit main strengths; attribution compatibility needs testing |
| DistilBERT | Distilled smaller encoder | Lower latency/memory | Possible loss of accuracy and different architecture assumptions |
| Other finance-adapted encoders | Domain-specific representations | Better target-language coverage | Corpus rights, contamination and domain-match uncertainty |
| Frozen sentence embeddings + linear head | Reuse dense sentence features | Cheap supervised fitting | Frozen representation may miss subtle label distinctions |

For a new architecture, first check the tokenizer, model output shape, label mapping and explanation layer. Then compare at a declared compute or latency budget. Published gains elsewhere do not predict gains here. [Chapter 11](11_improvements_and_future_research.md) explains how to test these alternatives fairly.

**Check:** Why might a long-context model not help this task automatically? **Answer:** The study uses short, length-64 inputs. Extra context capacity only helps if relevant omitted information exists and the new procedure learns to use it.

## Practice explaining it

1. **What does logistic regression calculate?**

   A weighted sum of input features plus a bias for each class, then class probabilities.

2. **What is CLS used for?**

   Its final contextual vector supplies the sentence representation used by this classifier.

3. **Why is ProsusAI separate?**

   Its earlier sentiment training can include the local test examples.

4. **Why inspect id2label?**

   Output index zero does not mean the same sentiment in every checkpoint.

5. **Why is architecture verification important for IG?**

   Different models expose different continuous layers; choosing the wrong hook changes the question or breaks the computation.

## Sources and next lesson

[Hugging Face BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert), [financial pretraining card](https://huggingface.co/yiyanghkust/finbert-pretrain), [ProsusAI card](https://huggingface.co/ProsusAI/finbert), [alternative-model papers](references.md). Continue to [training](04_baselines_and_training.md).
