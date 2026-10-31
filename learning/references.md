# Reading the research behind the project

Start with a question, not a bibliography. Read the chapter’s worked example first, then use these sources to go deeper. For attention, read Vaswani and then Devlin. For calibration, start with Guo. For IG, read Sundararajan, Taly and Yan after Chapter 07. Read both Jain/Wallace and Wiegreffe/Pinter before making a broad claim about attention as explanation. For future adaptation experiments, start with Gururangan and colleagues.

**Check:** Does citing an IG paper show that our heatmap is faithful? **Answer:** No. A paper motivates a method; our chosen layer, reference, numerical implementation and evaluation still need scrutiny.

## Verification and version limits

Primary pages and official documentation below were checked during earlier guide authoring on **30 October 2026**. This rewrite retains that record, not a claim of a fresh live check. “Verified” means identity and relevant location were checked, not independent replication. Initial preprint years are used unless a conference is stated.

Online docs may differ from the saved environment (Transformers 5.13.0, sklearn 1.9.0, SciPy 1.17.1, PyTorch 2.12.1+cu126). Local source and saved arguments govern what ran.

## Verified academic sources

| Source | Why read it here |
|---|---|
| Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer attention and architecture foundations; not every modern model uses its exact design |
| Devlin et al. (2019), [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) | Bidirectional encoder, MLM/NSP and transfer to downstream tasks |
| Malo et al. (2013 preprint), [Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts](https://arxiv.org/abs/1307.5336) | Financial phrase-level sentiment and PhraseBank context |
| Yang, Uy and Huang (2020), [FinBERT: A Pretrained Language Model for Financial Communications](https://arxiv.org/abs/2006.08097) | Financial pretraining background; distinguish this lineage from ProsusAI |
| Loshchilov and Hutter (2017 preprint), [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | Why AdamW decay differs from naively adding L2 to adaptive gradients |
| Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Confidence calibration and temperature scaling |
| Sundararajan, Taly and Yan (2017), [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) | IG's axiomatic motivation and path attribution; not a guarantee of natural-language causal validity |
| Jain and Wallace (2019), [Attention is not Explanation](https://aclanthology.org/N19-1357/) | Empirical challenges to treating attention weights as sufficient explanations |
| Wiegreffe and Pinter (2019), [Attention is not not Explanation](https://aclanthology.org/D19-1002/) | Explanatory definitions and additional experimental controls |
| Abnar and Zuidema (2020), [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/) | Attention flow and rollout alternatives |
| Ribeiro, Singh and Guestrin (2016), [“Why Should I Trust You?”: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) | LIME/local surrogate explanation |
| Lundberg and Lee (2017), [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) | SHAP framework and feature-allocation perspective |
| Shrikumar, Greenside and Kundaje (2017 preprint), [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685) | DeepLIFT reference-based attribution |
| Liu et al. (2019), [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | Alternative pretraining recipe; not tested in this project |
| He et al. (2020 preprint), [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) | Content/position attention alternative |
| Sanh et al. (2019 preprint), [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) | Smaller encoder through distillation |
| Warner et al. (2024), [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663) | ModernBERT architecture and efficiency motivation; no local performance claim |
| Gururangan et al. (2020), [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/) | DAPT/TAPT research motivation |
| Lin et al. (2017 preprint), [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | Original focal-loss mechanism; its application to this NLP task is a proposal |
| Hinton, Vinyals and Dean (2015), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Teacher/student distributions and distillation motivation |

## Verified official documentation and model/data cards

| Source | Use and caveat |
|---|---|
| [Hugging Face BERT](https://huggingface.co/docs/transformers/model_doc/bert) | Classifier architecture and model interface; local installed version is historical authority |
| [Captum layer attribution](https://captum.ai/api/layer.html#layer-integrated-gradients) | LayerIG API, baselines, layer input/output attribution and convergence delta |
| [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Seed/kernel/platform limitations; stable URL redirects to current version |
| [sklearn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) | Vectorizer semantics; installed defaults were also inspected |
| [sklearn evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) | Metric and averaging definitions |
| [SciPy Spearman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) | Rank association and constant-input caveats |
| [SciPy Kendall](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html) | Tau-b and tie handling |
| [Financial PhraseBank card](https://huggingface.co/datasets/takala/financial_phrasebank) | Current dataset description and CC BY-NC-SA 3.0 metadata; pinned ZIP is the actual input |
| [Financial pretraining card](https://huggingface.co/yiyanghkust/finbert-pretrain) | Upstream financial model identity and corpus description; missing explicit license metadata remains a blocker |
| [ProsusAI card](https://huggingface.co/ProsusAI/finbert) | PhraseBank training provenance and current CC BY-NC-SA 4.0 metadata; not independent local evaluation |

## Local primary evidence

- [Phase 2 results](../PHASE2_RESULTS.md), [comparison JSON](../artifacts/phase2/comparison/comparison.json), [config](../configs/phase2.json).
- [Phase 3 results](../PHASE3_RESULTS.md), [summary](../artifacts/phase3/summary.json), [current annotation CSV](../artifacts/phase3/human_annotation.csv).
- [Phase 4 results](../PHASE4_RESULTS.md), [IG core](../src/finbert_explainability/phase4/core.py), [initial sensitivity](../artifacts/phase4/initial/baseline_sensitivity.json).
- [Phase 5 results](../PHASE5_RESULTS.md), [full summary](../artifacts/phase5/full/summary.json), [analysis code](../src/finbert_explainability/phase5/analysis.py).
- [Phase 6 results](../PHASE6_RESULTS.md), [sensitivity JSON](../artifacts/phase6/audit-final/ig_sensitivity.json), [final report](../artifacts/phase6/MODEL_VALIDATION_REPORT.md), [model card](../publication/MODEL_CARD.md).
- [Data statistics](../artifacts/phase1/statistics.json), [leakage report](../artifacts/phase1/leakage_report.json), [original README](../README.md), [publication README](../publication/README.md), [experiment documentation](../PHASE2.md).

## What still needs verification

Exact rights for pinned upstream model revisions are not resolved by opening their current cards. Upstream training corpus membership, complete original training recipes and exact per-row annotation votes were not independently recovered. These are missing evidence, not fabricated references. The financial-pretrain checkpoint's license and the project's code-license choice require owner review before publication. See the [existing license review](../publication/LICENSE_REVIEW.md).

The initial attempted dataset URL without `/datasets/` failed; the correct official dataset URL above was successfully opened. No conclusion depends on the failed URL. This catalog intentionally avoids unsupported bibliographic details for documents not inspected.
