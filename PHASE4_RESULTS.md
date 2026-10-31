# Phase 4 measured results

Checkpoint: `artifacts\phase2\finbert\seed-42\model`. Seed is fixed by configuration, not chosen by test performance. No training or tuning was performed.

All 727 frozen test predictions were checked against Phases 2 and 3. Maximum probability difference: 0; maximum logit difference against Phase 3: 1.43e-06.

Selection: lexicographically smallest five row IDs among correct predictions and among errors, using the fixed checkpoint. This balanced explanation subset is not a population estimate. Target = each original predicted-class logit.

Measured attribution records: 10 PAD and 4 MASK. Experiment time: 107.14 s (includes integrity checks and rendering). Attribution time: 44.21 s; throughput: 0.317 records/s. Peak CUDA allocated/reserved: 0.567/0.641 GiB.

Final absolute residual (all references): median 0.000328735, max 0.000757853. Maximum relative residual: 0.000795425. Escalated beyond 50 steps: 10. Unresolved tolerance failures: 0.

Residual = sum of all embedding-dimension attributions minus (target input logit - target reference logit). Relative residual divides its absolute value by max(abs(logit difference), 1e-6). Both absolute <= 0.001 and relative <= 0.01 are required. Every 50/100/200-step attempt is retained; special tokens and padding remain in completeness sums. Failures are retained, not filtered out.

## Examples

| Sample | True / predicted | Probability | Steps | Delta | Top absolute WordPiece contributions | Figure |
|---|---|---:|---:|---:|---|---|
| fpb50-00001 | neutral / neutral | 0.9990 | 50 | 0.0004292 | said (-0.626), stages (+0.361), host (+0.328), statement (+0.312) | [PNG](artifacts/phase4/initial/figures/fpb50-00001.png) |
| fpb50-00002 | negative / negative | 0.9985 | 200 | -5.627e-05 | off (+2.254), employees (+0.763), has (+0.746), tens (+0.630) | [PNG](artifacts/phase4/initial/figures/fpb50-00002.png) |
| fpb50-00018 | positive / positive | 0.9936 | 200 | 2.337e-05 | double (+2.783), revenues (+0.726), plans (+0.721), its (+0.592) | [PNG](artifacts/phase4/initial/figures/fpb50-00018.png) |
| fpb50-00019 | positive / positive | 0.9601 | 200 | 0.0004416 | up (+0.908), dividend (+0.669), from (+0.446), eur (+0.417) | [PNG](artifacts/phase4/initial/figures/fpb50-00019.png) |
| fpb50-00022 | positive / positive | 0.9990 | 200 | 0.0007579 | increased (+1.678), share (+0.242), market (+0.181), . (+0.177) | [PNG](artifacts/phase4/initial/figures/fpb50-00022.png) |
| fpb50-00059 | neutral / negative | 0.9925 | 200 | 0.0002284 | down (+2.784), sales (+1.270), up (+1.173), % (+0.980) | [PNG](artifacts/phase4/initial/figures/fpb50-00059.png) |
| fpb50-00081 | neutral / positive | 0.7581 | 50 | 3.529e-05 | signed (+3.982), months (-1.390), contracts (+1.321), delivery (+1.001) | [PNG](artifacts/phase4/initial/figures/fpb50-00081.png) |
| fpb50-00150 | positive / neutral | 0.9941 | 100 | -0.0004551 | turnover (+2.545), according (-0.596), ##kk (-0.554), over (+0.414) | [PNG](artifacts/phase4/initial/figures/fpb50-00150.png) |
| fpb50-00403 | neutral / positive | 0.9234 | 50 | -0.0007443 | agreement (+0.921), companies (+0.625), signed (+0.468), of (-0.457) | [PNG](artifacts/phase4/initial/figures/fpb50-00403.png) |
| fpb50-00423 | negative / neutral | 0.5718 | 100 | 5.341e-05 | rose (-2.451), compared (-0.542), change (+0.383), - (+0.376) | [PNG](artifacts/phase4/initial/figures/fpb50-00423.png) |

## Baseline sensitivity

First two correct and first two incorrect selected examples use MASK as the additional reference. Target class, positions, attention mask, and model weights are held fixed. Rankings use absolute signed token sums; tied ranks use average ranks for Spearman, with position order breaking top-k ties.

| Sample | Absolute-rank Spearman | Top-5 Jaccard | Sign-change fraction | Both converged |
|---|---:|---:|---:|---|
| fpb50-00001 | 0.1980392156862745 | 0.42857142857142855 | 0.2 | True |
| fpb50-00002 | 0.4006968641114983 | 0.25 | 0.38095238095238093 | True |
| fpb50-00059 | 0.6442687747035574 | 0.6666666666666666 | 0.3181818181818182 | True |
| fpb50-00081 | 0.6202270381836945 | 0.6666666666666666 | 0.2777777777777778 | True |

## Limitations and audit notes

- PAD and MASK references are artificial attended sequences, not neutral financial sentences. Neither reference is universally correct. Sensitivity does not identify a preferred baseline.
- Integration is at the complete embedding output, after LayerNorm; it does not separately attribute lexical, position, and token-type components. Constant endpoint embeddings receive zero attribution even when those positions are used by the model.
- Positive/negative scores support/oppose the selected target logit, not positive/negative sentiment in general. Small numerical residuals verify integration accuracy, not explanation faithfulness.
- Original WordPiece records are retained. Optional word displays merge adjacent ## continuations and sum signed scores; punctuation is separate. Sum of absolute pieces and absolute signed word sum are distinct.
- Maximum length is the original 64-token protocol. Truncation is recorded per example. Float32 eager attention is used for gradient compatibility and compared with prior inference; bitwise cross-device equivalence is not promised.
- No human causal interpretations or attention-faithfulness conclusions are asserted. No Phase 5 comparison was performed.
- Earlier files preserved: True. Any tolerated pre-existing annotation manifest difference is recorded below.

Warnings: [{'artifact': 'human_annotation.csv', 'reason': 'Known user-authorized annotation edits after sealing; not an inference input', 'sealed_sha256': '5617c819ad50fcebd442f813ae09bd1d85eea2d2cf316c5f49d32147900f18e3', 'current_sha256': 'dd8529ee87b77cc4ef7877b59660c33ebcf018b14e14089d1b09878276c69331'}]

References: [Captum LayerIntegratedGradients](https://captum.ai/api/layer.html#layer-integrated-gradients), [Captum memory guidance](https://captum.ai/docs/faq), [Integrated Gradients paper](https://arxiv.org/abs/1703.01365).
