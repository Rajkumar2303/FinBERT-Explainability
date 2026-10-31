# Phase 5 measured results

Run type: full frozen test experiment. Checkpoint: `artifacts\phase2\finbert\seed-42\model`; fixed seed 42. No model retraining, retuning, or weight changes.

Processed 727 sentences; all 727 frozen predictions verified. Maximum probability/logit differences: 0/1.43e-06. Reused 10 Phase 4 IG records. Converged IG: 687/727.

Integration-step counts: {'50': 151, '100': 400, '200': 176}. Unresolved convergence row IDs: ['fpb50-00040', 'fpb50-00089', 'fpb50-00093', 'fpb50-00152', 'fpb50-00167', 'fpb50-00170', 'fpb50-00181', 'fpb50-00187', 'fpb50-00202', 'fpb50-00243', 'fpb50-00323', 'fpb50-00348', 'fpb50-00349', 'fpb50-00368', 'fpb50-00374', 'fpb50-00378', 'fpb50-00550', 'fpb50-00586', 'fpb50-00602', 'fpb50-00677', 'fpb50-00701', 'fpb50-00727', 'fpb50-00960', 'fpb50-00962', 'fpb50-01725', 'fpb50-01755', 'fpb50-02174', 'fpb50-02276', 'fpb50-02300', 'fpb50-02803', 'fpb50-04019', 'fpb50-04057', 'fpb50-04074', 'fpb50-04401', 'fpb50-04418', 'fpb50-04533', 'fpb50-04661', 'fpb50-04689', 'fpb50-04765', 'fpb50-04772']. All records, including failures, are retained. Primary IG comparisons use the converged subset; all-record sensitivity summaries are in summary.json. Truncated examples: 15.

Wall time: 951.44 s. Recorded IG time (includes reused work): 830.38 s. Attention/deletion time: 80.15 s. Peak allocated/reserved CUDA memory: 0.567/0.641 GiB. Earlier files preserved: True (909 files).

## Attribution agreement

Final-layer CLS-to-token attention is averaged across all 12 heads. Absolute and signed IG comparisons use identical content WordPieces. Padding, all special token IDs, and tokens whose characters (after removing ##) are entirely Unicode punctuation are excluded. Currency/math symbols and numeric WordPieces remain content. Ties use average ranks for Spearman and tau-b for Kendall. Constant vectors and fewer than three eligible tokens return undefined, never zero.

| Comparison | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |
|---|---:|---:|---:|---:|---|
| absolute IG: spearman | 687 | 0.2406 | 0.2681 | 0.2632 | [0.2214, 0.2607] |
| absolute IG: kendall_tau_b | 687 | 0.1724 | 0.1868 | 0.1940 | [0.1584, 0.1870] |
| signed IG: spearman | 687 | 0.1008 | 0.1045 | 0.2983 | [0.0781, 0.1229] |
| signed IG: kendall_tau_b | 687 | 0.0741 | 0.0735 | 0.2193 | [0.0575, 0.0902] |

Correlation status counts (all records): {'absolute': {'valid': 727}, 'signed': {'valid': 727}}.

![Correlation distributions](artifacts/phase5/full/figures/correlations.png)

## MASK deletion interventions

Rankings and the predicted-class target are fixed from unmodified input. Budgets use ceil(fraction × eligible count), with repeated counts deduplicated for inference. Masks preserve positions and the original attention mask. AUC integrates the actual masked fraction using the trapezoid rule. Lower probability AUC means larger suppression under this intervention; negative drops are retained. Random results average five fixed orderings per sentence before statistics.

| Method: probability AUC | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |
|---|---:|---:|---:|---:|---|
| ig_absolute | 687 | 0.6555 | 0.9865 | 0.4186 | [0.6249, 0.6858] |
| attention | 687 | 0.6608 | 0.9942 | 0.4229 | [0.6297, 0.6912] |
| random | 687 | 0.7866 | 0.9148 | 0.2390 | [0.7690, 0.8035] |

| Paired AUC difference | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |
|---|---:|---:|---:|---:|---|
| ig_absolute_minus_attention | 687 | -0.0053 | -0.0002 | 0.1043 | [-0.0130, 0.0024] |
| ig_absolute_minus_random | 687 | -0.1311 | -0.0005 | 0.2360 | [-0.1482, -0.1143] |
| attention_minus_random | 687 | -0.1258 | 0.0001 | 0.2431 | [-0.1440, -0.1085] |

For ig_absolute_minus_attention, the paired interval is spans zero. This describes the specified MASK intervention and does not establish general explanation fidelity.

For ig_absolute_minus_random, the paired interval is below zero. This describes the specified MASK intervention and does not establish general explanation fidelity.

For attention_minus_random, the paired interval is below zero. This describes the specified MASK intervention and does not establish general explanation fidelity.

Mean within-sentence random-order probability-AUC SD: 0.0991.

| Method | Any-budget label flip fraction | Final-budget flip fraction | Mean logit-drop AUC |
|---|---:|---:|---:|
| ig_absolute | 0.4309 | 0.3916 | 2.0677 |
| attention | 0.4178 | 0.3916 | 1.9840 |
| random | 0.4684 | 0.3916 | 1.2460 |

Positive-only IG masks strictly positive scores and stops when exhausted. Mean maximum eligible-token coverage: 0.6628. Partial AUCs are exported but are not ranked against complete-coverage AUCs. Its curve is a sensitivity analysis, not a matched-budget winner comparison.

![Deletion curves](artifacts/phase5/full/figures/deletion_curves.png)

![AUC distributions](artifacts/phase5/full/figures/deletion_auc.png)

![Paired AUC intervals](artifacts/phase5/full/figures/paired_auc.png)

![Positive-only sensitivity](artifacts/phase5/full/figures/positive_ig_sensitivity.png)

## Attention mass

| Token type | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |
|---|---:|---:|---:|---:|---|
| CLS | 727 | 0.1046 | 0.1034 | 0.0255 | [0.1028, 0.1066] |
| SEP | 727 | 0.2980 | 0.2961 | 0.0872 | [0.2919, 0.3047] |
| punctuation | 727 | 0.0704 | 0.0601 | 0.0496 | [0.0668, 0.0740] |
| content | 727 | 0.5270 | 0.5212 | 0.0973 | [0.5198, 0.5336] |
| other_special | 727 | 0.0000 | 0.0000 | 0.0008 | [0.0000, 0.0001] |
| padding | 727 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |

![Attention token types](artifacts/phase5/full/figures/attention_token_types.png)

![Head means](artifacts/phase5/full/figures/attention_heads.png)

Mass is measured before content filtering, with special tokens retained. The query is CLS; padding mass is checked separately. All-layer, all-head raw CLS rows are stored per sentence. Special-token mass alone does not establish that attention is unfaithful: these representations can carry contextual information.

## Examples and disagreements

![Example-level differences](artifacts/phase5/full/figures/example_disagreements.png)

Token heatmaps are in figures/ and listed in figure_selection.json. Representative examples use the first three correct and first three incorrect row IDs; the two lowest-correlation cases are separately labeled descriptive extremes. Each heatmap column has its own scale, so it displays relative patterns rather than equating attention probability and logit attribution units.

## Interpretation against the literature

[Jain and Wallace (2019)](https://aclanthology.org/N19-1357/) question attention explanations using gradient agreement and alternative attention distributions with similar outputs. Our rank correlations and MASK tests address related diagnostics, but we do not reproduce their attention-distribution interventions. This BERT checkpoint and final-layer CLS summary differ from the attention mechanisms in their experiments. Correlation with IG alone cannot establish faithfulness.

[Wiegreffe and Pinter (2019)](https://aclanthology.org/D19-1002/) argue that the answer depends on the definition of explanation and model context, proposing uniform-attention, random-seed, diagnostic, and adversarial-training tests. Our random token-deletion baseline and paired uncertainty provide narrower controls, not those tests. We do not train adversarial models or evaluate attention variability across model seeds. These measurements cannot prove that attention is never useful as an explanation.

## Uncertainty and limitations

- Percentile bootstrap: 2000 resamples, seed 2042, sampling sentences with replacement. Paired comparisons resample within-sentence differences. Random-order replicates and heads are not treated as independent examples. Curve intervals are pointwise; no multiplicity adjustment is made.
- The frozen benchmark is not a random sample of all financial language. Shared topics and phrasing can create dependence; sentence bootstrap intervals do not eliminate that limitation. One fixed model seed and one attention aggregation limit generalizability.
- MASK replacement introduces distribution shift. Correlations and deletion AUC are diagnostic evidence, not ground-truth causal fidelity. Absolute IG can prioritize inhibitory tokens; removing them may increase confidence. Positive-only sensitivity has different coverage.
- The original Phase 4 PAD reference and 50/100/200-step schedule remain fixed. Baseline sensitivity was observed in Phase 4. No test finding changes model or attribution settings. Numerical failures are retained and disclosed.
- WordPiece deletion can remove part of a word. Punctuation filtering may retain numeric context that matters. The 64-token truncation protocol is unchanged.
- Attention extraction uses eager attention in eval mode. The 12×12 CLS rows are retained across layers; complete query-key matrices are optional and disabled to bound disk use. No large tensor collection is kept on the GPU.
- Existing input warnings: [{'artifact': 'human_annotation.csv', 'reason': 'Known user-authorized annotation edits after sealing; not an inference input', 'sealed_sha256': '5617c819ad50fcebd442f813ae09bd1d85eea2d2cf316c5f49d32147900f18e3', 'current_sha256': 'dd8529ee87b77cc4ef7877b59660c33ebcf018b14e14089d1b09878276c69331'}]. The authorized Phase 3 annotation edits are preserved without rewriting its original manifest.
- No Phase 6 report or GitHub publication work was performed.
