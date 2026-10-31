import html
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["negative", "neutral", "positive"]


def figure(record, output):
    output=Path(output)
    items=[t for t in record["tokens"] if t["attention_mask"]]
    scores=[t["signed"] for t in items]
    fig,ax=plt.subplots(figsize=(11,max(5,len(items)*.26)))
    ax.barh(range(len(items)),scores,color=["#167b53" if s>=0 else "#bd3851" for s in scores])
    ax.set_yticks(range(len(items)),[f'{t["position"]}: {t["token"]}' for t in items],fontsize=9)
    ax.invert_yaxis(); ax.axvline(0,color="black",linewidth=.6)
    ax.set_xlabel("Signed contribution to predicted-class logit (PAD reference)")
    last=record["attempts"][-1]
    title=(f'{record["row_id"]} | true: {LABELS[record["true_label"]]} | predicted: {LABELS[record["predicted_label"]]} '
           f'(p={record["probabilities"][record["predicted_label"]]:.4f})\n'
           f'Captum delta={last["captum_delta"]:.3g}; steps={last["n_steps"]}; converged={record["converged"]}\n'+
           '\n'.join(textwrap.wrap(record["sentence"],105)))
    ax.set_title(title,fontsize=10,pad=15)
    fig.tight_layout(); fig.savefig(output,dpi=140); plt.close(fig)


def gallery(records,path):
    parts=['<!doctype html><meta charset="utf-8"><title>Phase 4 IG</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}section{border-bottom:1px solid #bbb;padding:20px}span.token{display:inline-block;padding:8px;margin:3px;border-radius:4px}small{display:block}</style>',
           '<h1>Predicted-class logit contributions</h1><p>Green supports the target logit relative to PAD; pink opposes it. Colors are scaled within each example. WordPiece tokens remain separate. Hover for raw scores. Padding is hidden here, retained in JSON. Attribution is baseline-dependent and is not a causal claim.</p>']
    for r in records:
        last=r['attempts'][-1]
        parts.append(f'<section><h2>{r["row_id"]}</h2><p>{html.escape(r["sentence"])}</p><p>True: {LABELS[r["true_label"]]}; predicted/target: {LABELS[r["predicted_label"]]}; probability: {r["probabilities"][r["predicted_label"]]:.6f}; delta: {last["captum_delta"]:.6g}; steps: {last["n_steps"]}; converged: {r["converged"]}</p>')
        scale=max(abs(t['signed']) for t in r['tokens']) or 1
        for t in r['tokens']:
            if not t['attention_mask']: continue
            rgb='30,155,100' if t['signed']>=0 else '215,60,90'
            parts.append(f'<span class="token" style="background:rgba({rgb},{.1+.7*abs(t["signed"])/scale:.3f})" title="position {t["position"]}; signed {t["signed"]:.8f}; absolute {t["absolute"]:.8f}">{html.escape(t["token"])}<small>{t["signed"]:+.4f}</small></span>')
        parts.append('</section>')
    Path(path).write_text(''.join(parts),encoding='utf8')


def write_report(path, summary, records, sensitivities, output):
    lines=['# Phase 4 measured results','',
        f'Checkpoint: `{summary["checkpoint"]}`. Seed is fixed by configuration, not chosen by test performance. No training or tuning was performed.', '',
        f'All {summary["verified_test_rows"]} frozen test predictions were checked against Phases 2 and 3. Maximum probability difference: {summary["prediction_max_probability_difference"]:.3g}; maximum logit difference against Phase 3: {summary["prediction_max_logit_difference"]:.3g}.', '',
        'Selection: lexicographically smallest five row IDs among correct predictions and among errors, using the fixed checkpoint. This balanced explanation subset is not a population estimate. Target = each original predicted-class logit.', '',
        f'Measured attribution records: {summary["records"]} PAD and {summary["sensitivity_records"]} MASK. Experiment time: {summary["wall_seconds"]:.2f} s (includes integrity checks and rendering). Attribution time: {summary["attribution_seconds"]:.2f} s; throughput: {summary["attributions_per_second"]:.3f} records/s. Peak CUDA allocated/reserved: {summary["peak_allocated_bytes"]/2**30:.3f}/{summary["peak_reserved_bytes"]/2**30:.3f} GiB.', '',
        f'Final absolute residual (all references): median {summary["median_absolute_residual"]:.6g}, max {summary["max_absolute_residual"]:.6g}. Maximum relative residual: {summary["max_relative_residual"]:.6g}. Escalated beyond 50 steps: {summary["escalated_records"]}. Unresolved tolerance failures: {summary["unconverged_records"]}.', '',
        'Residual = sum of all embedding-dimension attributions minus (target input logit - target reference logit). Relative residual divides its absolute value by max(abs(logit difference), 1e-6). Both absolute <= 0.001 and relative <= 0.01 are required. Every 50/100/200-step attempt is retained; special tokens and padding remain in completeness sums. Failures are retained, not filtered out.', '',
        '## Examples','', '| Sample | True / predicted | Probability | Steps | Delta | Top absolute WordPiece contributions | Figure |','|---|---|---:|---:|---:|---|---|']
    for r in records[:10]:
        ranked=[r['tokens'][i] for i in r['ranking_positions'][:4]]
        top=', '.join(f'{t["token"]} ({t["signed"]:+.3f})' for t in ranked).replace('|','\\|')
        last=r['attempts'][-1]
        lines.append(f'| {r["row_id"]} | {LABELS[r["true_label"]]} / {LABELS[r["predicted_label"]]} | {r["probabilities"][r["predicted_label"]]:.4f} | {last["n_steps"]} | {last["captum_delta"]:.4g} | {top} | [PNG]({Path(output).as_posix()}/figures/{r["row_id"]}.png) |')
    lines.extend(['','## Baseline sensitivity','','First two correct and first two incorrect selected examples use MASK as the additional reference. Target class, positions, attention mask, and model weights are held fixed. Rankings use absolute signed token sums; tied ranks use average ranks for Spearman, with position order breaking top-k ties.','', '| Sample | Absolute-rank Spearman | Top-5 Jaccard | Sign-change fraction | Both converged |','|---|---:|---:|---:|---|'])
    for s in sensitivities:
        lines.append(f'| {s["row_id"]} | {s["spearman_absolute_rank"]} | {s["top_k_jaccard"]} | {s["sign_change_fraction"]} | {s["both_converged"]} |')
    lines.extend(['','## Limitations and audit notes','',
        '- PAD and MASK references are artificial attended sequences, not neutral financial sentences. Neither reference is universally correct. Sensitivity does not identify a preferred baseline.',
        '- Integration is at the complete embedding output, after LayerNorm; it does not separately attribute lexical, position, and token-type components. Constant endpoint embeddings receive zero attribution even when those positions are used by the model.',
        '- Positive/negative scores support/oppose the selected target logit, not positive/negative sentiment in general. Small numerical residuals verify integration accuracy, not explanation faithfulness.',
        '- Original WordPiece records are retained. Optional word displays merge adjacent ## continuations and sum signed scores; punctuation is separate. Sum of absolute pieces and absolute signed word sum are distinct.',
        '- Maximum length is the original 64-token protocol. Truncation is recorded per example. Float32 eager attention is used for gradient compatibility and compared with prior inference; bitwise cross-device equivalence is not promised.',
        '- No human causal interpretations or attention-faithfulness conclusions are asserted. No Phase 5 comparison was performed.',
        f'- Earlier files preserved: {summary["earlier_files_preserved"]}. Any tolerated pre-existing annotation manifest difference is recorded below.',
        '', 'Warnings: '+str(summary['warnings']), '',
        'References: [Captum LayerIntegratedGradients](https://captum.ai/api/layer.html#layer-integrated-gradients), [Captum memory guidance](https://captum.ai/docs/faq), [Integrated Gradients paper](https://arxiv.org/abs/1703.01365).', ''])
    with Path(path).open('x',encoding='utf8') as f: f.write('\n'.join(lines))
