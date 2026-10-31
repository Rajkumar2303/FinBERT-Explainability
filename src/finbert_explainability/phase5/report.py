import textwrap
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..phase4.artifacts import exclusive_json


def save(fig,path):
    fig.tight_layout()
    if not path.exists(): fig.savefig(path,dpi=150)
    plt.close(fig)


def heatmap(r,path):
    pos=r['eligible_positions']
    if not pos: return
    arrays=np.array([[r[k][i] for i in pos] for k in ('ig_absolute','attention','ig_signed')]).T
    scales=np.max(np.abs(arrays),axis=0); scales[scales==0]=1
    fig,ax=plt.subplots(figsize=(8,max(4,len(pos)*.24)))
    im=ax.imshow(arrays/scales,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    ax.set_yticks(range(len(pos)),[f'{i}: {r["tokens"][i]}' for i in pos],fontsize=8)
    ax.set_xticks(range(3),['Absolute IG','Attention','Signed IG'])
    ax.set_title(f'{r["row_id"]}: true={r["true_label"]}, predicted={r["predicted_label"]}\n'+
                 '\n'.join(textwrap.wrap(r['sentence'],85)),fontsize=9)
    fig.colorbar(im,ax=ax,label='Each column divided by its own maximum absolute score')
    save(fig,path)


def render(records,s,cfg,output):
    figures=output/'figures'; valid=[r for r in records if r['ig_converged']]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,('spearman','kendall_tau_b')):
        for mode in ('absolute','signed'):
            vals=[r['agreement'][mode][metric] for r in valid if r['agreement'][mode][metric] is not None]
            ax.hist(vals,bins=np.linspace(-1,1,21),alpha=.55,label=mode+' IG')
        ax.set_xlabel(metric); ax.set_ylabel('Sentences'); ax.legend()
    save(fig,figures/'correlations.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for method in ('ig_absolute','attention','random'):
        curve=s['curves'][method]; x=[p['budget'] for p in curve]
        for ax,metric in zip(axes,('target_probability','logit_drop','label_flip')):
            y=[p[metric]['mean'] for p in curve]; ci=np.array([p[metric]['mean_ci95'] for p in curve])
            if any(v is None for v in y): continue
            ax.plot(x,y,label=method); ax.fill_between(x,ci[:,0],ci[:,1],alpha=.15)
            ax.set_xlabel('Requested fraction of eligible tokens masked'); ax.set_ylabel(metric); ax.legend()
    save(fig,figures/'deletion_curves.png')
    fig,ax=plt.subplots(figsize=(7,4))
    for method in ('ig_positive','ig_absolute','attention','random'):
        curve=s['curves'][method]
        ax.plot([p['actual_fraction']['mean'] for p in curve],[p['target_probability']['mean'] for p in curve],marker='.',label=method)
    ax.set_xlabel('Mean actual masked fraction (positive-only curve is capped)'); ax.set_ylabel('Mean original-class probability'); ax.legend()
    save(fig,figures/'positive_ig_sensitivity.png')
    fig,ax=plt.subplots(figsize=(7,4))
    data=[[r['auc'][method]['probability_auc'] for r in valid if r['auc'][method]['probability_auc'] is not None] for method in ('ig_absolute','attention')]
    data.append([r['random_summary']['probability_auc']['mean'] for r in valid if 'probability_auc' in r['random_summary']])
    ax.boxplot(data,tick_labels=['Absolute IG','Attention','Random mean'],showfliers=False)
    ax.set_ylabel('Original-class probability deletion AUC (lower = larger drop)')
    save(fig,figures/'deletion_auc.png')
    fig,ax=plt.subplots(figsize=(8,4))
    names=list(s['paired_auc_differences']); means=[s['paired_auc_differences'][n]['mean'] for n in names]
    ci=np.array([s['paired_auc_differences'][n]['mean_ci95'] for n in names])
    if all(v is not None for v in means):
        ax.errorbar(means,range(len(names)),xerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt='o')
    ax.set_yticks(range(len(names)),names); ax.axvline(0,color='grey'); ax.set_xlabel('Paired mean probability AUC difference; 95% bootstrap CI')
    save(fig,figures/'paired_auc.png')
    kinds=['CLS','SEP','punctuation','content','other_special']
    fig,ax=plt.subplots(figsize=(8,4))
    ax.boxplot([[np.mean(r['head_token_type_mass'][k]) for r in records] for k in kinds],tick_labels=kinds,showfliers=False)
    ax.set_ylabel('Final-layer CLS attention mass, head mean')
    save(fig,figures/'attention_token_types.png')
    fig,ax=plt.subplots(figsize=(8,4))
    mat=np.array([s['head_token_type_mass'][k]['mean'] for k in kinds])
    im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1,cmap='Blues'); ax.set_yticks(range(len(kinds)),kinds)
    ax.set_xlabel('Head (zero-based)'); fig.colorbar(im,ax=ax,label='Mean mass across sentences')
    save(fig,figures/'attention_heads.png')
    fig,ax=plt.subplots(figsize=(7,4))
    usable=[r for r in valid if r['agreement']['absolute']['spearman'] is not None and r['auc']['ig_absolute']['probability_auc'] is not None]
    ax.scatter([r['agreement']['absolute']['spearman'] for r in usable],
        [r['auc']['ig_absolute']['probability_auc']-r['auc']['attention']['probability_auc'] for r in usable],s=12,alpha=.5)
    ax.axhline(0,color='grey'); ax.set_xlabel('Absolute IG–attention Spearman'); ax.set_ylabel('IG minus attention probability AUC')
    save(fig,figures/'example_disagreements.png')
    ordered=sorted(records,key=lambda r:r['row_id'])
    count=cfg['representative_examples']//2
    representatives=[r for r in ordered if r['true_label']==r['predicted_label']][:count]+[r for r in ordered if r['true_label']!=r['predicted_label']][:count]
    disagreements=sorted(usable,key=lambda r:(r['agreement']['absolute']['spearman'],r['row_id']))[:2]
    for r in {r['row_id']:r for r in representatives+disagreements}.values(): heatmap(r,figures/f'{r["row_id"]}.png')
    if not (output/'figure_selection.json').exists(): exclusive_json(output/'figure_selection.json',{
        'representative_policy':'First three correct and first three incorrect row IDs; no attribution-based selection',
        'representative_ids':[r['row_id'] for r in representatives],
        'disagreement_policy':'Two lowest absolute Spearman values among converged eligible examples; descriptive extremes, not representative',
        'disagreement_ids':[r['row_id'] for r in disagreements]})


def fmt(v):
    return 'undefined' if v is None else f'{v:.4f}'


def stats_row(name,v):
    ci=v['mean_ci95']; interval='undefined' if ci is None else f'[{fmt(ci[0])}, {fmt(ci[1])}]'
    return f'| {name} | {v["n"]} | {fmt(v["mean"])} | {fmt(v["median"])} | {fmt(v["sd"])} | {interval} |'


def write_report(path,s,cfg,output):
    figure_base=Path(os.path.relpath(output/'figures',Path(path).resolve().parent)).as_posix()
    lines=['# Phase 5 measured results','',
        f'Run type: {"10-example CUDA smoke" if s["smoke"] else "full frozen test experiment"}. Checkpoint: `{s["checkpoint"]}`; fixed seed 42. No model retraining, retuning, or weight changes.', '',
        f'Processed {s["n"]} sentences; all {s["verified_test_predictions"]} frozen predictions verified. Maximum probability/logit differences: {s["probability_max_difference"]:.3g}/{s["logit_max_difference"]:.3g}. Reused {s["ig_reused_from_phase4"]} Phase 4 IG records. Converged IG: {s["converged_n"]}/{s["n"]}.', '',
        f'Integration-step counts: {s["ig_steps_histogram"]}. Unresolved convergence row IDs: {s["convergence_failures"]}. All records, including failures, are retained. Primary IG comparisons use the converged subset; all-record sensitivity summaries are in summary.json. Truncated examples: {s["truncated_examples"]}.', '',
        f'Wall time: {s["wall_seconds"]:.2f} s. Recorded IG time (includes reused work): {s["ig_total_recorded_seconds"]:.2f} s. Attention/deletion time: {s["attention_deletion_seconds"]:.2f} s. Peak allocated/reserved CUDA memory: {s["peak_allocated_bytes"]/2**30:.3f}/{s["peak_reserved_bytes"]/2**30:.3f} GiB. Earlier files preserved: {s["earlier_files_preserved"]} ({s["earlier_file_count"]} files).', '',
        '## Attribution agreement','',
        'Final-layer CLS-to-token attention is averaged across all 12 heads. Absolute and signed IG comparisons use identical content WordPieces. Padding, all special token IDs, and tokens whose characters (after removing ##) are entirely Unicode punctuation are excluded. Currency/math symbols and numeric WordPieces remain content. Ties use average ranks for Spearman and tau-b for Kendall. Constant vectors and fewer than three eligible tokens return undefined, never zero.', '',
        '| Comparison | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |','|---|---:|---:|---:|---:|---|']
    for mode,metrics in s['agreement'].items():
        for metric,v in metrics.items(): lines.append(stats_row(f'{mode} IG: {metric}',v))
    lines += ['',f'Correlation status counts (all records): {s["correlation_status"]}.', '',
              f'![Correlation distributions]({figure_base}/correlations.png)', '',
              '## MASK deletion interventions','',
              'Rankings and the predicted-class target are fixed from unmodified input. Budgets use ceil(fraction × eligible count), with repeated counts deduplicated for inference. Masks preserve positions and the original attention mask. AUC integrates the actual masked fraction using the trapezoid rule. Lower probability AUC means larger suppression under this intervention; negative drops are retained. Random results average five fixed orderings per sentence before statistics.', '',
              '| Method: probability AUC | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |','|---|---:|---:|---:|---:|---|']
    for method,v in s['auc'].items(): lines.append(stats_row(method,v['probability_auc']))
    lines += ['','| Paired AUC difference | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |','|---|---:|---:|---:|---:|---|']
    for method,v in s['paired_auc_differences'].items(): lines.append(stats_row(method,v))
    for name,v in s['paired_auc_differences'].items():
        if v['mean_ci95'] is None: continue
        lo,hi=v['mean_ci95']
        direction='below zero' if hi<0 else ('above zero' if lo>0 else 'spans zero')
        lines.append(f'\nFor {name}, the paired interval is {direction}. This describes the specified MASK intervention and does not establish general explanation fidelity.')
    lines += ['',f'Mean within-sentence random-order probability-AUC SD: {fmt(s["random_within_example_auc_sd"]["mean"])}.', '',
              '| Method | Any-budget label flip fraction | Final-budget flip fraction | Mean logit-drop AUC |','|---|---:|---:|---:|']
    for method,v in s['auc'].items(): lines.append(f'| {method} | {fmt(v["any_flip"]["mean"])} | {fmt(v["final_flip"]["mean"])} | {fmt(v["logit_drop_auc"]["mean"])} |')
    lines += ['',f'Positive-only IG masks strictly positive scores and stops when exhausted. Mean maximum eligible-token coverage: {fmt(s["positive_coverage"]["mean"])}. Partial AUCs are exported but are not ranked against complete-coverage AUCs. Its curve is a sensitivity analysis, not a matched-budget winner comparison.', '',
              f'![Deletion curves]({figure_base}/deletion_curves.png)', '',f'![AUC distributions]({figure_base}/deletion_auc.png)', '',f'![Paired AUC intervals]({figure_base}/paired_auc.png)', '',
              f'![Positive-only sensitivity]({figure_base}/positive_ig_sensitivity.png)', '',
              '## Attention mass','', '| Token type | N | Mean | Median | Sample SD | Mean 95% bootstrap CI |','|---|---:|---:|---:|---:|---|']
    for kind,v in s['token_type_mass'].items(): lines.append(stats_row(kind,v))
    lines += ['',f'![Attention token types]({figure_base}/attention_token_types.png)', '',f'![Head means]({figure_base}/attention_heads.png)', '',
              'Mass is measured before content filtering, with special tokens retained. The query is CLS; padding mass is checked separately. All-layer, all-head raw CLS rows are stored per sentence. Special-token mass alone does not establish that attention is unfaithful: these representations can carry contextual information.', '',
              '## Examples and disagreements','',f'![Example-level differences]({figure_base}/example_disagreements.png)', '',
              'Token heatmaps are in figures/ and listed in figure_selection.json. Representative examples use the first three correct and first three incorrect row IDs; the two lowest-correlation cases are separately labeled descriptive extremes. Each heatmap column has its own scale, so it displays relative patterns rather than equating attention probability and logit attribution units.', '',
              '## Interpretation against the literature','',
              '[Jain and Wallace (2019)](https://aclanthology.org/N19-1357/) question attention explanations using gradient agreement and alternative attention distributions with similar outputs. Our rank correlations and MASK tests address related diagnostics, but we do not reproduce their attention-distribution interventions. This BERT checkpoint and final-layer CLS summary differ from the attention mechanisms in their experiments. Correlation with IG alone cannot establish faithfulness.', '',
              '[Wiegreffe and Pinter (2019)](https://aclanthology.org/D19-1002/) argue that the answer depends on the definition of explanation and model context, proposing uniform-attention, random-seed, diagnostic, and adversarial-training tests. Our random token-deletion baseline and paired uncertainty provide narrower controls, not those tests. We do not train adversarial models or evaluate attention variability across model seeds. These measurements cannot prove that attention is never useful as an explanation.', '',
              '## Uncertainty and limitations','',
              f'- Percentile bootstrap: {cfg["bootstrap_samples"]} resamples, seed {cfg["bootstrap_seed"]}, sampling sentences with replacement. Paired comparisons resample within-sentence differences. Random-order replicates and heads are not treated as independent examples. Curve intervals are pointwise; no multiplicity adjustment is made.',
              '- The frozen benchmark is not a random sample of all financial language. Shared topics and phrasing can create dependence; sentence bootstrap intervals do not eliminate that limitation. One fixed model seed and one attention aggregation limit generalizability.',
              '- MASK replacement introduces distribution shift. Correlations and deletion AUC are diagnostic evidence, not ground-truth causal fidelity. Absolute IG can prioritize inhibitory tokens; removing them may increase confidence. Positive-only sensitivity has different coverage.',
              '- The original Phase 4 PAD reference and 50/100/200-step schedule remain fixed. Baseline sensitivity was observed in Phase 4. No test finding changes model or attribution settings. Numerical failures are retained and disclosed.',
              '- WordPiece deletion can remove part of a word. Punctuation filtering may retain numeric context that matters. The 64-token truncation protocol is unchanged.',
              '- Attention extraction uses eager attention in eval mode. The 12×12 CLS rows are retained across layers; complete query-key matrices are optional and disabled to bound disk use. No large tensor collection is kept on the GPU.',
              f'- Existing input warnings: {s["warnings"]}. The authorized Phase 3 annotation edits are preserved without rewriting its original manifest.',
              '- No Phase 6 report or GitHub publication work was performed.', '']
    with Path(path).open('x',encoding='utf8') as f: f.write('\n'.join(lines))
