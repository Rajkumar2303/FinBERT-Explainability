from collections import Counter

import numpy as np

from .analysis import describe, paired


def summarize(records,cfg):
    options={'seed':cfg['bootstrap_seed'],'replicates':cfg['bootstrap_samples']}
    d=lambda x:describe(x,**options)
    valid=[r for r in records if r['ig_converged']]
    summary={'n':len(records),'converged_n':len(valid),
        'convergence_failures':[r['row_id'] for r in records if not r['ig_converged']],
        'uncertainty':'2000 fixed-seed percentile bootstrap resamples of sentences; paired differences resample sentence differences; intervals descriptive, not multiplicity-adjusted',
        'agreement':{},'agreement_all_records':{},'correlation_status':{},'auc':{},'auc_all_records':{},'paired_auc_differences':{},
        'token_type_mass':{},'head_token_type_mass':{},'curves':{},'positive_coverage':d([r['auc']['ig_positive']['coverage'] for r in valid])}
    for mode in ('absolute','signed'):
        summary['agreement'][mode]={metric:d([r['agreement'][mode][metric] for r in valid]) for metric in ('spearman','kendall_tau_b')}
        summary['agreement_all_records'][mode]={metric:d([r['agreement'][mode][metric] for r in records]) for metric in ('spearman','kendall_tau_b')}
        summary['correlation_status'][mode]=dict(Counter(r['agreement'][mode]['status'] for r in records))
    for method in ('ig_absolute','attention','random'):
        get=lambda r,metric: r['random_summary'].get(metric,{}).get('mean') if method=='random' else r['auc'][method][metric]
        summary['auc'][method]={metric:d([get(r,metric) for r in valid]) for metric in ('probability_auc','probability_drop_auc','logit_drop_auc','any_flip','final_flip')}
        summary['auc_all_records'][method]={metric:d([get(r,metric) for r in records]) for metric in ('probability_auc','probability_drop_auc','logit_drop_auc')}
    summary['random_within_example_auc_sd']=d([r['random_summary']['probability_auc']['sd'] for r in valid if 'probability_auc' in r['random_summary']])
    for method in ('attention','random'):
        a=[r['auc']['ig_absolute']['probability_auc'] for r in valid]
        b=[r['auc']['attention']['probability_auc'] if method=='attention' else r['random_summary'].get('probability_auc',{}).get('mean') for r in valid]
        summary['paired_auc_differences'][f'ig_absolute_minus_{method}']=paired(a,b,**options)
    summary['paired_auc_differences']['attention_minus_random']=paired(
        [r['auc']['attention']['probability_auc'] for r in valid],
        [r['random_summary'].get('probability_auc',{}).get('mean') for r in valid],**options)
    for kind in ('CLS','SEP','punctuation','content','other_special','padding'):
        summary['token_type_mass'][kind]=d([float(np.mean(r['head_token_type_mass'][kind])) for r in records])
        values=np.array([r['head_token_type_mass'][kind] for r in records])
        summary['head_token_type_mass'][kind]={'mean':values.mean(0).tolist(),'sd':values.std(0,ddof=1).tolist() if len(values)>1 else None}
    for method in ('ig_absolute','attention','random','ig_positive'):
        curve=[]
        for budget in cfg['deletion_fractions']:
            per_example=[]; within=[]
            for r in valid:
                points=[p for p in r['deletion'] if p['budget']==budget and (p['method'].startswith('random_') if method=='random' else p['method']==method)]
                per_example.append({metric:float(np.mean([p[metric] for p in points])) for metric in ('target_probability','probability_drop','logit_drop','label_flip','actual_fraction')})
                within.append(float(np.std([p['target_probability'] for p in points],ddof=1)) if len(points)>1 else 0.)
            curve.append({'budget':budget,**{metric:d([p[metric] for p in per_example]) for metric in ('target_probability','probability_drop','logit_drop','label_flip','actual_fraction')},
                          'mean_within_random_probability_sd':float(np.mean(within)) if within else None})
        summary['curves'][method]=curve
    return summary
