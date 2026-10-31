import hashlib
import math
import unicodedata

import numpy as np
from scipy.stats import kendalltau, spearmanr


def token_types(tokens, tokenizer):
    result=[]
    for t in tokens:
        token_id=t['token_id']; text=t['token'].removeprefix('##')
        if not t['attention_mask']: kind='padding'
        elif token_id==tokenizer.cls_token_id: kind='CLS'
        elif token_id==tokenizer.sep_token_id: kind='SEP'
        elif token_id in tokenizer.all_special_ids: kind='other_special'
        elif text and all(unicodedata.category(c).startswith('P') for c in text): kind='punctuation'
        else: kind='content'
        result.append(kind)
    return result


def correlations(ig, attention, minimum=3):
    x,y=np.asarray(ig,float),np.asarray(attention,float)
    if x.shape!=y.shape or x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Correlation vectors must be aligned, finite, one-dimensional')
    reason='too_few_tokens' if len(x)<minimum else ('constant_vector' if np.ptp(x)==0 or np.ptp(y)==0 else None)
    if reason: return {'spearman':None,'kendall_tau_b':None,'status':reason,'n':len(x)}
    return {'spearman':float(spearmanr(x,y).statistic),
            'kendall_tau_b':float(kendalltau(x,y,variant='b').statistic),'status':'valid','n':len(x)}


def ordering(scores, eligible):
    return sorted(eligible,key=lambda i:(-float(scores[i]),i))


def random_order(eligible,row_id,seed):
    stable=int.from_bytes(hashlib.sha256(f'{row_id}:{seed}'.encode()).digest()[:8],'little')
    return np.random.default_rng(stable).permutation(eligible).tolist()


def deletion_plan(ig, attention, types, row_id, cfg):
    eligible=[i for i,k in enumerate(types) if k=='content']
    signed=[t['signed'] for t in ig['tokens']]
    orders={'ig_absolute':ordering([abs(v) for v in signed],eligible),
            'attention':ordering(attention,eligible),
            'ig_positive':ordering(signed,[i for i in eligible if signed[i]>0])}
    orders.update({f'random_{seed}':random_order(eligible,row_id,seed) for seed in cfg['random_seeds']})
    plan=[]
    for method,order in orders.items():
        for fraction in cfg['deletion_fractions']:
            requested=0 if fraction==0 else min(len(eligible),math.ceil(fraction*len(eligible)))
            k=min(requested,len(order))
            plan.append({'method':method,'budget':fraction,'requested_k':requested,'k':k,
                'actual_fraction':k/len(eligible) if eligible else 0.,
                'saturated':k<requested,'positions':sorted(order[:k])})
    return plan,orders,eligible


def auc_metrics(curve):
    # Duplicate rounded budgets have identical perturbations; keep a single x.
    points={r['actual_fraction']:r for r in curve}
    x=np.array(sorted(points)); p=np.array([points[f]['target_probability'] for f in x])
    drop=np.array([points[f]['probability_drop'] for f in x]); logit=np.array([points[f]['logit_drop'] for f in x])
    integrate=lambda y:float(np.trapezoid(y,x)) if len(x)>1 else 0.
    coverage=float(x[-1]) if len(x) else 0.
    return {'probability_auc':integrate(p) if coverage==1 else None,
            'probability_drop_auc':integrate(drop) if coverage==1 else None,
            'logit_drop_auc':integrate(logit) if coverage==1 else None,
            'partial_probability_auc':integrate(p),'coverage':coverage,
            'final_flip':curve[-1]['label_flip'],'any_flip':any(r['label_flip'] for r in curve)}


def describe(values,seed=2042,replicates=2000):
    a=np.array([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(a): return {'n':0,'mean':None,'median':None,'sd':None,'mean_ci95':None}
    rng=np.random.default_rng(seed)
    # Bounded memory; sentence is the sampling unit, not token/head/random replicate.
    means=np.array([rng.choice(a,size=len(a),replace=True).mean() for _ in range(replicates)])
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'sd':float(a.std(ddof=1)) if len(a)>1 else None,
            'mean_ci95':np.quantile(means,[.025,.975]).tolist()}


def paired(a,b,**kwargs):
    if len(a)!=len(b): raise ValueError('Paired samples differ in length')
    return describe([x-y for x,y in zip(a,b) if x is not None and y is not None],**kwargs)
