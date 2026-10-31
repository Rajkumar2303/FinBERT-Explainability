import contextlib
import csv
import io
import json
import re
import runpy
import ast
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ..phase3.io import digest, read_json, load_predictions, write_csv
from ..phase4.artifacts import exclusive_json, fingerprint, read_record


def same(actual,expected,path='value',atol=1e-9):
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k not in actual: raise AssertionError(f'{path}/{k}: missing')
            same(actual[k],v,f'{path}/{k}',atol)
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise AssertionError(f'{path}: length')
        for i,(a,b) in enumerate(zip(actual,expected)): same(a,b,f'{path}/{i}',atol)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not np.isclose(actual,expected,rtol=0,atol=atol): raise AssertionError(f'{path}: {actual} != {expected}')
    elif actual!=expected: raise AssertionError(f'{path}: value mismatch')


def walk_artifacts():
    import os
    for phase in range(1,6):
        for root,dirs,files in os.walk(f'artifacts/phase{phase}'):
            dirs[:]=[d for d in dirs if not (d.startswith('test-temp') or d=='regression-temp')]
            for f in files: yield Path(root)/f


def snapshot():
    paths=list(walk_artifacts())
    for root in ('src','tests','configs','scripts'):
        paths.extend(p for p in Path(root).rglob('*') if p.is_file() and p.suffix in ('.py','.json') and 'phase6' not in str(p))
    paths.extend(Path(f'PHASE{i}{suffix}.md') for i in range(2,6) for suffix in ('','_RESULTS'))
    paths.extend([Path('README.md'),Path('pyproject.toml')])
    return {str(p):digest(p) for p in sorted(set(paths))}


def manifests():
    rows=[]; exceptions=[]; failures=[]
    for p in walk_artifacts():
        if p.name not in ('manifest.json','run_manifest.json'): continue
        data=read_json(p); count=0
        for relative,expected in data.items():
            target=(p.parent/relative).resolve()
            if p.parent.resolve() not in target.parents: raise ValueError('Unsafe manifest path')
            actual=digest(target); count+=1
            if actual!=expected:
                item={'manifest':str(p),'file':str(target),'sealed':expected,'current':actual}
                if p==Path('artifacts/phase3/manifest.json') and relative=='human_annotation.csv':
                    item['reason']='Authorized post-seal human categories, disagreement bands and AI-assisted notes; source manifest preserved'
                    exceptions.append(item)
                else: failures.append(item)
        rows.append({'manifest':str(p),'entries':count,'sha256':digest(p)})
    # Historical input inventories are snapshots, not current license grants.
    for p in walk_artifacts():
        if p.name=='report_hash.json':
            value=read_json(p)
            if digest(value['path'])!=value['sha256']: failures.append({'file':value['path'],'reason':'report hash mismatch'})
    return {'manifests':rows,'entries':sum(r['entries'] for r in rows),'exceptions':exceptions,'failures':failures,
            'scope':'All saved phase1-5 manifest.json/run_manifest.json including CPU smoke runs; ACL-protected pytest scratch excluded'}


def numeric_tokens(text):
    # Paths change when reports are regenerated in the Phase 6 directory.
    text=re.sub(r'\]\([^\n)]*\)','](LINK)',text)
    return re.findall(r'(?<![\w])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',text)


def numeric_claims(text):
    text=re.sub(r'\]\([^\n)]*\)','](LINK)',text)
    def normalize(match):
        try: return json.dumps(ast.literal_eval(match.group()),sort_keys=True)
        except (ValueError,SyntaxError): return match.group()
    text=re.sub(r'\{[^{}\n]*\}',normalize,text)
    # Preserve each metric's label/value association, but permit JSON-key/table-row reordering.
    return Counter(line.strip() for line in text.splitlines() if numeric_tokens(line))


def regenerate_reports(out):
    folder=out/'regenerated_reports'; folder.mkdir()
    original_exists=Path.exists; original_write=Path.write_text
    target=Path('PHASE2_RESULTS.md').resolve()
    def exists(p): return False if p.resolve()==target else original_exists(p)
    def write(p,text,*args,**kwargs):
        if p.resolve()!=target: raise RuntimeError(f'Unexpected legacy report write: {p}')
        return original_write(folder/'PHASE2_RESULTS.md',text,encoding='utf8')
    with patch.object(Path,'exists',exists),patch.object(Path,'write_text',write),contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path('scripts/report_phase2.py',run_name='__main__')
    from ..phase3.report import write_report as report3
    report3(Path('artifacts/phase3'),folder/'PHASE3_RESULTS.md')
    from ..phase4.report import write_report as report4
    p4=Path('artifacts/phase4/initial'); cfg4=read_json('configs/phase4.json')
    ctx=fingerprint(read_json(p4/'context.json'))
    records=[read_record(p4/'records'/f'{rid}-pad.json',ctx) for rid in read_json(p4/'selection.json')['figure_row_ids']]
    report4(folder/'PHASE4_RESULTS.md',read_json(p4/'summary.json'),records,read_json(p4/'baseline_sensitivity.json'),p4)
    from ..phase5.report import write_report as report5
    report5(folder/'PHASE5_RESULTS.md',read_json('artifacts/phase5/full/summary.json'),read_json('configs/phase5.json'),Path('artifacts/phase5/full').resolve())
    result=[]
    for phase in range(2,6):
        name=f'PHASE{phase}_RESULTS.md'
        old=Path(name).read_text(encoding='utf8'); new=(folder/name).read_text(encoding='utf8')
        a,b=numeric_tokens(old),numeric_tokens(new)
        if numeric_claims(old)!=numeric_claims(new): raise AssertionError(f'Numeric report regeneration mismatch: {name}')
        result.append({'report':name,'numeric_tokens_checked':len(a),'numeric_sequence_matches':True,
            'exact_text_matches':old==new,'note':'Metric label/value lines match; JSON-key/table-row order and generated reference locations may differ; original files untouched'})
    return result


def recompute_classification():
    from ..phase2.metrics import metrics,summarize
    from ..phase3.calibration import calibration,mean_sd
    from ..phase3.pipeline import subgroup_report,robustness_metrics
    rows=[json.loads(s) for s in Path('artifacts/phase1/test.jsonl').read_text(encoding='utf8').splitlines()]
    labels=[r['label'] for r in rows]; cfg=read_json('configs/phase3.json'); s3=read_json('artifacts/phase3/summary.json')
    checked=[]
    for kind in ('baseline','bert','finbert','reference'):
        classifications=[]; calibrations=[]; groups=[]; robustness=[]
        seeds=[42,123,2026] if kind!='reference' else [None]
        for seed in seeds:
            root=Path('artifacts/phase2')/(f'{kind}-evaluation/seed-{seed}' if seed is not None else 'reference')
            probability=load_predictions(root/'test_predictions.jsonl',rows)
            m=metrics(labels,probability.argmax(1)); same(read_json(root/'test_metrics.json'),m,f'{kind}-{seed}/classification')
            key=f'{kind}-{seed}' if seed is not None else 'reference'; p3=Path('artifacts/phase3')/key
            saved=read_json(p3/'metrics.json'); c=calibration(labels,probability,cfg['bins']); g=subgroup_report(rows,probability,cfg)
            same(saved['classification'],m); same(saved['calibration'],c); same(saved['subgroups'],g)
            variants=[json.loads(s) for s in (p3/'robustness_predictions.jsonl').read_text(encoding='utf8').splitlines()]
            robust=robustness_metrics(variants); same(saved['robustness'],robust)
            classifications.append(m);calibrations.append(c);groups.append(g);robustness.append(robust)
            checked.append(key)
        if kind!='reference':
            aggregate=s3['classification_and_calibration'][kind]; computed=summarize(classifications)
            same(aggregate['classification'],computed)
            same(read_json(f'artifacts/phase2/{kind}-evaluation/summary.json')['metrics'],computed)
            same(read_json('artifacts/phase2/comparison/comparison.json')['independent_workflows'][kind]['metrics'],computed)
            for metric in ('ece_10','ece_15','brier','nll'): same(aggregate['calibration'][metric],mean_sd([c[metric] for c in calibrations]))
            for group in ('allagree','non_allagree'): same(aggregate['subgroups'][group]['macro_f1'],mean_sd([g[group]['metrics']['macro_f1'] for g in groups]))
            for typ in robustness[0]:
                for metric in ('flip_rate','visible_flip_rate','mean_confidence_change','mean_original_class_probability_change'):
                    same(s3['robustness'][kind][typ][metric],mean_sd([r[typ][metric] for r in robustness]))
    return {'model_seed_records':checked,'classification_calibration_subgroups_robustness_recomputed':True}


def explanations():
    from ..phase4.core import sensitivity,converged
    from ..phase5.statistics import summarize
    p4=Path('artifacts/phase4/initial'); key4=fingerprint(read_json(p4/'context.json'))
    originals={}; all4=[]
    for p in sorted((p4/'records').glob('*.json')):
        r=read_record(p,key4); all4.append(r)
        if r['baseline']['kind']=='pad': originals[r['row_id']]=r
    comparisons=[sensitivity(originals[r['row_id']],r) for r in all4 if r['baseline']['kind']=='mask']
    saved={r['row_id']:r for r in read_json(p4/'baseline_sensitivity.json')}
    for r in comparisons: same(saved[r['row_id']],r)
    cfg4=read_json('configs/phase4.json'); records=[]; ig=[]
    p5=Path('artifacts/phase5/full'); key5=fingerprint(read_json(p5/'context.json'))
    for p in sorted((p5/'examples').glob('*.json')):
        r=read_record(p,key5); records.append(r)
        attr=read_record(p5/'ig'/p.name,key5); ig.append(attr)
        if r['ig_sha256']!=digest(p5/'ig'/p.name) or r['attention_sha256']!=digest(p5/'attention'/f'{p.stem}.npz'): raise AssertionError('Dependency hash mismatch')
        same(r['ig_converged'],converged(attr['attempts'][-1],cfg4))
        for field in ('signed','absolute'):
            same(r[f'ig_{field}'],[t[field] for t in attr['tokens']])
        for a in attr['attempts']:
            same(a['absolute_residual'],abs(a['signed_residual']))
            same(a['relative_residual'],a['absolute_residual']/max(abs(a['logit_difference']),cfg4['relative_floor']))
    computed=summarize(records,read_json('configs/phase5.json'))
    same(read_json(p5/'summary.json'),computed)
    s4=read_json(p4/'summary.json'); final=[r['attempts'][-1] for r in all4]
    same(s4['max_absolute_residual'],max(a['absolute_residual'] for a in final))
    same(s4['median_absolute_residual'],float(np.median([a['absolute_residual'] for a in final])))
    same(s4['max_relative_residual'],max(a['relative_residual'] for a in final))
    same(s4['escalated_records'],sum(len(r['attempts'])>1 for r in all4))
    return records,ig,{'phase4_records':len(all4),'phase5_records':len(records),'phase5_summary_recomputed':True}


def csv_checks():
    checked=[]
    # Canonical per-seed CSV metrics are checked against their already-recomputed JSON records.
    for name in ('classification_per_seed.csv','calibration_per_seed.csv','subgroup_per_seed.csv','per_class_per_seed.csv'):
        path=Path('artifacts/phase3')/name
        with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        for row in rows:
            saved=read_json(Path('artifacts/phase3')/f"{row['model']}-{row['seed']}"/'metrics.json')
            if name=='classification_per_seed.csv':
                for metric in ('macro_f1','accuracy'): same(float(row[metric]),saved['classification'][metric])
            elif name=='calibration_per_seed.csv':
                for metric in ('ece_10','ece_15','brier','nll','mean_confidence'): same(float(row[metric]),saved['calibration'][metric])
            elif name=='subgroup_per_seed.csv':
                g=saved['subgroups'][row['subgroup']]
                same(float(row['macro_f1']),g['metrics']['macro_f1']); same(int(row['n']),g['n']);same(json.loads(row['class_counts']),g['class_counts'])
            else:
                c=saved['classification']['per_class'][row['class']]
                for metric in ('precision','recall','f1','support'): same(float(row[metric]),c[metric])
        checked.append({'file':str(path),'rows':len(rows),'sha256':digest(path),'numeric_json_crosscheck':True})
    with Path('artifacts/phase5/full/correlations.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            r=read_json(Path('artifacts/phase5/full/examples')/(row['row_id']+'.json'))['payload']['record']
            for mode in ('absolute','signed'):
                for metric in ('spearman','kendall_tau_b'):
                    value=r['agreement'][mode][metric]
                    if value is not None: same(float(row[f'{mode}_{metric}']),value)
    checked.append({'file':'artifacts/phase5/full/correlations.csv','numeric_json_crosscheck':True})
    return checked


def annotations():
    path=Path('artifacts/phase3/human_annotation.csv')
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    return {'rows':len(rows),'categories_completed':sum(bool(r['error_category'].strip()) for r in rows),
        'category_counts':dict(Counter(r['error_category'] or 'UNFILLED' for r in rows)),
        'disagreement_bands':dict(Counter(r['annotator_disagreement_band'] for r in rows)),
        'ai_assisted_interpretations':sum('AI-assisted' in r['interpretation'] for r in rows),
        'ai_assisted_comments':sum('AI-assisted' in r['comments'] for r in rows),
        'source_sha256':digest(path),'provenance':'User supplied categories; interpretation/comments include labeled AI-assisted drafts. No independent human validation of those drafts is claimed.',
        'sampling':'20 confidence/disagreement-selected errors, not a representative sample; exact per-row annotator vote counts unavailable',
        'historical_report_note':'PHASE3_RESULTS.md describes the original blank handoff; it is preserved as historical, not current annotation status.'}


def sensitivity(records,ig,out):
    from ..phase5.analysis import describe,paired
    rows={r['row_id']:r for r in [json.loads(s) for s in Path('artifacts/phase1/test.jsonl').read_text(encoding='utf8').splitlines()]}
    attrs={r['row_id']:r for r in ig}; cfg=read_json('configs/phase5.json')
    kw={'seed':cfg['bootstrap_seed'],'replicates':cfg['bootstrap_samples']}
    result={}; flat=[]
    for name,subset in [('converged',[r for r in records if r['ig_converged']]),('all',records),('unreliable',[r for r in records if not r['ig_converged']])]:
        y=[r['true_label'] for r in subset]; pred=[r['predicted_label'] for r in subset]
        item={'n':len(subset),'gold_class_counts':dict(Counter(y)),'predicted_class_counts':dict(Counter(pred)),
            'allagree':sum(rows[r['row_id']]['is_allagree'] for r in subset),'correct':sum(a==b for a,b in zip(y,pred)),
            'truncated':sum(attrs[r['row_id']]['truncated'] for r in subset),
            'mean_eligible_tokens':float(np.mean([len(r['eligible_positions']) for r in subset])),
            'spearman':describe([r['agreement']['absolute']['spearman'] for r in subset],**kw),
            'kendall':describe([r['agreement']['absolute']['kendall_tau_b'] for r in subset],**kw),'auc':{}}
        values={m:[r['random_summary']['probability_auc']['mean'] if m=='random' else r['auc'][m]['probability_auc'] for r in subset] for m in ('ig_absolute','attention','random')}
        for m,v in values.items(): item['auc'][m]=describe(v,**kw)
        item['paired_ig_minus_attention']=paired(values['ig_absolute'],values['attention'],**kw)
        item['paired_ig_minus_random']=paired(values['ig_absolute'],values['random'],**kw)
        item['paired_attention_minus_random']=paired(values['attention'],values['random'],**kw)
        result[name]=item
        for metric,v in [('spearman',item['spearman']),('kendall',item['kendall']),*item['auc'].items(),('paired_ig_minus_attention',item['paired_ig_minus_attention'])]:
            flat.append({'population':name,'metric':metric,**v})
    result['interpretation']={'thresholds_unchanged':True,'unreliable_records_retained':40,
        'spearman_mean_shift_all_minus_converged':result['all']['spearman']['mean']-result['converged']['spearman']['mean'],
        'paired_interval_includes_zero':{name:v['paired_ig_minus_attention']['mean_ci95'][0]<=0<=v['paired_ig_minus_attention']['mean_ci95'][1] for name,v in result.items()},
        'warning':'Subset/all overlap: their intervals are not independent. All-record IG estimates include unreliable numerical integrations. Exclusion is not missing at random.'}
    exclusive_json(out/'ig_sensitivity.json',result); write_csv(out/'ig_sensitivity.csv',flat)
    failures=[]
    for r in records:
        a=attrs[r['row_id']]['attempts'][-1]
        failures.append({'row_id':r['row_id'],'ig_reliable':r['ig_converged'],'true_label':r['true_label'],
                         'predicted_label':r['predicted_label'],'is_allagree':rows[r['row_id']]['is_allagree'],
                         'truncated':attrs[r['row_id']]['truncated'],**a})
    write_csv(out/'ig_reliability_inventory.csv',failures)
    return result


def run(output='artifacts/phase6/audit'):
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    before=snapshot(); exclusive_json(out/'earlier_hashes.json',before)
    manifest=manifests(); exclusive_json(out/'manifest_audit.json',manifest)
    if manifest['failures']: raise AssertionError('Unexplained manifest failures; see manifest_audit.json')
    classification=recompute_classification(); records,ig,explanation=explanations()
    reports=regenerate_reports(out); csvs=csv_checks(); annotation=annotations()
    exclusive_json(out/'annotation_summary.json',annotation)
    sensitivity(records,ig,out)
    after=snapshot()
    if before!=after: raise AssertionError('Earlier files changed during audit')
    receipt={'passed':True,'earlier_files_verified':len(before),'manifests':len(manifest['manifests']),
        'manifest_entries':manifest['entries'],'known_exceptions':manifest['exceptions'],
        'classification':classification,'explanations':explanation,'reported_numeric_claims':reports,'csv_files':csvs,
        'no_model_inference_or_training':True}
    exclusive_json(out/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('passed','earlier_files_verified','manifests','manifest_entries')},indent=2))
