import os
from pathlib import Path

from ..phase3.io import digest, read_json, verify_manifest
from ..phase4.artifacts import exclusive_json, fingerprint, read_record

IG_KEYS=('max_length','n_steps','internal_batch_size','absolute_tolerance','relative_tolerance',
         'relative_floor','random_seed','model_kind','model_seed','phase1_manifest_sha256','test_sha256')


def output_path(value):
    path=Path(value).resolve(); root=Path('artifacts/phase5').resolve()
    if root not in path.parents: raise ValueError('Outputs must be under artifacts/phase5/<run>')
    return path


def snapshot():
    paths=[]; excluded=[]
    for root in ('artifacts/phase1','artifacts/phase2','artifacts/phase3','artifacts/phase4'):
        for folder,dirs,files in os.walk(root):
            # Failed pytest's Windows ACL-protected scratch is not scientific output.
            skip=[d for d in dirs if root=='artifacts/phase4' and (d.startswith('test-temp') or d=='regression-temp')]
            excluded.extend(str(Path(folder)/d) for d in skip)
            dirs[:]=[d for d in dirs if d not in skip]
            paths.extend(Path(folder)/f for f in files)
    for root in ('src','tests','configs'):
        paths.extend(p for p in Path(root).rglob('*') if p.is_file() and p.suffix in ('.py','.json') and 'phase5' not in str(p))
    paths.extend(p for p in Path('.').glob('*.md') if 'PHASE5' not in p.name)
    paths.append(Path('pyproject.toml'))
    return {'files':{str(p):digest(p) for p in sorted(set(paths))},'excluded_pytest_scratch':excluded}


def validate_config(cfg):
    output_path(cfg['output_dir'])
    if Path(cfg['report_path']).resolve()!=Path('PHASE5_RESULTS.md').resolve(): output_path(cfg['report_path'])
    b=cfg['deletion_fractions']
    if b!=sorted(set(b)) or b[0]!=0 or b[-1]!=1 or any(not 0<=v<=1 for v in b): raise ValueError('Invalid deletion budgets')
    if len(set(cfg['random_seeds']))<2 or len(set(cfg['random_seeds']))!=len(cfg['random_seeds']): raise ValueError('Need multiple unique random seeds')
    if cfg['expected_examples']!=727 or cfg['inference_batch_size']<1 or cfg['bootstrap_samples']<100:
        raise ValueError('Invalid experiment size/batching/bootstrap settings')
    if cfg['minimum_correlation_tokens']<3: raise ValueError('Require at least three eligible tokens')


def validated_ig_sources(cfg,ig_cfg):
    sources={}
    for folder in [cfg['phase4_run'],*cfg['additional_ig_runs']]:
        root=Path(folder)
        context=read_json(root/'context.json')
        if root==Path(cfg['phase4_run']):
            verify_manifest(root,'manifest.json')
            report=read_json(root/'report_hash.json')
            if digest(report['path'])!=report['sha256']: raise ValueError('Phase 4 report hash mismatch')
        elif (root/'manifest.json').exists(): verify_manifest(root,'manifest.json')
        for key in IG_KEYS:
            if context['config'][key]!=ig_cfg[key]: raise ValueError(f'Phase 4 IG configuration mismatch: {key}')
        source=Path(__file__).parents[1]/'phase4'
        for name,checksum in context['code'].items():
            if digest(source/name)!=checksum: raise ValueError(f'Phase 4 source changed: {name}')
        for file in sorted((root/'records').glob('*-pad.json')):
            record=read_record(file,fingerprint(context))
            if record['baseline']['kind']!='pad': raise ValueError('Invalid reused reference')
            sources.setdefault(record['row_id'],(record,str(file),digest(file)))
    return sources


def check_ig(record,row,tokenizer,ig_cfg,probabilities,logits):
    import numpy as np
    from ..phase4.core import converged
    if record['row_id']!=row['row_id'] or record['sentence']!=row['sentence'] or record['true_label']!=row['label']:
        raise ValueError('IG sample identity mismatch')
    if record['predicted_label']!=int(np.argmax(probabilities)) or record['target_class']!=record['predicted_label']:
        raise ValueError('IG target/prediction mismatch')
    if not np.allclose(record['probabilities'],probabilities,atol=ig_cfg['probability_tolerance'],rtol=0) or not np.allclose(record['logits'],logits,atol=ig_cfg['logit_tolerance'],rtol=0):
        raise ValueError('IG prediction/logit mismatch')
    batch=tokenizer(row['sentence'],padding='max_length',truncation=True,max_length=ig_cfg['max_length'],return_special_tokens_mask=True)
    for i,t in enumerate(record['tokens']):
        if t['token_id']!=batch['input_ids'][i] or t['attention_mask']!=batch['attention_mask'][i]: raise ValueError('IG token alignment mismatch')
        expected=tokenizer.pad_token_id if batch['attention_mask'][i] and not batch['special_tokens_mask'][i] else batch['input_ids'][i]
        if t['baseline_token_id']!=expected: raise ValueError('IG PAD baseline mismatch')
    if len(record['tokens'])!=ig_cfg['max_length'] or record['baseline']['kind']!='pad': raise ValueError('IG length/reference mismatch')
    if record['converged']!=converged(record['attempts'][-1],ig_cfg): raise ValueError('IG convergence flag mismatch')
    if record['attempts'][-1]['n_steps'] not in ig_cfg['n_steps']: raise ValueError('IG integration settings mismatch')
    if abs(sum(t['signed'] for t in record['tokens'])-record['attempts'][-1]['attribution_sum'])>1e-5: raise ValueError('IG token sum mismatch')


def seal(output,report):
    value={'path':str(report),'sha256':digest(report)}
    if (output/'report_hash.json').exists():
        if read_json(output/'report_hash.json')!=value: raise ValueError('Resume report mismatch')
    else: exclusive_json(output/'report_hash.json',value)
    exclusive_json(output/'manifest.json',{p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'})
