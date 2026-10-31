import json
import platform
import sys
import time
import traceback
from pathlib import Path

import captum
import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..phase3.io import digest,read_json,verify_manifest,write_csv
from ..phase4.artifacts import exclusive_json,fingerprint,load_verified,read_record,save_record
from ..phase4.core import deterministic,embedding_layer,explain
from ..phase4.pipeline import check_inference
from .artifacts import check_ig,output_path,seal,snapshot,validate_config,validated_ig_sources
from .inference import attention_and_deletion
from .statistics import summarize


def run(cfg,resume=False,smoke=False):
    start=time.perf_counter(); validate_config(cfg)
    if Path(sys.executable).parent.name.lower()!='rag-env': raise RuntimeError('Use rag-env Python')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    output=output_path(cfg['output_dir']); report=Path(cfg['report_path'])
    if output.exists() and any(output.iterdir()) and not resume: raise FileExistsError(output)
    if report.exists() and not resume: raise FileExistsError(report)
    output.mkdir(parents=True,exist_ok=True)
    print('Verifying preservation snapshot and sealed Phase 4 sources...',flush=True)
    before=snapshot()
    ig_cfg=read_json(cfg['phase4_config'])
    if ig_cfg['model_kind']!='finbert' or ig_cfg['model_seed']!=42: raise ValueError('Primary checkpoint must be fixed FinBERT seed 42')
    deterministic(ig_cfg['random_seed'])
    env={'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,
         'transformers':transformers.__version__,'captum':captum.__version__,'numpy':np.__version__,
         'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
    context={'config':cfg,'ig_config':ig_cfg,'smoke':smoke,'environment':env,'earlier':before,
             'code':{p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    key=fingerprint(context)
    if (output/'context.json').exists():
        if read_json(output/'context.json')!=context: raise ValueError('Resume input/config/source/environment mismatch')
    else: exclusive_json(output/'context.json',context)
    if (output/'manifest.json').exists():
        verify_manifest(output,'manifest.json'); saved=read_json(output/'report_hash.json')
        if digest(saved['path'])!=saved['sha256']: raise ValueError('Report checksum mismatch')
        print('Completed run verified; nothing overwritten.',flush=True)
        return
    try:
        sources=validated_ig_sources(cfg,ig_cfg)
        rows,checkpoint,probabilities,logits,warnings=load_verified(ig_cfg)
        if len(rows)!=cfg['expected_examples']: raise ValueError('Expected exactly 727 frozen test examples')
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager')
        embedding_layer(model); model.requires_grad_(False).cuda().eval()
        if model.config.num_hidden_layers!=12 or model.config.num_attention_heads!=12: raise ValueError('Expected 12-layer/12-head FinBERT')
        torch.cuda.reset_peak_memory_stats()
        print('Checking all frozen predictions before attention extraction...',flush=True)
        _,pdiff,ldiff=check_inference(model,tokenizer,rows,probabilities,logits,ig_cfg,'cuda')
        selected=read_json(Path(cfg['phase4_run'])/'selection.json')['figure_row_ids'] if smoke else [r['row_id'] for r in rows]
        indices=[i for i,r in enumerate(rows) if r['row_id'] in set(selected)]
        if smoke and len(indices)!=10: raise ValueError('Smoke requires exactly ten Phase 4 examples')
        for sub in ('ig','examples','attention','figures'): (output/sub).mkdir(exist_ok=True)
        records=[]; ig_records=[]; newly_computed=0; reused_external=0; resumed=0
        for j,i in enumerate(indices):
            row=rows[i]; row_id=row['row_id']; ig_path=output/'ig'/f'{row_id}.json'
            if ig_path.exists(): ig=read_record(ig_path,key)
            else:
                if row_id in sources:
                    source,path,checksum=sources[row_id]; ig=dict(source)
                    ig['phase5_reuse']={'source':path,'sha256':checksum}; reused_external+=1
                else:
                    ig=explain(model,tokenizer,row,ig_cfg,'pad')
                    ig.update(configuration=ig_cfg,checkpoint=str(checkpoint)); newly_computed+=1
                check_ig(ig,row,tokenizer,ig_cfg,probabilities[i],logits[i])
                save_record(ig_path,ig,key)
            check_ig(ig,row,tokenizer,ig_cfg,probabilities[i],logits[i]); ig_records.append(ig)
            example=output/'examples'/f'{row_id}.json'; attention_file=output/'attention'/f'{row_id}.npz'
            if example.exists():
                result=read_record(example,key)
                if result['attention_sha256']!=digest(attention_file) or result['ig_sha256']!=digest(ig_path): raise ValueError('Resumed example dependency mismatch')
                resumed+=1
            else:
                result=attention_and_deletion(model,tokenizer,row,ig,cfg,ig_cfg,attention_file,(12,12))
                result.update(attention_sha256=digest(attention_file),ig_sha256=digest(ig_path))
                save_record(example,result,key)
            records.append(result)
            message={'completed':j+1,'total':len(indices),'row_id':row_id,'ig_converged':ig['converged'],
                'ig_steps':ig['attempts'][-1]['n_steps'],'absolute_spearman':result['agreement']['absolute']['spearman'],
                'elapsed_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            print(json.dumps(message),flush=True)
            with (output/'progress.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(message)+'\n')
        print('Aggregating sentence-level statistics and paired bootstrap intervals...',flush=True)
        summary=summarize(records,cfg)
        summary.update(checkpoint=str(checkpoint),verified_test_predictions=len(rows),probability_max_difference=pdiff,
            logit_max_difference=ldiff,environment=env,smoke=smoke,warnings=warnings,
            ig_reused_from_phase4=sum('phase5_reuse' in r for r in ig_records),
            ig_newly_computed_this_session=newly_computed,examples_resumed_this_session=resumed,
            ig_total_recorded_seconds=sum(r['runtime_seconds'] for r in ig_records),
            attention_deletion_seconds=sum(r['runtime_seconds'] for r in records),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            truncated_examples=sum(r['truncated'] for r in ig_records),
            ig_steps_histogram={str(s):sum(r['attempts'][-1]['n_steps']==s for r in ig_records) for s in ig_cfg['n_steps']})
        from .report import render,write_report
        if not (output/'correlations.csv').exists():
            write_csv(output/'correlations.csv',[{'row_id':r['row_id'],'ig_converged':r['ig_converged'],
                'eligible_tokens':len(r['eligible_positions']),**{f'{mode}_{k}':v for mode,values in r['agreement'].items() for k,v in values.items()}} for r in records])
        if not (output/'deletion.csv').exists(): write_csv(output/'deletion.csv',[{'row_id':r['row_id'],'ig_converged':r['ig_converged'],**p} for r in records for p in r['deletion']])
        if not (output/'auc.csv').exists(): write_csv(output/'auc.csv',[{'row_id':r['row_id'],'ig_converged':r['ig_converged'],'method':m,**v} for r in records for m,v in r['auc'].items()])
        if not (output/'attention_mass.csv').exists(): write_csv(output/'attention_mass.csv',[{'row_id':r['row_id'],'head':h,'type':kind,'mass':mass} for r in records for kind,values in r['head_token_type_mass'].items() for h,mass in enumerate(values)])
        if not (output/'ig_convergence.csv').exists(): write_csv(output/'ig_convergence.csv',[{'row_id':r['row_id'],**attempt} for r in ig_records for attempt in r['attempts']])
        render(records,summary,cfg,output)
        print('Verifying earlier files remain unchanged...',flush=True)
        if snapshot()!=before: raise RuntimeError('Earlier-file preservation failed')
        summary.update(earlier_files_preserved=True,earlier_file_count=len(before['files']),wall_seconds=time.perf_counter()-start,
            timing_note='Wall time is this invocation. Per-record runtimes include reused work. Initial Phase 4 records are not recomputed.')
        if not (output/'summary.json').exists(): exclusive_json(output/'summary.json',summary)
        else: summary=read_json(output/'summary.json')
        if not report.exists(): write_report(report,summary,cfg,output)
        seal(output,report)
        print(json.dumps({k:summary[k] for k in ('n','converged_n','ig_reused_from_phase4','wall_seconds','peak_allocated_bytes','earlier_files_preserved')},indent=2),flush=True)
    except Exception as error:
        exclusive_json(output/f'failure-{time.time_ns()}.json',{'error':str(error),'traceback':traceback.format_exc(),
            'earlier_files_preserved':snapshot()==before})
        raise
