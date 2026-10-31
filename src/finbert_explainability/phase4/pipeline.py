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

from ..phase3.io import digest, read_json, write_csv
from .artifacts import (exclusive_json, fingerprint, load_verified, phase4_path,
                        preservation_snapshot, read_record, save_record, selection, validate_config)
from .core import deterministic, embedding_layer, explain, sensitivity
from .report import figure, gallery, write_report


def check_inference(model, tokenizer, rows, previous, previous_logits, cfg, device):
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(rows),8):
            inputs=tokenizer([r['sentence'] for r in rows[start:start+8]],padding=True,truncation=True,
                             max_length=cfg['max_length'],return_tensors='pt').to(device)
            chunks.append(model(**inputs).logits.cpu().numpy())
    logits=np.concatenate(chunks)
    exp=np.exp(logits.astype(float)-logits.max(1,keepdims=True)); p=exp/exp.sum(1,keepdims=True)
    pdiff=float(np.max(np.abs(p-previous))); ldiff=float(np.max(np.abs(logits-previous_logits)))
    if not np.isfinite(logits).all() or pdiff>cfg['probability_tolerance'] or ldiff>cfg['logit_tolerance'] or not np.array_equal(p.argmax(1),previous.argmax(1)):
        raise ValueError(f'Frozen prediction reproduction failed: probability diff={pdiff}, logit diff={ldiff}')
    return p,pdiff,ldiff


def run(cfg,cpu=False,full_test=False,resume=False):
    started=time.perf_counter()
    validate_config(cfg)
    output=phase4_path(cfg['output_dir']); report=Path(cfg['report_path'])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'Existing run; choose a new Phase 4 directory or --resume: {output}')
    if report.exists() and not resume:
        raise FileExistsError(f'Report exists: {report}')
    output.mkdir(parents=True,exist_ok=True)
    device='cpu' if cpu else 'cuda'
    if not cpu and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if Path(sys.executable).parent.name.lower()!='rag-env': raise RuntimeError('Use the existing rag-env interpreter')
    deterministic(cfg['random_seed'])
    print('Snapshotting earlier artifacts and verifying frozen inputs...',flush=True)
    before=preservation_snapshot()
    code={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    environment={'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,
        'captum':captum.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0) if not cpu else 'cpu'}
    context_data={'config':cfg,'full_test':full_test,'device':device,'code':code,'earlier_hashes':before,'environment':environment}
    context=fingerprint(context_data)
    if (output/'context.json').exists():
        if read_json(output/'context.json')!=context_data: raise ValueError('Resume context differs from original run')
    else: exclusive_json(output/'context.json',context_data)
    if (output/'manifest.json').exists():
        from ..phase3.io import verify_manifest
        verify_manifest(output,'manifest.json')
        sealed_report=read_json(output/'report_hash.json')
        if digest(sealed_report['path'])!=sealed_report['sha256']:
            raise ValueError('Completed report hash mismatch')
        print('Completed run verified; nothing overwritten.',flush=True)
        return read_json(output/'summary.json')
    try:
        rows,checkpoint,previous,previous_logits,warnings=load_verified(cfg)
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager')
        embedding_layer(model)
        model.requires_grad_(False).to(device).eval()
        if not cpu: torch.cuda.reset_peak_memory_stats()
        print('Reproducing all frozen test predictions...',flush=True)
        p,pdiff,ldiff=check_inference(model,tokenizer,rows,previous,previous_logits,cfg,device)
        correct,incorrect=selection(rows,p,cfg['examples_per_outcome'])
        selected=correct+incorrect
        attribution_indices=selected if not full_test else selected+[i for i in sorted(range(len(rows)),key=lambda i:rows[i]['row_id']) if i not in selected]
        sensitivity_indices=correct[:cfg['sensitivity_per_outcome']]+incorrect[:cfg['sensitivity_per_outcome']]
        plan={'policy':'lexicographic row_id; first 5 correct, first 5 incorrect; fixed model seed, no confidence ranking',
              'figure_row_ids':[rows[i]['row_id'] for i in selected], 'attribution_row_ids':[rows[i]['row_id'] for i in attribution_indices],
              'sensitivity_row_ids':[rows[i]['row_id'] for i in sensitivity_indices], 'checkpoint':str(checkpoint)}
        if (output/'selection.json').exists():
            if read_json(output/'selection.json')!=plan: raise ValueError('Resume selection mismatch')
        else: exclusive_json(output/'selection.json',plan)
        records=[]; refs=[]
        for kind,indices,destination in [('pad',attribution_indices,records),('mask',sensitivity_indices,refs)]:
            for index in indices:
                row=rows[index]; path=output/'records'/f'{row["row_id"]}-{kind}.json'
                if path.exists(): record=read_record(path,context)
                else:
                    print(f'IG {row["row_id"]} reference={kind}',flush=True)
                    record=explain(model,tokenizer,row,cfg,kind)
                    if record['predicted_label']!=int(previous[index].argmax()) or np.max(np.abs(np.array(record['probabilities'])-previous[index]))>cfg['probability_tolerance']:
                        raise ValueError(f'Per-example probability mismatch: {row["row_id"]}')
                    if np.max(np.abs(np.array(record['logits'])-previous_logits[index]))>cfg['logit_tolerance']:
                        raise ValueError(f'Per-example logit mismatch: {row["row_id"]}')
                    record.update({'configuration':cfg,'checkpoint':str(checkpoint),'context_fingerprint':context})
                    save_record(path,record,context)
                destination.append(record)
                last=record['attempts'][-1]
                print(f'  steps={last["n_steps"]} abs={last["absolute_residual"]:.3g} rel={last["relative_residual"]:.3g} converged={record["converged"]}',flush=True)
        by_id={r['row_id']:r for r in records}
        comparisons=[sensitivity(by_id[r['row_id']],r) for r in refs]
        # Final products are created once. Partial runs resume from sealed per-example records.
        (output/'figures').mkdir(exist_ok=True)
        shown=[by_id[rows[i]['row_id']] for i in selected]
        for record in shown:
            path=output/'figures'/f'{record["row_id"]}.png'
            if not path.exists(): figure(record,path)
        if not (output/'figures'/'index.html').exists(): gallery(shown,output/'figures'/'index.html')
        if not (output/'baseline_sensitivity.json').exists(): exclusive_json(output/'baseline_sensitivity.json',comparisons)
        if not (output/'baseline_sensitivity.csv').exists(): write_csv(output/'baseline_sensitivity.csv',comparisons)
        all_records=records+refs
        token_rows=[{'row_id':r['row_id'],'reference':r['baseline']['kind'],'target_class':r['target_class'],**t} for r in all_records for t in r['tokens']]
        if not (output/'token_attributions.csv').exists(): write_csv(output/'token_attributions.csv',token_rows)
        attempts=[{'row_id':r['row_id'],'reference':r['baseline']['kind'],**a} for r in all_records for a in r['attempts']]
        if not (output/'convergence.csv').exists(): write_csv(output/'convergence.csv',attempts)
        print('Checking preservation of all earlier files...',flush=True)
        after=preservation_snapshot()
        if before!=after: raise RuntimeError('Earlier artifact/code preservation check failed')
        final=[r['attempts'][-1] for r in all_records]
        attr_seconds=sum(r['runtime_seconds'] for r in all_records)
        summary={'checkpoint':str(checkpoint),'verified_test_rows':len(rows),'records':len(records),'sensitivity_records':len(refs),
            'prediction_max_probability_difference':pdiff,'prediction_max_logit_difference':ldiff,
            'wall_seconds':time.perf_counter()-started,'attribution_seconds':attr_seconds,
            'attributions_per_second':len(all_records)/attr_seconds,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if not cpu else 0,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if not cpu else 0,
            'median_absolute_residual':float(np.median([a['absolute_residual'] for a in final])),
            'max_absolute_residual':max(a['absolute_residual'] for a in final),
            'max_relative_residual':max(a['relative_residual'] for a in final),
            'escalated_records':sum(len(r['attempts'])>1 for r in all_records),
            'unconverged_records':sum(not r['converged'] for r in all_records),
            'earlier_files_preserved':True,'earlier_file_count':len(before),'warnings':warnings,
            'truncated_selected_examples':sum(r['truncated'] for r in records), 'environment':environment,
            'resume_used':resume,'timing_note':'Record times include original work when resumed; wall time covers this invocation only'}
        if not (output/'summary.json').exists(): exclusive_json(output/'summary.json',summary)
        else: summary=read_json(output/'summary.json')
        if not report.exists(): write_report(report,summary,shown,comparisons,Path(cfg['output_dir']))
        report_record={'path':str(report),'sha256':digest(report)}
        if (output/'report_hash.json').exists():
            if read_json(output/'report_hash.json')!=report_record: raise ValueError('Resume report hash mismatch')
        else: exclusive_json(output/'report_hash.json',report_record)
        exclusive_json(output/'manifest.json',{str(f.relative_to(output)).replace('\\','/'):digest(f) for f in sorted(output.rglob('*')) if f.is_file() and f.name!='manifest.json'})
        print(json.dumps(summary,indent=2),flush=True)
        return summary
    except Exception as error:
        failure={'exception':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
                 'earlier_files_preserved':preservation_snapshot()==before}
        stamp=time.time_ns()
        exclusive_json(output/f'failure-{stamp}.json',failure)
        raise
