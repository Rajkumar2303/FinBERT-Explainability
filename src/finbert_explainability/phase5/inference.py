import time

import numpy as np
import torch

from .analysis import auc_metrics, correlations, deletion_plan, token_types


def attention_and_deletion(model,tokenizer,row,ig,cfg,ig_cfg,save_path=None,expected_architecture=None):
    started=time.perf_counter(); model.eval()
    if any(m.training for m in model.modules()): raise RuntimeError('Dropout/training active')
    device=next(model.parameters()).device
    batch=tokenizer(row['sentence'],padding='max_length',max_length=ig_cfg['max_length'],
                    truncation=True,return_tensors='pt').to(device)
    ids=batch['input_ids'][0].tolist(); mask=batch['attention_mask'][0].tolist()
    if ids!=[t['token_id'] for t in ig['tokens']] or mask!=[t['attention_mask'] for t in ig['tokens']]:
        raise ValueError('Attention/IG token or mask mismatch')
    with torch.no_grad(): out=model(**batch,output_attentions=True)
    if out.attentions is None: raise ValueError('Attention unavailable; eager attention required')
    layers=len(out.attentions); heads=model.config.num_attention_heads; length=len(ids)
    if layers!=model.config.num_hidden_layers or any(a.shape!=(1,heads,length,length) for a in out.attentions):
        raise ValueError('Unexpected attention architecture/tensor dimensions')
    if expected_architecture and (layers,heads)!=expected_architecture:
        raise ValueError(f'Expected {expected_architecture} layer/head counts, got {(layers,heads)}')
    logits=out.logits[0].double().cpu().numpy(); probability=np.exp(logits-logits.max()); probability/=probability.sum()
    if np.max(np.abs(logits-ig['logits']))>ig_cfg['logit_tolerance'] or np.max(np.abs(probability-ig['probabilities']))>ig_cfg['probability_tolerance']:
        raise ValueError('output_attentions changed frozen inference')
    # All layer/head CLS rows are small; full query/key matrices are optional.
    active=np.flatnonzero(mask)
    cls_index=ids.index(tokenizer.cls_token_id)
    cls=np.stack([a[0,:,cls_index,:].float().cpu().numpy() for a in out.attentions])
    if not np.isfinite(cls).all() or not np.allclose(cls.sum(-1),1,atol=1e-5):
        raise ValueError('Nonfinite/unnormalized attention')
    if np.any(np.abs(cls[:,:,np.array(mask)==0])>1e-7): raise ValueError('Nonzero padding attention')
    attention=cls[-1].mean(0)
    if save_path is not None:
        data={'cls_attention':cls[:,:,active],'positions':active,'token_ids':np.array(ids)[active],
              'final_head_mean':attention[active]}
        if cfg['save_full_attention']:
            data['full_attention']=np.stack([a[0].float().cpu().numpy()[:,active][:,:,active] for a in out.attentions])
        if save_path.exists():
            with np.load(save_path) as previous:
                if set(previous.files)!=set(data) or any(not np.array_equal(previous[k],v) for k,v in data.items()):
                    raise ValueError('Existing attention file differs; refusing overwrite')
        else:
            with save_path.open('xb') as f: np.savez_compressed(f,**data)
    types=token_types(ig['tokens'],tokenizer)
    mass={kind:cls[-1,:,np.array(types)==kind].sum(0).tolist()
          for kind in ('CLS','SEP','punctuation','content','other_special','padding')}
    # Advanced indexing above yields (positions, heads), so sum positions.
    if any(len(v)!=heads for v in mass.values()) or not np.allclose(np.array(list(mass.values())).sum(0),1,atol=1e-5):
        raise ValueError('Attention mass categories do not partition head mass')
    plan,orders,eligible=deletion_plan(ig,attention,types,row['row_id'],cfg)
    absolute=[ig['tokens'][i]['absolute'] for i in eligible]; signed=[ig['tokens'][i]['signed'] for i in eligible]
    agreement={'absolute':correlations(absolute,attention[eligible],cfg['minimum_correlation_tokens']),
               'signed':correlations(signed,attention[eligible],cfg['minimum_correlation_tokens'])}
    target=ig['predicted_label']
    if target!=int(logits.argmax()) or ig['target_class']!=target: raise ValueError('Original target mismatch')
    unique=list(dict.fromkeys(tuple(p['positions']) for p in plan))
    predictions={(): (probability,logits)}
    remaining=[p for p in unique if p]
    del out
    with torch.no_grad():
        for start in range(0,len(remaining),cfg['inference_batch_size']):
            positions=remaining[start:start+cfg['inference_batch_size']]
            copies={k:v.repeat(len(positions),1) for k,v in batch.items()}
            for j,pos in enumerate(positions): copies['input_ids'][j,list(pos)]=tokenizer.mask_token_id
            values=model(**copies,output_attentions=False).logits.double().cpu().numpy()
            exp=np.exp(values-values.max(1,keepdims=True)); probabilities=exp/exp.sum(1,keepdims=True)
            for pos,pr,lo in zip(positions,probabilities,values): predictions[pos]=(pr,lo)
    curve=[]
    for item in plan:
        pr,lo=predictions[tuple(item['positions'])]
        curve.append({**item,'target_probability':float(pr[target]),'target_logit':float(lo[target]),
            'probability_drop':float(probability[target]-pr[target]),'logit_drop':float(logits[target]-lo[target]),
            'probabilities':pr.tolist(),'logits':lo.tolist(),'predicted_label':int(pr.argmax()),
            'label_flip':bool(int(pr.argmax())!=target)})
    auc={method:auc_metrics([p for p in curve if p['method']==method]) for method in orders}
    randoms=[auc[f'random_{s}'] for s in cfg['random_seeds']]
    random_summary={metric:{'mean':float(np.mean([r[metric] for r in randoms])),
                            'sd':float(np.std([r[metric] for r in randoms],ddof=1))}
                    for metric in ('probability_auc','probability_drop_auc','logit_drop_auc','any_flip','final_flip')
                    if all(r[metric] is not None for r in randoms)}
    return {'row_id':row['row_id'],'sentence':row['sentence'],'true_label':row['label'],
        'predicted_label':target,'original_probabilities':probability.tolist(),'original_logits':logits.tolist(),
        'ig_converged':ig['converged'],'ig_final_attempt':ig['attempts'][-1],
        'token_ids':ids,'tokens':[t['token'] for t in ig['tokens']],'attention_mask':mask,'token_types':types,
        'ig_signed':[t['signed'] for t in ig['tokens']],'ig_absolute':[t['absolute'] for t in ig['tokens']],
        'attention':attention.tolist(),'eligible_positions':eligible,'agreement':agreement,
        'head_token_type_mass':mass,'layers':layers,'heads':heads,'orders':orders,'deletion':curve,
        'auc':auc,'random_summary':random_summary,'runtime_seconds':time.perf_counter()-started}
