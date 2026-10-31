import time

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from .artifacts import exclusive_json, phase4_path
from .core import deterministic, explain


def tiny(device='cpu'):
    deterministic(42)
    words=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','profit','rose','fell','company','loss','##es','.']
    tokenizer=BertTokenizer(vocab={w:i for i,w in enumerate(words)})
    config=BertConfig(vocab_size=len(words),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
        intermediate_size=32,num_labels=3,hidden_dropout_prob=.2,attention_probs_dropout_prob=.2)
    config._attn_implementation='eager'
    model=BertForSequenceClassification(config).to(device).eval()
    model.requires_grad_(False)
    cfg={'max_length':12,'n_steps':[50,100,200],'internal_batch_size':4,
         'absolute_tolerance':.001,'relative_tolerance':.01,'relative_floor':1e-6}
    return model,tokenizer,cfg


def run_smoke(output,cuda=False):
    output=phase4_path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    model,tokenizer,cfg=tiny('cuda' if cuda else 'cpu')
    row={'row_id':'synthetic-0','sentence':'profit rose .','label':2}
    records=[explain(model,tokenizer,row,cfg,kind) for kind in ('pad','mask')]
    assert all(r['converged'] for r in records)
    assert all(r['attribution_shape']==[1,12,16] for r in records)
    assert all(abs(t['signed'])<1e-8 for r in records for t in r['tokens'] if not t['rank_eligible'])
    repeated=explain(model,tokenizer,row,cfg)
    assert [t['signed'] for t in records[0]['tokens']]==[t['signed'] for t in repeated['tokens']]
    exclusive_json(output/'smoke.json',{'passed':True,'device':'cuda' if cuda else 'cpu','records':records,
        'runtime_seconds':time.perf_counter()-started,'no_training':True})
    print(f'Smoke passed on {"CUDA" if cuda else "CPU"}; {time.perf_counter()-started:.2f}s',flush=True)
