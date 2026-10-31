import time

from ..phase4.artifacts import exclusive_json
from ..phase4.core import explain
from ..phase4.smoke import tiny
from .artifacts import output_path
from .inference import attention_and_deletion


def run_smoke(output,cfg):
    output=output_path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    model,tokenizer,ig_cfg=tiny()
    ig_cfg.update(probability_tolerance=1e-5,logit_tolerance=1e-4)
    texts=['profit rose .','company profit fell .','company losses rose .','profit fell .','company profit rose .']*2
    results=[]
    for i,text in enumerate(texts):
        row={'row_id':f'synthetic-{i:02d}','sentence':text,'label':i%3}
        ig=explain(model,tokenizer,row,ig_cfg)
        assert ig['converged']
        r=attention_and_deletion(model,tokenizer,row,ig,cfg,ig_cfg,output/f'{i}.npz')
        assert all(abs(p['probability_drop'])<1e-10 for p in r['deletion'] if p['budget']==0)
        assert all(r['token_types'][i]=='content' for p in r['deletion'] for i in p['positions'])
        results.append(r)
    exclusive_json(output/'smoke.json',{'passed':True,'examples':len(results),'device':'cpu','tiny_random_model':True,
        'runtime_seconds':time.perf_counter()-started,'records':results})
    print(f'10-example CPU smoke passed in {time.perf_counter()-started:.2f}s',flush=True)
