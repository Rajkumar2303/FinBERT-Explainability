import copy
import json
from pathlib import Path

import numpy as np
import pytest

from finbert_explainability.phase4.core import explain
from finbert_explainability.phase4.smoke import tiny
from finbert_explainability.phase5.analysis import (auc_metrics,correlations,deletion_plan,describe,ordering,paired,random_order,token_types)
from finbert_explainability.phase5.artifacts import output_path,validate_config
from finbert_explainability.phase5.inference import attention_and_deletion


@pytest.fixture(scope='module')
def config(): return json.loads(Path('configs/phase5.json').read_text())


@pytest.fixture(scope='module')
def example(config):
    model,tok,igcfg=tiny(); igcfg.update(probability_tolerance=1e-5,logit_tolerance=1e-4)
    row={'row_id':'synthetic-1','sentence':'company profit rose .','label':2}
    ig=explain(model,tok,row,igcfg)
    result=attention_and_deletion(model,tok,row,ig,config,igcfg)
    return model,tok,igcfg,row,ig,result


def test_correlations_known_ties_and_undefined():
    assert correlations([1,2,3],[3,2,1])['spearman']==pytest.approx(-1)
    assert correlations([1,1,3],[1,1,3])['kendall_tau_b']==pytest.approx(1)
    assert correlations([1,1,1],[1,2,3])['status']=='constant_vector'
    assert correlations([1,2],[2,3])['status']=='too_few_tokens'
    with pytest.raises(ValueError): correlations([1,2,float('nan')],[1,2,3])


def test_random_and_tie_determinism():
    assert ordering([2,2,1],[2,1,0])==[0,1,2]
    a=random_order(list(range(30)),'row-1',11)
    assert a==random_order(list(range(30)),'row-1',11)
    assert a!=random_order(list(range(30)),'row-2',11)
    assert sorted(a)==list(range(30))


def test_punctuation_and_special_categories(example):
    _,tok,_,_,ig,r=example
    types=token_types(ig['tokens'],tok)
    assert types[0]=='CLS' and 'SEP' in types and 'padding' in types and 'punctuation' in types
    assert all(types[i]=='content' for i in r['eligible_positions'])
    fake=[{'token_id':999,'token':s,'attention_mask':1} for s in ['##,','%','123','$','##growth']]
    assert token_types(fake,tok)==['punctuation','punctuation','content','content','content']


def test_attention_shapes_head_mass_and_zero_budget(example):
    _,_,_,_,_,r=example
    assert r['layers']==1 and r['heads']==2
    assert np.allclose(np.array(list(r['head_token_type_mass'].values())).sum(0),1,atol=1e-6)
    assert all(abs(p['probability_drop'])<1e-12 and not p['label_flip'] for p in r['deletion'] if p['budget']==0)
    assert all(r['token_types'][i]=='content' for p in r['deletion'] for i in p['positions'])


def test_mask_endpoints_match_across_methods(example):
    *_,r=example
    points=[p for p in r['deletion'] if p['budget']==1 and p['method']!='ig_positive']
    assert len({tuple(p['positions']) for p in points})==1
    assert len({tuple(p['logits']) for p in points})==1
    assert all(p['actual_fraction']==1 for p in points)


def test_positive_only_does_not_mask_negative_scores(example,config):
    _,_,_,_,ig,r=example
    forced=copy.deepcopy(ig)
    for i,t in enumerate(forced['tokens']): t['signed']=1 if i==1 else -1
    plan,orders,eligible=deletion_plan(forced,r['attention'],r['token_types'],'test',config)
    points=[p for p in plan if p['method']=='ig_positive']
    assert orders['ig_positive']==[1]
    assert points[-1]['saturated'] and points[-1]['k']==1


def test_actual_fraction_auc_and_incomplete_coverage():
    curve=[{'actual_fraction':x,'target_probability':1-x,'probability_drop':x,'logit_drop':2*x,'label_flip':x==1} for x in (0,.5,.5,1)]
    out=auc_metrics(curve)
    assert out['probability_auc']==pytest.approx(.5)
    assert out['logit_drop_auc']==pytest.approx(1)
    assert auc_metrics(curve[:-1])['probability_auc'] is None


def test_bootstrap_sentence_pairing():
    result=paired([1,2,3],[0,1,2],seed=3,replicates=100)
    assert result['mean_ci95']==[1,1]
    assert result['sd']==0
    assert describe([None])['n']==0
    assert describe([1,2,3],seed=3,replicates=100)==describe([1,2,3],seed=3,replicates=100)


def test_attention_export_and_resume_validation(example,config,tmp_path):
    model,tok,igcfg,row,ig,r=example
    path=tmp_path/'attention.npz'
    again=attention_and_deletion(model,tok,row,ig,config,igcfg,path)
    with np.load(path) as data:
        assert data['cls_attention'].shape==(1,2,sum(r['attention_mask']))
        assert len(data['token_ids'])==sum(r['attention_mask'])
    attention_and_deletion(model,tok,row,ig,config,igcfg,path)
    assert again['auc']==r['auc']
    bad=copy.deepcopy(ig); bad['tokens'][1]['token_id']=999
    with pytest.raises(ValueError,match='token or mask'): attention_and_deletion(model,tok,row,bad,config,igcfg)


def test_output_config_protection(config):
    validate_config(config)
    with pytest.raises(ValueError): output_path('artifacts/phase4/initial')
    bad={**config,'deletion_fractions':[0,.5,.1,1]}
    with pytest.raises(ValueError): validate_config(bad)
