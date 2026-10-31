import copy

import numpy as np
import pytest
import torch
from captum.attr import LayerIntegratedGradients

from finbert_explainability.phase4.artifacts import (fingerprint, read_record, save_record, selection, phase4_path)
from finbert_explainability.phase4.core import (aggregate_words, completeness, converged, embedding_layer,
    explain, reference_ids, sensitivity, target_forward)
from finbert_explainability.phase4.smoke import tiny


@pytest.fixture(scope='module')
def example():
    model,tok,cfg=tiny()
    row={'row_id':'tiny','sentence':'profit rose .','label':2}
    return model,tok,cfg,row,explain(model,tok,row,cfg)


def test_shapes_determinism_and_dropout(example):
    model,tok,cfg,row,r=example
    assert r['attribution_shape']==[1,12,16]
    assert not any(m.training for m in model.modules())
    assert [t['signed'] for t in r['tokens']]==[t['signed'] for t in explain(model,tok,row,cfg)['tokens']]
    assert all(t['absolute']==abs(t['signed']) for t in r['tokens'])


def test_special_padding_and_variable_lengths(example):
    model,tok,cfg,row,r=example
    for text in ('profit .','company profit rose .'):
        encoded=tok(text,max_length=12,padding='max_length',return_special_tokens_mask=True,return_tensors='pt')
        baseline,content=reference_ids(encoded['input_ids'],encoded['attention_mask'],encoded['special_tokens_mask'],tok.pad_token_id)
        assert torch.equal(baseline[~content],encoded['input_ids'][~content])
        assert (baseline[content]==tok.pad_token_id).all()
    assert all(abs(t['signed'])<1e-8 for t in r['tokens'] if not t['rank_eligible'])
    assert all(r['tokens'][i]['rank_eligible'] for i in r['ranking_positions'])


def test_target_is_logit_and_can_be_changed(example):
    model,tok,cfg,row,r=example
    batch=tok(row['sentence'],return_tensors='pt')
    with torch.no_grad():
        logits=model(**batch).logits
        for target in range(3):
            actual=target_forward(model,target)(batch['input_ids'],batch['attention_mask'],batch['token_type_ids'])
            assert torch.equal(actual,logits[:,target])
    target=(r['predicted_label']+1)%3
    other=explain(model,tok,row,cfg,target=target)
    assert other['target_class']==target
    assert other['target_logit']==pytest.approx(other['logits'][target])
    assert other['target_logit']!=pytest.approx(other['probabilities'][target])
    with pytest.raises(ValueError): target_forward(model,3)


def test_linear_analytic_signed_completeness():
    layer=torch.nn.Embedding.from_pretrained(torch.tensor([[0.,0.],[2.,-1.],[-3.,1.]]))
    def forward(ids): return layer(ids).sum((1,2))
    attr,delta=LayerIntegratedGradients(forward,layer).attribute(torch.tensor([[1,2]]),
        baselines=torch.tensor([[0,0]]),n_steps=50,return_convergence_delta=True)
    assert attr.sum(-1).tolist()[0]==pytest.approx([1.,-2.])
    assert delta.item()==pytest.approx(0,abs=1e-6)
    assert completeness(attr,-1,0,1e-6)['absolute_residual']<1e-6


def test_completeness_includes_every_dimension_and_floor(example):
    _,_,cfg,_,r=example
    last=r['attempts'][-1]
    assert sum(t['signed'] for t in r['tokens'])==pytest.approx(last['logit_difference'],abs=1e-5)
    assert last['captum_delta']==pytest.approx(last['signed_residual'],abs=1e-6)
    bad=completeness(torch.tensor([1.]),0.,0.,1e-6)
    assert bad['relative_residual']==1e6
    assert not converged(bad,cfg)


def test_escalation_retains_failures(example):
    model,tok,cfg,row,_=example
    strict={**cfg,'absolute_tolerance':1e-20,'relative_tolerance':1e-20,'n_steps':[2,3]}
    r=explain(model,tok,row,strict)
    assert len(r['attempts'])==2
    assert not r['converged']


def test_wordpieces_preserve_signed_sums():
    tokens=[{'position':i,'token':t,'signed':v,'absolute':abs(v),'rank_eligible':ok}
            for i,(t,v,ok) in enumerate([('[CLS]',0,False),('loss',2,True),('##es',-3,True),('.',.5,True)])]
    words=aggregate_words(tokens)
    assert words[0]=={'text':'losses','positions':[1,2],'signed':-1,'sum_token_absolute':5,'absolute_of_signed_sum':1}


def test_reference_sensitivity(example):
    model,tok,cfg,row,r=example
    alternate=explain(model,tok,row,cfg,'mask')
    assert alternate['target_class']==r['target_class']
    result=sensitivity(r,r)
    assert result['spearman_absolute_rank']==pytest.approx(1)
    assert result['top_k_jaccard']==1
    assert result['sign_change_fraction']==0
    assert any(a['baseline_token_id']!=b['baseline_token_id'] for a,b in zip(r['tokens'],alternate['tokens']))


def test_deterministic_selection():
    rows=[{'row_id':f'id-{i}','label':0} for i in reversed(range(6))]
    p=np.array([[1,0,0] if int(r['row_id'][-1])%2==0 else [0,1,0] for r in rows])
    correct,error=selection(rows,p,2)
    assert [rows[i]['row_id'] for i in correct]==['id-0','id-2']
    assert [rows[i]['row_id'] for i in error]==['id-1','id-3']


def test_resume_integrity_and_no_overwrite(tmp_path):
    path=tmp_path/'record.json'
    save_record(path,{'row_id':'example'},'context')
    assert read_record(path,'context')=={'row_id':'example'}
    with pytest.raises(FileExistsError): save_record(path,{},'context')
    with pytest.raises(ValueError): read_record(path,'different')
    path.write_text(path.read_text().replace('example','tampered'))
    with pytest.raises(ValueError): read_record(path,'context')
    with pytest.raises(ValueError): phase4_path('artifacts/phase1')
    with pytest.raises(TypeError): embedding_layer(torch.nn.Linear(2,3))
