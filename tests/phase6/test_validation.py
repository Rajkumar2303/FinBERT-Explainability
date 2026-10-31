import copy
import json
from pathlib import Path

import pytest

from finbert_explainability.phase6.audit import numeric_claims,same
from finbert_explainability.phase6.publication import COMMANDS,command_text,scan_files,validate_commands


def test_numeric_claims_preserve_metric_associations():
    a='| BERT | 0.8476 |\n| FinBERT | 0.8339 |'
    assert numeric_claims(a)==numeric_claims('\n'.join(reversed(a.splitlines())))
    assert numeric_claims(a)!=numeric_claims('| BERT | 0.8339 |\n| FinBERT | 0.8476 |')


def test_dictionary_order_not_a_metric_change():
    assert numeric_claims("Counts: {'50': 151, '100': 400}")==numeric_claims("Counts: {'100': 400, '50': 151}")
    assert numeric_claims('Value 0.24 [figure](a/phase5.png)')==numeric_claims('Value 0.24 [figure](../../a/phase5.png)')


def test_nested_metric_mismatch_fails():
    same({'a':[.1,.2]},{'a':[.1,.2]})
    with pytest.raises(AssertionError): same({'a':[.1,.3]},{'a':[.1,.2]})


def test_all_readme_commands_parse_without_dispatch():
    results=validate_commands(COMMANDS)
    assert len(results)==19 and all(r['parsed'] and not r['workflow_executed'] for r in results)


def test_secret_scan_redacts_values(tmp_path):
    file=tmp_path/'example.txt'; value='ghp_'+'a'*36
    file.write_text(value,encoding='utf8')
    findings=scan_files([file]); assert len(findings)==1
    assert value not in json.dumps(findings) and findings[0]['value']=='REDACTED'


def test_saved_audit_and_sensitivity_if_available():
    root=Path('artifacts/phase6/audit-final')
    if not (root/'receipt.json').exists(): pytest.skip('Private historical artifacts not distributed')
    receipt=json.loads((root/'receipt.json').read_text(encoding='utf8'))
    assert receipt['passed'] and receipt['manifests']==30 and receipt['manifest_entries']==3025
    s=json.loads((root/'ig_sensitivity.json').read_text(encoding='utf8'))
    assert s['converged']['n']==687 and s['all']['n']==727 and s['unreliable']['n']==40
    assert s['unreliable']['gold_class_counts'].get('1',0)==0
    for key in ('converged','all'):
        lo,hi=s[key]['paired_ig_minus_attention']['mean_ci95']; assert lo<0<hi


def test_publication_readme_commands_and_links_if_available():
    p=Path('publication/README.md')
    if not p.exists(): pytest.skip('Publication files not generated')
    text=p.read_text(encoding='utf8')
    for c in COMMANDS: assert command_text(c) in text
    assert '0.8476' in text and '0.8339' in text and '0.2406' in text
