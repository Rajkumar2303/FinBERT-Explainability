"""Read-only repository checks; writes documentation evidence only under learning/.

Run from the project root with rag-env Python. No model loading, inference,
training, experiment regeneration, package installation or Git mutation.
"""
import ast
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'learning'
sys.path.insert(0, str(ROOT / 'src'))


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def parameters(path):
    with (ROOT / path).open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    return sum(math.prod(v['shape']) for k, v in header.items() if k != '__metadata__')


def main():
    errors = []
    checks = []

    def check(condition, label):
        checks.append({'check': label, 'passed': bool(condition)})
        if not condition:
            errors.append(label)

    sources = [
        'artifacts/phase1/statistics.json', 'artifacts/phase1/leakage_report.json',
        'artifacts/phase1/metadata.json', 'artifacts/phase2/audit/audit.json',
        'artifacts/phase2/comparison/comparison.json', 'artifacts/phase3/summary.json',
        'artifacts/phase4/initial/summary.json', 'artifacts/phase4/initial/baseline_sensitivity.json',
        'artifacts/phase5/full/summary.json', 'artifacts/phase6/audit-final/ig_sensitivity.json',
        'publication/commands.json',
    ]
    p1, leak, meta, audit, p2, p3, p4, refs, p5, sens, commands = [read(p) for p in sources]
    test_hash = digest(ROOT / 'artifacts/phase1/test.jsonl')
    manifest_hash = digest(ROOT / 'artifacts/phase1/manifest.json')
    check(test_hash == '97ad15089bd04e0d61726a6218bfcc2bdb18a4bcc1f5e8a299e1da41607e36a9', 'frozen test SHA256')
    check(manifest_hash == 'c5b8f2b32453b22c9f5038ba844b3cc5c025bce603e27f7899e7d62e00a8c782', 'Phase 1 manifest SHA256')
    check([p1['splits'][s]['rows'] for s in ['train', 'validation', 'test']] == [3391, 728, 727], 'split sizes')
    check(audit['rows_reassigned'] == 2199 and audit['moved_singletons'] == 2143 and audit['moved_duplicate_group_rows'] == 56, 'moved-row interpretation')
    check(audit['deterministic_replay'] and audit['saved_assignments_match_replay'] and audit['final_cross_split_pairs'] == 0, 'saved duplicate-group audit')
    cfg2, cfg4, cfg5 = [read(f'configs/phase{i}.json') for i in [2,4,5]]
    sources.extend(['configs/phase2.json','configs/phase4.json','configs/phase5.json'])
    check(cfg2['seeds'] == [42,123,2026] and cfg2['per_device_batch_size']*cfg2['gradient_accumulation_steps'] == 16, 'training seeds and effective batch configuration')
    check(cfg4['model_seed'] == 42 and cfg4['n_steps'] == [50,100,200] and cfg4['absolute_tolerance'] == .001 and cfg4['relative_tolerance'] == .01, 'fixed IG seed schedule and tolerances')
    check(cfg5['random_seeds'] == [11,23,37,53,71] and cfg5['bootstrap_samples'] == 2000 and not cfg5['save_full_attention'], 'attention experiment configuration')

    chapter_names = [
        '00_project_overview.md', '01_foundations_and_theory.md', '02_dataset_and_preprocessing.md',
        '03_model_architecture.md', '04_baselines_and_training.md', '05_evaluation_and_validation.md',
        '06_calibration_and_robustness.md', '07_integrated_gradients.md', '08_attention_mechanisms.md',
        '09_explainability_experiments.md', '10_results_and_discussion.md', '11_improvements_and_future_research.md',
    ]
    required = chapter_names + ['README.md', 'interview_preparation.md', 'technical_glossary.md', 'references.md']
    for name in required:
        check((HERE / name).is_file(), f'required file {name}')
    chapters = {p: (HERE / p).read_text(encoding='utf8') for p in chapter_names}
    stages = ['Why it matters', 'Build the intuition', 'A concrete example',
              'Walk through it', 'Give the idea its technical name', 'Add the mathematics',
              'Connect it to our project', 'Alternatives and trade-offs',
              'Check your understanding', 'Deeper dive and common confusion']
    lesson_counts = {}
    for name, text in chapters.items():
        lessons = re.findall(r'^## Lesson:.*?(?=^## |\Z)', text, re.M | re.S)
        lesson_counts[name] = len(lessons)
        check(len(lessons) >= 3, f'at least three progressive lessons {name}')
        check(all(re.findall(r'^### (.+)$', x, re.M) == stages for x in lessons), f'ten teaching stages in order {name}')
        check(all('**Question:**' in x and '**Answer explained:**' in x for x in lessons), f'answered concept checks every lesson {name}')
        check('## Learning objectives' in text, f'explicit learning objectives {name}')
        qa = text.split('## Practice explaining it')[1].split('## Sources')[0]
        check(len(re.findall(r'^\d+\. \*\*', qa, re.M)) >= 5, f'five answered self-assessment questions {name}')
    interview = (HERE / 'interview_preparation.md').read_text(encoding='utf8')
    check(len(re.findall(r'^### Q\d+\.', interview, re.M)) == 60, '60 interview questions')
    check(interview.count('**Spoken answer:**') == 60 and interview.count('**Deeper explanation:**') == 60 and interview.count('**Follow-up:**') == 60, 'interview answer depth and follow-ups')

    # Label-bound numeric checks, not a claim that every numeral in prose is a metric.
    measured = []

    def claim(chapter, label, value, expected, decimals=4):
        formatted = f'{value:.{decimals}f}'
        check(formatted == expected and expected in chapters[chapter], f'{label}: {expected}')
        measured.append({'chapter': chapter, 'metric': label, 'exact': value, 'display': expected})

    expected_f1 = {'baseline': '0.6713', 'bert': '0.8476', 'finbert': '0.8339'}
    expected_acc = {'baseline': '0.7442', 'bert': '0.8556', 'finbert': '0.8469'}
    for model in expected_f1:
        a = p2['independent_workflows'][model]['metrics']
        b = p3['classification_and_calibration'][model]['classification']
        check(a['macro_f1'] == b['macro_f1'] and a['accuracy'] == b['accuracy'], f'{model} Phase 2/3 consistency')
        claim('05_evaluation_and_validation.md', model + ' macro F1', a['macro_f1']['mean'], expected_f1[model])
        claim('05_evaluation_and_validation.md', model + ' accuracy', a['accuracy']['mean'], expected_acc[model])
        for label in ['negative','neutral','positive']:
            for metric in ['precision','recall','f1']:
                value = a['per_class'][label][metric]['mean']
                claim('05_evaluation_and_validation.md',model+' '+label+' '+metric,value,f'{value:.4f}')
        for metric in ['macro_f1','accuracy']:
            value = a[metric]['std']
            claim('05_evaluation_and_validation.md',model+' '+metric+' sample SD',value,f'{value:.4f}')
    calibration = {'baseline': ['0.0701','0.0730','0.3674','0.6320'], 'bert': ['0.1064','0.1088','0.2468','0.5115'], 'finbert': ['0.0923','0.0944','0.2620','0.6022']}
    flips = {'baseline': ['0.2019','0.0769','0.0640'], 'bert':['0.0128','0.0031','0.2255'], 'finbert':['0.0192','0.0000','0.1889']}
    for model in calibration:
        for metric, value in zip(['ece_10','ece_15','brier','nll'], calibration[model]):
            claim('06_calibration_and_robustness.md', model + ' ' + metric, p3['classification_and_calibration'][model]['calibration'][metric]['mean'], value)
        for kind, value in zip(['company','numerical','negation'], flips[model]):
            claim('06_calibration_and_robustness.md', model + ' visible ' + kind, p3['robustness'][model][kind]['visible_flip_rate']['mean'], value)
    for population, values in [('converged',['0.2406','0.1724','0.6555','0.6608','0.7866']), ('all',['0.2465','0.1764','0.6353','0.6372','0.7778'])]:
        s = sens[population]
        actual = [s['spearman']['mean'],s['kendall']['mean'],*[s['auc'][k]['mean'] for k in ['ig_absolute','attention','random']]]
        for label, value, expected in zip(['Spearman','Kendall','IG AUC','attention AUC','random AUC'],actual,values):
            claim('09_explainability_experiments.md',population+' '+label,value,expected)
        for value in s['paired_ig_minus_attention']['mean_ci95']:
            claim('09_explainability_experiments.md',population+' paired CI endpoint',value,f'{value:.4f}')
    check(sens['converged']['n'] == 687 and sens['unreliable']['n'] == 40 and sens['all']['n'] == 727, 'IG populations')
    check(sens['unreliable']['gold_class_counts'] == {'0':11,'2':29} and sens['unreliable']['allagree'] == 31 and sens['unreliable']['correct'] == 36, 'non-random exclusion composition')
    import csv
    with (ROOT/'artifacts/phase3/human_annotation.csv').open(encoding='utf-8-sig',newline='') as f:
        annotations = list(csv.DictReader(f))
    categories = collections.Counter(row['error_category'] for row in annotations)
    check(dict(categories) == {'ENTITY_CONTEXT':7,'FINANCIAL_CONTEXT':4,'IMPLICIT_SENTIMENT':4,'NEGATION':2,'COMPLEX_SENTENCE':1,'NUMERICAL_REASONING':1,'LABEL_AMBIGUITY':1}, 'current user-supplied category counts')
    claim('07_integrated_gradients.md','initial IG wall seconds',p4['wall_seconds'],'107.14',2)
    claim('09_explainability_experiments.md','full explanation wall seconds',p5['wall_seconds'],'951.44',2)
    check(p5['agreement']['absolute']['spearman']['mean'] == sens['converged']['spearman']['mean'], 'Phase 5/6 correlation agreement')

    best = {}
    counts = {}
    for model in ['bert','finbert']:
        counts[model] = parameters(f'artifacts/phase2/{model}/seed-42/model/model.safetensors')
        check(str(counts[model]) in chapters['03_model_architecture.md'], model+' parameter count from safetensors header')
        best[model] = {}
        for seed in [42,123,2026]:
            path = f'artifacts/phase2/{model}/seed-{seed}/training_record.json'
            sources.append(path)
            record = read(path)
            step = int(record['best_checkpoint'].split('-')[-1])
            best[model][str(seed)] = {'step':step,'epoch':step/212,'validation_f1':record['best_validation_macro_f1']}
    check([best['finbert'][str(s)]['epoch'] for s in [42,123,2026]] == [4,1,4], 'FinBERT actual best epochs')

    # Parse real commands, stopping immediately after argparse, before dispatch.
    from finbert_explainability.phase6.publication import validate_commands, command_text
    parsed = validate_commands(commands)
    check(len(parsed) == 19 and all(x['parsed'] and not x['workflow_executed'] for x in parsed), '19 real CLI parses without workflow execution')
    command_doc = '# Read the historical commands\n\nA CLI command is a way to call a workflow with explicit settings. In these commands, `python -m` selects a module; the following words select its action and arguments. A config path supplies recorded settings, while output paths identify where results go.\n\nThese are the exact supported historical commands, not commands executed for this course. Existing outputs must be preserved. A separately authorized reproduction needs fresh output locations, corresponding config paths, lawful data/model access, the recorded environment and passing smoke checks. Broad Phase 4/5 context fingerprints may require the archived source inventory.\n\nSource: [audited inventory](../publication/commands.json). The validator runs the real argument parsers and stops before workflow dispatch. Parsing verifies supported syntax; it does not verify a fresh experiment would finish.\n\n**Check:** Does a parser check run training? **Answer:** No. This validator blocks dispatch immediately after parsing, before the workflow can load a model.\n'
    for c in commands:
        command_doc += '\n## ' + c['id'].replace('_',' ') + '\n\n```powershell\n' + command_text(c) + '\n```\n'
    command_doc += '\n## Before a future reproduction\n\nRead the [training chapter](04_baselines_and_training.md) for the existing rag-env, CUDA, validation selection and measured resources. Trace every input path and every output path before executing a command in a new workspace.\n\n**Check:** May you reuse old output directories because the flags parse? **Answer:** No. Syntax validity does not authorize overwriting completed evidence.\n'
    (HERE/'reproduction_commands.md').write_text(command_doc, encoding='utf8')

    # Check actual source symbols named in the tutorial.
    symbols = {
        'src/finbert_explainability/data.py':['read_subset','annotate_membership','load_source'],
        'src/finbert_explainability/splitting.py':['stratified_split','grouped_stratified_split'],
        'src/finbert_explainability/leakage.py':['detect_duplicates','connected_groups'],
        'src/finbert_explainability/phase4/core.py':['embedding_layer','reference_ids','target_forward','explain','completeness','aggregate_words'],
        'src/finbert_explainability/phase5/analysis.py':['correlations','token_types','ordering','random_order','deletion_plan','auc_metrics','describe','paired'],
        'src/finbert_explainability/phase5/inference.py':['attention_and_deletion'],
    }
    for path, names in symbols.items():
        tree = ast.parse((ROOT/path).read_text(encoding='utf8'))
        present = {n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        check(set(names) <= present, 'source symbols '+path)

    source_hashes = {p:digest(ROOT/p) for p in sources}
    ledger = {'source_files_sha256':source_hashes,'test_sha256':test_hash,'phase1_manifest_sha256':manifest_hash,
              'verified_numeric_claims':measured,'parameter_counts':counts,'best_checkpoints':best,
              'phase1_statistics':p1,'phase2_comparison':p2['independent_workflows'],
              'phase3_calibration':{k:v['calibration'] for k,v in p3['classification_and_calibration'].items()},
              'ig_sensitivity':sens,'baseline_sensitivity':refs,
              'scope':'Read-only documentation validation; no new experiment or causal conclusion.'}
    save('evidence_ledger.json',ledger)

    links, missing, diagrams = 0, [], []
    files = []
    for path in sorted(HERE.glob('*.md')):
        text = path.read_text(encoding='utf8')
        check(not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text), 'no accidental control characters '+path.name)
        files.append({'file':path.name,'bytes':path.stat().st_size,'words':len(re.findall(r"\b[\w'-]+\b",text))})
        check(text.count('```') % 2 == 0, 'balanced fences '+path.name)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if re.match(r'^(https?://|mailto:)',target):
                continue
            links += 1
            name, _, anchor = target.partition('#')
            resolved = (path.parent/name).resolve() if name else path
            # The results file is created after this check; permit that one
            # declared generated target on a fresh run, not arbitrary missing files.
            if resolved == HERE / 'validation_results.json' and not resolved.exists():
                continue
            if not resolved.exists():
                missing.append({'file':path.name,'target':target})
            elif anchor and resolved.suffix == '.md':
                headings = re.findall(r'^#{1,6}\s+(.+)$',resolved.read_text(encoding='utf8'),re.M)
                slugs = [re.sub(r'[^\w\- ]','',h.lower()).replace(' ','-') for h in headings]
                if anchor not in slugs:
                    missing.append({'file':path.name,'anchor':target})
        for graph in re.findall(r'```mermaid\n(.*?)```',text,re.S):
            # Deliberately restricted flowchart subset used by this guide.
            lines = [line.strip() for line in graph.strip().splitlines()]
            valid = lines[0] in ('flowchart TD','flowchart LR') and all(re.fullmatch(r'[A-Z]\[[^\[\]]+\] --> [A-Z]\[[^\[\]]+\]|[A-Z] --> [A-Z]\[[^\[\]]+\]|[A-Z] --> [A-Z]',line) for line in lines[1:])
            # Also permit an existing target after a newly labeled source.
            if not valid:
                valid = lines[0] in ('flowchart TD','flowchart LR') and all(re.fullmatch(r'[A-Z](?:\[[^\[\]]+\])? --> [A-Z](?:\[[^\[\]]+\])?',line) for line in lines[1:])
            diagrams.append({'file':path.name,'basic_syntax_valid':valid})
            check(valid,'restricted Mermaid flowchart grammar '+path.name)
    check(not missing,'all local Markdown targets and used heading anchors resolve')
    check(len(diagrams) <= 4,'at most four Mermaid diagrams')
    check(len([d for d in diagrams if d['file']=='09_explainability_experiments.md']) <= 2,'at most two experiment diagrams')

    # Preserve an inventory at first validation, then compare at each subsequent pass.
    # Excludes only learning, caches, Git internals and inaccessible pytest scratch.
    inventory = {}
    skipped = []
    for base, dirs, names in os.walk(ROOT, onerror=lambda e: skipped.append(str(e))):
        dirs[:] = [d for d in dirs if d not in {'.git','.cache','.tmp','.deepeval','learning','__pycache__','.pytest_cache'} and 'temp' not in d and not d.startswith('pytest-cache')]
        for name in names:
            path = Path(base)/name
            try:
                inventory[path.relative_to(ROOT).as_posix()] = digest(path)
            except PermissionError:
                skipped.append(str(path))
    prior_path = HERE/'preservation_baseline.json'
    historical = read('artifacts/phase6/audit-final/earlier_hashes.json')
    historical_changed = [p for p,h in historical.items() if inventory.get(p.replace('\\','/')) != h]
    check(not historical_changed,'historical Phase 6 inventory remains byte-identical')
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding='utf8'))
        changed = [p for p,h in prior.items() if inventory.get(p) != h]
        added = sorted(set(inventory)-set(prior))
        check(not changed and not added,'earlier readable non-cache files unchanged since first documentation validation')
    else:
        save('preservation_baseline.json',inventory)
        changed, added = [], []

    output = {'completed_on':'2026-10-31','passed':not errors,'checks':checks,'errors':errors,'local_links_checked':links,'missing_links':missing,
              'main_chapters':len(chapter_names),'required_files':len(required),'interview_questions':60,
              'self_assessment_questions':60,'mermaid':diagrams,
              'teaching_lessons':lesson_counts,'total_teaching_lessons':sum(lesson_counts.values()),
              'mermaid_validation_scope':'Restricted flowchart grammar and balanced fences only; Mermaid rendering was not run.',
              'numeric_validation_scope':'Label-bound headline/calibration/robustness/sensitivity assertions, source ledger, cross-phase consistency and manual review; not a parser proving every natural-language claim.',
              'cli_commands':parsed,'files':files,'markdown_files':len(files),
              'total_markdown_bytes':sum(f['bytes'] for f in files),'total_markdown_words':sum(f['words'] for f in files),
              'preservation':{'readable_files':len(inventory),'changed':changed,'added_outside_learning':added,'skipped':skipped,
              'historical_phase6_files':len(historical),'historical_changed':historical_changed,
              'scope':'Baseline captured at first completed documentation validation after drafting; all authoring writes confined to learning. Historical Phase 6 inventory independently checked. Cache/Git/temporary annotation and pytest scratch excluded.'},
              'runtime_python':sys.executable,'no_training_or_inference':True,
              'known_limitations':['External reference pages verified during authoring, not re-fetched by this script.','Old annotation manifest exception remains; not rewritten.','Historical resume context may change when new documentation is added.']}
    save('validation_results.json',output)
    print(json.dumps({k:output[k] for k in ['passed','errors','local_links_checked','main_chapters','markdown_files','total_markdown_bytes','total_markdown_words']},indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
