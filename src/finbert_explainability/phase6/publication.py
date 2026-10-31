import argparse
import contextlib
import importlib.metadata
import io
import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from ..phase3.io import digest,read_json
from ..phase4.artifacts import exclusive_json

COMMANDS=[
    {'id':'phase1','module':'finbert_explainability','args':['--config','configs/phase1.json']},
    {'id':'phase2_audit','module':'finbert_explainability.phase2','args':['audit','--phase1','artifacts/phase1','--output','artifacts/phase2/audit']},
    {'id':'phase2_smoke','module':'finbert_explainability.phase2','args':['smoke','--output','artifacts/phase2/smoke-final']},
    *[{'id':f'phase2_train_{k}','module':'finbert_explainability.phase2','args':['train',k,'--config','configs/phase2.json','--output',f'artifacts/phase2/{k}','--approve-training','--smoke-receipt','artifacts/phase2/smoke-final/smoke_receipt.json']} for k in ('baseline','bert','finbert')],
    *[{'id':f'phase2_evaluate_{k}','module':'finbert_explainability.phase2','args':['evaluate','--run',f'artifacts/phase2/{k}','--output',f'artifacts/phase2/{k}-evaluation']} for k in ('baseline','bert','finbert')],
    {'id':'phase2_reference','module':'finbert_explainability.phase2','args':['reference','--config','configs/phase2.json','--output','artifacts/phase2/reference']},
    {'id':'phase2_compare','module':'finbert_explainability.phase2','args':['compare','--runs','artifacts/phase2/baseline-evaluation','artifacts/phase2/bert-evaluation','artifacts/phase2/finbert-evaluation','artifacts/phase2/reference','--output','artifacts/phase2/comparison']},
    {'id':'phase3_smoke','module':'finbert_explainability.phase3','args':['smoke','--output','artifacts/phase3-smoke']},
    {'id':'phase3','module':'finbert_explainability.phase3','args':['analyze','--config','configs/phase3.json','--smoke-receipt','artifacts/phase3-smoke/smoke_receipt.json']},
    {'id':'phase4_smoke','module':'finbert_explainability.phase4','args':['smoke','--output','artifacts/phase4/smoke']},
    {'id':'phase4','module':'finbert_explainability.phase4','args':['run','--config','configs/phase4.json']},
    {'id':'phase5_cpu','module':'finbert_explainability.phase5','args':['cpu-smoke','--output','artifacts/phase5/cpu-smoke']},
    {'id':'phase5_cuda','module':'finbert_explainability.phase5','args':['run','--config','configs/phase5/cuda-smoke.json','--smoke']},
    {'id':'phase5','module':'finbert_explainability.phase5','args':['run','--config','configs/phase5.json']},
    {'id':'phase6','module':'finbert_explainability.phase6','args':['audit','--output','artifacts/phase6/audit-final']}
]


def command_text(c): return 'python -m '+c['module']+' '+' '.join(c['args'])


def validate_commands(commands):
    original=argparse.ArgumentParser.parse_args
    class Parsed(Exception): pass
    result=[]
    for c in commands:
        def intercept(parser,args=None,namespace=None):
            original(parser,c['args'],namespace)
            raise Parsed()
        with patch.object(argparse.ArgumentParser,'parse_args',intercept),contextlib.redirect_stdout(io.StringIO()):
            try: runpy.run_module(c['module']+'.__main__',run_name='__main__')
            except Parsed: result.append({'id':c['id'],'parsed':True,'workflow_executed':False})
            else: raise AssertionError('Command did not invoke expected parser')
    return result


def write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f: f.write(text.rstrip()+'\n')


def prepare():
    pub=Path('publication'); pub.mkdir(exist_ok=True)
    for name in ('figures','docs'): (pub/name).mkdir(exist_ok=True)
    audit=Path('artifacts/phase6/audit-final'); sens=read_json(audit/'ig_sensitivity.json')
    metrics=read_json('artifacts/phase3/summary.json')['classification_and_calibration']
    performance='\n'.join(f'| {k} | {v["classification"]["macro_f1"]["mean"]:.4f} +/- {v["classification"]["macro_f1"]["std"]:.4f} | {v["classification"]["accuracy"]["mean"]:.4f} +/- {v["classification"]["accuracy"]["std"]:.4f} |' for k,v in metrics.items())
    commands='\n'.join(command_text(c) for c in COMMANDS)
    write(pub/'README.md',f'''# FinBERT Explainability

**Does financial pretraining improve sentiment classification, and do attention weights behave like Integrated Gradients explanations?**

A six-phase, frozen-split evaluation of Financial PhraseBank using TF-IDF, general BERT and financial BERT. All findings below are measured results, not an assumed advantage for FinBERT. Publication is prepared locally; no repository has been pushed.

## Findings

| Independent workflow | Macro-F1: mean +/- sample SD | Accuracy: mean +/- sample SD |
|---|---:|---:|
{performance}

All workflows use the same 727-row test set and seeds 42, 123, 2026. General BERT has the higher mean macro-F1; three seeds do not justify a significance claim. ProsusAI/finbert is a **non-independent external reference** because Financial PhraseBank was part of its fine-tuning data; its 0.8797 macro-F1 is not a fair held-out ranking.

- BERT macro-F1 is 0.9583 on allagree versus 0.7500 on non-allagree examples.
- FinBERT seed-42 absolute IG versus final-layer head-mean CLS attention: mean per-sentence Spearman **0.2406** on the 687 converged examples, not a pooled token correlation.
- Mean original-class probability deletion AUC: IG **0.6555**, attention **0.6608**, random **0.7866**. Lower means stronger suppression under this MASK intervention.
- Paired IG-minus-attention 95% bootstrap CI **[-0.0130, 0.0024]** includes zero: no clear winner, not proof of equivalence.
- Forty IG records failed the unchanged absolute tolerance. Including all 727 changes Spearman to **0.2465** and the paired CI to **[-0.0097, 0.0061]**. The conclusion remains cautious; the excluded class mix is not random.
- PAD versus MASK baseline choice materially changes token rankings. MASK deletion introduces distribution shift and does not prove causality.

[Five-page validation report (PDF)](docs/model-validation.pdf) · [Report Markdown](docs/model-validation.md) · [Model card](MODEL_CARD.md) · [Reproducibility checklist](REPRODUCIBILITY.md) · [Licenses and release gates](LICENSE_REVIEW.md)

![Deletion comparison](figures/deletion_curves.png)
![Rank agreement](figures/correlations.png)

## Pipeline

```mermaid
flowchart TD
  A[Financial PhraseBank 50agree + agreement subsets] --> B[Phase 1: deterministic duplicate-group splits]
  B --> C[Phase 2: train-only fitting and validation-only selection]
  C --> D[TF-IDF / general BERT / financial BERT: three seeds]
  D --> E[Phase 3: reliability, calibration, subgroups, robustness]
  D --> F[Fixed FinBERT seed-42 checkpoint]
  F --> G[Phase 4: logit Layer IG and baseline sensitivity]
  G --> H[Phase 5: 727-example attention and MASK comparison]
  E --> I[Phase 6: audit, validation report, publication preparation]
  H --> I
```

`src/finbert_explainability/phase2` through `phase6` contain modular CLI workflows. Phase 1 is the package's root CLI. Configurations pin model revisions, data hashes, seeds and experiment settings. A completed output directory is protected against overwrite. The Phase 4 and 5 pipelines support checksummed record resumption.

## Dataset and intended use

Financial PhraseBank `sentences_50agree`: 4,846 sentences; train/validation/test = **3,391/728/727**. Canonical labels are negative=0, neutral=1, positive=2. Exact/near duplicate grouping leaves zero detected cross-split pairs under the declared lexical detector, not a guarantee against semantic or pretraining overlap. The deterministic regrouping reassigned 2,199 rows, including 2,143 singletons; no rows were lost.

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) is listed as **CC BY-NC-SA 3.0**. Obtain the pinned source through the supplied configuration and respect attribution, noncommercial and share-alike terms. Raw sentences, derived row-level exports, downloaded datasets and checkpoints are not in this publication package. Model and data rights are separate from the owner-selected code license, which remains pending. See [license review](LICENSE_REVIEW.md).

Use for financial-language research or supervised evaluation. This is not a validated trading, credit, compliance or investment decision system. Temporal/domain validation, risk calibration, human oversight and rights clearance are needed before deployment.

## Environment and installation

Historical environment: Python 3.11.15; Windows `rag-env`; NVIDIA RTX 4070 Laptop GPU (8 GB); CUDA 12.6; PyTorch **2.12.1+cu126**. Do not replace PyTorch in the existing environment.

```powershell
conda activate rag-env
$env:PYTHONPATH = (Resolve-Path src).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
python -c "import sys,torch; print(sys.executable,torch.__version__,torch.version.cuda); print(torch.cuda.is_available())"
python -m pip install -e . --no-deps --no-build-isolation
```

The editable install is optional; PYTHONPATH is sufficient. `requirements/research-pinned.txt` records tested non-Torch direct dependencies, and `requirements/constraints-observed.txt` pins the observed Torch build. These are not a universal cross-platform lock or proof that every historical wheel remains available. A fresh environment requires a compatible CUDA driver and verified availability of the exact Torch build; consult the [official PyTorch installation guidance](https://pytorch.org/get-started/locally/). Do not silently substitute a different build. Report generation additionally uses ReportLab 4.4.9 and the pinned PDF tools. The original project pyproject has broad bounds and omits several later-phase dependencies; use the accompanying pins for a reviewed reconstruction.

## Reproduction commands for every phase

The following are documented historical workflows, **not commands executed by Phase 6**. Run only in a fresh workspace or use new output/config paths. The `--approve-training` flags below are for a separately approved Phase 2 reproduction; they do not authorize any training during this audit.

```powershell
{commands}
```

Full private artifact access is required to replay the exact historical evaluation and audit. The public package intentionally omits models/data. Rebuilding data can reproduce assignments with matching software/source but metadata and manifest bytes may differ; never change the historical hashes to hide that difference. Use copied configs for genuinely new runs. The Phase 3 smoke receipt binds its source code; Phase 4/5 resumption binds input and implementation hashes. Do not retune from this already-examined test set.

```powershell
python -m finbert_explainability.phase5 run --config configs/phase5.json --resume
python -m pytest tests -q -p no:cacheprovider --basetemp artifacts/phase6/tests-fresh
```

The numerical audit has no model inference or training. It recomputes metrics from saved records and verifies manifests. README workflow flags are automatically parsed by the actual CLI parsers with dispatch blocked.

## Explainability example

Measured example `fpb50-00018`, fixed FinBERT seed 42: true/predicted positive, probability 0.9936. Under the PAD reference, the WordPiece `double` contributes approximately +2.783 to the **positive-class logit difference**. This is a local, baseline-dependent attribution, not proof of a causal financial effect. Full sentence/token exports remain local; the small attribution fragment is dataset-derived and subject to the source rights review. The report also covers misclassifications rather than presenting only successful examples.

## Limitations and annotation provenance

- Three training seeds, one explanation checkpoint, one benchmark and no new temporal holdout.
- 64-token cap; 15 Phase 5 examples truncated. Calibration and subgroup variation matter.
- ECE is bin-dependent; no test-set calibration fitting. Rule-based perturbations have unvalidated gold labels.
- All 20 selected error categories were supplied by the user; interpretation/comments contain labeled AI-assisted drafts. This is not independent human validation, and the selected sample is not representative.
- Forty numerically unreliable IG records are retained; primary and all-record sensitivity results are distinguished.
- Attention is class-agnostic while IG targets a class logit. Correlation is not faithfulness; deletion tests are imperfect interventions. Special-token attention is not inherently unfaithful.
- Sentence bootstrap intervals ignore possible remaining topical dependence and are not multiplicity-adjusted.
- A known authorized annotation CSV change differs from the original Phase 3 manifest; that manifest remains unchanged.

## References

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) · [General BERT](https://huggingface.co/google-bert/bert-base-uncased) · [Financial pretraining checkpoint](https://huggingface.co/yiyanghkust/finbert-pretrain) · [ProsusAI reference](https://huggingface.co/ProsusAI/finbert) · [Guo et al., calibration](https://proceedings.mlr.press/v70/guo17a.html) · [Sundararajan et al., IG](https://arxiv.org/abs/1703.01365) · [Jain & Wallace, Attention is not Explanation](https://aclanthology.org/N19-1357/) · [Wiegreffe & Pinter, Attention is not not Explanation](https://aclanthology.org/D19-1002/).

This experiment evaluates related diagnostics, not every control proposed in those papers. It supports neither a universal dismissal nor a universal endorsement of attention explanations.
''')
    write(pub/'MODEL_CARD.md','''# Model card: frozen FinBERT explanation checkpoint

## Identity and intended use

Primary artifact: `artifacts/phase2/finbert/seed-42/model`, a three-class BERT sequence classifier fine-tuned from `yiyanghkust/finbert-pretrain`, revision `88ab954a39ea6d3ce2b62cff086dd5ad1172c664`. It has 12 layers and 12 attention heads. Seed 42 was fixed for explanations, not chosen by test performance. Weights are excluded from the publication package.

English financial sentence sentiment research: negative=0, neutral=1, positive=2. Not validated for returns forecasting, investment decisions, document-level analysis, multilingual use, lending or automated high-stakes deployment. Misclassification costs and temporal/domain drift have not been prospectively validated.

## Training and evaluation

Financial PhraseBank 50agree; frozen duplicate-group split 3391/728/727. Train-only preprocessing; validation-only search. Selected financial BERT setting: learning rate 2e-5, four epochs, max length 64, per-device batch 8 and accumulation 2, FP16 CUDA. Final training seeds: 42, 123, 2026. This card documents completed training; Phase 6 did not retrain or retune.

Three-seed financial BERT macro-F1 0.8339 +/- 0.0093 and accuracy 0.8469 +/- 0.0097; these are aggregate results, not seed-42-only claims. General BERT has higher mean macro-F1 0.8476. Allagree/non-allagree financial BERT macro-F1 0.9553/0.7262. ECE-10 mean 0.0923 with SD 0.0556. See the validation report for per-class, subgroup and calibration context. ProsusAI/finbert is a non-independent reference, not the same model as this checkpoint.

## Explanation protocol and caveats

Captum Layer IG of original predicted-class logits, embedding-layer output, fixed PAD-content reference and 50/100/200 steps. Forty of 727 records fail absolute completeness tolerance; they are retained and flagged. Alternative MASK references materially alter rankings. Final-layer CLS attention is head-averaged; agreement and token MASK deletion do not prove causality or faithfulness. WordPiece interventions and mask-induced distribution shift limit interpretation.

## Data, rights, provenance and release status

Financial PhraseBank is listed as CC BY-NC-SA 3.0. The financial pretraining card lacks explicit license metadata; rights for the exact pinned weights and redistribution require clarification. The project code license has not been selected. No license is invented or granted by this card. No checkpoints or raw dataset are released by default. Raw source data can contain formatting/encoding defects and annotator ambiguity. Human error categories exist for a selected 20-row sample; accompanying AI-assisted notes are not independent human assessments.

Maintainer/contact and final license must be set by the repository owner before publication. This card is evidence documentation, not production certification.
''')
    write(pub/'LICENSE_REVIEW.md','''# License and publication review

Status checked 2026-10-29. This is a release inventory, not a substitute for rights-holder/legal review.

| Component | Observed source | Release action |
|---|---|---|
| Project source | No code LICENSE selected | Owner must choose a license and confirm ownership; no permission grant is fabricated |
| Financial PhraseBank | [Dataset card](https://huggingface.co/datasets/takala/financial_phrasebank): CC BY-NC-SA 3.0 | Raw data and row-level derivatives excluded; obtain source separately; retain attribution/noncommercial/share-alike obligations |
| General BERT | [Model card](https://huggingface.co/google-bert/bert-base-uncased): Apache-2.0 | Review pinned revision notices before any weight redistribution |
| Financial pretraining | [Model card](https://huggingface.co/yiyanghkust/finbert-pretrain): missing explicit license metadata | Clarify exact pinned checkpoint and corpus rights before distribution/commercial use |
| ProsusAI reference | [Current model card](https://huggingface.co/ProsusAI/finbert): CC BY-NC-SA 4.0 | Non-independent reference; weights excluded; review exact historical revision rights |
| Aggregate results and explanation fragments | Derived from benchmark/model evaluation | Retain provenance/citations; confirm any applicable derivative obligations before release |

Attempts to retrieve the pinned model-card revisions through the web tool were unavailable; current card labels are not presented as proof of the historical revision's rights. Original packages and notices remain in the environment and are not redistributed. Dataset/checkpoint access is not automatically licensed by a code license.

The package contains code, configs, tests, aggregate figures, the report, and documentation. No raw sentences, original per-example prediction exports, model weights, Hugging Face cache, credentials, environment binaries or private artifact snapshots are included. Do not use `git add .` on the working research directory; review the prepared allowlist package and its file inventory instead.

Manual gates: code license; maintainer/contact; exact model/data rights; publication visibility; final secret/privacy review; repository destination; explicit approval to push. No push, remote creation or commit was performed.
''')
    write(pub/'REPRODUCIBILITY.md','''# Reproducibility checklist

- [x] Frozen test ID/text/label alignment and hashes checked; no retraining or result changes.
- [x] All 30 scientific input/output manifests audited, including smoke outputs; known annotation exception documented.
- [x] Reported numeric claims regenerated and classification/calibration/robustness/statistics recomputed from saved records.
- [x] Seed means and sample SD use ddof=1; explanation uncertainty uses 2000 sentence-level bootstrap resamples, seed 2042.
- [x] Forty unreliable IG records retained, thresholds unchanged; 687 versus 727 sensitivity and exclusion composition reported.
- [x] User error categories counted without inventing assessments; AI-assisted notes labeled separately.
- [x] Original files preserved; original root README remains intact and publication README is prepared separately.
- [x] Direct dependency pins, observed Torch constraint, pinned model revisions and source configuration supplied.
- [x] Reproduction commands checked against real parsers without dispatching training/evaluation.
- [x] Dataset, weights, caches and secret-file patterns ignored by default; curated publication copy provided.
- [ ] Owner chooses the code license and maintainer/contact.
- [ ] Resolve exact pinned-model rights and any data-derived redistribution obligations.
- [ ] Arrange lawful access to original private artifacts for exact historical audit; public code-only bundle is not self-contained model reproduction.
- [ ] Confirm driver/wheel availability for a fresh environment; cross-platform bitwise identity is not promised.
- [ ] Conduct prospective temporal/out-of-domain validation and independent human review before deployment.
- [ ] Review all candidate files, repository destination/visibility, and approve publication explicitly.

Tests and exact completed checks are listed in PHASE6_RESULTS.md and the Phase 6 JSON receipts. Automated secret scans are heuristic, not a guarantee of absence.
''')
    pins=['transformers','datasets','accelerate','captum','numpy','scipy','scikit-learn','huggingface-hub','joblib','matplotlib','pytest','pypdf','pypdfium2']
    write(pub/'requirements/research-pinned.txt','# Observed non-Torch direct dependencies; installation was not performed during Phase 6.\n'+'\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in pins))
    write(pub/'requirements/report-pinned.txt','reportlab==4.4.9\npypdf==6.14.2\npypdfium2==5.11.0')
    write(pub/'requirements/constraints-observed.txt','torch==2.12.1+cu126\n'+ '\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in pins))
    exclusive_json(pub/'commands.json',COMMANDS)
    for filename in ('deletion_curves.png','correlations.png','paired_auc.png'):
        shutil.copyfile(Path('artifacts/phase5/full/figures')/filename,pub/'figures'/filename)
    shutil.copyfile('output/pdf/finbert-model-validation.pdf',pub/'docs/model-validation.pdf')
    text=Path('artifacts/phase6/MODEL_VALIDATION_REPORT.md').read_text(encoding='utf8').replace('../../artifacts/phase5/full/figures/deletion_curves.png','../figures/deletion_curves.png')
    write(pub/'docs/model-validation.md',text)
    exclusive_json(Path('artifacts/phase6')/'command_validation.json',validate_commands(COMMANDS))


SECRET_PATTERNS=[r'ghp_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'hf_[A-Za-z0-9]{25,}',r'AKIA[0-9A-Z]{16}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']


def scan_files(paths):
    findings=[]
    for p in paths:
        if p.suffix.lower() not in ('.py','.md','.json','.txt','.toml','.yml','.yaml','.ps1','.ini'): continue
        text=p.read_text(encoding='utf8',errors='replace')
        for i,line in enumerate(text.splitlines(),1):
            if any(re.search(pattern,line) for pattern in SECRET_PATTERNS): findings.append({'path':str(p),'line':i,'type':'possible_secret','value':'REDACTED'})
    return findings


def stage():
    pub=Path('publication'); target=Path('artifacts/phase6/github-ready')
    if target.exists(): raise FileExistsError(target)
    shutil.copytree(pub,target)
    for folder in ('src','tests','configs','scripts'):
        shutil.copytree(folder,target/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for file in ('pyproject.toml','.gitignore'):
        shutil.copyfile(file,target/file)
    files=sorted(p for p in target.rglob('*') if p.is_file())
    prohibited=[str(p.relative_to(target)) for p in files if p.suffix.lower() in ('.safetensors','.bin','.pt','.pth','.joblib','.arrow','.parquet','.npz','.jsonl','.zip') or p.stat().st_size>50*2**20]
    secrets=scan_files(files)
    links=re.findall(r'\]\(([^)]+)\)',(target/'README.md').read_text(encoding='utf8'))
    missing=[link for link in links if not link.startswith(('http://','https://','#')) and not (target/link.split('#')[0]).exists()]
    inventory=[{'path':p.relative_to(target).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]
    weights=[{'path':str(p),'bytes':p.stat().st_size} for p in Path('artifacts/phase2').rglob('model.safetensors')]
    receipt={'files':len(files),'bytes':sum(r['bytes'] for r in inventory),'prohibited_files':prohibited,'possible_secrets':secrets,'missing_readme_links':missing,
        'checkpoints_excluded':weights,'no_git_mutations_or_network_publication':True,'publication_blocked_pending_owner_approval_and_license_review':True}
    exclusive_json('artifacts/phase6/publication_inventory.json',inventory)
    exclusive_json('artifacts/phase6/publication_check.json',receipt)
    if prohibited or secrets or missing: raise AssertionError('Publication checks failed; review receipt')
    print(json.dumps({k:v for k,v in receipt.items() if k!='checkpoints_excluded'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','stage']);args=parser.parse_args()
    prepare() if args.command=='prepare' else stage()
