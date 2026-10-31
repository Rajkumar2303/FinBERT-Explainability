import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from ..phase3.io import read_json
from ..phase4.artifacts import exclusive_json


def f(x): return f'{x:.4f}'
def ms(x): return f'{x["mean"]:.4f} +/- {x["std"]:.4f}'
def ci(x): return '['+', '.join(f(v) for v in x['mean_ci95'])+']'


def content(audit):
    p3=read_json('artifacts/phase3/summary.json'); p4=read_json('artifacts/phase4/initial/summary.json')
    p5=read_json('artifacts/phase5/full/summary.json'); sens=read_json(audit/'ig_sensitivity.json')
    ann=read_json(audit/'annotation_summary.json'); receipt=read_json(audit/'receipt.json')
    stats=read_json('artifacts/phase1/statistics.json'); cls=p3['classification_and_calibration']
    pages=[]
    def page(title):
        blocks=[];pages.append((title,blocks));return blocks
    b=page('Decision and benchmark performance')
    b += [('p','Model validation report | FinBERT Explainability | 29 October 2026 | Research evaluation, not production certification'),
          ('h','Executive decision'),
          ('p','Recommend supervised research or a monitored human-review pilot only. The evidence does not support autonomous trading, lending, or other consequential financial decisions. General BERT is the stronger mean-performing candidate on this benchmark; financial pretraining did not improve the measured three-seed mean. Prioritize temporal and out-of-domain validation, validation-only calibration, error-cost analysis, and rights clearance before deployment.'),
          ('h','Intended use and dataset'),
          ('p','English financial sentence sentiment: negative (0), neutral (1), positive (2). Financial PhraseBank sentences_50agree has 4,846 rows, with 604 negative, 2,879 neutral and 1,363 positive labels; 2,264 rows belong to allagree. Labels reflect annotator judgments, not realized returns or investment value. The task does not establish multilingual, document-level, current-market, or predictive-return capability.'),
          ('t',[['Split','Rows','Negative','Neutral','Positive']]+[[k.title(),str(stats['splits'][k]['rows'])]+[str(stats['splits'][k]['class_counts'][label]) for label in ('negative','neutral','positive')] for k in ('train','validation','test')]),
          ('p','The initial stratified 70/15/15 split was rebuilt deterministically with intact lexical duplicate groups: 8 exact and 49 near pairs, 25 initial crossings and zero final crossings under the declared detector (character 3-5-gram cosine >=0.90). All 2,199 reassigned rows are retained: 2,143 singletons and 56 grouped rows moved because the algorithm rebuilt all assignments, not because there were 2,199 duplicates. Semantic or pretraining overlap is not ruled out.'),
          ('h','Same frozen 727-row test set; three seeds'),
          ('t',[['Independent workflow','Macro-F1, mean +/- SD','Accuracy, mean +/- SD']]+[[{'baseline':'TF-IDF + logistic regression','bert':'General BERT','finbert':'Financial BERT'}[k],ms(v['classification']['macro_f1']),ms(v['classification']['accuracy'])] for k,v in cls.items()]),
          ('p','Sample SD uses ddof=1 across seeds 42, 123, 2026; it is not a confidence interval. TF-IDF fits training text only; model selection uses validation only. The identical baseline scores across seeds do not establish universal determinism. Three seeds give limited variance information. No significance claim about BERT versus FinBERT is made.'),
          ('p','ProsusAI/finbert scores macro-F1 0.8797 and accuracy 0.8900, but was fine-tuned on Financial PhraseBank. Potential overlap makes it a non-independent external reference, not a valid winner over the held-out workflows. It has no three-seed SD.')]
    b=page('Reliability and qualitative review')
    b += [('h','Annotator-agreement subgroups'),
          ('t',[['Workflow','Allagree macro-F1','Non-allagree macro-F1']]+[[k,ms(cls[k]['subgroups']['allagree']['macro_f1']),ms(cls[k]['subgroups']['non_allagree']['macro_f1'])] for k in ('baseline','bert','finbert')]),
          ('p','Allagree: 338 rows (50 negative / 209 neutral / 79 positive). Non-allagree: 389 (41 / 223 / 125). Small class counts and different composition limit conclusions. Agreement is observational; the gap does not show that disagreement causes model errors. Exact annotator vote counts were not released; membership bands cannot recover those counts.'),
          ('h','Raw probability calibration'),
          ('t',[['Workflow','ECE-10','ECE-15','Brier','NLL']]+[[k]+[f(cls[k]['calibration'][m]['mean']) for m in ('ece_10','ece_15','brier','nll')] for k in ('baseline','bert','finbert')]),
          ('p',f'Calibration entries are three-seed means; complete SDs and bin counts are in Phase 3. FinBERT ECE-10 SD is {f(cls["finbert"]["calibration"]["ece_10"]["std"])}. ECE uses equal-width confidence bins; Brier sums squared error across the three classes (range 0-2); NLL clips at 1e-15. No temperature scaling was fitted. Bin-sensitive ECE alone is not a calibrated-risk guarantee.'),
          ('h','Deterministic robustness probes'),
          ('t',[['Workflow','Company flip rate','Numerical flip rate','Negation flip rate']]+[[k]+[f(p3['robustness'][k][typ]['visible_flip_rate']['mean']) for typ in ('company','numerical','negation')] for k in ('baseline','bert','finbert')]),
          ('p','Eligible counts are 104 company substitutions, 109 numerical edits and 375 negations. Rates use model-visible changes; denominators differ by tokenizer. Company and amount replacements are conditional-invariance probes; negation has no prescribed target class. Grammar and sentiment changes are unvalidated. Low flip rates do not establish semantic robustness, and negation need not reverse financial sentiment.'),
          ('h','Completed categories; AI-assisted explanatory notes'),
          ('p',f'All {ann["categories_completed"]}/{ann["rows"]} user-supplied error categories are filled. Counts: '+', '.join(f'{k.lower().replace("_"," ")} {v}' for k,v in ann['category_counts'].items())+'.'),
          ('p','The 20 rows were selected for confident errors and model disagreement, not sampled randomly. Fourteen have at least 25% annotator disagreement; nine are in the 34%-to-less-than-50% band. These proportions cannot describe the full test population. Interpretations and comments in all 20 rows contain labeled AI-assisted drafts; they are not independent human-validated causal assessments. Categories were not recoded to fit a hypothesis. The original Phase 3 report still describes its earlier blank handoff and is preserved as historical.')]
    b=page('Integrated Gradients and exclusion sensitivity')
    b += [('h','Fixed explanation protocol'),
          ('p','Primary checkpoint: saved yiyanghkust/finbert-pretrain fine-tune, seed 42, fixed before explanation generation rather than selected by test performance. Captum LayerIntegratedGradients targets the original predicted-class logit at model.bert.embeddings in eval mode. The PAD-content reference preserves CLS, SEP, padding, positions and attention masks. Signed embedding-dimension sums are retained separately from their absolute values. Content rankings exclude specials/padding; Phase 5 also excludes punctuation.'),
          ('p','Integration uses the unchanged 50/100/200-step schedule, absolute residual <=0.001 and relative residual <=0.01, with denominator floor 1e-6. Completeness includes every embedding position. No thresholds were changed post hoc. Forty of 727 records fail the absolute tolerance; their maximum absolute residual is 0.015395. All relative residuals remain below 1%. These records are saved and explicitly marked unreliable, not silently discarded.'),
          ('p','Phase 4 compared PAD and MASK references on four fixed examples. Absolute-rank Spearman ranged from 0.1980 to 0.6443; top-5 Jaccard ranged from 0.2500 to 0.6667. Reference choice materially changes rankings even when numerical completeness passes. Both references are artificial attended sequences; neither is a neutral financial sentence or a universally correct baseline.'),
          ('h','Who is excluded?'),
          ('t',[['Population','N','Gold neg/neu/pos','Allagree','Correct','Truncated']]+[[name,str(sens[name]['n']),'/'.join(str(sens[name]['gold_class_counts'].get(str(i),0)) for i in range(3)),str(sens[name]['allagree']),str(sens[name]['correct']),str(sens[name]['truncated'])] for name in ('converged','unreliable','all')]),
          ('p','The excluded records contain no neutral gold labels: 11 negative and 29 positive, with 31/40 allagree and 36/40 correct predictions. Mean eligible-token length is 28.6 versus 24.7 in the converged subset. Exclusion is not random and shifts the analyzed class and agreement mix; primary results should not be described as automatically representative of all 727 examples.'),
          ('t',[['Population','Mean Spearman','Mean Kendall','IG AUC','Attention AUC','Random AUC']]+[[name,f(sens[name]['spearman']['mean']),f(sens[name]['kendall']['mean'])]+[f(sens[name]['auc'][m]['mean']) for m in ('ig_absolute','attention','random')] for name in ('converged','all')]),
          ('p',f'Adding all 40 flagged records shifts mean Spearman by {f(sens["interpretation"]["spearman_mean_shift_all_minus_converged"])}. The paired IG-minus-attention probability-AUC mean is {f(sens["converged"]["paired_ig_minus_attention"]["mean"])} with 95% CI {ci(sens["converged"]["paired_ig_minus_attention"])} for 687 rows, versus {f(sens["all"]["paired_ig_minus_attention"]["mean"])} with {ci(sens["all"]["paired_ig_minus_attention"])} for all 727. Both intervals include zero. The main directional conclusion is unchanged, but metric magnitudes and composition do change; no practical-materiality margin was prespecified.'),
          ('p','The unreliable 40 alone have a positive IG-minus-attention AUC difference, but their IG integration is unreliable and the subset is small and selected. That result is diagnostic, not a basis for declaring attention superior. The full and converged populations overlap; their intervals are not independent comparisons.')]
    b=page('Attention comparison and intervention evidence')
    b += [('h','Correlation is not faithfulness'),
          ('p','The final layer CLS-to-token attention is averaged across 12 heads. All 12 layers and heads have raw CLS rows saved per example. Attention is not conditioned on the chosen class, whereas IG is. Rankings use identical content WordPieces: special IDs, padding and all-Unicode-punctuation tokens are excluded; symbols and numerical content remain. Spearman uses tied average ranks and Kendall uses tau-b; fewer than three tokens or constant vectors would be undefined.'),
          ('p',f'On 687 converged rows, absolute IG-attention Spearman mean is {f(p5["agreement"]["absolute"]["spearman"]["mean"])} (95% CI {ci(p5["agreement"]["absolute"]["spearman"])}), Kendall mean {f(p5["agreement"]["absolute"]["kendall_tau_b"]["mean"])}. Signed-IG Spearman is {f(p5["agreement"]["signed"]["spearman"]["mean"])}. These are modest average rank associations, not proof of causal explanation quality.'),
          ('h','Fixed-rank MASK deletion'),
          ('p','Budgets are 0%, 5%, 10%, 20%, 30%, 50%, 75% and 100% of eligible tokens, rounded upward. Positions and masks remain fixed. The original predicted-class probability/logit are measured after masking. AUC integrates actual masked fractions; lower probability AUC means stronger suppression. Five fixed random orderings are averaged per sentence. Positive-only IG stops when positive scores run out and is a separate partial-coverage sensitivity analysis.'),
          ('image','artifacts/phase5/full/figures/deletion_curves.png',122),
          ('p','Mean probability AUC: absolute IG 0.6555, attention 0.6608, random 0.7866. Paired mean differences versus random are -0.1311 for IG and -0.1258 for attention, with both bootstrap intervals below zero. IG versus attention is not resolved: CI [-0.0130, 0.0024] includes zero. A null-including interval is not proof of equivalence. Heterogeneous per-example behavior remains important.'),
          ('p',f'Any-budget flip rates are {100*p5["auc"]["ig_absolute"]["any_flip"]["mean"]:.1f}% (IG), {100*p5["auc"]["attention"]["any_flip"]["mean"]:.1f}% (attention), and {100*p5["auc"]["random"]["any_flip"]["mean"]:.1f}% (random ordering mean). Every method shares the same 100%-mask endpoint. Different diagnostics need not rank methods identically.'),
          ('h','Special tokens and scope of claims'),
          ('p','Across all 727 examples, final-layer head-mean CLS attention mass is 10.46% on CLS, 29.80% on SEP, 7.04% on punctuation and 52.70% on content (other specials approximately 0.005%; padding zero). Contextual special-token representations can carry information; mass there alone does not establish unfaithfulness.'),
          ('p','MASK replacements induce distribution shift and can remove part of a WordPiece word. These are imperfect interventions, not proof of causality or ground-truth fidelity. Results concern one checkpoint, one layer aggregation and this benchmark. They are related diagnostics to Jain and Wallace (2019), but do not reproduce alternate-attention interventions. Wiegreffe and Pinter (2019) emphasize explanatory definitions and model-level controls; their adversarial-training and multi-seed attention tests were not performed here.')]
    b=page('Reproducibility, release controls and references')
    b += [('h','Audit verdict'),
          ('p',f'{receipt["manifests"]} saved input/output manifests and {receipt["manifest_entries"]} checksum entries were audited, including smoke runs. Classification, calibration, subgroup and robustness metrics were recomputed from saved predictions. Phase 4 sensitivity and Phase 5 bootstrap/statistical summaries were recomputed from saved records; no models were run or trained. All numeric report claims were regenerated with metric labels preserved. {receipt["earlier_files_verified"]} earlier files remain byte-identical.'),
          ('p','Known exception: the Phase 3 human_annotation.csv changed after its manifest was sealed, as authorized by the user. Its current and sealed hashes are both recorded; the old manifest was not rewritten. The historical blank-annotation statement is no longer current. Pytest scratch is outside the scientific manifest inventory. The final audit retains this distinction rather than claiming every old manifest entry currently matches.'),
          ('p','Frozen test SHA256: 97ad15089bd04e0d61726a6218bfcc2bdb18a4bcc1f5e8a299e1da41607e36a9. The existing Python 3.11 rag-env and PyTorch 2.12.1+cu126 were used without replacement. Pinned dependency inventory and model revisions are supplied. Phase 5 used 951.44 s wall time and 0.567 GiB peak allocated CUDA memory; 10 completed IG records were reused.'),
          ('h','Uncertainty and next validation'),
          ('p','Seed SD is not statistical significance. Explainability intervals use 2,000 fixed-seed percentile bootstrap resamples of sentences; paired comparisons resample within-sentence differences. Tokens, heads and random orderings are not independent sampling units. Similar texts may remain dependent, intervals are pointwise and not multiplicity-adjusted, and no temporal/external holdout was evaluated. The 64-token cap truncates 15 Phase 5 examples.'),
          ('p','Next steps: prospectively evaluate a new time period and target domain; validate error costs and abstention; fit calibration only on validation data; obtain independent human review of qualitative notes; assess longer text and semantic leakage; pre-register further reference/attention controls. Preserve this test set as already examined. Do not tune model or attribution settings from these findings.'),
          ('h','Publication readiness and rights'),
          ('p','GitHub publication is prepared, not performed. Raw text, per-example exports, caches and weights are excluded by default. Financial PhraseBank is listed as CC BY-NC-SA 3.0; the current ProsusAI card lists CC BY-NC-SA 4.0; the financial pretraining model has unclear license metadata. Current cards do not establish the exact pinned revision rights. Owner selection of a code license, upstream/pinned-model rights review, final secret review, repository/visibility choice and explicit publication approval remain manual gates.'),
          ('h','References'),
          ('p','[1] Financial PhraseBank, Malo et al. Dataset and source license: https://huggingface.co/datasets/takala/financial_phrasebank\n[2] General BERT and financial pretraining model cards: https://huggingface.co/google-bert/bert-base-uncased ; https://huggingface.co/yiyanghkust/finbert-pretrain\n[3] ProsusAI reference: https://huggingface.co/ProsusAI/finbert\n[4] Guo et al. (2017), calibration: https://proceedings.mlr.press/v70/guo17a.html\n[5] Sundararajan et al. (2017), IG: https://arxiv.org/abs/1703.01365 ; Captum: https://captum.ai/api/layer.html\n[6] Jain & Wallace (2019): https://aclanthology.org/N19-1357/\n[7] Wiegreffe & Pinter (2019): https://aclanthology.org/D19-1002/')]
    return pages


def build(audit='artifacts/phase6/audit-final',reportlab_path=None):
    sys.dont_write_bytecode=True
    if reportlab_path: sys.path.append(reportlab_path)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    from reportlab.lib.pagesizes import A4
    audit=Path(audit); pages=content(audit)
    out=Path('artifacts/phase6'); pdf=Path('output/pdf/finbert-model-validation.pdf')
    md=out/'MODEL_VALIDATION_REPORT.md'
    if pdf.exists() or md.exists(): raise FileExistsError('Report exists; refusing overwrite')
    pdf.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body6',fontName='Helvetica',fontSize=9.1,leading=12,spaceAfter=6,textColor=colors.HexColor('#263445')))
    styles.add(ParagraphStyle(name='ResearchHeading6',fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=6,spaceAfter=5,textColor=colors.HexColor('#126c78')))
    styles.add(ParagraphStyle(name='Title6',fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=14,textColor=colors.HexColor('#142e4c')))
    styles.add(ParagraphStyle(name='Cell6',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=0))
    story=[]; markdown=['# FinBERT Explainability: model validation report','']
    for number,(title,blocks) in enumerate(pages,1):
        if number>1: story.append(PageBreak())
        story.append(Paragraph(f'{number:02d} / {escape(title)}',styles['Title6']))
        markdown+=['## '+title,'']
        for block in blocks:
            if block[0]=='h': story.append(Paragraph(escape(block[1]),styles['ResearchHeading6']));markdown+=['### '+block[1],'']
            elif block[0]=='p': story.append(Paragraph(escape(block[1]).replace('\n','<br/>'),styles['Body6']));markdown+=[block[1],'']
            elif block[0]=='t':
                rows=block[1]; cells=[[Paragraph(escape(str(c)),styles['Cell6']) for c in row] for row in rows]
                widths=([145,170,170] if len(rows[0])==3 else [485/len(rows[0])]*len(rows[0]))
                table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2eef2')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#b9c9d4'))]))
                story.extend([table,Spacer(1,8)])
                markdown+=['| '+' | '.join(str(c) for c in row)+' |' for row in rows[:1]]+['| '+' | '.join('---' for _ in rows[0])+' |']+['| '+' | '.join(str(c) for c in row)+' |' for row in rows[1:]]+['']
            else:
                story.extend([Image(block[1],width=485,height=block[2]),Spacer(1,8)])
                markdown += [f'![Deletion curves](../../{block[1]})','']
    def footer(canvas,doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#ccd8e0'));canvas.line(54,42,541,42)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526577'))
        canvas.drawString(54,29,'FINBERT EXPLAINABILITY  |  MODEL VALIDATION  |  RESEARCH USE')
        canvas.drawRightString(541,29,str(doc.page)); canvas.restoreState()
    SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=40,bottomMargin=54,
                      title='FinBERT Explainability - Model Validation',author='FinBERT Explainability Project').build(story,onFirstPage=footer,onLaterPages=footer)
    with md.open('x',encoding='utf8') as stream: stream.write('\n'.join(markdown))
    import pypdfium2 as pdfium
    from pypdf import PdfReader
    count=len(PdfReader(pdf).pages)
    if count!=5: raise AssertionError(f'Expected five report pages, got {count}')
    render=out/'pdf-review';render.mkdir(exist_ok=True)
    doc=pdfium.PdfDocument(str(pdf))
    for i in range(len(doc)): doc[i].render(scale=1.5).to_pil().save(render/f'page-{i+1}.png')
    exclusive_json(out/'pdf_receipt.json',{'pages':count,'pdf':str(pdf),'markdown':str(md),'renderer':'pypdfium2 (existing rag-env; Poppler unavailable)','reportlab':'4.4.9; read-only bundled package path, no environment installation'})
    print(f'Created {count}-page PDF and matching Markdown.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reportlab-path');parser.add_argument('--audit',default='artifacts/phase6/audit-final')
    args=parser.parse_args();build(args.audit,args.reportlab_path)
