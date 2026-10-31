"""Final readability pass; writes documentation only."""
from pathlib import Path
import re

here=Path(__file__).resolve().parent
objectives={
'00':['Explain the task using a financial sentence.','Describe the six-phase workflow and its boundaries.','Give the project story in 30 seconds, 2 minutes and 10 minutes.'],
'01':['Turn words into token IDs and continuous vectors.','Explain contextual mixing, positions and the BERT block.','Calculate probabilities, loss and a small gradient update before using advanced notation.'],
'02':['Interpret agreement thresholds without inventing votes.','Explain stratification, duplicate groups and the 2199 moved assignments.','Distinguish frozen-file integrity from complete protection against leakage.'],
'03':['Trace a sentence through the lexical baseline and BERT classifier.','Check tensor shapes, special tokens and class mappings.','Explain why the financial encoder and ProsusAI reference have different evaluation roles.'],
'04':['Explain a weight update, effective batch size and mixed precision.','Separate validation search, checkpoint selection and final seed fits.','Read the actual configuration, runtime records and reproduction commands.'],
'05':['Calculate accuracy, precision, recall and all three class F1 scores from one real matrix.','Explain macro averaging, seed variation and resampling uncertainty.','Interpret subgroup results and selected errors without overstating them.'],
'06':['Distinguish confidence from correctness using groups of predictions.','Calculate ECE, Brier score and NLL in small examples.','Explain what calibration and text-edit tests do—and do not—establish.'],
'07':['Build IG from local sensitivity and a reference-to-input path.','Explain the exact layer, target, baseline and completeness checks.','Read signed token contributions while preserving numerical and reference limitations.'],
'08':['Explain attention through token relationships before Q, K and V.','Trace the exact final-layer CLS summary used here.','Discuss special-token mass and alternatives without treating weights as automatic causal explanations.'],
'09':['Calculate rank agreement and a deletion area from short examples.','Explain paired bootstrap sampling and the random baseline.','Interpret the 687/727 comparison with the 40 unreliable records visible.'],
}
for path in sorted(here.glob('[0-9][0-9]_*.md')):
    text=path.read_text(encoding='utf8')
    if '## Learning objectives' not in text:
        marker='## Lesson:'
        objective='## Learning objectives\n\n'+ '\n'.join('- '+x for x in objectives[path.name[:2]])+'\n\n'
        text=text.replace(marker,objective+marker,1)
    text=text.replace('A missed negative and a false negative flag','A missed negative sentence and an incorrect negative flag')
    path.write_text(text,encoding='utf8')

# Break the old compact roadmap into readable stages, retaining its concrete controls.
p=here/'11_improvements_and_future_research.md'
t=p.read_text(encoding='utf8')
start=t.index('1. **P0 — Independent evaluation',t.index('## Research notebook D:'))
end=t.index('**Check:** Why is collecting a holdout',start)
roadmap=[
('P0: build an independent evaluation set',
'The historical benchmark may look easier than later, more varied news.',
'Collect lawful, dated news with event IDs and multiple annotators. Define labels and adjudication rules before reviewing model outputs. Deduplicate against all old data; reserve a locked temporal/event-group holdout.',
'Keep the same label policy and model versions. Do not use holdout labels for development.',
'Complete provenance, no detected cross-group overlap, and documented class/agreement support. These are data-quality requirements, not a model-score win.'),
('P1: compare equal-budget optimization',
'A wider, multi-seed validation search may reduce FinBERT’s gap or variation.',
'Keep training rows, input length and optimizer-step accounting comparable. Give both encoders the same declared compute budget and prespecified seeds. Select on development data; confirm once on the new holdout.',
'Match data access and resources; log every trial, including failures.',
'A prespecified useful macro-F1 improvement with paired uncertainty and no unacceptable minority-recall loss. Set the numeric margin from use-case needs before seeing results.'),
('P1: test calibration and a review policy',
'A validation-fitted temperature may improve probability quality without changing class choices.',
'Reserve calibration data or use an appropriate development cross-fitting scheme. Fit temperature while keeping model weights fixed. Evaluate NLL, Brier, ECE and accepted-case error versus coverage on independent data.',
'Use identical model logits and evaluation rows before/after fitting; do not tune on the holdout.',
'Improved held-out probability scores with uncertainty and an agreed review workload/error target. One favorable ECE binning alone is insufficient.'),
('P1: test quantities and relations directly',
'Some errors may concern comparisons, entity roles or negation rather than missing financial vocabulary.',
'Create human-reviewed matched pairs: revenue up/down, losses up/down, percentages with clear denominators, swapped entity roles and conflicting clauses. Keep challenge-test labels out of training.',
'Match vocabulary, length and amount format where feasible. Blind reviewers to model predictions.',
'Prespecified contrast accuracy/consistency with class-specific results, plus no unacceptable deterioration on natural independent news.'),
('P2: test one motivated intervention at a time',
'One diagnosed bottleneck may respond to warmup, weighting, soft labels or news adaptation.',
'Use the previous diagnostics to choose an intervention. Start with one-factor comparisons; use a factorial design only if interactions matter and resources allow.',
'Common initialization/tokenizer where feasible, equal steps and data access, paired seeds. Document unavoidable differences.',
'A result that replicates on a second independent slice while meeting calibration and compute safeguards. Report negative outcomes too.'),
('P2: broaden explanation checks',
'The ordering of explanation methods may depend on reference and intervention choices.',
'Prespecify PAD/MASK or multiple-reference comparisons, whole-word masking and natural contrast tests on a new sample. Add parameter-randomization checks and more than one checkpoint seed.',
'Fixed targets, sampling units, budgets and convergence thresholds. A numerical investigation gets a new version; historical records remain untouched.',
'Transparent effect/stability intervals across protocols. Do not choose a winning protocol afterward or silently redefine numerical success.'),
('P2: measure a deployable cost–quality trade-off',
'A student or cascade may satisfy a latency target with acceptable error.',
'Measure warm and cold starts, input-length mixtures, batch size and concurrency. Evaluate abstention/review and a rollback plan alongside prediction performance.',
'Fixed hardware, precision, workload and independent traffic-like data.',
'Prespecified p95 latency, memory, calibration and error-cost bounds. Passing them would support a later deployment decision, not automatically authorize publication or deployment.'),
]
parts=[]
for title,hyp,steps,controls,success in roadmap:
    parts.append(f'### {title}\n\n**Question to test:** {hyp}\n\n**Steps:** {steps}\n\n**Keep controlled:** {controls}\n\n**Acceptance rule:** {success}\n\n**Check:** What must be fixed before the final evaluation? **Answer:** The intervention, selection rule, comparison, acceptance criteria and access to the independent holdout.')
t=t[:start]+'\n\n'.join(parts)+'\n\n'+t[end:]
p.write_text(t,encoding='utf8')

# Every substantial non-lesson teaching section should close with a check.
missing=[]
for p in sorted(here.glob('[0-9][0-9]_*.md')):
    t=p.read_text(encoding='utf8')
    for block in re.findall(r'^## .+?(?=^## |\Z)',t,re.M|re.S):
        title=block.splitlines()[0]
        if any(x in title for x in ['Learning objectives','Practice explaining','Sources and next']):
            continue
        if not ('**Check' in block or '**Question:**' in block):
            missing.append((p.name,title))
print('Major sections without a concept check:',missing)
