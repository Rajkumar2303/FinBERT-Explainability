# Model card: frozen FinBERT explanation checkpoint

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
