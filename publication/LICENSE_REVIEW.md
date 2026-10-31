# License and publication review

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
