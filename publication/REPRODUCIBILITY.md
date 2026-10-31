# Reproducibility checklist

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
