# Validation notes

- Real data download and preparation: GSE37250 (537 arrays, 20,750 mapped gene symbols) and GSE19491 (498 arrays, 19,593 mapped gene symbols).
- GSE19491's 498 arrays have NOT been treated as 498 independent eligible whole-blood participants. No external GSE19491 model evaluation is bundled.
- Six software unit tests passed.
- End-to-end software-only fixture test exercised elastic-net, ordinary MLP, cohort-balanced MLP, compact-candidate selection, 18 result files and model serialization/reload. Artificial fixture data and scores were discarded, not bundled.
- Actual source-curated GSE37250 pilot ran once with default model settings: seed 42, 500 variance-selected genes, 200 MLP epochs, three fixed initialization seeds, 1,000 bootstrap replicates. Software bug fixes before this run did not use test performance to select models.
- Training: 163 South African samples. Separate threshold calibration: 41 South African samples. Held-out Malawi: 166 samples (102 TB; 64 other disease).
- AUROC: elastic-net 0.829810; MLP 0.817402; compact candidate approximately 0.847. Exact full-precision values and confidence intervals are in EXAMPLE_REAL_PILOT/heldout_metrics.csv and take precedence over these rounded notes.
- No pathway-constrained model, viral-inflammation auxiliary model or clinical assay validation is represented by these results. Cohort-balanced ablation skipped in the real country-held-out run because development has only one country.
- Modest specificity and any failure to retain calibration sensitivity should be reported. Do not select favorable models or thresholds after this held-out evaluation and claim a fresh independent test.
- Public processed matrices had source-study normalization; this is a preliminary country-held-out reanalysis, not fully independent acquisition/preprocessing validation.
