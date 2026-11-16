# Preliminary computational results

Comparison: TB versus OD.

Train: 163; threshold calibration: 41; held-out test: 166.

Thresholds were set on a separate development subset; test labels were used only for reporting.

The development CV scores are tuning diagnostics, not independent validation.

## Held-out results

- elastic_net: AUROC 0.830; sensitivity 0.931; specificity 0.484; TP/FN/TN/FP 95/7/31/33.

- mlp: AUROC 0.817; sensitivity 0.843; specificity 0.625; TP/FN/TN/FP 86/16/40/24.

- compact_candidate: AUROC 0.847; sensitivity 0.892; specificity 0.625; TP/FN/TN/FP 91/11/40/24.

## Interpretation limits

- Deposited TB and OD labels do not establish pulmonary-only TB or respiratory-only comparator disease. GSE37250 blood was collected before or within 24h of treatment.

- Confidence intervals are participant-bootstrap intervals conditional on the fitted model and cohort; they do not quantify between-study variation.

- No claim of clinical utility, causal TB-specific biology, or qPCR transfer is supported.

- Published processed arrays may have been normalized together; country-held-out results are preliminary, not fully independent preprocessing.

- Baseline MLP and cohort-balanced MLP are implemented; pathway-constrained or viral-inflammation auxiliary models are not.

- Compact genes use ranks relative to the full common-gene universe. A new qPCR score and independent assay bridge are required.

- Sensitivity target applies to calibration data only; inspect actual held-out sensitivity.

- Feature stability uses only three training folds and is exploratory.

- MLP probabilities have not been post-hoc probability calibrated; a reliability plot is provided.

- Model selection must not be revised after viewing the held-out test. New experiments need new untouched data.