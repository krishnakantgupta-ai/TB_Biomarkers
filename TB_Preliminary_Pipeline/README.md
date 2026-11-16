# TB host-response preliminary-results pipeline

Prepared for Dr. Krishna Kant Gupta's proposed ICMR study.

## What this package does

Downloads real GEO processed microarrays, parses metadata, maps unambiguous probes to gene symbols, takes a fixed shared gene universe, performs sample-wise percentile ranking, and generates an auditable computational pilot. It trains elastic-net logistic regression, a two-hidden-layer neural network (64/16), and a compact candidate-gene classifier. A cohort-balanced neural-network ablation runs only if training contains at least two countries with both classes.

**Implemented pilot, not the entire proposed novel method:** pathway-constrained networks, viral-inflammation auxiliary tasks, published-signature benchmarking, uncertainty/abstention and RT-qPCR translation are NOT implemented. The code does not fabricate results or require a GPU. Deep learning uses scikit-learn's multilayer neural network; PyTorch is not required.

## Mac / Linux setup

Unzip the archive, open Terminal and change into its `TB_Preliminary_Pipeline` directory. Use your normal Python 3.11 or 3.12 installation. For example, on your Homebrew Mac:

```bash
cd ~/Downloads/TB_Preliminary_Pipeline
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Homebrew Python is elsewhere, use `python3 -m venv .venv` instead. Run within this virtual environment.

Windows PowerShell setup:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively use `.venv\Scripts\python.exe` directly if activation is restricted.

## Run the complete first pilot

After installation, the simplest option is:

```bash
python run_all.py
```

This runs the four stages below sequentially. To run each stage yourself:

```bash
python tb_pipeline.py --work tb_work fetch --series GSE37250
python tb_pipeline.py --work tb_work prepare
python tb_pipeline.py --work tb_work curate37250
python tb_pipeline.py --work tb_work analyze
```

- `fetch`: downloads ~100 MB compressed expression data over HTTPS. Allow a few hundred MB disk space and roughly 2-4 GB available RAM. Speed depends on GEO/network availability.
- `prepare`: downloads the GPL annotation, maps genes, exports raw metadata and an unapproved sample manifest.
- `curate37250`: applies a narrowly scoped, source-checked rule for this exact dataset. It checks 537 samples and the observed diagnosis/country counts; derives study-code identifiers from the unique `WB_group_site_number` title suffixes; records the source and the assumption explicitly. If counts/title patterns change, it stops for review. Original manifest is backed up.
- `analyze`: uses South Africa for model training/threshold calibration and **Malawi as the held-out country**. Only TB and OD enter this primary pilot. It does not use test outcomes to select genes, parameters or thresholds.

No edits are required for this specific deposited GSE37250 pilot if source checks pass. This is cohort-level source curation, not independent verification of original clinical records. The study-code interpretation and exact-profile duplicate screen cannot exclude every hidden duplicate or metadata error.

The paper says blood was collected **before OR within 24 hours of starting TB treatment**. The software records `baseline_or_within_24h`, not 'all untreated'. 'OD' is broad other disease, not necessarily respiratory infection. TB labels are not treated as pulmonary-only. Thus the first result is active TB versus deposited other diseases, not definitive validation of the narrower prospective respiratory-triage target.

Default seed: 42; 500 training-selected high-variance genes; 200 neural-network epochs; three initialization seeds averaged; 1,000 participant-bootstrap replicates. Hyperparameters for the neural network are fixed in advance. SAGA elastic-net C/l1-ratio selection uses training-only 3-fold grouped CV. Threshold selection uses a separate development subset, targeting 90% sensitivity there. Actual held-out sensitivity can be lower.

The cohort-balanced MLP is normally **skipped in this country-held-out pilot**, because only South Africa is in training. This is intentional; two countries cannot both train a cohort-aware model and provide a new held-out country. More independent development cohorts are needed for that ablation.

You can run a quick software check with `--epochs 10 --bootstrap 100 --skip-neural`, but do not change settings repeatedly after viewing held-out results. A new setting chosen in response to the test invalidates its untouched-test status. The code refuses to overwrite an existing results folder.

## Outputs

Look in `tb_work/results/`:

| File | Meaning |
|---|---|
| PRELIMINARY_REPORT.md | Actual computed results and interpretation limits |
| heldout_metrics.csv | AUROC/AUPRC, sensitivity/specificity, counts, Brier score and bootstrap intervals |
| heldout_predictions.csv | Every held-out sample's prediction and deposited label |
| sample_partitions.csv | Train, threshold-calibration, test and unused membership |
| analysis_sample_counts.csv | Diagnostic/country/HIV counts by partition |
| all_sample_counts.csv | Manifest counts before comparison selection |
| heldout_curves.png | ROC, precision-recall and reliability plots |
| confusion_matrices.png | True/false positive and negative counts |
| pca_audit.png | PCA fitted on training only; all included rows projected |
| sample_flow.png | Counts through analysis partitioning |
| gene_stability.csv / .png | Exploratory selection frequency across three training folds |
| candidate_panel.csv | Up to eight development-selected candidate genes |
| subgroup_metrics.csv | HIV, country, class and available inflammation subgroup summaries |
| thresholds.json | Locked thresholds plus calibration-set diagnostics |
| training_tuning_results.csv | Training CV parameter search; NOT independent validation |
| run_provenance.json | Versions, arguments, input checksums and limitations |
| models.joblib | Fitted models, gene order and thresholds; load only trusted files |

Preprocessing is fitted on original training samples before any cohort-balanced resampling. The compact panel uses within-sample ranks relative to the FULL common-gene set. It is a computational shortlist; its score cannot be measured directly from eight qPCR values. An assay bridge and a newly validated qPCR-compatible score are required.

Candidate selection is based on training-fold coefficient stability. It is exploratory, not proof of biological specificity. No causal claims are made from coefficient magnitudes.

## Independent GSE19491 analysis

Plan this BEFORE viewing its outcomes. Use a separate workspace from the start:

```bash
python tb_pipeline.py --work external_work fetch --series GSE37250 GSE19491
python tb_pipeline.py --work external_work prepare
python tb_pipeline.py --work external_work curate37250
```

Now review `external_work/prepared/metadata_raw.csv` and the original publication/subseries for the GSE19491 rows. Edit `sample_manifest.csv` in a spreadsheet editor and save as CSV. For each included external row, establish:

- accurate label TB / OD / LTBI / HC;
- globally consistent participant ID across accessions;
- whole-blood specimen;
- baseline treatment status (`pretreatment`, `untreated`, `baseline_untreated`, or documented `baseline_or_within_24h`);
- one eligible sample per participant;
- include=yes, reviewed=yes, provenance_verified=yes, with a source note.

Do not mark rows reviewed solely to bypass a stop. Longitudinal follow-up, separated cell types and uncertain labels must be excluded. Site, HIV and respiratory subtype information must not be invented.

For a verified TB-versus-LTBI comparison:

```bash
python tb_pipeline.py --work external_work analyze --development-series GSE37250 --test-series GSE19491 --test-country "" --controls LTBI
```

The empty `--test-country ""` means hold out the entire external series. Change `--controls` only according to the prespecified, verified comparison. Availability of LTBI/healthy controls does not validate TB-vs-respiratory-disease specificity. A study/site inventory and cross-accession linkage review remain necessary.

External platform mapping is based on the intersection of annotated genes (no external expression statistics or labels). Sample-wise rankings use that fixed universe. For prospective deployment the same universe and measurement platform assumptions must hold.

If a platform's automatic GEO annotation is unavailable, supply `WORK/raw/GPLxxxx_probe_to_gene.csv` with columns `probe_id,gene`, document its source, then rerun `prepare`. Ambiguous multi-gene probes are dropped. Annotation symbols are from the deposited GPL annotation; no claim of current HGNC remapping is made.

## Interpretation limits

1. GSE37250 is a case-control study with deposited processed arrays; those arrays were normalized together in the source study. A held-out country is independent of model fitting but not necessarily of original normalization. An independent accession is a stronger next test.
2. These cohorts are not a prospective Indian respiratory-triage population. Report source spectrum and sampling explicitly.
3. High sensitivity on the calibration subset is not a guarantee on a new cohort.
4. The reported probability reliability curve is an evaluation, not a fitted calibration procedure.
5. Bootstrap intervals condition on this model/cohort, exclude assay and between-study uncertainty, and are not evidence of clinical readiness.
6. The model does not distinguish viral inflammation unless validated subtype labels and a dedicated model are added later. Generic OD is never automatically relabelled viral or bacterial.
7. ICMR proposals should report measured values, exclusions and negative findings, not select the most favorable seed or contrast.
8. Institutional review of public-data secondary analysis requirements remains your institution's responsibility; this package does not issue ethics approval.

## Tests and reproducibility

```bash
python -m unittest discover -s . -p 'test_pipeline.py' -v
```

Software tests use tiny artificial fixtures clearly labelled as software-only. They are not research data and produce no proposal results. Six unit checks cover parsing, labels, ambiguous-probe exclusion, sample-local ranking, group separation and threshold selection. The analysis workflow was also checked end-to-end, including saved-model reload, on a discarded artificial fixture. Real GEO download, parsing and probe annotation were checked for both GSE37250 and GSE19491.

`AUDIT_SNAPSHOT/` contains actual metadata audit outputs from the development run, not fabricated training scores. `EXAMPLE_REAL_PILOT/`, if present, contains a clearly labelled real GSE37250 run with its exact arguments and caveats. Source data are not bundled; download from GEO using the script.

Do not load pickle/joblib objects obtained from untrusted sources. Use the prepared matrix generated by this script.

## Sources

- GEO GSE37250: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37250
- GEO GSE19491: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19491
- Kaforou et al., 2013: https://doi.org/10.1371/journal.pmed.1001538 (sampling/treatment timing and diagnostic group definitions).
- scikit-learn grouped CV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

Research software supplied for inspection and adaptation. Preserve data-provider citations and disclose the actual methods in your grant application.
