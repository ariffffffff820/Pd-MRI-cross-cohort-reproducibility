# Cross-cohort reproducibility of structural MRI findings in Parkinson's disease

Regional brain volumes and pretrained embeddings in two public cohorts (Neurocon and Taowu).

**Research use only. Nothing here is a diagnostic tool.**

## What this project asks

Do PD-versus-control differences in regional brain volumes, and classifiers built on them, carry over between two independent public cohorts?

- 83 participants: Neurocon (27 PD, 16 controls) and Taowu (20 PD, 20 controls), T1-weighted MRI.
- Features: FastSurfer regional volumes, normalised by brain-mask volume (20 subcortical, 62 cortical, 94 structures in total).
- Models: L2 logistic regression with tuning inside the training cohort only; train on one cohort and test on the other, in both directions; bootstrap intervals and permutation tests.
- Baselines: sex only, age only, age + sex. Cohort-identity check. ComBat harmonisation (fitted without diagnosis). Effect-size replication between cohorts.
- Pretrained arm: frozen BrainIAC embeddings with logistic regression, under a decision rule written down before the embeddings were computed (`docs/pre_declaration_2026-11-01.md`).

## Main results (all 83 participants)

| Features | Internal, Neurocon | Internal, Taowu | Neurocon to Taowu | Taowu to Neurocon |
| --- | --- | --- | --- | --- |
| Sex only | 0.62 | 0.32 | 0.48 [0.32, 0.64] | 0.31 [0.16, 0.46] |
| Subcortical volumes (20) | 0.58 | 0.61 | 0.57 [0.38, 0.74] | 0.60 [0.42, 0.79] |
| Cortical volumes (62) | 0.56 | 0.45 | 0.54 [0.36, 0.72] | 0.56 [0.39, 0.74] |
| All structures (94) | 0.52 | 0.48 | 0.55 [0.37, 0.73] | 0.52 [0.34, 0.70] |
| BrainIAC embeddings + LR | 0.25 | 0.47 | 0.54 [0.36, 0.71] | 0.50 [0.30, 0.70] |

AUC values; external cells show bootstrap 95% intervals. Internal values below 0.5 for features without signal are a cross-validation artefact. The sex-only value of 0.31 (Taowu to Neurocon) is a real reversal, because the sex-diagnosis association differs between the cohorts.

- No model met the pre-declared criterion for a transferable signal.
- PD-versus-control effect sizes did not correlate between cohorts (Spearman r = 0.00 across 20 subcortical regions).
- The cohorts are separable from volumes (AUC 0.81 to 0.86) and from embeddings (0.91). ComBat removed this without changing classification.
- With 40 to 43 participants per test cohort, only AUCs of about 0.74 or higher are detectable. The conclusion is the absence of evidence at this sample size, not evidence of absence.

## Repository layout

```
code/
  pd_pipeline.py          download, master table, metadata dump, registration and QC (Kaggle-oriented)
  pd_analysis.py          baselines, volume models, cohort identity, ComBat, effect-size replication
  pd_embedding_eval.py    evaluation of embeddings under the pre-declared protocol
notebooks/
  01_fastsurfer_volumes.ipynb               FastSurfer segmentation and volumes.csv (GPU, about 4 hours)
  02_brainiac_embeddings_full_run.ipynb    preprocessing and BrainIAC embedding extraction (GPU)
docs/
  pre_declaration_2026-11-01.md
results/
  figures/                three manuscript figures
  tables/                 Table 1 (demographics)
data/                     see data/README.md (raw data are not redistributed)
```

## Reproducing

1. Run `notebooks/01_fastsurfer_volumes.ipynb` (it downloads the public data and runs FastSurfer in segmentation-only mode on the original T1-weighted scans to produce `volumes.csv`). The per-scan command is: `run_fastsurfer.sh --t1 <scan> --sid <id> --sd <output> --seg_only --allow_root --vox_size 1 --no_cc --no_cereb --no_hypothal`.
2. Run the main analysis (CPU):
   `python code/pd_analysis.py --vol volumes.csv --demo subject_table_with_demographics.csv --out results/main`
   and the sensitivity run without the two outlier scans:
   `python code/pd_analysis.py --vol volumes.csv --demo subject_table_with_demographics.csv --out results/excluded --exclude sub-patient032037 sub-patient032078`
3. Extract BrainIAC embeddings with the notebook (GPU, internet on, BrainIAC checkpoint attached), then evaluate:
   `python code/pd_embedding_eval.py --vol volumes.csv --demo subject_table_with_demographics.csv --emb brainiac=embeddings_brainiac.csv --out results/embeddings`

All random seeds are 42. Package versions and settings are written to a run-information file with each result.

## Data

Neurocon and Taowu are public datasets released under CC BY-NC-SA. Raw images are not included here. See `data/README.md`.

## Limitations

Two cohorts and 83 participants; cohort and site cannot be separated; the brain-mask volume is a proxy for intracranial volume; volumes come from a segmentation-only pipeline; ComBat was fitted on both cohorts together without labels; the embeddings were frozen. Full list in the thesis.

## Licence

Code: MIT (see `LICENSE`). Data and derived data tables: CC BY-NC-SA, as for the source datasets.

## Citation

[Add the thesis or preprint reference here when available.] Dataset source: Badea et al., 2017, PLoS ONE (confirm details).
