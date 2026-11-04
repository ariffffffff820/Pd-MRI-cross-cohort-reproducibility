# Pre-declaration for the embedding arm (1 November 2026)

Written before any embedding was computed.

```
PRE-DECLARATION, 1 November 2026 (written before any embeddings are computed)

Question: do frozen pretrained brain-MRI embeddings plus a linear classifier separate
PD from controls in a cohort never used for training (Neurocon <-> Taowu)?

Models (fixed list, nothing added after seeing results):
 M1  BrainIAC frozen embeddings -> logistic regression      (PRIMARY)
 M2  MedicalNet ResNet-18 frozen embeddings -> logistic regression
 M3  volumes + BrainIAC embeddings, PCA inside the folds -> logistic regression
 M4  volume-only logistic regression (reference, already run)

Protocol: no fine-tuning. Scaling, PCA and C are chosen by 5-fold CV inside the
training cohort only. The test cohort is used once per model and direction. Seed 42.
Baselines in every table: sex only, age only, age + sex.

Pre-checks, reported whatever they show: cohort-identity AUC from the embeddings;
sex and age predictability from the embeddings; whether BrainIAC's pretraining
data include PD datasets or Neurocon/Taowu.

Primary outcome: the two external AUCs for M1, with bootstrap 95% intervals.
A signal is claimed only if BOTH directions have permutation p < 0.0125
(0.05 divided by 4 models) AND beat the sex-only and age + sex baselines.
Sensitivity run without the two outlier scans. All models are reported in one table.
Any deviation from this plan is stated in the paper.
```

## Deviations
- Model M2 was not completed and is not reported.
- The question about BrainIAC's pretraining data could not be settled from the information used.
- The analysis code compares each external AUC with the age + sex baseline only.
