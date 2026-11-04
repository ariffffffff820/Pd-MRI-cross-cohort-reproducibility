# Data

Raw images are not redistributed here.

## Sources
- Neurocon (43 participants) and Taowu (40 participants), T1-weighted MPRAGE, listed on the Canadian Open Neuroscience Platform (CONP) under CC BY-NC-SA.
- Both were downloaded from the INDI project's public Amazon S3 bucket (`fcp-indi`, prefix `data/Projects/INDI/umf_pd/`, files `neurocon.tar.gz` and `taowu.tar.gz`). The scripts download them with unsigned requests, no account needed.

## What the analysis needs
- `volumes.csv`: FastSurfer regional volumes per participant (columns include cohort, sid, label and about 110 volumes).
- `subject_table_with_demographics.csv`: cohort, subject, age, sex. Age and sex come from the metadata files shipped with each dataset (`neurocon_patients.tsv`, `taowu_patients.tsv`).

## Before sharing derived tables
Derived tables are adaptations of CC BY-NC-SA data, so they must carry the same licence and attribution. Do not publish dates of birth or scan dates, which appear in the Neurocon metadata file. The analysis does not use them.
