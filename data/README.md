# Data (not redistributed — download from the original repositories)

All four datasets are public under open licences. Use the versions below (the ones analysed); `code/download_figshare.py <article_id> <out_dir>` downloads a figshare article with resume and MD5 checks.

| Folder | Dataset | figshare article (version DOI) | Licence |
|---|---|---|---|
| `data/stroke_acute/` | Liu & Lv, *EEG datasets of stroke patients* (acute stroke, 50 patients; Liu et al., Sci Data 2024) | 21679035 (10.6084/m9.figshare.21679035.v5) | CC BY 4.0 |
| `data/stroke_chronic_hs/` | Luo, *HS Stroke dataset: an upper-limb motor imagery EEG dataset of chronic stroke patients* (14 patients) | 30747806 (10.6084/m9.figshare.30747806.v6) | CC BY 4.0 |
| `data/stroke_tj_lowerlimb/` | Gui et al., *Lower limb motor imagery EEG dataset based on the multi-paradigm and longitudinal-training of stroke patients* (27 patients) | 27130299 (10.6084/m9.figshare.27130299.v1) | CC BY |
| `data/stroke_tj_sixthfinger/` | Wang & Liu, *A multi-paradigm and longitudinal EEG dataset including the "sixth-finger" and "affected-hand" motor imagery of stroke patients* (24 patients) | 30509165 (10.6084/m9.figshare.30509165.v11) | CC BY 4.0 |

```bash
python3 code/download_figshare.py 21679035 data/stroke_acute
python3 code/download_figshare.py 30747806 data/stroke_chronic_hs
python3 code/download_figshare.py 27130299 data/stroke_tj_lowerlimb      # 21.8 GB (rar archives)
python3 code/download_figshare.py 30509165 data/stroke_tj_sixthfinger
```

Extract the archives so that the scripts find these paths:

| Dataset | Archive(s) | Expected layout |
|---|---|---|
| acute | `sourcedata.zip` | `data/stroke_acute/sourcedata/sub-XX/*.mat`, plus `data/stroke_acute/participants.tsv` |
| chronic | `sourcedata.zip` | `data/stroke_chronic_hs/sub-XX/ses-YY/eeg/*run-*_eeg.mat` (session folders directly under the dataset folder) |
| lower-limb | `sourcedata.rar` (main analysis, raw 64 channels); `PEEG.rar` (preprocessed release, sensitivity only) | `data/stroke_tj_lowerlimb/sourcedata/sub-XX/ses-N/*_ori.set`; `data/stroke_tj_lowerlimb/PEEG/sub-XX/ses-*/eeg/*.set` |
| paretic-hand / sixth-finger | `sourcedata.zip` | `data/stroke_tj_sixthfinger/sourcedata/sub-XX/{pre,post,follow}/*_original.set` |

**Duplicated records.** A signal-level audit (`code/duplicate_audit.py`, `results/duplicate_audit.json`) found point-wise identical recordings in the two Tianjin releases: lower-limb `sub-23` pre = `sub-24` pre, paretic-hand `sub-21` pre = `sub-22` pre, and paretic-hand `sub-02` follow-up = `sub-02` post-training. The main analysis excludes both members of each cross-patient pair and the duplicated follow-up (`results/exclusions.json`); results without exclusion are kept as `results/tianjin/*_full.json`.
