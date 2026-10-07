# stroke-mi-period-detection-benchmark

Analysis code and per-patient results for:

> **Benchmarking motor-imagery period detection in stroke EEG across four public datasets: fixed rules, trained classifiers, calibration budgets and a specificity audit.** Yao Xie and Lirui Sun, School of Information Science and Technology, University of Science and Technology of China, 2026 (manuscript submitted).

The study compares a fixed C3/C4 power rule, two-feature LDA, CSP+LDA and an untuned Riemannian tangent-space classifier for detecting cued motor-imagery periods in four public stroke EEG datasets, under one leakage-controlled protocol (physical-trial folds, out-of-sample reference estimation, time-prefix evaluation across sessions), and audits how spatially and physiologically specific the discrimination is.

## What is here

| Path | Content |
|---|---|
| `code/` | All analysis, statistics, figure and supplement scripts (Python; comments partly in Chinese) |
| `results/` | Per-patient result files (JSON) for every analysis, the statistical families, the duplicate audit, and the final figures (`results/figs_paper2/`) |
| `results/tianjin/*_full.json` | Tianjin results **before** excluding duplicated records; the files without `_full` are the main (filtered) analysis |
| `docs/` | Generated tables: single-source number table (`numbers_paper2.md/json`), statistical families (`tables_paper2.md`, `tianjin_stats.md`, `strengthen_stats.md`), duplicate audit and its sensitivity, and the supplementary material in Markdown (`paper2_supplement.md`) |
| `data/README.md` | Where to download the four datasets (not redistributed) and the expected folder layout |
| `reproduce.sh` | The pipeline: `stage1` (needs the datasets) and `stage2` (from the JSON files in `results/`) |

Some intermediate tables in `docs/` carry Chinese row labels; the English versions of all tables are in `docs/paper2_supplement.md`.

## Quick check (no data needed)

```bash
pip install -r requirements.txt          # Python 3.11; versions as used for the paper
bash reproduce.sh stage2
git status                               # should show no changes
```

Stage 2 recomputes all statistical families (patient-level Wilcoxon signed-rank tests with Holm correction, patient-bootstrap intervals, Hodges–Lehmann estimates), the number table, the seven figures and the supplementary Markdown from the result files. Before release this was run in a clean copy and reproduced every committed table byte-for-byte and every figure pixel-for-pixel.

## Full pipeline

1. Download and extract the datasets as described in `data/README.md`.
2. `bash reproduce.sh all` (stage 1 takes several hours on 8 cores). Stage-1 commands are taken from the scripts' documented options and the settings recorded in the result files; the main acute analyses use the cue reference (`REST_MODE=cue`), the main paretic-hand analysis the late rest windows (`SF_REST=late`), and the main lower-limb analysis the raw release (`LL_SRC=raw`).

Main scripts: `stroke_mi_vs_rest.py` (acute), `stroke_chronic_mi_vs_rest.py` (chronic), `tianjin_analysis.py` (lower-limb and paretic-hand), `paper2_followups.py`, `rule_montage_variants.py`, `paper2_strengthen.py` (Riemannian baseline, band/occipital, lateralisation, EOG regression, filter transient, held-out window sensitivity), `baseline_stroke.py` (re-implementation of the original left/right baselines), `duplicate_audit.py` and `apply_exclusions.py`.

## Data integrity note

A signal-level audit found point-wise identical recordings under different patient identifiers in the two Tianjin releases (lower-limb `sub-23`/`sub-24` pre-training; paretic-hand `sub-21`/`sub-22` pre-training) and an identical follow-up/post-training pair (paretic-hand `sub-02`). The main analysis excludes them; see `data/README.md` and Supplementary S6.6.

## Citation

Please cite the paper (once published), this software archive (see `CITATION.cff`) and the four datasets listed in `data/README.md`.

## Licence

Code and result files: MIT (see `LICENSE`). The datasets keep their original CC BY licences.
