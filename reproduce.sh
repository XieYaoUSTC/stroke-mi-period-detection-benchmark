#!/bin/bash
# Full pipeline. Stage 1 needs the four public datasets under data/ (see data/README.md) and takes several hours on 8 cores.
# Stage 2 regenerates every statistic, number table, figure and the supplementary Markdown from the JSON files in results/ (minutes).
# Usage:  bash reproduce.sh [stage1|stage2|all]      (default: stage2)
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1791331200}   # fixed timestamp in figure PDFs, so reruns are byte-identical
STAGE=${1:-stage2}

if [ "$STAGE" = stage1 ] || [ "$STAGE" = all ]; then
  # --- acute dataset (Liu et al. 2024): three reference windows; cue = main setting
  for m in cue break late; do REST_MODE=$m $PY code/stroke_mi_vs_rest.py; done
  $PY code/stroke_clinical_covariates.py
  $PY code/baseline_stroke.py 10 5                                # left/right baseline re-implementation (S2), notch Q = 30
  NOTCH_Q=6 OUT_SUFFIX=_q6 $PY code/baseline_stroke.py 10 5       # same with the original notch Q = 6
  # --- chronic dataset (HS Stroke v6)
  $PY code/stroke_chronic_mi_vs_rest.py
  # --- follow-up analyses (acute analyses use the cue reference)
  REST_MODE=cue $PY code/rule_montage_variants.py both
  REST_MODE=cue $PY code/paper2_followups.py prospective rawref predict paretic matched controls controls2 farref
  REST_MODE=cue $PY code/paper2_supp_checks.py
  # --- strengthening analyses (Riemannian baseline, band/occipital, lateralisation, EOG regression, filter transient, held-out windows)
  REST_MODE=cue   $PY code/paper2_strengthen.py acute acute_eogreg acute_transient
  REST_MODE=break $PY code/paper2_strengthen.py acute acute_eogreg
  REST_MODE=cue   EOG_PAD=3 $PY code/paper2_strengthen.py acute_eogreg
  REST_MODE=break EOG_PAD=3 $PY code/paper2_strengthen.py acute_eogreg
  $PY code/paper2_strengthen.py chronic splithalf
  # --- Tianjin datasets: paretic-hand / sixth-finger (both rest-window settings) and lower-limb (raw = main, PEEG = sensitivity)
  SF_REST=both $PY code/tianjin_analysis.py sf
  SF_REST=late $PY code/tianjin_analysis.py sf
  LL_SRC=raw   $PY code/tianjin_analysis.py ll
  LL_SRC=peeg  $PY code/tianjin_analysis.py ll
  # --- signal-level duplicate audit and exclusions (writes *_full.json and the filtered main files)
  $PY code/duplicate_audit.py
  $PY code/apply_exclusions.py
fi

if [ "$STAGE" = stage2 ] || [ "$STAGE" = all ]; then
  $PY code/stats_paper2.py            # acute/chronic families F1–F14
  $PY code/tianjin_stats.py           # Tianjin families F15–F65
  $PY code/strengthen_stats.py        # strengthening families G1–G8
  $PY code/sensitivity_keep_one.py    # duplicate-handling sensitivity
  $PY code/numbers_paper2.py          # single-source number table
  $PY code/make_figures_paper2.py; $PY code/make_fig1_protocol.py; $PY code/make_fig6_four_datasets.py; $PY code/make_fig7_specificity.py
  $PY code/make_supplement_paper2.py  # docs/paper2_supplement.md (PDF: bash code/build_supplement_pdf.sh <outdir>, needs pandoc + xelatex)
fi
