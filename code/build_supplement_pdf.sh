#!/bin/bash
# 论文二补充材料 PDF(横版):先由结果文件生成 md,再 pandoc+xelatex。用法:bash code/build_supplement_pdf.sh [输出目录]
set -e
cd "$(dirname "$0")/.."
OUT=${1:-docs/投稿包_BSPC_20261006}
.venv/bin/python code/make_supplement_paper2.py
pandoc docs/paper2_supplement.md -f markdown -o "$OUT/paper2_supplement.pdf" --pdf-engine=xelatex -V mainfont="DejaVu Serif" -V CJKmainfont="Noto Sans CJK SC" -V geometry:margin=1.6cm -V geometry:landscape -V fontsize=8pt -V colorlinks 2> paper_bspc/supp.log || { tail -20 paper_bspc/supp.log; exit 1; }
echo "supplement pages: $(pdfinfo "$OUT/paper2_supplement.pdf" | awk '/Pages/{print $2}')"
