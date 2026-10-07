#!/usr/bin/env python3
"""急性卒中 50 人:逐人「想象 vs 静息」AUC 与临床指标的关系(论文二 4.x 节)。

输入: results/stroke_mi_vs_rest/summary_{cue,break,late}.json(逐人 AUC)
      data/stroke_acute/participants.tsv(NIHSS/MBI/mRS/病程/年龄/性别/瘫痪侧/首发)
输出: results/stroke_mi_vs_rest/clinical_covariates.json
用法: .venv/bin/python code/stroke_clinical_covariates.py
"""
import json, csv
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "stroke_mi_vs_rest"
part = {r["Participant_ID"]: r for r in csv.DictReader(open(ROOT / "data/stroke_acute/participants.tsv"), delimiter="\t")}
num_cols = ["NIHSS", "MBI", "mRS", "Duration", "Age"]
cat_cols = ["Gender", "ParalysisSide", "IsFirstTime"]
methods = ["rule_oos_auc_8-30", "lda_auc_8-30", "csp_lda_auc_8-30"]
out = {"n": len(part), "windows": {}}

for win in ["cue", "break", "late"]:
    rows = json.load(open(RES / f"summary_{win}.json"))["rows"]
    subs = [r["sub"] for r in rows]
    res = {"spearman": {}, "usable_vs_not": {}}
    for m in methods:
        auc = np.array([r[m] for r in rows])
        res["spearman"][m] = {}
        for c in num_cols:
            x = np.array([float(part[s][c]) for s in subs])
            rho, p = stats.spearmanr(x, auc)
            res["spearman"][m][c] = {"rho": round(float(rho), 3), "p": round(float(p), 3)}
    # 以主口径 CSP AUC>0.7 定「可用」
    auc = np.array([r["csp_lda_auc_8-30"] for r in rows]); usable = auc > 0.7
    res["usable_vs_not"]["n_usable"] = int(usable.sum())
    for c in num_cols:
        x = np.array([float(part[s][c]) for s in subs])
        u, p = stats.mannwhitneyu(x[usable], x[~usable], alternative="two-sided")
        res["usable_vs_not"][c] = {"usable_mean": round(float(x[usable].mean()), 2), "not_mean": round(float(x[~usable].mean()), 2), "p_mwu": round(float(p), 3)}
    for c in cat_cols:
        levels = sorted(set(part[s][c] for s in subs))
        tab = np.array([[sum(1 for s, u in zip(subs, usable) if u == uu and part[s][c] == lv) for lv in levels] for uu in (True, False)])
        try:
            p = stats.fisher_exact(tab)[1] if tab.shape == (2, 2) else stats.chi2_contingency(tab)[1]
        except Exception:
            p = None
        res["usable_vs_not"][c] = {"levels": levels, "table_usable_then_not": tab.tolist(), "p": (round(float(p), 3) if p is not None else None)}
    out["windows"][win] = res

# 两种静息窗下「可用」判定的一致性(单次 40 试次稳不稳)
cue_rows = {r["sub"]: r for r in json.load(open(RES / "summary_cue.json"))["rows"]}; brk_rows = {r["sub"]: r for r in json.load(open(RES / "summary_break.json"))["rows"]}
uc = {s for s in cue_rows if cue_rows[s]["csp_lda_auc_8-30"] > 0.7}; ub = {s for s in brk_rows if brk_rows[s]["csp_lda_auc_8-30"] > 0.7}
both = sorted(uc & ub); only_one = sorted(uc ^ ub)
json.dump({"usable_cue": sorted(uc), "usable_break": sorted(ub)}, open(RES / "usable_subjects.json", "w"), indent=1)
out["usable_consistency_cue_vs_break"] = {"both": len(both), "only_one": len(only_one), "neither": len(part) - len(both) - len(only_one)}
json.dump(out, open(RES / "clinical_covariates.json", "w"), ensure_ascii=False, indent=1)
w = out["windows"]["cue"]
print("cue window, usable n =", w["usable_vs_not"]["n_usable"])
for c in num_cols: print(f"  {c}: usable {w['usable_vs_not'][c]['usable_mean']} vs not {w['usable_vs_not'][c]['not_mean']}, p={w['usable_vs_not'][c]['p_mwu']}")
for c in cat_cols: print(f"  {c}: {w['usable_vs_not'][c]}")
print("spearman (cue):", json.dumps(w["spearman"], ensure_ascii=False))
print("usable consistency:", out["usable_consistency_cue_vs_break"])
