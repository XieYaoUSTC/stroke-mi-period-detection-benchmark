#!/usr/bin/env python3
"""强化分析的逐人统计(急性/慢性:黎曼基线、频带与枕区、偏侧化;家族编号接在天津家族之后),输出 docs/strengthen_stats.md 与 results/paper2_strengthen/stats.json。
用法:.venv/bin/python code/strengthen_stats.py(先跑 tianjin_stats.py 以确定编号起点)"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_paper2 import paired, one_sample, holm
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results" / "paper2_strengthen"
start = 0
fam = {}; L = ["# 强化分析(急性/慢性) · 逐人配对统计(code/strengthen_stats.py 生成;家族编号 G 系列,与 F 系列独立)\n"]
FAM_NO = [start]

def table(name, rows):
    FAM_NO[0] += 1; name = f"G{FAM_NO[0]} {name}"; fam[name] = holm(rows)
    L.append(f"\n## {name}\n\n| 比较 | n | A | B | 均值差 | bootstrap 95% CI | HL 位置估计 [95% CI] | +/− | p | Holm |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in fam[name]:
        L.append(f"| {r['comparison']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['hl']:+.3f} [{r['hl_ci95'][0]:+.3f}, {r['hl_ci95'][1]:+.3f}] | {r['n_pos']}/{r['n_neg']} | {r['p']:.3g} | {r['p_holm']:.3g} |")

def rows_from(rows_json, pairs, ones):
    out = []
    for a, b, lab in pairs:
        ok = [r for r in rows_json if isinstance(r.get(a), float) and isinstance(r.get(b), float)]
        if len(ok) >= 5: out.append(paired([r[a] for r in ok], [r[b] for r in ok], lab))
    for a, lab in ones:
        v = [r[a] for r in rows_json if isinstance(r.get(a), float)]
        if len(v) >= 5: out.append(one_sample(v, 0.5, lab))
    return out

PAIRS = [("riem_ts_lr", "rule_c34", "黎曼 TS+LR − 规则 C3/C4"), ("rule_c34_foldmean", "rule_c34", "规则 C3/C4:折内均值 − 拼接"), ("riem_ts_lr_foldmean", "riem_ts_lr", "黎曼 TS+LR:折内均值 − 拼接"), ("riem_central", "riem_distal", "同管线黎曼:中央 8 导 − 远端 8 导"), ("riem_ts_lr", "riem_central", "黎曼:全导 − 中央 8 导"), ("riem_mdm", "rule_c34", "黎曼 MDM − 规则 C3/C4"), ("rule_c34_beta", "rule_c34_mu", "C3/C4 beta 规则 − C3/C4 mu 规则"),
         ("rule_contra_hand", "rule_ipsi_hand", "按想象手:对侧单导 − 同侧单导")]
PAIRS_OCC = [("rule_c34_beta", "rule_occ_mu", "C3/C4 beta 规则 − 枕区 alpha 规则"), ("rule_c34_mu", "rule_occ_mu", "C3/C4 mu 规则 − 枕区 alpha 规则"), ("lda_occ_mu_c34_beta", "lda_occ_mu", "LDA(枕区 alpha + C3/C4 beta) − LDA(枕区 alpha):中央 beta 增量")]
ONES = [("lr_rule_auc", "左右手判别(C3−C4 固定分数) vs 0.5"), ("lr_lda_auc", "左右手判别 LDA vs 0.5")]
ONES_OCC = [("rule_occ_mu", "枕区 alpha 规则 vs 0.5")]

for fname, title in [("acute_cue.json", "急性(提示期参考)·强化"), ("acute_break.json", "急性(休息期参考)·强化"), ("chronic.json", "慢性(CAR,逐会话均值)·强化")]:
    if (R / fname).exists():
        J = json.load(open(R / fname)); rows = J["rows"]; occ = any("rule_occ_mu" in r for r in rows)
        lat = fname != "acute_break.json"            # 偏侧化只用想象窗,与参考窗无关:唯一推断归属 G1(急性)与 G3(慢性),G2 不重复
        pairs = [p for p in PAIRS if lat or "hand" not in p[0]] + (PAIRS_OCC if occ else []); ones = (ONES if lat else []) + (ONES_OCC if occ else [])
        table(f"{title} (n={len(rows)})", rows_from(rows, pairs, ones))
for m, title in [("cue", "急性(提示期参考)·眼电回归对照"), ("break", "急性(休息期参考)·眼电回归对照")]:
    f = R / f"acute_eogreg_{m}.json"
    if f.exists():
        rows = json.load(open(f))["rows"]
        table(f"{title} (n={len(rows)})", rows_from(rows, [("riem_ts_lr_eogreg", "riem_ts_lr_filt", "黎曼:回归去眼电 − 仅滤波对照"), ("rule_c34_eogreg", "rule_c34_filt", "规则:回归去眼电 − 仅滤波对照"), ("riem_ts_lr_filt", "riem_ts_lr_raw", "黎曼:仅多一道 0.5–40 Hz 滤波 − 原始"), ("riem_ts_lr_eogreg", "riem_ts_lr_raw", "黎曼:回归去眼电 − 原始")], []))
f = R / "acute_transient_cue.json"
if f.exists():
    rows = json.load(open(f))["rows"]
    table(f"急性(提示期参考)·滤波瞬态针对性检验 (n={len(rows)})", rows_from(rows, [("riem_ts_lr_pre_pad1", "riem_ts_lr_raw", "黎曼:多一道 0.5–40 Hz(补段 1 s) − 原始"), ("riem_ts_lr_pre_pad3", "riem_ts_lr_raw", "黎曼:多一道 0.5–40 Hz(补段 3 s) − 原始"), ("riem_ts_lr_pre_zerophase", "riem_ts_lr_raw", "黎曼:多一道 0.5–40 Hz(零相位) − 原始"), ("riem_ts_lr_pre_pad3", "riem_ts_lr_pre_pad1", "黎曼:补段 3 s − 补段 1 s"), ("rule_c34_pre_pad1", "rule_c34_raw", "规则:多一道 0.5–40 Hz(补段 1 s) − 原始"), ("rule_c34_pre_pad3", "rule_c34_raw", "规则:多一道 0.5–40 Hz(补段 3 s) − 原始")], []))
for m, title in [("cue", "急性(提示期参考)·眼电回归对照(补段 3 s)"), ("break", "急性(休息期参考)·眼电回归对照(补段 3 s)")]:
    f = R / f"acute_eogreg_{m}_pad3.json"
    if f.exists():
        rows = json.load(open(f))["rows"]
        table(f"{title} (n={len(rows)})", rows_from(rows, [("riem_ts_lr_eogreg", "riem_ts_lr_filt", "黎曼:回归去眼电 − 仅滤波对照"), ("rule_c34_eogreg", "rule_c34_filt", "规则:回归去眼电 − 仅滤波对照"), ("riem_ts_lr_filt", "riem_ts_lr_raw", "黎曼:仅多一道 0.5–40 Hz 滤波(补段 3 s) − 原始"), ("riem_ts_lr_eogreg", "riem_ts_lr_raw", "黎曼:回归去眼电 − 原始")], []))
json.dump(fam, open(R / "stats.json", "w"), ensure_ascii=False, indent=1)
open(ROOT / "docs" / "strengthen_stats.md", "w").write("\n".join(L) + "\n"); print("written", len(fam), "G families")
