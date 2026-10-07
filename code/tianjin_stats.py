#!/usr/bin/env python3
"""天津两数据集:逐人配对统计(Wilcoxon + Holm + bootstrap 区间),输出 docs/tianjin_stats.md 与 results/tianjin/stats.json。
用法:.venv/bin/python code/tianjin_stats.py"""
import json, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_paper2 import paired, one_sample, holm, wilson   # 复用论文二的统计函数
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results" / "tianjin"
fam = {}; L = ["# 天津两数据集 · 逐人配对统计(code/tianjin_stats.py 生成)\n"]

FAM_NO = [14]
def table(name, rows):
    FAM_NO[0] += 1; name = f"F{FAM_NO[0]} {name}"
    fam[name] = holm(rows)
    L.append(f"\n## {name}\n\n| 比较 | n | A | B | 均值差 | bootstrap 95% CI | HL 位置估计 [95% CI] | +/− | p | Holm |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in fam[name]:
        L.append(f"| {r['comparison']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['hl']:+.3f} [{r['hl_ci95'][0]:+.3f}, {r['hl_ci95'][1]:+.3f}] | {r['n_pos']}/{r['n_neg']} | {r['p']:.3g} | {r['p_holm']:.3g} |")

def block_rows(res, getter, keys_pairs, title, extra_one=()):
    """res: {sub: {...}};getter(sub) → dict 指标;keys_pairs: [(a, b, label)]"""
    subs = [s for s in sorted(res) if getter(s) is not None]
    rows = []
    for a, b, lab in keys_pairs:
        isnum = lambda v: isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))
        ok = [s for s in subs if isnum(getter(s).get(a)) and isnum(getter(s).get(b))]
        va = [getter(s)[a] for s in ok]; vb = [getter(s)[b] for s in ok]
        if len(va) >= 5: rows.append(paired(va, vb, lab))
    for a, lab in extra_one:
        va = [getter(s)[a] for s in subs if isinstance(getter(s).get(a), (int, float)) and not (isinstance(getter(s).get(a), float) and np.isnan(getter(s).get(a)))]
        if len(va) >= 5: rows.append(one_sample(va, 0.5, lab))
    if rows: table(f"{title} (n={len(subs)})", rows)
    return subs

PAIRS_WITHIN = [("rule_c34", "lda_c34", "规则 C3/C4 − LDA 2特征"), ("rule_c34", "csp_all", "规则 C3/C4 − 全导 CSP"), ("lda_all_abs", "csp_all", "全导功率 LDA(绝对) − 全导 CSP"),
                ("lda_all_abs", "lda_all_rel", "全导功率 LDA 绝对 − 相对"), ("rule_far", "rule_c34", "远端规则 − C3/C4 规则"), ("lda_far_c34", "lda_far", "LDA(远端+C3/C4) − LDA(远端):中央增量"),
                ("lda_far_c34", "lda_c34", "LDA(远端+C3/C4) − LDA(C3/C4):远端增量"), ("rule_c34", "rule_c34_minus_far", "C3/C4 规则 − C3/C4 减远端"), ("rule_m7", "lda_m7", "7导规则 − 7导 LDA(同输入)"),
                ("rule_contra", "rule_ipsi", "患侧对侧单导 − 同侧单导")]
ONE_WITHIN = [("rule_c34", "规则 C3/C4 vs 0.5"), ("rule_c34_minus_far", "C3/C4 减远端 vs 0.5"), ("lda_rel_c34", "相对特征 LDA vs 0.5"), ("eog_rule_1-8", "EOG 1–8 Hz 规则 vs 0.5"), ("eog_rule_8-30", "EOG 8–30 Hz 规则 vs 0.5")]
PAIRS_DEPLOY = [("pro10_youden_bacc", "pro10_youden_fixed_on_test", "前缀10参考试次重标 − 固定参考(同评价集,Youden)"), ("pro10_youden_offline_on_test", "pro10_youden_bacc", "整场离线 − 前缀10(同评价集)"),
                ("pro10_restq_bacc", "pro10_restq_fixed_on_test", "前缀10重标 − 固定(参考分位数阈值)"), ("lda_drop", "rule_drop", "跨时点掉幅:LDA − 规则"), ("lda_within_auc", "lda_fixed_auc", "LDA 时点内 − pre 固定"),
                ("rule_within_auc", "rule_fixed_auc", "规则 时点内 − pre 固定"), ("lda_fixed_bacc", "fixed_youden_bacc", "pre 训练 LDA 固定 − 固定规则(BA)")]
ONE_DEPLOY = [("fixed_youden_bacc", "固定参考+Youden 阈值 BA vs 0.5"), ("fixed_restq_bacc", "固定参考+参考分位数阈值 BA vs 0.5")]

# ---------------- 第六指
SF_FILE = R / ("sf_restlate.json" if (R / "sf_restlate.json").exists() else "sf.json")      # 主口径:静息段后半窗
LL_FILE = R / ("ll_raw.json" if (R / "ll_raw.json").exists() else "ll.json")              # 主口径:原始 64 导
L.append(f"主口径文件:{SF_FILE.name}、{LL_FILE.name}\n")
if SF_FILE.exists():
    SF = json.load(open(SF_FILE))
    for para in ["AH", "SF"]:
        for tp in ["pre", "post", "follow"]:
            block_rows(SF, lambda s, tp=tp, para=para: SF[s].get(tp, {}).get(para), PAIRS_WITHIN, f"第六指·{para} vs 静息·{tp}", ONE_WITHIN)
        for tp in ["post", "follow"]:
            block_rows(SF, lambda s, tp=tp, para=para: SF[s].get(f"deploy_{para}_pre_to_{tp}"), PAIRS_DEPLOY, f"第六指·{para}·pre→{tp} 部署", ONE_DEPLOY)
    # 过线比例
    for para in ["AH"]:
        for tp in ["pre"]:
            v = [SF[s][tp][para]["rule_c34"] for s in SF if tp in SF[s] and para in SF[s][tp]]; k = sum(x > 0.7 for x in v)
            L.append(f"\n第六指 {para} {tp}:规则 C3/C4 >0.7 {k}/{len(v)} Wilson {wilson(k, len(v))[0]*100:.0f}%–{wilson(k, len(v))[1]*100:.0f}%")
# ---------------- 下肢
if LL_FILE.exists():
    LL = json.load(open(LL_FILE))
    pw = [p for p in PAIRS_WITHIN if "contra" not in p[0]] + [("rule_cz", "rule_c34", "Cz 规则 − C3/C4 规则"), ("rule_far", "rule_cz", "远端规则 − Cz 规则"), ("rule_cz", "lda_c34", "Cz 规则 − LDA(C3/C4)"), ("rule_cz", "csp_all", "Cz 规则 − 全导 CSP")]
    ow = ONE_WITHIN + [("rule_cz", "Cz 规则 vs 0.5")]
    for design in ["task_vs_idle", "task_vs_idle_mixed_runs", "task_vs_fixation"]:
        for tp in ["pre", "post", "follow"]:
            block_rows(LL, lambda s, tp=tp, d=design: LL[s].get(tp, {}).get(d), pw, f"下肢·{design}·{tp}", ow)
    # 设计对照:同一人同一时点 任务 vs 空闲 与 任务 vs 注视 的差
    for tp in ["pre"]:
        subs = [s for s in sorted(LL) if tp in LL[s] and "task_vs_idle" in LL[s][tp] and "task_vs_fixation" in LL[s][tp]]
        rows = [paired([LL[s][tp]["task_vs_fixation"][k] for s in subs], [LL[s][tp]["task_vs_idle"][k] for s in subs], f"{k}:vs注视 − vs空闲") for k in ["rule_cz", "rule_c34", "rule_far", "rule_global", "lda_all_abs", "csp_all"]]
        table(f"下肢·设计对照(任务 vs 注视 − 任务 vs 空闲)·{tp} (n={len(subs)})", rows)
    for tp in ["post", "follow"]:
        block_rows(LL, lambda s, tp=tp: LL[s].get(f"deploy_cz_pre_to_{tp}"), PAIRS_DEPLOY, f"下肢·Cz/FCz·pre→{tp} 部署", ONE_DEPLOY)
# ---------------- 强化分析家族(2026-10-06:黎曼基线、频带/枕区、k=5/20、跨时点黎曼);单独成家族,不改动 F15–F36 的 Holm
PAIRS_STRONG = [("riem_ts_lr", "rule_c34", "黎曼 TS+LR − 规则 C3/C4"), ("riem_ts_lr", "csp_all", "黎曼 TS+LR − 全导 CSP"), ("riem_ts_lr", "lda_all_abs", "黎曼 TS+LR − 全导功率 LDA"), ("riem_mdm", "rule_c34", "黎曼 MDM − 规则 C3/C4"),
                ("rule_c34_beta", "rule_c34_mu", "C3/C4 beta 规则 − C3/C4 mu 规则"), ("rule_c34_beta", "rule_occ_mu", "C3/C4 beta 规则 − 枕区 alpha 规则"), ("rule_c34_mu", "rule_occ_mu", "C3/C4 mu 规则 − 枕区 alpha 规则"),
                ("lda_occ_mu_c34_beta", "lda_occ_mu", "LDA(枕区 alpha + C3/C4 beta) − LDA(枕区 alpha):中央 beta 增量")]
ONE_STRONG = [("rule_occ_mu", "枕区 alpha 规则 vs 0.5"), ("rule_c34_beta", "C3/C4 beta 规则 vs 0.5")]
PAIRS_DEPLOY_STRONG = [("pro5_youden_bacc", "pro5_youden_fixed_on_test", "前缀5重标 − 固定(同评价集,Youden)"), ("pro20_youden_bacc", "pro20_youden_fixed_on_test", "前缀20重标 − 固定(同评价集,Youden)"),
                       ("pro5_restq_bacc", "pro5_restq_fixed_on_test", "前缀5重标 − 固定(参考分位数)"), ("pro20_restq_bacc", "pro20_restq_fixed_on_test", "前缀20重标 − 固定(参考分位数)"),
                       ("riem_within_auc", "riem_fixed_auc", "黎曼 时点内 − pre 固定"), ("riem_fixed_auc", "rule_fixed_auc", "pre 固定:黎曼 − 规则(AUC)"), ("riem_within_auc", "rule_within_auc", "时点内:黎曼 − 规则(AUC)")]
if SF_FILE.exists():
    for para in ["AH", "SF"]:
        for tp in ["pre", "post", "follow"]:
            block_rows(SF, lambda s, tp=tp, para=para: SF[s].get(tp, {}).get(para), PAIRS_STRONG, f"第六指·{para} vs 静息·{tp}·强化", ONE_STRONG)
        for tp in ["post", "follow"]:
            block_rows(SF, lambda s, tp=tp, para=para: SF[s].get(f"deploy_{para}_pre_to_{tp}"), PAIRS_DEPLOY_STRONG, f"第六指·{para}·pre→{tp} 部署·强化")
if LL_FILE.exists():
    for design in ["task_vs_idle", "task_vs_fixation"]:
        for tp in ["pre", "post", "follow"]:
            block_rows(LL, lambda s, tp=tp, d=design: LL[s].get(tp, {}).get(d), PAIRS_STRONG, f"下肢·{design}·{tp}·强化", ONE_STRONG)
    for tp in ["pre"]:
        subs = [s for s in sorted(LL) if tp in LL[s] and "task_vs_idle" in LL[s][tp] and "task_vs_fixation" in LL[s][tp] and "riem_ts_lr" in LL[s][tp]["task_vs_fixation"]]
        if len(subs) >= 5:
            rows = [paired([LL[s][tp]["task_vs_fixation"][k] for s in subs], [LL[s][tp]["task_vs_idle"][k] for s in subs], f"{k}:vs注视 − vs空闲") for k in ["riem_ts_lr", "riem_mdm", "lda_c34", "lda_m7", "lda_all_rel"]]
            table(f"下肢·设计对照·强化(黎曼/其他 LDA)·{tp} (n={len(subs)})", rows)
    for tp in ["post", "follow"]:
        block_rows(LL, lambda s, tp=tp: LL[s].get(f"deploy_cz_pre_to_{tp}"), PAIRS_DEPLOY_STRONG, f"下肢·Cz/FCz·pre→{tp} 部署·强化")
# ---------------- 补充家族(2026-10-07):折内 vs 拼接 AUC;同管线黎曼中央 vs 远端;公共后缀评价集上的 k 比较
PAIRS_FOLD = [("rule_c34_foldmean", "rule_c34", "规则 C3/C4:折内均值 − 拼接"), ("lda_c34_foldmean", "lda_c34", "LDA 2特征:折内均值 − 拼接"), ("csp_all_foldmean", "csp_all", "全导 CSP:折内均值 − 拼接"), ("riem_ts_lr_foldmean", "riem_ts_lr", "黎曼 TS+LR:折内均值 − 拼接"),
              ("riem_ts_lr_foldmean", "rule_c34_foldmean", "折内均值:黎曼 − 规则"), ("riem_central", "riem_distal", "同管线黎曼:中央 9 导 − 远端 9 导"), ("riem_ts_lr", "riem_central", "黎曼:全导 − 中央 9 导")]
PAIRS_COMMON = [("pro5_youden_bacc_common", "fixed_youden_bacc_common", "公共后缀:前缀5 − 固定(Youden)"), ("pro10_youden_bacc_common", "fixed_youden_bacc_common", "公共后缀:前缀10 − 固定(Youden)"), ("pro20_youden_bacc_common", "fixed_youden_bacc_common", "公共后缀:前缀20 − 固定(Youden)"),
                ("pro20_youden_bacc_common", "pro10_youden_bacc_common", "公共后缀:前缀20 − 前缀10"), ("pro10_youden_bacc_common", "pro5_youden_bacc_common", "公共后缀:前缀10 − 前缀5"), ("offline_youden_bacc_common", "pro10_youden_bacc_common", "公共后缀:整场离线 − 前缀10"),
                ("pro10_youden_sens", "pro10_youden_fixed_sens", "前缀10 − 固定:灵敏度(同评价集)"), ("pro10_youden_spec", "pro10_youden_fixed_spec", "前缀10 − 固定:特异度(同评价集)")]
if SF_FILE.exists():
    for para in ["AH"]:
        for tp in ["pre", "post", "follow"]:
            block_rows(SF, lambda s, tp=tp, para=para: SF[s].get(tp, {}).get(para), PAIRS_FOLD, f"第六指·{para} vs 静息·{tp}·折内/黎曼子集")
        for tp in ["post", "follow"]:
            block_rows(SF, lambda s, tp=tp, para=para: SF[s].get(f"deploy_{para}_pre_to_{tp}"), PAIRS_COMMON, f"第六指·{para}·pre→{tp} 部署·公共后缀")
if LL_FILE.exists():
    for tp in ["pre", "post", "follow"]:
        block_rows(LL, lambda s, tp=tp: LL[s].get(tp, {}).get("task_vs_idle"), PAIRS_FOLD, f"下肢·task_vs_idle·{tp}·折内/黎曼子集")
    for tp in ["post", "follow"]:
        block_rows(LL, lambda s, tp=tp: LL[s].get(f"deploy_cz_pre_to_{tp}"), PAIRS_COMMON, f"下肢·Cz/FCz·pre→{tp} 部署·公共后缀")
json.dump(fam, open(R / "stats.json", "w"), ensure_ascii=False, indent=1)
open(ROOT / "docs" / "tianjin_stats.md", "w").write("\n".join(L) + "\n"); print("written", len(fam), "families")
