#!/usr/bin/env python3
"""论文二统计:逐人配对 Wilcoxon + 家族内 Holm + 逐人 bootstrap 95% 区间,单源出表。

输入: results/stroke_mi_vs_rest/summary_{cue,break,late}.json、results/stroke_chronic_mi_vs_rest/summary.json
输出: results/stats_paper2.json、docs/tables_paper2.md
用法: .venv/bin/python code/stats_paper2.py
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)
NB = 10000

def boot_ci(d):
    d = np.asarray(d, float); n = len(d)
    bs = RNG.choice(d, (NB, n), replace=True).mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def hl_ci(d):
    """Hodges–Lehmann 中位差(配对差的 Walsh 均值中位数)及其 bootstrap 95% 区间——与 Wilcoxon 的检验目标一致。"""
    d = np.asarray(d, float); n = len(d)
    def hl(x):
        w = (x[:, None] + x[None, :]) / 2
        return float(np.median(w[np.triu_indices(len(x))]))
    bs = [hl(RNG.choice(d, n, replace=True)) for _ in range(2000)]
    return float(hl(d)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]

def paired(a, b, label):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    if np.allclose(d, 0): p = 1.0
    else: p = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact" if len(d) <= 25 else "auto").pvalue)
    hl, hlci = hl_ci(d)
    return dict(comparison=label, n=int(len(d)), mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(d.mean()),
                ci95=boot_ci(d), hl=hl, hl_ci95=hlci, n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), p=p)

def one_sample(a, mu, label):
    a = np.asarray(a, float); d = a - mu
    p = float(stats.wilcoxon(d, alternative="two-sided", method="exact" if len(d) <= 25 else "auto").pvalue)
    hl, hlci = hl_ci(d)
    return dict(comparison=label, n=int(len(d)), mean_a=float(a.mean()), mean_b=mu, mean_diff=float(d.mean()), ci95=boot_ci(d), hl=hl, hl_ci95=hlci,
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), p=p)

def holm(rows):
    ps = np.array([r["p"] for r in rows]); m = len(ps); order = np.argsort(ps); adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, ps[idx] * (m - rank))); adj[idx] = running
    for r, pa in zip(rows, adj): r["p_holm"] = float(pa)
    return rows

families = {}
# ---- 急性:三种静息窗,各自一个家族 ----
for win, name in [("cue", "F1 急性·提示期窗(主)"), ("break", "F2 急性·休息期 0.25–1.75 s"), ("late", "F3 急性·休息期 0.5–2.0 s")]:
    rows = json.load(open(ROOT / f"results/stroke_mi_vs_rest/summary_{win}.json"))["rows"]
    g = lambda k: [r[k] for r in rows]
    fam = [paired(g("rule_oos_auc_8-30"), g("lda_auc_8-30"), "规则(样本外) − 两特征LDA"),
           paired(g("rule_oos_auc_8-30"), g("csp_lda_auc_8-30"), "规则(样本外) − CSP+LDA"),
           paired(g("lda_auc_8-30"), g("csp_lda_auc_8-30"), "两特征LDA − CSP+LDA"),
           paired(g("rule_oos_auc_8-30"), g("rule_auc_8-30"), "规则(样本外) − 规则(同源参考)")]
    if win == "cue":
        fam += [one_sample(g("rule_oos_auc_8-30"), 0.5, "规则(样本外) vs 0.5"),
                paired(g("rule_oos_auc_beta"), g("rule_oos_auc_mu"), "规则 beta − 规则 mu")]
    families[name] = holm(fam)

# ---- 慢性 ----
ps = json.load(open(ROOT / "results/stroke_chronic_mi_vs_rest/summary.json"))["per_subject"]
subs = sorted(ps); g = lambda k: [ps[s][k] for s in subs]
families["F4 慢性·会话内(n=14)"] = holm([
    paired(g("rule_oos_auc_mean"), g("lda_auc_mean"), "规则(样本外) − 两特征LDA"),
    paired(g("rule_oos_auc_mean"), g("csp_lda_auc_mean"), "规则(样本外) − CSP+LDA"),
    one_sample(g("rule_oos_auc_mean"), 0.5, "规则(样本外) vs 0.5"),
    paired(g("rule_oos_auc_mean"), g("rule_auc_mean"), "规则(样本外) − 规则(同源参考)")])
families["F5 慢性·部署口径(第2次会话起,n=14)"] = holm([
    one_sample(g("rule_fixed_bacc_later_mean"), 0.5, "固定阈值平衡准确率 vs 0.5"),
    paired(g("rule_recal_bacc_later_mean"), g("rule_fixed_bacc_later_mean"), "重标参考 − 固定阈值(平衡准确率)"),
    paired(g("rule_fixed_auc_later_mean"), g("lda_fixed_auc_later_mean"), "固定规则AUC − 固定LDA AUC"),
    paired(np.array(g("lda_auc_later_mean")) - np.array(g("lda_fixed_auc_later_mean")), np.array(g("rule_oos_auc_later_mean")) - np.array(g("rule_fixed_auc_later_mean")), "跨会话掉幅:LDA − 规则(会话内样本外−首会话固定)"),
    paired(g("lda_auc_later_mean"), g("lda_fixed_auc_later_mean"), "LDA 会话内 − 首会话固定"),
    paired(g("rule_oos_auc_later_mean"), g("rule_fixed_auc_later_mean"), "规则 会话内(样本外) − 首会话固定")])

# ---- 补实验家族(文件存在才算) ----
import os
FU = ROOT / "results/paper2_followups"
if (FU / "prospective.json").exists():
    P = json.load(open(FU / "prospective.json"))["per_subject"]; ps_ = sorted(P); gp = lambda k: [P[s]["summary"][k] for s in ps_]
    ok = [s for s in ps_ if P[s]["summary"].get("pro10_youden_bacc") is not None]
    gq = lambda k: [P[s]["summary"][k] for s in ok]
    families["F6 慢性·前瞻重标(第2次会话起,n=14)"] = holm([
        paired(gq("pro10_youden_bacc"), gq("pro10_youden_fixed_on_test"), "前瞻10试次重标 − 固定参考(同评价集,Youden阈值)"),
        paired(gq("pro10_youden_offline_on_test"), gq("pro10_youden_bacc"), "整场离线重标 − 前瞻10试次(同评价集)"),
        paired(gq("pro5_youden_bacc"), gq("pro5_youden_fixed_on_test"), "前瞻5试次重标 − 固定参考"),
        paired(gq("pro20_youden_bacc"), gq("pro20_youden_fixed_on_test"), "前瞻20试次重标 − 固定参考"),
        paired(gq("pro10_restq_bacc"), gq("pro10_restq_fixed_on_test"), "无标签阈值:前瞻10重标 − 固定参考"),
        paired(gp("fixed_restq_bacc"), gp("fixed_youden_bacc"), "无标签阈值 − Youden阈值(固定参考)"),
        one_sample(gq("pro10_restq_bacc"), 0.5, "无标签阈值+前瞻10重标 vs 0.5"),
        paired(gp("lda_fixed_bacc"), gp("fixed_youden_bacc"), "首会话LDA固定 − 固定规则(平衡准确率)")])
if (FU / "paretic.json").exists():
    R = json.load(open(FU / "paretic.json"))["rows"]; gr = lambda k: [r[k] for r in R]
    families["F7 急性·患手 vs 健手(n=50)"] = holm([
        paired(gr("paretic_rule_oos"), gr("nonparetic_rule_oos"), "规则:患手 − 健手"),
        paired(gr("paretic_csp"), gr("nonparetic_csp"), "CSP:患手 − 健手"),
        paired(gr("paretic_rule_contra"), gr("paretic_rule_ipsi"), "患手:对侧单导 − 同侧单导规则"),
        one_sample(gr("paretic_rule_oos"), 0.5, "患手规则 vs 0.5")])
if (FU / "matched.json").exists():
    M = json.load(open(FU / "matched.json"))["rows"]; gm = lambda k: [r[k] for r in M]
    rows_cue = json.load(open(ROOT / "results/stroke_mi_vs_rest/summary_cue.json"))["rows"]; rule7 = {r["sub"]: r for r in json.load(open(ROOT / "results/rule_montage_variants/acute_cue.json"))["rows"]}
    families["F8 急性·同输入训练对照(n=50)"] = holm([
        paired(gm("lda_7ch"), [rule7[r["sub"]]["rule_m7"] for r in M], "7导LDA − 7导规则(同输入)"),
        paired(gm("csp_7ch"), gm("lda_7ch"), "7导CSP − 7导功率LDA"),
        paired(gm("csp_29ch"), gm("csp_7ch"), "29导CSP − 7导CSP"),
        paired(gm("lda_29ch_power"), gm("lda_7ch"), "29导功率LDA − 7导功率LDA")])
if (FU / "controls.json").exists():
    K = json.load(open(FU / "controls.json"))["rows"]; gk = lambda k: [r[k] for r in K]
    kk = [r for r in K if r.get("c34_rule_clean") is not None]
    families["F9 急性·对照(n=50;剔窗一项 n=47)"] = holm([
        one_sample(gk("eog_rule_1-8"), 0.5, "EOG 1–8 Hz 规则 vs 0.5"),
        one_sample(gk("eog_rule_8-30"), 0.5, "EOG 8–30 Hz 规则 vs 0.5"),
        paired(gk("c34_rule"), gk("fp_rule"), "C3/C4 规则 − Fp1/Fp2 规则"),
        paired(gk("c34_rule"), gk("occ_rule"), "C3/C4 规则 − O1/O2 规则"),
        paired([r["c34_rule_clean"] for r in kk], [r["c34_rule"] for r in kk], "剔窗后 − 剔窗前(C3/C4 规则)")])

if (FU / "controls2.json").exists():
    C2 = json.load(open(FU / "controls2.json"))
    for mode, name in [("cue", "F10 急性·来源分解·提示期窗(n=50)"), ("break", "F11 急性·来源分解·休息期窗(n=50)")]:
        rows2 = C2[mode]["rows"]; g2 = lambda k: [r[k] for r in rows2]
        families[name] = holm([
            paired(g2("eog_reg_c34"), g2("c34_filt_only"), "去眼电回归后 − 同滤波不回归(匹配对照)"),
            paired(g2("c34_filt_only"), g2("c34_rule"), "同滤波不回归 − 原 C3/C4 规则(滤波本身的作用)"),
            paired(g2("global_rule"), g2("c34_rule"), "29导均值(全局)规则 − C3/C4 规则"),
            paired(g2("c34_rule"), g2("motor_rel_global"), "C3/C4 规则 − C3/C4 相对全局"),
            one_sample(g2("motor_rel_global"), 0.5, "C3/C4 相对全局 vs 0.5"),
            one_sample(g2("motor_rel_occ"), 0.5, "C3/C4 相对枕区 vs 0.5")])
if (FU / "farref.json").exists():
    FR = json.load(open(FU / "farref.json"))
    for mode, name in [("cue", "F12 急性·远端参照与增量信息·提示期窗(n=50)"), ("break", "F13 急性·远端参照与增量信息·休息期窗(n=50)")]:
        rr = FR[mode]["rows"]; gf = lambda k: [r[k] for r in rr]
        families[name] = holm([
            paired(gf("lda_far_c34"), gf("lda_far"), "LDA(远端+C3/C4) − LDA(远端):中央增量"),
            paired(gf("lda_far_c34"), gf("lda_c34"), "LDA(远端+C3/C4) − LDA(C3/C4):远端增量"),
            one_sample(gf("rule_c34_minus_far"), 0.5, "C3/C4 减远端 固定规则 vs 0.5"),
            one_sample(gf("lda_rel_c34"), 0.5, "相对特征 LDA vs 0.5"),
            paired(gf("rule_c34"), gf("rule_c34_minus_far"), "C3/C4 规则 − C3/C4 减远端规则"),
            paired(gf("lda_29_abs"), gf("lda_29_rel"), "29导绝对功率 LDA − 29导相对功率 LDA"),
            paired(gf("rule_far"), gf("rule_c34"), "远端均值规则 − C3/C4 规则")])
if (FU / "predict.json").exists():
    PR = json.load(open(FU / "predict.json")); key = "rule_oos_auc" if "rule_oos_auc" in PR else "rule_auc"
    rows_p = []
    for kk, lab in [(key, "规则"), ("csp_lda_auc", "CSP")]:
        for m in [1, 2, 3]:
            r = PR[kk][f"first{m}"]
            rows_p.append(dict(comparison=f"{lab}:前{m}次会话 vs 第4次起均值(Spearman)", n=r["n"], mean_a=r["spearman_rho"], mean_b=0.0, mean_diff=r["spearman_rho"], ci95=r["rho_ci95"], hl=r["spearman_rho"], hl_ci95=r["rho_ci95"], n_pos=r["agree_gt_0.7"], n_neg=r["n"] - r["agree_gt_0.7"], p=r["p"]))
    families["F14 慢性·早期−后续关联(n=14;'差'列为ρ,+/−列为过线一致/不一致人数)"] = holm(rows_p)
# ---- 过线比例的 Wilson 区间 ----
props = {}
cue = json.load(open(ROOT / "results/stroke_mi_vs_rest/summary_cue.json"))["summary"]
for k, lab in [("rule_oos_auc_8-30", "急性 规则(oos) >0.7"), ("lda_auc_8-30", "急性 两特征LDA >0.7"), ("csp_lda_auc_8-30", "急性 CSP >0.7")]:
    props[lab] = {"k": cue[k]["n_auc_gt_0.7"], "n": 50, "wilson95": wilson(cue[k]["n_auc_gt_0.7"], 50)}
ch = json.load(open(ROOT / "results/stroke_chronic_mi_vs_rest/summary.json"))["summary"]
for k, lab in [("rule_oos_auc", "慢性 规则(样本外,逐人会话均值) >0.7"), ("csp_lda_auc", "慢性 CSP >0.7")]:
    props[lab] = {"k": ch[k]["n_gt_0.7"], "n": 14, "wilson95": wilson(ch[k]["n_gt_0.7"], 14)}
families["_proportions"] = props
json.dump(families, open(ROOT / "results/stats_paper2.json", "w"), ensure_ascii=False, indent=1)
families.pop("_proportions")
with open(ROOT / "docs/tables_paper2.md", "w") as f:
    f.write("# 论文二统计表(由 code/stats_paper2.py 生成,勿手改)\n\n以患者为单位的配对 Wilcoxon 符号秩检验(n≤25 用 SciPy 精确法;零差按 wilcox 法剔除;并列秩按 SciPy 默认平均秩),家族内 Holm 校正。两种区间:均值差的逐人 bootstrap 95%(10,000 次,未作多重校正)与 Hodges–Lehmann 中位差(与 Wilcoxon 检验目标一致)的 bootstrap 95%(2,000 次)。\n\n")
    f.write("## 过线比例(AUC>0.7)的 Wilson 95% 区间\n\n| 口径 | k/n | 区间 |\n|---|---|---|\n")
    for lab, v in props.items(): f.write(f"| {lab} | {v['k']}/{v['n']} | {v['wilson95'][0]*100:.0f}%–{v['wilson95'][1]*100:.0f}% |\n")
    f.write("\n")
    for name, rows in families.items():
        f.write(f"## {name}\n\n| 比较 | n | 均值 A | 均值 B | 均值差 | 均值差 bootstrap 95% CI | HL 中位差 [95% CI] | +/− 人数 | p | p_Holm |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['comparison']} | {r['n']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {r['mean_diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['hl']:+.3f} [{r['hl_ci95'][0]:+.3f}, {r['hl_ci95'][1]:+.3f}] | {r['n_pos']}/{r['n_neg']} | {r['p']:.3g} | {r['p_holm']:.3g} |\n")
        f.write("\n")
for name, rows in families.items():
    print("==", name)
    for r in rows: print(f"  {r['comparison']:34s} diff {r['mean_diff']:+.3f} CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] {r['n_pos']}/{r['n_neg']} p={r['p']:.3g} Holm={r['p_holm']:.3g}")
