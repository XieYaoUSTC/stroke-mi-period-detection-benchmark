#!/usr/bin/env python3
"""论文二正文数字单一来源:从全部结果文件生成 docs/numbers_paper2.md(+json)。正文只许引用这里的数字。
用法:.venv/bin/python code/numbers_paper2.py
"""
import json, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
J = lambda *p: json.load(open(R.joinpath(*p)))
out = {}; lines = ["# 论文二正文数字表(code/numbers_paper2.py 生成,勿手改)\n"]
def sec(t): lines.append(f"\n## {t}\n")
def row(label, val, note=""):
    out[label] = val; lines.append(f"- **{label}**: {val}{('  ('+note+')') if note else ''}")
def f3(x): return f"{x:.3f}"
def ms(s): return f"{s['mean']:.3f} ± {s['sd']:.3f}"

# ---- 急性三窗 ----
sec("急性 50 人 · 三种参考窗(summary_{cue,break,late}.json;规则=样本外参考)")
for w, lab in [("cue", "提示期"), ("break", "休息期0.25–1.75"), ("late", "休息期0.5–2.0")]:
    S = J("stroke_mi_vs_rest", f"summary_{w}.json")["summary"]
    for k, kl in [("rule_oos_auc_8-30", "规则oos"), ("rule_auc_8-30", "规则同源"), ("lda_auc_8-30", "LDA2"), ("csp_lda_auc_8-30", "CSP29"), ("rule_oos_auc_mu", "规则oos mu"), ("rule_oos_auc_beta", "规则oos beta")]:
        row(f"{lab}·{kl}", f"{ms(S[k])} >0.7:{S[k]['n_auc_gt_0.7']} >0.8:{S[k]['n_auc_gt_0.8']}")
# ---- 慢性 ----
sec("慢性 14 人(stroke_chronic_mi_vs_rest/summary.json;CAR)")
C = J("stroke_chronic_mi_vs_rest", "summary.json"); S = C["summary"]
for k, kl in [("rule_oos_auc", "会话内规则oos"), ("rule_auc", "会话内规则同源"), ("lda_auc", "会话内LDA2"), ("csp_lda_auc", "会话内CSP11"), ("rule_fixed_bacc", "固定阈值BA"), ("rule_recal_bacc", "整场重标BA"), ("rule_fixed_auc", "固定规则AUC"), ("lda_fixed_auc", "固定LDA AUC")]:
    row(kl, f"人均 {ms(S[k])} 第2次起 {S[k]['later_mean']:.3f} >0.7:{S[k]['n_gt_0.7']} 会话间SD {S[k]['within_subject_session_sd']:.3f}")
ps = C["per_subject"]; n_ses = [len(ps[s]["sessions"]) for s in ps]; n_tr = sum(x["n_trials"] for s in ps for x in ps[s]["sessions"])
row("慢性纳入", f"{len(ps)} 人, {sum(n_ses)} 会话 (每人 {min(n_ses)}–{max(n_ses)}), {n_tr} 试次")
row("慢性 FMA", f"ρ(基线)={S['fma']['rho_auc_vs_fma_pre']:.2f} p={S['fma']['p']:.2f}; ρ(增量)={S['fma']['rho_auc_vs_fma_gain']:.2f} p={S['fma']['p_gain']:.2f}; 趋势ρ均值 {S['trend_rho_mean']:.2f}", "均为 CSP AUC")
row("sub-01 规则oos / 重标BA(第2次起)", f"{ps['sub-01']['rule_oos_auc_mean']:.2f} / {ps['sub-01']['rule_recal_bacc_later_mean']:.2f}")
row("sub-05 规则oos / CSP", f"{ps['sub-05']['rule_oos_auc_mean']:.2f} / {ps['sub-05']['csp_lda_auc_mean']:.2f}")
# ---- 前瞻 ----
sec("前瞻重标(paper2_followups/prospective.json;第2次会话起)")
P = J("paper2_followups", "prospective.json")["summary"]
for k in ["fixed_youden_bacc", "fixed_youden_sens", "fixed_youden_spec", "pro5_youden_bacc", "pro10_youden_bacc", "pro20_youden_bacc", "pro10_youden_sens", "pro10_youden_spec", "pro10_youden_fixed_on_test", "pro10_youden_offline_on_test", "offline_youden_bacc",
          "fixed_restq_bacc", "fixed_restq_sens", "fixed_restq_spec", "pro5_restq_bacc", "pro10_restq_bacc", "pro20_restq_bacc", "pro10_restq_sens", "pro10_restq_spec", "pro10_restq_fixed_on_test", "lda_fixed_bacc", "lda_fixed_auc", "rule_fixed_auc", "lda_within_auc", "rule_within_auc", "lda_drop", "rule_drop"]:
    row(f"前瞻·{k}", f"{P[k]['mean']:.3f} (中位 {P[k]['median']:.3f}, n={P[k]['n']})")
PP = J("paper2_followups", "prospective.json")["per_subject"]
d10 = [PP[s]["summary"]["pro10_youden_bacc"] - PP[s]["summary"]["pro10_youden_fixed_on_test"] for s in PP if PP[s]["summary"].get("pro10_youden_bacc") is not None]
row("前瞻10 − 固定(同评价集)·逐人", f"均值 {np.mean(d10):.3f} 中位 {np.median(d10):.3f} 去sub-01均值 {np.mean([v for s, v in zip(sorted(PP), d10) if s != 'sub-01']):.3f}")
# ---- 原始参考 ----
sec("慢性 · 原始双乳突参考(paper2_followups/rawref.json;CSP 与主分析同实现)")
RW = J("paper2_followups", "rawref.json")["summary"]
for k, v in RW.items(): row(f"原始参考·{k}", f"{v['mean']:.3f} >0.7:{v['n_gt_0.7']}")
MV = J("rule_montage_variants", "chronic.json")["summary"]
row("CAR·Laplacian 规则 会话内", f"{MV['rule_lap']['within_auc']:.3f}"); row("CAR·rule2(oos) 会话内(montage)", f"{MV['rule2']['within_auc']:.3f}")
# ---- 预测 ----
sec("早期−后续关联(paper2_followups/predict.json;目标=第4次会话起均值)")
PR = J("paper2_followups", "predict.json")
for key in ["rule_oos_auc", "csp_lda_auc"]:
    for m in [1, 2, 3]:
        r = PR[key][f"first{m}"]; row(f"预测·{key}·前{m}次", f"ρ={r['spearman_rho']:.2f} [{r['rho_ci95'][0]:.2f},{r['rho_ci95'][1]:.2f}] p={r['p']:.3f} MAE={r['mean_abs_err']:.3f} 过线一致 {r['agree_gt_0.7']}/14 (早期过线 {r['early_gt_0.7']}, 目标过线 {r['target_gt_0.7']})")
    row(f"预测·{key}·患者内两两会话差", f"均值 {PR[key]['pairwise_session_abs_diff_patient_mean']:.3f} 中位 {PR[key]['pairwise_session_abs_diff_patient_median']:.3f}")
    row(f"预测·{key}·分半/ICC样", f"ρ={PR[key]['split_half']['spearman_rho']:.2f} [{PR[key]['split_half']['rho_ci95'][0]:.2f},{PR[key]['split_half']['rho_ci95'][1]:.2f}]; ICC样 {PR[key]['icc_like_single_session']:.2f}")
# ---- 患手/同输入/对照/来源/远端 ----
sec("急性 · 患手(paper2_followups/paretic.json)")
for k, v in J("paper2_followups", "paretic.json")["summary"].items(): row(f"患手·{k}", f"{ms(v)} >0.7:{v['n_gt_0.7']}")
sec("急性 · 同输入对照(matched.json)与导联变体(rule_montage_variants/acute_cue.json)")
for k, v in J("paper2_followups", "matched.json")["summary"].items(): row(f"同输入·{k}", f"{ms(v)} >0.7:{v['n_gt_0.7']}")
for k, v in J("rule_montage_variants", "acute_cue.json")["summary"].items(): row(f"导联变体·{k}", f"{ms(v)} >0.7:{v['n_gt_0.7']} >0.8:{v['n_gt_0.8']}")
sec("急性 · 对照(controls.json / controls2.json)")
K = J("paper2_followups", "controls.json")
for k, v in K["summary"].items():
    if "mean" in v and "sd" in v: row(f"对照·{k}", f"{ms(v)} >0.7:{v['n_gt_0.7']}/{v['n']}")
rows = K["rows"]; row("剔窗", f"全剔 {sum(r['n_windows_kept']==0 for r in rows)} 人, 部分剔 {sum(0<r['n_windows_kept']<120 for r in rows)} 人, 无剔 {sum(r['n_windows_kept']==120 for r in rows)} 人")
C2 = J("paper2_followups", "controls2.json")
for mode in ["cue", "break"]:
    for k, v in C2[mode]["summary"].items(): row(f"来源·{mode}·{k}", f"{ms(v)} >0.7:{v['n_gt_0.7']}")
sec("急性 · 远端参照与增量信息(farref.json;远端=Fp1 Fp2 F7 F8 T5 T6 Oz O1 O2)")
FR = J("paper2_followups", "farref.json")
for mode in ["cue", "break"]:
    for k, v in FR[mode]["summary"].items(): row(f"远端·{mode}·{k}", f"{ms(v)} >0.7:{v['n_gt_0.7']}")
# ---- 临床 ----
sec("急性 · 临床指标(clinical_covariates.json;主口径,分组=CSP>0.7)")
CL = J("stroke_mi_vs_rest", "clinical_covariates.json"); w = CL["windows"]["cue"]
row("可用人数", w["usable_vs_not"]["n_usable"])
for c in ["NIHSS", "MBI", "mRS", "Duration", "Age"]: row(f"临床·{c}", f"{w['usable_vs_not'][c]['usable_mean']} vs {w['usable_vs_not'][c]['not_mean']}, p={w['usable_vs_not'][c]['p_mwu']}")
for c in ["Gender", "ParalysisSide", "IsFirstTime"]: row(f"临床·{c}", f"表 {w['usable_vs_not'][c]['table_usable_then_not']} p={w['usable_vs_not'][c]['p']}")
sig = [(m, c, v) for m in w["spearman"] for c, v in w["spearman"][m].items() if v["p"] < 0.05]
row("临床·Spearman 名义显著(15 项)", f"{len(sig)} 项: {sig}")
row("临床·最小 p 的相关", min(((m, c, v) for m in w["spearman"] for c, v in w["spearman"][m].items()), key=lambda t: t[2]["p"]))
row("一致性 cue vs break(CSP>0.7)", CL["usable_consistency_cue_vs_break"])
# ---- 交叉表 ----
sec("过线交叉表")
cue = {r["sub"]: r for r in J("stroke_mi_vs_rest", "summary_cue.json")["rows"]}; brk = {r["sub"]: r for r in J("stroke_mi_vs_rest", "summary_break.json")["rows"]}
rr = np.array([cue[s]["rule_oos_auc_8-30"] > 0.7 for s in sorted(cue)]); cc = np.array([cue[s]["csp_lda_auc_8-30"] > 0.7 for s in sorted(cue)]); bb = np.array([brk[s]["csp_lda_auc_8-30"] > 0.7 for s in sorted(cue)])
row("规则 vs CSP(cue)", f"都过 {(rr&cc).sum()} 仅规则 {(rr&~cc).sum()} 仅CSP {(~rr&cc).sum()} 都不过 {(~rr&~cc).sum()}")
row("CSP cue vs break", f"都过 {(cc&bb).sum()} 仅cue {(cc&~bb).sum()} 仅break {(~cc&bb).sum()} 都不过 {(~cc&~bb).sum()} 总一致 {(cc==bb).sum()}")
# ---- 天津两数据集 ----
TJ = R / "tianjin"
def tj_block(res, getter, keys, label):
    for k in keys:
        v = [getter(s)[k] for s in sorted(res) if getter(s) is not None and isinstance(getter(s).get(k), (int, float)) and not (isinstance(getter(s).get(k), float) and np.isnan(getter(s).get(k)))]
        if v: row(f"{label}·{k}", f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} >0.7:{sum(x > 0.7 for x in v)}/{len(v)}" if ('auc' in k or 'rule' in k or 'lda' in k or 'csp' in k or 'riem' in k) else f"{np.mean(v):.3f} (中位 {np.median(v):.3f}, n={len(v)})")
WK = ["rule_c34", "lda_c34", "csp_all", "rule_cz", "rule_m7", "lda_m7", "rule_far", "rule_c34_minus_far", "lda_far", "lda_far_c34", "lda_rel_c34", "rule_global", "lda_all_abs", "lda_all_rel", "rule_contra", "rule_ipsi", "eog_rule_1-8", "eog_rule_8-30", "riem_ts_lr", "riem_mdm", "rule_c34_mu", "rule_c34_beta", "lda_c34_mu", "lda_c34_beta", "rule_occ_mu", "rule_occ_beta", "lda_occ_mu", "lda_occ_mu_c34_beta", "rule_c34_foldmean", "lda_c34_foldmean", "csp_all_foldmean", "riem_ts_lr_foldmean", "riem_central", "riem_distal"]
DK = ["fixed_youden_bacc", "pro10_youden_bacc", "pro10_youden_fixed_on_test", "pro10_youden_offline_on_test", "offline_youden_bacc", "fixed_restq_bacc", "pro10_restq_bacc", "pro10_restq_fixed_on_test", "rule_fixed_auc", "lda_fixed_auc", "lda_fixed_bacc", "rule_within_auc", "lda_within_auc", "rule_drop", "lda_drop", "pro5_youden_bacc", "pro5_youden_fixed_on_test", "pro20_youden_bacc", "pro20_youden_fixed_on_test", "pro5_restq_bacc", "pro5_restq_fixed_on_test", "pro20_restq_bacc", "pro20_restq_fixed_on_test", "riem_fixed_auc", "riem_within_auc", "pro5_n_test_mi", "pro20_n_test_mi", "fixed_youden_sens", "fixed_youden_spec", "pro10_youden_sens", "pro10_youden_spec", "pro10_youden_fixed_sens", "pro10_youden_fixed_spec", "fixed_youden_bacc_common", "pro5_youden_bacc_common", "pro10_youden_bacc_common", "pro20_youden_bacc_common", "offline_youden_bacc_common", "n_common_mi", "n_common_ref"]
for fname, lab in [("sf_restlate.json", "第六指(主口径:静息后半窗)"), ("sf.json", "第六指(敏感性:静息两窗)")]:
    if (TJ / fname).exists():
        SFJ = json.load(open(TJ / fname)); sec(f"{lab}(tianjin/{fname})")
        for para in ["AH", "SF"]:
            for tp in ["pre", "post", "follow"]:
                tj_block(SFJ, lambda s, tp=tp, para=para: SFJ[s].get(tp, {}).get(para), WK, f"{lab}·{para}·{tp}")
            for tp in ["post", "follow"]:
                tj_block(SFJ, lambda s, tp=tp, para=para: SFJ[s].get(f"deploy_{para}_pre_to_{tp}"), DK, f"{lab}·{para}·pre→{tp}")
        n = {tp: sum(1 for s in SFJ if tp in SFJ[s]) for tp in ["pre", "post", "follow"]}; row(f"{lab}·人数", n)
for fname, lab in [("ll_raw.json", "下肢(主口径:原始 64 导)"), ("ll.json", "下肢(敏感性:PEEG 40 导)")]:
    if (TJ / fname).exists():
        LLJ = json.load(open(TJ / fname)); sec(f"{lab}(tianjin/{fname})")
        for design in ["task_vs_idle", "task_vs_fixation"]:
            for tp in ["pre", "post", "follow"]:
                tj_block(LLJ, lambda s, tp=tp, d=design: LLJ[s].get(tp, {}).get(d), WK, f"{lab}·{design}·{tp}")
        for tp in ["post", "follow"]:
            tj_block(LLJ, lambda s, tp=tp: LLJ[s].get(f"deploy_cz_pre_to_{tp}"), DK, f"{lab}·{'Cz/FCz' if '原始' in lab else 'Cz/CPz'}·pre→{tp}")
        n = {tp: sum(1 for s in LLJ if tp in LLJ[s]) for tp in ["pre", "post", "follow"]}; row(f"{lab}·人数", n)
        fs = set(LLJ[s][tp]["task_vs_idle"].get("far_set", "") for s in LLJ for tp in ["pre"] if tp in LLJ[s]); row(f"{lab}·远端集合", fs)
# ---- 强化分析(paper2_strengthen:急性/慢性黎曼、频带、偏侧化;对半验证) ----
ST = R / "paper2_strengthen"
for fname, lab in [("acute_cue.json", "强化·急性(提示期参考)"), ("acute_break.json", "强化·急性(休息期参考)"), ("chronic.json", "强化·慢性(CAR,逐会话均值)"), ("acute_eogreg_cue.json", "强化·急性眼电回归(提示期,补段1s)"), ("acute_eogreg_break.json", "强化·急性眼电回归(休息期,补段1s)"), ("acute_eogreg_cue_pad3.json", "强化·急性眼电回归(提示期,补段3s)"), ("acute_eogreg_break_pad3.json", "强化·急性眼电回归(休息期,补段3s)"), ("acute_transient_cue.json", "强化·急性滤波瞬态检验(提示期)")]:
    if (ST / fname).exists():
        JJ = json.load(open(ST / fname)); sec(f"{lab}(paper2_strengthen/{fname})")
        for k, v in JJ["summary"].items(): row(f"{lab}·{k}", f"{v['mean']:.3f} ± {v['sd']:.3f} >0.7:{v['n_gt_0.7']}/{v['n']}")
if (ST / "splithalf.json").exists():
    JJ = json.load(open(ST / "splithalf.json")); sec("强化·对半验证(paper2_strengthen/splithalf.json)")
    for ds, v in JJ.items():
        row(f"对半·{ds}·n", str(v["n"])); row(f"对半·{ds}·候选均值", ", ".join(f"{k}={x:.3f}" for k, x in v["candidates"].items())); row(f"对半·{ds}·主口径", v["main"])
        row(f"对半·{ds}·选中率", ", ".join(f"{k}={x:.2f}" for k, x in v["pick_rate"].items())); row(f"对半·{ds}·A选B评", f"{v['selected_on_A_eval_on_B_mean']:.3f}"); row(f"对半·{ds}·主口径B评", f"{v['main_eval_on_B_mean']:.3f}"); row(f"对半·{ds}·乐观偏差", f"{v['optimism_mean']:+.3f}")
if (TJ / "stats.json").exists():
    TS = json.load(open(TJ / "stats.json")); sec("天津统计家族(tianjin/stats.json;见 docs/tianjin_stats.md)")
    for fam_, rows_ in TS.items():
        for r in rows_: row(f"{fam_}·{r['comparison']}", f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] | {r['n_pos']}/{r['n_neg']} | p={r['p']:.3g} | Holm={r['p_holm']:.3g}")

# ---- 左右手 ----
sec("左右手基线(stroke_baseline)")
for suf, lab in [("", "Q=30"), ("_q6", "Q=6")]:
    p = R / "stroke_baseline" / f"summary_rep10{suf}.json"
    if p.exists():
        B = json.load(open(p))["summary"]
        row(f"左右手·{lab}", "; ".join(f"{k} {v['mean']:.2f}±{v['sd']:.2f}" for k, v in B.items()))
# ---- 统计家族 ----
sec("统计家族(stats_paper2.json;均值差 [bootstrap 95%] | HL [95%] | +/− | p | Holm)")
ST = J("stats_paper2.json")
for fam, rows_ in ST.items():
    if fam.startswith("_"): continue
    lines.append(f"\n### {fam}")
    for r in rows_:
        row(f"{fam[:3]}·{r['comparison']}", f"{r['mean_diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] | HL {r['hl']:+.3f} [{r['hl_ci95'][0]:+.3f},{r['hl_ci95'][1]:+.3f}] | {r['n_pos']}/{r['n_neg']} | p={r['p']:.3g} | Holm={r['p_holm']:.3g}")
lines.append("\n### 过线比例 Wilson 95%")
for lab, v in ST["_proportions"].items(): row(f"比例·{lab}", f"{v['k']}/{v['n']} = {v['wilson95'][0]*100:.0f}%–{v['wilson95'][1]*100:.0f}%")
json.dump(out, open(ROOT / "docs" / "numbers_paper2.json", "w"), ensure_ascii=False, indent=1, default=str)
open(ROOT / "docs" / "numbers_paper2.md", "w").write("\n".join(lines) + "\n")
print("written", len(out), "entries →", ROOT / "docs" / "numbers_paper2.md")
