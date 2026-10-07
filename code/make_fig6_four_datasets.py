#!/usr/bin/env python3
"""fig6:四数据集并排汇总(急性 cue / 慢性 CAR / 下肢 任务 vs 空闲 / 第六指 患手 MI vs 静息后半窗)。
A 三种检测器逐人 AUC;B 远端规则 vs 中央规则 与 LDA 中央增量;C 跨会话/时点:固定阈值 BA、前缀重标 BA、规则/LDA 掉幅;D 眼电规则(有眼电的数据集)。
输出 results/figs_paper2/fig6_four_datasets.{png,pdf}"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = lambda *p: os.path.join(ROOT, 'results', *p)
OUT = R('figs_paper2'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
C_RULE, C_LDA, C_CSP, C_FAR, C_RIEM = '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728'

def load():
    D = {}
    cue = json.load(open(R('stroke_mi_vs_rest', 'summary_cue.json')))['rows']
    D['Acute\n(cue ref., n=50)'] = dict(rule=[r['rule_oos_auc_8-30'] for r in cue], lda=[r['lda_auc_8-30'] for r in cue], csp=[r['csp_lda_auc_8-30'] for r in cue])
    fr = json.load(open(R('paper2_followups', 'farref.json')))['cue']['rows']; c2 = json.load(open(R('paper2_followups', 'controls2.json')))['cue']['rows']
    stA = {r['sub']: r for r in json.load(open(R('paper2_strengthen', 'acute_cue.json')))['rows']} if os.path.exists(R('paper2_strengthen', 'acute_cue.json')) else {}
    if stA: D['Acute\n(cue ref., n=50)']['riem'] = [stA[r['sub']]['riem_ts_lr'] for r in cue]
    D['Acute\n(cue ref., n=50)'].update(far=[r['rule_far'] for r in fr], c_minus_far=[r['rule_c34_minus_far'] for r in fr], inc=[r['lda_far_c34'] - r['lda_far'] for r in fr], eog18=[r['eog_rule_1-8'] for r in c2], eog830=[r['eog_rule_8-30'] for r in c2])
    ch = json.load(open(R('stroke_chronic_mi_vs_rest', 'summary.json')))['per_subject']; cs = sorted(ch)
    pr = json.load(open(R('paper2_followups', 'prospective.json')))['per_subject']
    stC = {r['sub']: r for r in json.load(open(R('paper2_strengthen', 'chronic.json')))['rows']} if os.path.exists(R('paper2_strengthen', 'chronic.json')) else {}
    D['Chronic\n(CAR, n=14)'] = dict(rule=[ch[s]['rule_oos_auc_mean'] for s in cs], lda=[ch[s]['lda_auc_mean'] for s in cs], csp=[ch[s]['csp_lda_auc_mean'] for s in cs], riem=[stC[s]['riem_ts_lr'] for s in cs] if stC else None,
                                    fixed=[pr[s]['summary']['fixed_youden_bacc'] for s in cs if s in pr], pro=[pr[s]['summary']['pro10_youden_bacc'] for s in cs if s in pr and pr[s]['summary'].get('pro10_youden_bacc') is not None],
                                    pro_fixed=[pr[s]['summary']['pro10_youden_fixed_on_test'] for s in cs if s in pr and pr[s]['summary'].get('pro10_youden_fixed_on_test') is not None],
                                    rule_drop=[pr[s]['summary']['rule_drop'] for s in cs if s in pr], lda_drop=[pr[s]['summary']['lda_drop'] for s in cs if s in pr])
    llf = R('tianjin', 'll_raw.json') if os.path.exists(R('tianjin', 'll_raw.json')) else R('tianjin', 'll.json'); LL = json.load(open(llf)); ls = sorted(s for s in LL if 'pre' in LL[s])
    g = lambda s, k: LL[s]['pre']['task_vs_idle'][k]
    lab_ll = f"Lower-limb\nMI vs idle\n(n={len(ls)}, C3/C4)"
    D[lab_ll] = dict(rule=[g(s, 'rule_c34') for s in ls], lda=[g(s, 'lda_c34') for s in ls], csp=[g(s, 'csp_all') for s in ls], riem=[g(s, 'riem_ts_lr') for s in ls] if 'riem_ts_lr' in LL[ls[0]]['pre']['task_vs_idle'] else None, far=[g(s, 'rule_far') for s in ls], c_minus_far=[g(s, 'rule_c34_minus_far') for s in ls], inc=[g(s, 'lda_far_c34') - g(s, 'lda_far') for s in ls])
    if 'eog_rule_1-8' in LL[ls[0]]['pre']['task_vs_idle']: D[lab_ll].update(eog18=[g(s, 'eog_rule_1-8') for s in ls], eog830=[g(s, 'eog_rule_8-30') for s in ls])
    dp = [LL[s]['deploy_cz_pre_to_post'] for s in LL if 'deploy_cz_pre_to_post' in LL[s]]
    D[lab_ll].update(fixed=[d['fixed_youden_bacc'] for d in dp], pro=[d['pro10_youden_bacc'] for d in dp if d.get('pro10_youden_bacc') is not None], pro_fixed=[d['pro10_youden_fixed_on_test'] for d in dp if d.get('pro10_youden_fixed_on_test') is not None], rule_drop=[d['rule_drop'] for d in dp], lda_drop=[d['lda_drop'] for d in dp])
    sff = R('tianjin', 'sf_restlate.json') if os.path.exists(R('tianjin', 'sf_restlate.json')) else R('tianjin', 'sf.json'); SF = json.load(open(sff)); ss = sorted(s for s in SF if 'pre' in SF[s] and 'AH' in SF[s]['pre'])
    h = lambda s, k: SF[s]['pre']['AH'][k]
    lab_sf = f"Paretic-hand\nMI vs rest\n(n={len(ss)})"
    D[lab_sf] = dict(rule=[h(s, 'rule_c34') for s in ss], lda=[h(s, 'lda_c34') for s in ss], csp=[h(s, 'csp_all') for s in ss], riem=[h(s, 'riem_ts_lr') for s in ss] if 'riem_ts_lr' in SF[ss[0]]['pre']['AH'] else None, far=[h(s, 'rule_far') for s in ss], c_minus_far=[h(s, 'rule_c34_minus_far') for s in ss], inc=[h(s, 'lda_far_c34') - h(s, 'lda_far') for s in ss], eog18=[h(s, 'eog_rule_1-8') for s in ss], eog830=[h(s, 'eog_rule_8-30') for s in ss])
    dp = [SF[s]['deploy_AH_pre_to_post'] for s in SF if 'deploy_AH_pre_to_post' in SF[s]]
    D[lab_sf].update(fixed=[d['fixed_youden_bacc'] for d in dp], pro=[d['pro10_youden_bacc'] for d in dp if d.get('pro10_youden_bacc') is not None], pro_fixed=[d['pro10_youden_fixed_on_test'] for d in dp if d.get('pro10_youden_fixed_on_test') is not None], rule_drop=[d['rule_drop'] for d in dp], lda_drop=[d['lda_drop'] for d in dp])
    return D

def dots(ax, x, v, c, w=0.18):
    v = np.asarray(v); ax.scatter(x + np.random.default_rng(int(x * 10)).uniform(-w * 0.6, w * 0.6, len(v)), v, s=9, color=c, alpha=0.8, zorder=3); ax.hlines(v.mean(), x - w, x + w, color='k', lw=2, zorder=4)

def main():
    D = load(); names = list(D); fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # A
    ax = axes[0, 0]
    for i, n in enumerate(names):
        cols = [('rule', C_RULE), ('lda', C_LDA), ('csp', C_CSP)] + ([('riem', C_RIEM)] if D[n].get('riem') else [])
        for j, (k, c) in enumerate(cols): dots(ax, i * 5 + j, D[n][k], c, 0.3); ax.text(i * 5 + j, 1.02 + 0.045 * (j % 2), f"{np.mean(D[n][k]):.2f}", ha='center', fontsize=6.3)
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.axhline(0.7, ls='--', color='grey', lw=0.8); ax.set_xticks([i * 5 + 1.5 for i in range(len(names))]); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(0.3, 1.12); ax.set_ylabel('AUC')
    ax.set_title('A  Four detectors\n    (rule / LDA / CSP+LDA / Riemannian TS+LR)', loc='left', fontweight='bold')
    # B
    ax = axes[0, 1]
    for i, n in enumerate(names):
        if 'far' not in D[n]: ax.text(i * 4 + 1, 0.55, 'n/a\n(11 motor ch.)', ha='center', fontsize=7); continue
        dots(ax, i * 4 + 0, D[n]['rule'], C_RULE, 0.3); dots(ax, i * 4 + 1, D[n]['far'], C_FAR, 0.3); dots(ax, i * 4 + 2, D[n]['c_minus_far'], '#2ca02c', 0.3)
        ax.text(i * 4 + 1, 1.02, f"inc. {np.mean(D[n]['inc']):+.3f}", ha='center', fontsize=7)
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.set_xticks([i * 4 + 1 for i in range(len(names))]); ax.set_xticklabels(names, fontsize=7.5); ax.set_ylim(0.3, 1.08)
    ax.set_title('B  Central / distal / central − distal rules', loc='left', fontweight='bold')
    # C
    ax = axes[1, 0]; k = 0; xt = []
    for i, n in enumerate(names):
        if 'fixed' not in D[n]: continue
        x0 = k * 4; xt.append((x0 + 0.5, n.split('\n')[0]))
        ax.bar(x0, np.mean(D[n]['pro_fixed']), 0.8, color='k', alpha=0.6); ax.bar(x0 + 1, np.mean(D[n]['pro']), 0.8, color=C_CSP, alpha=0.8)      # 两柱同评价集
        gain = np.mean(D[n]['pro']) - np.mean(D[n]['pro_fixed']); ax.text(x0 + 0.5, max(np.mean(D[n]['pro_fixed']), np.mean(D[n]['pro'])) + 0.01, f"Δ{gain:+.3f}", ha='center', fontsize=7.5)
        ax.text(x0 + 0.5, 0.42, f"within − fixed:\nrule {np.mean(D[n]['rule_drop']):+.3f}, LDA {np.mean(D[n]['lda_drop']):+.3f}", ha='center', fontsize=6.8, va='bottom')
        k += 1
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.set_ylim(0.4, 0.8); ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([l for _, l in xt], fontsize=7.5); ax.set_ylabel('balanced accuracy, later sessions')
    ax.set_title('C  Later sessions: fixed vs prefix-10 update\n    (same evaluation set)', loc='left', fontweight='bold')
    # D
    ax = axes[1, 1]; k = 0; xt = []
    for i, n in enumerate(names):
        if 'eog18' not in D[n]: continue
        x0 = k * 3; xt.append((x0 + 0.5, n.split('\n')[0]))
        dots(ax, x0, D[n]['eog18'], '#d62728', 0.3); dots(ax, x0 + 1, D[n]['eog830'], '#e377c2', 0.3)
        ax.text(x0, 1.02, f"{np.mean(D[n]['eog18']):.2f}", ha='center', fontsize=7); ax.text(x0 + 1, 1.02, f"{np.mean(D[n]['eog830']):.2f}", ha='center', fontsize=7); k += 1
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.set_ylim(0.3, 1.08); ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([l for _, l in xt], fontsize=7.5)
    ax.set_title('D  EOG power rules (1–8 Hz / 8–30 Hz)', loc='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig6_four_datasets.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig6_four_datasets.pdf'), bbox_inches='tight'); print('fig6 →', OUT)

if __name__ == '__main__': main()
