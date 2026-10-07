#!/usr/bin/env python3
"""论文二四张图(英文标注,供中文稿审阅与英文稿共用)。输出 results/figs_paper2/fig{1..4}.{png,pdf}
fig1 两数据集试次时序、分析窗、标定/评价分离示意
fig2 逐人配对 AUC:规则(样本外)/两特征 LDA/全导 CSP × 急性(提示期窗)、慢性(会话内均值)
fig3 慢性部署口径:固定参考 vs 前瞻 k 试次重标 vs 整场离线重标(同评价集),按会话序号;右:灵敏度/特异度
fig4 过线判定的交叉表(规则 vs CSP;提示期窗 vs 休息期窗)+ 前 1 次会话 vs 后续会话 AUC 散点
用法:.venv/bin/python code/make_figures_paper2.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, 'results', *p)
OUT = R('figs_paper2'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
C_RULE, C_LDA, C_CSP = '#1f77b4', '#ff7f0e', '#2ca02c'

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- fig1
def fig1():
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.6), gridspec_kw={'height_ratios': [1, 1, 1.3]})
    ax = axes[0]; ax.set_title('A  Acute dataset (Liu 2024): one 8-s trial', loc='left', fontweight='bold')
    for x0, w, lab, col in [(0, 2, 'cue', '#dddddd'), (2, 4, 'motor imagery', '#c6dbef'), (6, 2, 'break', '#eeeeee')]:
        ax.add_patch(Rectangle((x0, 0.45), w, 0.5, color=col)); ax.text(x0 + w / 2, 0.7, lab, ha='center', va='center')
    for x0, lab, col in [(0.25, 'reference\n(cue window, main)', C_RULE), (2.5, 'MI window 1', '#08519c'), (4.0, 'MI window 2', '#08519c'), (6.25, 'alt. reference\n(break 0.25–1.75 s)', '#9ecae1')]:
        ax.add_patch(Rectangle((x0, 0.05), 1.5, 0.3, fill=False, edgecolor=col, lw=1.5)); ax.text(x0 + 0.75, -0.25, lab, ha='center', va='top', fontsize=7.5, color=col)
    ax.set_xlim(0, 8); ax.set_ylim(-0.9, 1.05); ax.set_yticks([]); ax.set_xlabel('time (s)'); ax.spines['left'].set_visible(False)
    ax = axes[1]; ax.set_title('B  Chronic dataset (HS Stroke): one 17-s trial', loc='left', fontweight='bold')
    for x0, w, lab, col in [(0, 9, 'preparation (relax; tactile priming ends at 7.5 s, onset not reported)', '#dddddd'), (9, 4, 'motor imagery', '#c6dbef'), (13, 4, 'feedback', '#eeeeee')]:
        ax.add_patch(Rectangle((x0, 0.45), w, 0.5, color=col)); ax.text(x0 + w / 2, 0.7, lab, ha='center', va='center', fontsize=8)
    for x0, lab, col in [(0.5, 'reference (0.5–2.0 s)', C_RULE), (10.0, 'MI window 1', '#08519c'), (11.5, 'MI window 2', '#08519c')]:
        ax.add_patch(Rectangle((x0, 0.05), 1.5, 0.3, fill=False, edgecolor=col, lw=1.5)); ax.text(x0 + 0.75, -0.25, lab, ha='center', va='top', fontsize=7.5, color=col)
    ax.set_xlim(0, 17); ax.set_ylim(-0.9, 1.05); ax.set_yticks([]); ax.set_xlabel('time (s)'); ax.spines['left'].set_visible(False)
    ax = axes[2]; ax.set_title('C  Chronic deployment analysis: what is fitted where', loc='left', fontweight='bold'); ax.axis('off')
    rows = [('Session 1', 'rule formula fixed; reference = median rest power; threshold: Youden (labelled) or 80th pct of rest scores (label-free); LDA trained', '#fdd0a2'),
            ('Session s ≥ 2, fixed', 'reference & threshold from session 1 → balanced accuracy on all trials', '#deebf7'),
            ('Session s ≥ 2, prospective k', 'reference = rest windows of first k trials (k = 5/10/20); evaluate on trials k+1…end; threshold unchanged', '#c7e9c0'),
            ('Session s ≥ 2, offline', 'reference = all rest windows of the session (uses future data; upper reference)', '#eeeeee')]
    for i, (a, b, col) in enumerate(rows):
        y = 0.85 - i * 0.24
        ax.add_patch(Rectangle((0.0, y - 0.09), 0.22, 0.18, color=col, transform=ax.transAxes)); ax.text(0.11, y, a, ha='center', va='center', fontsize=8, transform=ax.transAxes, fontweight='bold')
        ax.text(0.24, y, b, ha='left', va='center', fontsize=7.8, transform=ax.transAxes)
    fig.tight_layout(); save(fig, 'fig1_paradigm')

# ---------------------------------------------------------------- fig2
def fig2():
    cue = json.load(open(R('stroke_mi_vs_rest', 'summary_cue.json')))['rows']
    ch = json.load(open(R('stroke_chronic_mi_vs_rest', 'summary.json')))['per_subject']
    st = json.load(open(R('stats_paper2.json')))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    def panel(ax, data, labels, title, fam, comps):
        n = len(data[0]); x = np.arange(3)
        for i in range(n): ax.plot(x, [d[i] for d in data], color='grey', alpha=0.25, lw=0.6)
        for j, (d, c) in enumerate(zip(data, [C_RULE, C_LDA, C_CSP])):
            ax.scatter(np.full(n, j) + np.random.default_rng(j).uniform(-0.08, 0.08, n), d, s=12, color=c, zorder=3)
            ax.hlines(np.mean(d), j - 0.25, j + 0.25, color='k', lw=2, zorder=4); ax.text(j, 1.0, f'{np.mean(d):.3f}', ha='center', fontsize=8)
        ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.axhline(0.7, ls='--', color='grey', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(title, loc='left', fontweight='bold'); ax.set_ylim(0.3, 1.05)
        fr = {r['comparison']: r for r in st[fam]}
        txt = '\n'.join(f"{lab}: {fr[k]['mean_diff']:+.3f} [{fr[k]['ci95'][0]:+.3f}, {fr[k]['ci95'][1]:+.3f}]{'*' if fr[k]['p_holm'] < 0.05 else ''}" for k, lab in comps)
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=6.8, va='bottom', family='monospace')
    panel(axes[0], [[r['rule_oos_auc_8-30'] for r in cue], [r['lda_auc_8-30'] for r in cue], [r['csp_lda_auc_8-30'] for r in cue]],
          ['fixed rule\n(2 ch, no training)', 'LDA\n(2 features)', 'CSP+LDA\n(29 ch)'], 'A  Acute (n = 50), cue reference', 'F1 急性·提示期窗(主)',
          [('规则(样本外) − 两特征LDA', 'rule−LDA'), ('规则(样本外) − CSP+LDA', 'rule−CSP'), ('两特征LDA − CSP+LDA', 'LDA−CSP')])
    subs = sorted(ch)
    panel(axes[1], [[ch[s]['rule_oos_auc_mean'] for s in subs], [ch[s]['lda_auc_mean'] for s in subs], [ch[s]['csp_lda_auc_mean'] for s in subs]],
          ['fixed rule\n(2 ch)', 'LDA\n(2 features)', 'CSP+LDA\n(11 ch)'], 'B  Chronic (n = 14), within-session', 'F4 慢性·会话内(n=14)',
          [('规则(样本外) − 两特征LDA', 'rule−LDA'), ('规则(样本外) − CSP+LDA', 'rule−CSP')])
    axes[0].set_ylabel('AUC (MI vs reference window)'); fig.tight_layout(); save(fig, 'fig2_paired_auc')

# ---------------------------------------------------------------- fig3
def fig3():
    p = R('paper2_followups', 'prospective.json')
    if not os.path.exists(p): print('fig3: prospective.json 不存在,跳过'); return
    P = json.load(open(p))['per_subject']; subs = sorted(P)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    # A: 按会话序号(第 2 次起)的均值曲线
    ax = axes[0]; maxs = max(len(P[s]['sessions']) for s in subs)
    for key, lab, c in [('fixed_youden_bacc', 'fixed reference (session 1)', 'k'), ('pro10_youden_bacc', 'prospective, first 10 trials', C_CSP), ('offline_youden_bacc', 'offline, whole session', 'grey')]:
        m = [np.mean([P[s]['sessions'][i][key] for s in subs if i < len(P[s]['sessions']) and P[s]['sessions'][i].get(key) is not None]) for i in range(maxs)]
        n = [sum(1 for s in subs if i < len(P[s]['sessions']) and P[s]['sessions'][i].get(key) is not None) for i in range(maxs)]
        xs = [i + 2 for i in range(maxs) if n[i] >= 5]; ax.plot(xs, [m[i] for i in range(maxs) if n[i] >= 5], marker='o', ms=3, color=c, label=lab, ls='--' if key.startswith('offline') else '-')
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.set_xlabel('session index'); ax.set_ylabel('balanced accuracy'); ax.legend(fontsize=7, loc='lower right'); ax.set_title('A  Across sessions (mean, n ≥ 5)', loc='left', fontweight='bold')
    # B: 逐人 固定 vs 前瞻 k
    ax = axes[1]
    keys = [('fixed_youden_bacc', 'fixed'), ('pro5_youden_bacc', 'k=5'), ('pro10_youden_bacc', 'k=10'), ('pro20_youden_bacc', 'k=20'), ('offline_youden_bacc', 'offline')]
    for s in subs:
        ax.plot(range(len(keys)), [P[s]['summary'].get(k) if P[s]['summary'].get(k) is not None else np.nan for k, _ in keys], color='grey', alpha=0.3, lw=0.7)
    for j, (k, _) in enumerate(keys):
        v = [P[s]['summary'][k] for s in subs if P[s]['summary'].get(k) is not None]; ax.hlines(np.mean(v), j - 0.25, j + 0.25, color='k', lw=2); ax.text(j, 0.98, f'{np.mean(v):.3f}', ha='center', fontsize=7.5)
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys]); ax.set_ylim(0.4, 1.0); ax.set_title('B  Per subject (labelled threshold)', loc='left', fontweight='bold')
    # C: 无标签阈值 灵敏度/特异度
    ax = axes[2]
    bars = [('fixed_restq_sens', 'fixed_restq_spec', 'fixed'), ('pro10_restq_sens', 'pro10_restq_spec', 'prospective k=10')]
    w = 0.35
    for j, (ks, kp, lab) in enumerate(bars):
        se = [P[s]['summary'][ks] for s in subs if P[s]['summary'].get(ks) is not None]; sp = [P[s]['summary'][kp] for s in subs if P[s]['summary'].get(kp) is not None]
        ax.bar(j - w / 2, np.mean(se), w, color=C_RULE, label='sensitivity' if j == 0 else None); ax.bar(j + w / 2, np.mean(sp), w, color=C_LDA, label='specificity' if j == 0 else None)
        ax.text(j - w / 2, np.mean(se) + 0.01, f'{np.mean(se):.2f}', ha='center', fontsize=7.5); ax.text(j + w / 2, np.mean(sp) + 0.01, f'{np.mean(sp):.2f}', ha='center', fontsize=7.5)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[2] for b in bars]); ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title('C  Label-free threshold (rest 80th pct)', loc='left', fontweight='bold')
    fig.tight_layout(); save(fig, 'fig3_recalibration')

# ---------------------------------------------------------------- fig4
def fig4():
    cue = {r['sub']: r for r in json.load(open(R('stroke_mi_vs_rest', 'summary_cue.json')))['rows']}
    brk = {r['sub']: r for r in json.load(open(R('stroke_mi_vs_rest', 'summary_break.json')))['rows']}
    subs = sorted(cue)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), gridspec_kw={'width_ratios': [1, 1, 1.25]})   # 2026-10-07 格式审查:C 栏注释移到图下、加大字号
    def mosaic(ax, a, b, la, lb, title):
        t = np.array([[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]])
        ax.imshow(t, cmap='Blues', vmin=0, vmax=max(25, t.max()))
        for i in range(2):
            for j in range(2): ax.text(j, i, str(t[i, j]), ha='center', va='center', fontsize=12, color='k' if t[i, j] < 15 else 'w')
        ax.set_xticks([0, 1]); ax.set_xticklabels([f'{lb} > 0.7', f'{lb} ≤ 0.7'], fontsize=8); ax.set_yticks([0, 1]); ax.set_yticklabels([f'{la} > 0.7', f'{la} ≤ 0.7'], fontsize=8)
        ax.set_title(title, loc='left', fontweight='bold')
    mosaic(axes[0], np.array([cue[s]['rule_oos_auc_8-30'] > 0.7 for s in subs]), np.array([cue[s]['csp_lda_auc_8-30'] > 0.7 for s in subs]), 'rule', 'CSP', 'A  Rule vs CSP (acute, cue)')
    mosaic(axes[1], np.array([cue[s]['csp_lda_auc_8-30'] > 0.7 for s in subs]), np.array([brk[s]['csp_lda_auc_8-30'] > 0.7 for s in subs]), 'cue', 'break', 'B  CSP: cue vs break ref.')
    ch = json.load(open(R('stroke_chronic_mi_vs_rest', 'summary.json')))['per_subject']; cs = sorted(ch)
    ax = axes[2]
    for key, c, lab in [('rule_oos_auc', C_RULE, 'fixed rule'), ('csp_lda_auc', C_CSP, 'CSP+LDA')]:
        e = [ch[s]['sessions'][0][key] for s in cs]; l = [np.mean([x[key] for x in ch[s]['sessions'][3:]]) for s in cs]
        ax.scatter(e, l, color=c, s=18, label=lab)
    ax.plot([0.35, 1], [0.35, 1], ls=':', color='k', lw=0.8); ax.axhline(0.7, ls='--', color='grey', lw=0.7); ax.axvline(0.7, ls='--', color='grey', lw=0.7)
    pr = json.load(open(R('paper2_followups', 'predict.json')))
    k0 = 'rule_oos_auc' if 'rule_oos_auc' in pr else 'rule_auc'
    ax.text(0.0, -0.22, f"Spearman ρ, session 1 → sessions ≥ 4: rule {pr[k0]['first1']['spearman_rho']:.2f} [{pr[k0]['first1']['rho_ci95'][0]:.2f}, {pr[k0]['first1']['rho_ci95'][1]:.2f}], CSP {pr['csp_lda_auc']['first1']['spearman_rho']:.2f}\nSpearman ρ, sessions 1–2 → sessions ≥ 4: rule {pr[k0]['first2']['spearman_rho']:.2f} [{pr[k0]['first2']['rho_ci95'][0]:.2f}, {pr[k0]['first2']['rho_ci95'][1]:.2f}], CSP {pr['csp_lda_auc']['first2']['spearman_rho']:.2f}", transform=ax.transAxes, fontsize=8, va='top', ha='left')
    ax.set_xlabel('AUC, session 1'); ax.set_ylabel('mean AUC, sessions ≥ 4'); ax.legend(fontsize=8, loc='lower right', frameon=False); ax.set_title('C  Chronic: session 1 → later', loc='left', fontweight='bold'); ax.set_xlim(0.35, 1); ax.set_ylim(0.35, 1)
    fig.tight_layout(); save(fig, 'fig4_screening')

# ---------------------------------------------------------------- fig5 来源分解
def fig5():
    p = R('paper2_followups', 'controls2.json')
    if not os.path.exists(p): print('fig5: controls2.json 不存在,跳过'); return
    C2 = json.load(open(p)); FR = json.load(open(R('paper2_followups', 'farref.json')))
    for mode in C2:
        fr = {r['sub']: r for r in FR[mode]['rows']}
        for r in C2[mode]['rows']:
            for k in ['rule_far', 'rule_c34_minus_far', 'rule_lap']: r[k] = fr[r['sub']][k]
    keys = [('eog_rule_1-8', 'EOG 1–8 Hz'), ('eog_rule_8-30', 'EOG 8–30 Hz'), ('fp_rule', 'Fp1/Fp2'), ('occ_rule', 'O1/O2'), ('rule_far', 'far 9-ch mean (no central)'), ('c34_rule', 'C3/C4'), ('c34_filt_only', 'C3/C4, 0.5–40 Hz pre-filter'), ('eog_reg_c34', 'C3/C4, pre-filter + EOG regression'), ('rule_lap', 'C3/C4 Laplacian'), ('global_rule', '29-ch mean (global)'), ('motor_rel_occ', 'C3/C4 − O1/O2'), ('motor_rel_global', 'C3/C4 − global'), ('rule_c34_minus_far', 'C3/C4 − far 9-ch')]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)
    for ax, mode, title in [(axes[0], 'cue', 'A  Cue reference window'), (axes[1], 'break', 'B  Break reference window')]:
        rows = C2[mode]['rows']; m = [np.mean([r[k] for r in rows]) for k, _ in keys]
        cols = ['#d62728'] * 2 + ['#9467bd'] * 3 + [C_RULE] * 4 + ['#7f7f7f'] + ['#2ca02c'] * 3
        ax.barh(range(len(keys)), m, color=cols)
        for i, (v, (k, _)) in enumerate(zip(m, keys)):
            vals = [r[k] for r in rows]; lo, hi = np.percentile([np.mean(np.random.default_rng(j).choice(vals, len(vals))) for j in range(2000)], [2.5, 97.5])
            ax.plot([lo, hi], [i, i], color='k', lw=1); ax.text(max(v, hi) + 0.01, i, f'{v:.3f}', va='center', fontsize=7.5)
        ax.axvline(0.5, ls=':', color='k', lw=0.8); ax.set_yticks(range(len(keys))); ax.set_yticklabels([l for _, l in keys], fontsize=8); ax.invert_yaxis()
        ax.set_xlim(0.45, 1.05); ax.set_xlabel('AUC of fixed power rule (8–30 Hz unless stated)'); ax.set_title(title, loc='left', fontweight='bold')
    fig.tight_layout(); save(fig, 'fig5_sources')

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); print('figs →', OUT)
