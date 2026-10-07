#!/usr/bin/env python3
"""fig7:特异性审计(2026-10-06 强化分析)。
A 按想象手的偏侧化:对侧单导 vs 同侧单导的阶段检测 AUC(急性、慢性);左右手判别 AUC(C3−C4 固定分数)。
B 频带与枕区:C3/C4 mu / C3/C4 beta / 枕区 alpha 规则(急性、下肢、患手;慢性无枕区),以及中央 beta 在枕区 alpha 之上的 LDA 增量。
C 下肢设计对照:「vs 同试次注视」−「vs 空闲试次」的逐人差,按检测器(固定规则 / 功率 LDA / CSP / 黎曼)。
输出 results/figs_paper2/fig7_specificity.{png,pdf}"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = lambda *p: os.path.join(ROOT, 'results', *p)
OUT = R('figs_paper2')
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})


def dots(ax, x, v, c, w=0.3, label=None):
    v = np.asarray([a for a in v if a is not None], float); ax.scatter(x + np.random.default_rng(int(abs(x) * 10) + 1).uniform(-w * 0.6, w * 0.6, len(v)), v, s=9, color=c, alpha=0.8, zorder=3)
    ax.hlines(v.mean(), x - w, x + w, color='k', lw=2, zorder=4); ax.text(x, ax.get_ylim()[1] - 0.02, f"{v.mean():.2f}", ha='center', va='top', fontsize=7.5)


def main():
    ac = json.load(open(R('paper2_strengthen', 'acute_cue.json')))['rows']; ch = json.load(open(R('paper2_strengthen', 'chronic.json')))['rows']
    LL = json.load(open(R('tianjin', 'll_raw.json'))); SF = json.load(open(R('tianjin', 'sf_restlate.json')))
    ls = sorted(s for s in LL if 'pre' in LL[s]); ss = sorted(s for s in SF if 'pre' in SF[s] and 'AH' in SF[s]['pre'])
    g = lambda s, k: LL[s]['pre']['task_vs_idle'].get(k); h = lambda s, k: SF[s]['pre']['AH'].get(k)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2))          # 2026-10-07 格式审查:三栏横排在双栏版面里缩到约 55%,改为三行通栏(版心宽 ≈ 7.1 in,1:1 印出)
    # A 偏侧化
    ax = axes[0]; ax.set_ylim(0.25, 1.06)
    for i, (rows, name) in enumerate([(ac, 'Acute (n=50)'), (ch, 'Chronic (n=14)')]):
        dots(ax, i * 4 + 0, [r['rule_contra_hand'] for r in rows], '#1f77b4'); dots(ax, i * 4 + 1, [r['rule_ipsi_hand'] for r in rows], '#9ecae1'); dots(ax, i * 4 + 2, [r['lr_rule_auc'] for r in rows if 'lr_rule_auc' in r], '#7f7f7f')
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.set_xticks([0, 1, 2, 4, 5, 6]); ax.set_xticklabels(['contra', 'ipsi', 'L vs R', 'contra', 'ipsi', 'L vs R'], fontsize=8)
    ax.text(1, -0.16, 'Acute (n = 50)', ha='center', va='top', fontsize=8.5, transform=ax.get_xaxis_transform()); ax.text(5, -0.16, 'Chronic (n = 14)', ha='center', va='top', fontsize=8.5, transform=ax.get_xaxis_transform()); ax.set_ylabel('AUC')
    ax.set_title('A  Lateralisation by imagined hand', loc='left', fontweight='bold')
    # B 频带与枕区
    ax = axes[1]; ax.set_ylim(0.25, 1.06)
    sets = [('Acute', [r['rule_c34_mu'] for r in ac], [r['rule_c34_beta'] for r in ac], [r['rule_occ_mu'] for r in ac], [r['lda_occ_mu_c34_beta'] - r['lda_occ_mu'] for r in ac]),
            ('Lower-limb', [g(s, 'rule_c34_mu') for s in ls], [g(s, 'rule_c34_beta') for s in ls], [g(s, 'rule_occ_mu') for s in ls], [g(s, 'lda_occ_mu_c34_beta') - g(s, 'lda_occ_mu') for s in ls if g(s, 'lda_occ_mu') is not None]),
            ('Paretic-hand', [h(s, 'rule_c34_mu') for s in ss], [h(s, 'rule_c34_beta') for s in ss], [h(s, 'rule_occ_mu') for s in ss], [h(s, 'lda_occ_mu_c34_beta') - h(s, 'lda_occ_mu') for s in ss if h(s, 'lda_occ_mu') is not None])]
    for i, (name, mu, be, occ, inc) in enumerate(sets):
        dots(ax, i * 4 + 0, mu, '#1f77b4'); dots(ax, i * 4 + 1, be, '#08519c'); dots(ax, i * 4 + 2, occ, '#e377c2')
        ax.text(i * 4 + 1, -0.30, f"{name}\nβ increment over occ. α: {np.mean(inc):+.3f}", ha='center', va='top', fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.axhline(0.5, ls=':', color='k', lw=0.8); ax.set_xticks([i * 4 + j for i in range(3) for j in range(3)]); ax.set_xticklabels(['C3/C4 μ', 'C3/C4 β', 'occ. α'] * 3, fontsize=8, rotation=25, ha='right'); ax.set_ylabel('AUC')
    ax.set_title('B  Band split and occipital-alpha control (rules)', loc='left', fontweight='bold')
    # C 设计对照
    ax = axes[2]
    subs = [s for s in ls if 'task_vs_fixation' in LL[s]['pre']]
    items = [('rule C3/C4', 'rule_c34', '#1f77b4'), ('rule distal', 'rule_far', '#9467bd'), ('LDA 2-feat', 'lda_c34', '#ff7f0e'), ('LDA all-ch', 'lda_all_abs', '#ffbb78'), ('CSP+LDA', 'csp_all', '#2ca02c'), ('Riem TS+LR', 'riem_ts_lr', '#d62728')]
    vals = []
    for j, (lab, k, c) in enumerate(items):
        d = [LL[s]['pre']['task_vs_fixation'][k] - LL[s]['pre']['task_vs_idle'][k] for s in subs if LL[s]['pre']['task_vs_fixation'].get(k) is not None and LL[s]['pre']['task_vs_idle'].get(k) is not None]
        vals.append(d)
    ax.set_ylim(-0.25, 0.48)
    for j, ((lab, k, c), d) in enumerate(zip(items, vals)): dots(ax, j, d, c, 0.3)
    ax.axhline(0, ls=':', color='k', lw=0.8); ax.set_xticks(range(len(items))); ax.set_xticklabels([x[0] for x in items], fontsize=8, rotation=20, ha='right'); ax.set_ylabel('ΔAUC (vs fixation − vs idle)')
    ax.set_title(f'C  Lower-limb design contrast (n={len(subs)})', loc='left', fontweight='bold')
    fig.tight_layout(h_pad=2.2); fig.savefig(os.path.join(OUT, 'fig7_specificity.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig7_specificity.pdf'), bbox_inches='tight'); print('fig7 →', OUT)


if __name__ == '__main__':
    main()
