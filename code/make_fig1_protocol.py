#!/usr/bin/env python3
"""Fig. 1 协议图:四个数据集的试次时序、本文窗口(主口径/敏感性)与屏幕视觉内容。时序与窗口来自各数据集说明与本文代码:
  急性  stroke_mi_vs_rest.py   (MI_ON=2 s, REST_ON=6 s;MI 窗 +0.5/+2.0;cue 参考 0.25;break 参考 6.25;late 6.5)
  慢性  stroke_chronic_mi_vs_rest.py (REST0=0.5, MI0=10.0, MI1=11.5)
  下肢  tianjin_analysis.ll     (任务 4–9 s;窗 +0.5/+2.0;注视对照 +0.5)
  患手  tianjin_analysis.sf     (静息 0–7 s;主口径静息窗 4.0/5.5;敏感性 1.0/4.0;MI 8 s 起 +1.0/+2.5)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})          # PDF 内嵌 TrueType(避免 Type 3)
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'figs_paper2'); os.makedirs(OUT, exist_ok=True)
WIN = 1.5
C_MI, C_REF, C_ALT = '#1f77b4', '#555555', '#bbbbbb'
PERIOD_COLORS = {'cue': '#fde6c4', 'MI': '#dbe9f6', 'break': '#eeeeee', 'prep': '#eeeeee', 'feedback': '#f3e5f5', 'fixation': '#eeeeee', 'text': '#fde6c4', 'task': '#dbe9f6', 'rest': '#e8f5e9', 'prep circle': '#f0f0f0'}

DS = [
    ('A  Acute (Liu 2024): n = 50', 8, [('cue', 0, 2), ('MI', 2, 6), ('break', 6, 8)],
     [(2.5, 'mi'), (4.0, 'mi'), (0.25, 'ref'), (6.25, 'alt'), (6.5, 'alt')],
     'cue: visual cue for left/right hand (2 s) · MI: 4 s · break: 2 s (screen content as in the dataset description)', 'main reference = cue period 0.25 s; alternatives = break 6.25 / 6.5 s'),
    ('B  Chronic (Lu 2026): n = 14', 17, [('prep', 0, 9), ('MI', 9, 13), ('feedback', 13, 17)],
     [(10.0, 'mi'), (11.5, 'mi'), (0.5, 'ref'), (3.0, 'alt'), (6.0, 'alt')],
     'prep: relax; tactile brush input ends at 7.5 s (onset not stated) · MI cue at 9 s · feedback 13–17 s', 'main reference = preparation 0.5 s; split-half candidates 3.0 / 6.0 s'),
    ('C  Lower-limb (Liu 2025): n = 27 released, 25 analysed at the first assessment', 12, [('fixation', 0, 3), ('text', 3, 4), ('task', 4, 9), ('break', 9, 12)],
     [(4.5, 'mi'), (6.0, 'mi'), (4.5, 'ref2'), (6.0, 'ref2'), (0.5, 'alt')],
     'task: MI trial shows gait-phase digit in orange circle; idle trial shows black circle', 'reference = same windows of idle trials; control = same-trial fixation 0.5 s'),
    ('D  Paretic-hand / sixth-finger (Wang 2026): n = 24 released, 22 analysed at the first assessment', 14, [('rest', 0, 7), ('prep circle', 7, 8), ('MI', 8, 14)],
     [(9.0, 'mi'), (10.5, 'mi'), (4.0, 'ref'), (5.5, 'ref'), (1.0, 'alt')],
     'rest: green fixation cross · prep: grey circle · MI: picture of hand / sixth finger', 'main reference = rest 4.0 + 5.5 s; sensitivity = 1.0 + 4.0 s'),
]


def main():
    fig, axes = plt.subplots(4, 1, figsize=(10, 8.6))
    for ax, (title, T, periods, wins, visual, note) in zip(axes, DS):
        for name, a, b in periods:
            ax.add_patch(Rectangle((a, 0.55), b - a, 0.4, facecolor=PERIOD_COLORS[name], edgecolor='k', lw=0.6))
            ax.text((a + b) / 2, 0.75, name, ha='center', va='center', fontsize=8.5)
        row = {'mi': 0.33, 'ref': 0.33, 'ref2': 0.12, 'alt': 0.12}
        for s, kind in wins:
            y = row[kind]; col = {'mi': C_MI, 'ref': C_REF, 'ref2': C_REF, 'alt': C_ALT}[kind]; hatch = '//' if kind == 'alt' else None
            ax.add_patch(Rectangle((s, y), WIN, 0.14, facecolor=col, edgecolor='k', lw=0.5, hatch=hatch, alpha=0.95 if kind != 'alt' else 0.7))
        ax.set_xlim(-0.2, T + 0.2); ax.set_ylim(0, 1.05); ax.set_yticks([]); ax.set_xticks(range(0, T + 1, 1)); ax.tick_params(labelsize=8)
        for sp in ['left', 'right', 'top']: ax.spines[sp].set_visible(False)
        ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
        ax.text(T + 0.15, 0.40, 'MI / reference windows', fontsize=7.5, va='center', ha='left', color='k')
        ax.text(T + 0.15, 0.19, 'alternative / control', fontsize=7.5, va='center', ha='left', color='#666666')
        ax.text(0, -0.30, 'screen: ' + visual, fontsize=7.8, va='top', color='#333333', transform=ax.get_xaxis_transform())
        ax.text(0, -0.56, note, fontsize=7.8, va='top', color='#333333', transform=ax.get_xaxis_transform())
    axes[-1].set_xlabel('time within trial (s)', fontsize=9, labelpad=36)
    h = [Rectangle((0, 0), 1, 1, facecolor=C_MI, edgecolor='k'), Rectangle((0, 0), 1, 1, facecolor=C_REF, edgecolor='k'), Rectangle((0, 0), 1, 1, facecolor=C_ALT, edgecolor='k', hatch='//')]
    fig.legend(h, ['MI window (1.5 s)', 'reference window (main)', 'alternative reference / control window'], loc='upper center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=3.4); fig.subplots_adjust(bottom=0.07, top=0.95, right=0.84)
    fig.savefig(os.path.join(OUT, 'fig1_protocol.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'fig1_protocol.pdf'), bbox_inches='tight')
    print('fig1 →', OUT)


if __name__ == '__main__':
    main()
