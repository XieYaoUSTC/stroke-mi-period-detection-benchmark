#!/usr/bin/env python3
"""敏感性:重复记录「每对保留一份」(results/exclusions_keep_one.json)与主口径「两份都剔」的关键均值对照;另给「不剔除(完整版)」。
输出 docs/sensitivity_duplicates.md。不改主结果文件。"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, 'results', 'tianjin')


def filt(D, units):
    D = json.loads(json.dumps(D))
    for sub, tp in units:
        if sub not in D: continue
        D[sub].pop(tp, None)
        for k in list(D[sub]):
            if k.startswith('deploy_') and (tp == 'pre' or k.endswith(f'_pre_to_{tp}')): del D[sub][k]
    return D


def summarise(D, getter_pre, getter_dep):
    out = {}
    pre = [getter_pre(D[s]) for s in sorted(D) if getter_pre(D[s]) is not None]
    out['n_pre'] = len(pre)
    for k in ['rule_c34', 'lda_c34', 'csp_all', 'riem_ts_lr', 'rule_far', 'lda_far_c34', 'lda_far']:
        v = [p[k] for p in pre if p.get(k) is not None]; out[k] = round(float(np.mean(v)), 4) if v else None
    for tp in ['post', 'follow']:
        d = [getter_dep(D[s], tp) for s in sorted(D) if getter_dep(D[s], tp) is not None and getter_dep(D[s], tp).get('pro10_youden_bacc') is not None]
        out[f'n_dep_{tp}'] = len(d)
        if d:
            out[f'gain10_{tp}'] = round(float(np.mean([x['pro10_youden_bacc'] - x['pro10_youden_fixed_on_test'] for x in d])), 4)
            out[f'fixed_{tp}'] = round(float(np.mean([x['fixed_youden_bacc'] for x in d])), 4)
            out[f'rule_within_minus_fixed_{tp}'] = round(float(np.mean([x['rule_within_auc'] - x['rule_fixed_auc'] for x in d])), 4)
            if all('riem_fixed_auc' in x for x in d): out[f'riem_within_minus_fixed_{tp}'] = round(float(np.mean([x['riem_within_auc'] - x['riem_fixed_auc'] for x in d])), 4)
    return out


def main():
    main_ex = json.load(open(os.path.join(ROOT, 'results', 'exclusions.json'))); keep1 = json.load(open(os.path.join(ROOT, 'results', 'exclusions_keep_one.json')))
    L = ['# 重复记录处理的敏感性(code/sensitivity_keep_one.py;2026-10-07)\n', '口径:完整版(不剔除)/ 主口径(每对两份都剔 + sub-02 随访剔)/ 保留其一(剔编号较大者 + sub-02 随访剔)。均为患者均值。\n']
    for fname, gp, gd in [('ll_raw.json', lambda s: s.get('pre', {}).get('task_vs_idle'), lambda s, tp: s.get(f'deploy_cz_pre_to_{tp}')), ('sf_restlate.json', lambda s: s.get('pre', {}).get('AH'), lambda s, tp: s.get(f'deploy_AH_pre_to_{tp}'))]:
        full = os.path.join(R, fname.replace('.json', '_full.json')); D = json.load(open(full if os.path.exists(full) else os.path.join(R, fname)))
        rows = {'完整版': summarise(D, gp, gd), '主口径(两份都剔)': summarise(filt(D, main_ex.get(fname, [])), gp, gd), '保留其一': summarise(filt(D, keep1.get(fname, [])), gp, gd)}
        keys = sorted({k for r in rows.values() for k in r})
        L.append(f"\n## {fname}\n\n| 指标 | " + " | ".join(rows) + " |\n|---|" + "---|" * len(rows))
        for k in keys: L.append(f"| {k} | " + " | ".join(str(rows[r].get(k)) for r in rows) + " |")
    open(os.path.join(ROOT, 'docs', 'sensitivity_duplicates.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
