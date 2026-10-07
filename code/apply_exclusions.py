#!/usr/bin/env python3
"""按 results/exclusions.json 从天津结果 JSON 中剔除不确定记录(审稿意见 1:信号级重复),写回主文件,完整版另存 *_full.json。
exclusions.json 格式:{"ll_raw.json": [["sub-24","pre"], ...], "sf_restlate.json": [["sub-22","pre"], ["sub-02","follow"]], "note": "..."}
规则:剔除 (sub, tp) 块;若 tp == 'pre',同时剔除该患者全部 deploy_* 条目(以 pre 为校准);若 tp 为 post/follow,剔除对应 deploy_*_pre_to_{tp}。
所有分析均逐患者独立,剔除患者块不影响其他患者的结果,故无需重跑。"""
import json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, 'results', 'tianjin')


def main():
    ex = json.load(open(os.path.join(ROOT, 'results', 'exclusions.json')))
    for fname, units in ex.items():
        if not fname.endswith('.json'): continue
        p = os.path.join(R, fname); full = p.replace('.json', '_full.json')
        if not os.path.exists(full): shutil.copy(p, full)                      # 完整版只备份一次
        D = json.load(open(full)); removed = []
        for sub, tp in units:
            if sub not in D: continue
            if tp in D[sub]: del D[sub][tp]; removed.append(f"{sub}/{tp}")
            for k in list(D[sub]):
                if k.startswith('deploy_') and (tp == 'pre' or k.endswith(f'_pre_to_{tp}')): del D[sub][k]; removed.append(f"{sub}/{k}")
        json.dump(D, open(p, 'w'), indent=1, ensure_ascii=False); print(fname, 'removed:', removed)


if __name__ == '__main__':
    main()
