#!/usr/bin/env python3
"""论文二补实验:不训练的规则法换导联组态,差距到底在「导联数」还是「是否训练」。

变体(全部不含有标签拟合;参考中位数一律样本外/按会话):
  rule2      C3、C4 原始参考(=主稿规则法,复现用)
  rule_lap   C3、C4 大 Laplacian(减四邻均值)
  rule_m7    运动区 7 导(FC3,FC4,C3,C4,CP3,CP4,Cz)逐导功率下降的均值
  rule_lapm  Laplacian 后 6 导(FC3,FC4,C3,C4,CP3,CP4)下降均值
对照:csp_lda(全导联,有训练,主稿数字)
急性:主口径提示期窗(REST_MODE=cue),逐人 AUC;慢性:会话内 AUC + 部署口径(第 1 次会话阈值;固定参考 vs 每会话重标)。
输出:results/rule_montage_variants/{acute_cue.json, chronic.json}
用法:.venv/bin/python code/rule_montage_variants.py [acute|chronic|both]
"""
import os, sys, glob, json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('REST_MODE', 'cue')
import stroke_mi_vs_rest as A
import stroke_chronic_mi_vs_rest as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'rule_montage_variants'); os.makedirs(OUT, exist_ok=True)

def lap(X, center, neigh):
    return X[:, center, :] - X[:, neigh, :].mean(1)

def logp(sig):                      # sig: (窗, 时间)
    return np.log(sig.var(-1))

def variant_feats(X, kind, mont):
    """返回 (窗, 导) 的对数功率矩阵;规则分数 = 相对参考的下降,取 min(两导) 或 mean(多导)。"""
    if kind == 'rule2':
        return np.c_[logp(X[:, mont['C3']]), logp(X[:, mont['C4']])], 'min'
    if kind == 'rule_lap':
        return np.c_[logp(lap(X, mont['C3'], mont['nC3'])), logp(lap(X, mont['C4'], mont['nC4']))], 'min'
    if kind == 'rule_m7':
        return np.stack([logp(X[:, c]) for c in mont['m7']], 1), 'mean'
    if kind == 'rule_lapm':
        return np.stack([logp(lap(X, c, n)) for c, n in mont['lapm']], 1), 'mean'
    raise ValueError(kind)

def score(f, ref, agg):
    d = f - ref
    return -(d.min(1) if agg == 'min' else d.mean(1))


def trial_folds(grp, n_splits=5, seed=0):
    """按试次分折:同一试次的全部窗(两个想象窗+一个静息窗)放同一折。
    修正 2026-10-04:此前用 StratifiedKFold(...).split(f, grp % 5) 只是按 grp%5 分层,同一试次的窗会跨折(审读核出 98/120 测试窗泄漏)。"""
    trials = np.unique(grp)
    perm = np.random.default_rng(seed).permutation(trials)
    fold_of = {t: i % n_splits for i, t in enumerate(perm)}
    fold = np.array([fold_of[t] for t in grp])
    for k in range(n_splits):
        te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
        assert not (set(grp[tr]) & set(grp[te])), '训练/测试试次有交集'
        yield tr, te

def oos_auc(f, y, grp, agg):
    pred = np.zeros(len(y))
    for tr, te in trial_folds(grp):
        pred[te] = score(f[te], np.median(f[tr][y[tr] == 0], 0), agg)
    return float(roc_auc_score(y, pred))

KINDS = ['rule2', 'rule_lap', 'rule_m7', 'rule_lapm']

# 急性 29 导(EEG_IDX 顺序):0Fp1 1Fp2 2Fz 3F3 4F4 5F7 6F8 7FCz 8FC3 9FC4 10FT7 11FT8 12Cz 13C3 14C4 15T3 16T4 17CP3 18CP4 19TP7 20TP8 21Pz 22P3 23P4 24T5 25T6 26Oz 27O1 28O2
MONT_A = dict(C3=13, C4=14, nC3=[8, 17, 15, 12], nC4=[9, 18, 16, 12], m7=[8, 9, 13, 14, 17, 18, 12],
              lapm=[(8, [3, 13, 10, 7]), (9, [4, 14, 11, 7]), (13, [8, 17, 15, 12]), (14, [9, 18, 16, 12]), (17, [13, 22, 19, 21]), (18, [14, 23, 20, 21])])
# 慢性 11 导:0FC3 1FC4 2C5 3C3 4C1 5Cz 6C2 7C4 8C6 9CP3 10CP4
MONT_C = dict(C3=3, C4=7, nC3=[0, 9, 2, 4], nC4=[1, 10, 8, 6], m7=[0, 1, 3, 7, 9, 10, 5],
              lapm=[(0, [3, 2, 4]), (1, [7, 8, 6]), (3, [0, 9, 2, 4]), (7, [1, 10, 8, 6]), (9, [3, 2, 4]), (10, [7, 8, 6])])

def acute():
    import scipy.io as sio
    rows = []
    for fpath in sorted(glob.glob(os.path.join(A.DATA, 'sub-*', '*.mat'))):
        x = sio.loadmat(fpath)['eeg'][0, 0]['rawdata'].astype(float)[:, A.EEG_IDX, :]
        X, y, grp = A.windows(A.filt(x, 8, 30))
        r = {'sub': os.path.basename(os.path.dirname(fpath))}
        for k in KINDS:
            f, agg = variant_feats(X, k, MONT_A)
            r[k] = oos_auc(f, y, grp, agg)
        rows.append(r); print(r['sub'], ' '.join(f'{k}={r[k]:.3f}' for k in KINDS), flush=True)
    main_rows = json.load(open(os.path.join(ROOT, 'results/stroke_mi_vs_rest/summary_cue.json')))['rows']
    csp = {m['sub']: m['csp_lda_auc_8-30'] for m in main_rows}
    for r in rows: r['csp_lda'] = csp[r['sub']]
    summ = {k: {'mean': float(np.mean([r[k] for r in rows])), 'sd': float(np.std([r[k] for r in rows], ddof=1)),
                'n_gt_0.7': int(sum(r[k] > 0.7 for r in rows)), 'n_gt_0.8': int(sum(r[k] > 0.8 for r in rows))} for k in KINDS + ['csp_lda']}
    json.dump({'rows': rows, 'summary': summ, 'rest_mode': os.environ['REST_MODE']}, open(os.path.join(OUT, 'acute_cue.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 急性 50 人(提示期窗,样本外参考)===')
    for k, s in summ.items(): print(f"{k:10s} {s['mean']:.3f} ± {s['sd']:.3f}  >0.7: {s['n_gt_0.7']}/50  >0.8: {s['n_gt_0.8']}/50")

def chronic():
    per = {}
    for sub in sorted(os.path.basename(p) for p in glob.glob(os.path.join(C.DATA, 'sub-*'))):
        sessions = []
        for sdir in sorted(glob.glob(os.path.join(C.DATA, sub, 'ses-*'))):
            x = C.load_session(sdir)
            if x is None or x.shape[0] < 10: continue
            X, y, grp = C.windows(C.filt(x))
            sessions.append((X, y, grp))
        if not sessions: continue
        res = {}
        for k in KINDS:
            feats = [variant_feats(X, k, MONT_C) for X, _, _ in sessions]; agg = feats[0][1]
            f0, y0 = feats[0][0], sessions[0][1]
            ref0 = np.median(f0[y0 == 0], 0)
            fpr, tpr, thr = roc_curve(y0, score(f0, ref0, agg)); thr0 = thr[np.argmax(tpr - fpr)]
            within, fixed_bacc, recal_bacc, fixed_auc = [], [], [], []
            for i, ((X, y, grp), (f, _)) in enumerate(zip(sessions, feats)):
                within.append(oos_auc(f, y, grp, agg))
                if i == 0: continue
                sc = score(f, ref0, agg); pred = (sc >= thr0).astype(int)
                fixed_bacc.append((pred[y == 1].mean() + (1 - pred[y == 0]).mean()) / 2); fixed_auc.append(float(roc_auc_score(y, sc)))
                sc2 = score(f, np.median(f[y == 0], 0), agg); pred2 = (sc2 >= thr0).astype(int)
                recal_bacc.append((pred2[y == 1].mean() + (1 - pred2[y == 0]).mean()) / 2)
            res[k] = {'within_auc': float(np.mean(within)), 'fixed_bacc': float(np.mean(fixed_bacc)), 'recal_bacc': float(np.mean(recal_bacc)), 'fixed_auc': float(np.mean(fixed_auc)), 'n_ses': len(sessions)}
        per[sub] = res
        print(sub, ' | '.join(f"{k}: in {res[k]['within_auc']:.3f} fix {res[k]['fixed_bacc']:.3f} recal {res[k]['recal_bacc']:.3f}" for k in KINDS), flush=True)
    summ = {k: {m: float(np.mean([per[s][k][m] for s in per])) for m in ['within_auc', 'fixed_bacc', 'recal_bacc', 'fixed_auc']} for k in KINDS}
    for k in KINDS: summ[k]['n_within_gt_0.7'] = int(sum(per[s][k]['within_auc'] > 0.7 for s in per))
    json.dump({'per_subject': per, 'summary': summ}, open(os.path.join(OUT, 'chronic.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 慢性 14 人 ===')
    for k, s in summ.items(): print(f"{k:10s} 会话内 {s['within_auc']:.3f} (>0.7: {s['n_within_gt_0.7']}/14)  固定阈值 bacc {s['fixed_bacc']:.3f}  重标 {s['recal_bacc']:.3f}  固定 AUC {s['fixed_auc']:.3f}")

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if what in ('acute', 'both'): acute()
    if what in ('chronic', 'both'): chronic()
