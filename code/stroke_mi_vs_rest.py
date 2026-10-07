#!/usr/bin/env python3
"""急性卒中数据集(50 人):「在不在想」的检测基线 —— 想象期 vs 静息期。

康复用的脑机接口真正要判的是「病人此刻有没有在尝试/想象运动」,不是左手还是右手。
每次试验 8 s = 提示 2 s + 想象 4 s + 休息 2 s。取:
  想象段:想象开始后 0.5–3.5 s(3 s)      静息段:休息开始后 0.25–1.75 s + 下一次提示期前 1.5 s 拼成的参考(这里只用休息段 0.25–2.0 s,1.75 s)
为公平比较,把两类都切成 1.5 s 的窗:想象段取 0.5–2.0 与 2.0–3.5 两窗,休息段取 0.25–1.75 一窗 → 每人 80 个想象窗、40 个休息窗。
特征:C3、C4 两导 8–30 Hz 的对数功率(也给出 mu 8–13 / beta 13–30 分开的版本)。
方法:
  rule_fixed   固定规则:功率低于「本人休息段中位数」的某个比例即判为想象 —— 不训练分类器,阈值比例全体病人统一(阈值用全体病人另一半数据定),输出 AUC 与固定阈值下的灵敏度/特异度
  lda_cv       逐人 5 折交叉验证的 LDA(有训练)
  csp_lda_cv   逐人 CSP(29 导)+LDA,5 折
指标:每人 AUC;报告均值、AUC>0.7 的人数(「可用」)。
"""
import glob
import json
import os
import warnings

import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'stroke_acute', 'sourcedata')
OUT = os.path.join(ROOT, 'results', 'stroke_mi_vs_rest')
FS = 500
EEG_IDX = list(range(17)) + list(range(18, 30))
C3, C4 = 13, 14                                   # 在 29 导列表里的位置(FP1,FP2,Fz,F3,F4,F7,F8,FCz,FC3,FC4,FT7,FT8,Cz,C3,C4,…)
WIN = int(1.5 * FS)
MI_ON, REST_ON = 1000, 3000                        # 想象、休息的起点(采样点)


def filt(x, lo, hi):
    """每次试验是单独的 8 s 段,因果滤波从第 0 点起有启动暂态;前面镜像补 1 s 再滤,去掉补的部分。"""
    bn, an = iirnotch(50, 30, FS)
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    pad = FS
    xp = np.concatenate([x[:, :, pad:0:-1], x], axis=-1)
    return lfilter(b, a, lfilter(bn, an, xp, axis=-1), axis=-1)[:, :, pad:]


REST_MODE = os.environ.get('REST_MODE', 'break')      # break=休息期 0.25–1.75 s;cue=提示期 0.25–1.75 s(想象开始前);late=休息期 0.5–2.0 s


def windows(xf):
    """xf: (40, 29, 4000) 已滤波。返回 (窗, 29, 750), 标签 1=想象 0=静息。"""
    mi1 = xf[:, :, MI_ON + 250: MI_ON + 250 + WIN]
    mi2 = xf[:, :, MI_ON + 1000: MI_ON + 1000 + WIN]
    if REST_MODE == 'cue':
        rest = xf[:, :, 125: 125 + WIN]
    elif REST_MODE == 'late':
        rest = xf[:, :, REST_ON + 250: REST_ON + 250 + WIN]
    else:
        rest = xf[:, :, REST_ON + 125: REST_ON + 125 + WIN]
    X = np.concatenate([mi1, mi2, rest], 0)
    y = np.r_[np.ones(80), np.zeros(40)].astype(int)
    grp = np.r_[np.arange(40), np.arange(40), np.arange(40)]      # 同一试验的窗放同一折,防止泄漏
    return X, y, grp



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

def logpow(X, ch):
    return np.log(X[:, ch, :].var(-1))


def csp_fit(e, y, m=2):
    c = np.einsum('ict,idt->icd', e, e)
    c = c / np.trace(c, axis1=1, axis2=2)[:, None, None]
    c1, c2 = c[y == 1].mean(0), c[y == 0].mean(0)
    d, u = np.linalg.eigh(c1 + c2)
    p = (u / np.sqrt(d)).T
    d1, u1 = np.linalg.eigh(p @ c1 @ p.T)
    w = (u1[:, np.argsort(d1)[::-1]]).T @ p
    return np.vstack([w[:m], w[-m:]])


def csp_feat(e, w):
    v = np.einsum('fc,ict->ift', w, e).var(-1)
    return np.log(v / v.sum(1, keepdims=True))


def per_subject(path):
    x = sio.loadmat(path)['eeg'][0, 0]['rawdata'].astype(float)[:, EEG_IDX, :]
    out = {}
    for band, (lo, hi) in {'8-30': (8, 30), 'mu': (8, 13), 'beta': (13, 30)}.items():
        X, y, grp = windows(filt(x, lo, hi))
        f = np.c_[logpow(X, C3), logpow(X, C4)]                                # (120, 2)
        # 固定规则:相对本人静息中位的功率下降量,取 C3/C4 中下降更多者;分数越低越像想象
        ref = np.median(f[y == 0], 0)
        score = -(f - ref).min(1)
        out[f'rule_auc_{band}'] = float(roc_auc_score(y, score))
        out[f'rule_drop_{band}'] = float(np.median((f[y == 1] - ref).mean(1)))  # 想象窗相对静息的平均对数功率变化
        # 规则法的样本外版本:静息参考中位数只用训练折的静息窗算,再给测试折打分(与 LDA 同一套折)
        pred_r = np.zeros(len(y))
        for tr, te in trial_folds(grp):
            ref_tr = np.median(f[tr][y[tr] == 0], 0)
            pred_r[te] = -(f[te] - ref_tr).min(1)
        out[f'rule_oos_auc_{band}'] = float(roc_auc_score(y, pred_r))
        # 有训练:LDA 在 2 个功率特征上,按试验分折
        pred = np.zeros(len(y))
        for tr, te in trial_folds(grp):
            pred[te] = LDA().fit(f[tr], y[tr]).decision_function(f[te])
        out[f'lda_auc_{band}'] = float(roc_auc_score(y, pred))
        if band == '8-30':
            pred = np.zeros(len(y))
            for tr, te in trial_folds(grp):
                w = csp_fit(X[tr], y[tr])
                pred[te] = LDA().fit(csp_feat(X[tr], w), y[tr]).decision_function(csp_feat(X[te], w))
            out['csp_lda_auc_8-30'] = float(roc_auc_score(y, pred))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, 'sub-*', '*.mat'))):
        r = per_subject(f)
        r['sub'] = os.path.basename(os.path.dirname(f))
        rows.append(r)
        print(r['sub'], ' '.join(f'{k}={v:.2f}' for k, v in r.items() if k.startswith(('rule_auc', 'lda_auc', 'csp'))), flush=True)
    keys = [k for k in rows[0] if k != 'sub']
    summ = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        summ[k] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1))}
        if 'auc' in k:
            summ[k]['n_auc_gt_0.7'] = int((v > 0.7).sum())
            summ[k]['n_auc_gt_0.8'] = int((v > 0.8).sum())
    json.dump({'rows': rows, 'summary': summ, 'rest_mode': REST_MODE}, open(os.path.join(OUT, f'summary_{REST_MODE}.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 想象期 vs 静息期,50 人 ===')
    for k in keys:
        s = summ[k]
        extra = f"  AUC>0.7: {s['n_auc_gt_0.7']}/50  AUC>0.8: {s['n_auc_gt_0.8']}/50" if 'auc' in k else ''
        print(f'{k:22s} {s["mean"]:.3f} ± {s["sd"]:.3f}{extra}')


if __name__ == '__main__':
    main()
