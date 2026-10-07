#!/usr/bin/env python3
"""急性卒中运动想象数据集(Liu et al., Sci Data 2024)基线复现。

对齐原作者 MATLAB 代码的口径:
  - 29 导脑电(去掉第 18 导与眼电、触发),50 Hz 陷波 + 2 阶巴特沃斯带通(因果滤波)
  - 想象开始后 0-4 s,滤波时前面多带 800 个采样点
  - 每名病人 40 次试验,随机 24 训练 / 16 测试,重复 10 次
方法:
  csp_lda      CSP(2 对)+LDA,8-30 Hz              原文 55.57%
  fbcsp_svm    5 个子带 CSP + 线性 SVM            原文 57.57%
  ts_lda       切空间 + LDA,8-30 Hz
  fgmdm        判别测地滤波 + 最小距离,8-30 Hz    (原文 TSLDA+DGFMDRM 61.20%)
  twfb_max     8 个频带各跑一次 fgmdm,取测试准确率最大者(原作者代码的做法,原文 72.21%)
  twfb_inner   8 个频带,用训练集内部交叉验证选带(无泄漏的做法)
每种方法另跑一遍打乱标签的对照,给出同口径下的随机水平。
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from joblib import Parallel, delayed
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'stroke_acute', 'sourcedata')
OUT = os.path.join(ROOT, 'results', 'stroke_baseline')
FS = 500
EEG_IDX = list(range(17)) + list(range(18, 30))          # 对应 MATLAB 的 [1:17 19:30]
FB_BANDS = [(10, 13), (13, 16), (16, 19), (19, 22), (22, 25)]
TW_BANDS = [(8, 12), (8, 20), (8, 30), (12, 20), (15, 20), (15, 30), (20, 30), (8, 15)]
COV = {}                                                 # 当前病人各频带的试次协方差(每个进程各自一份)


NOTCH_Q = float(os.environ.get('NOTCH_Q', '30'))
SUFFIX = os.environ.get('OUT_SUFFIX', '')


def load_subject(path):
    d = sio.loadmat(path)['eeg'][0, 0]
    return d['rawdata'].astype(float), d['label'].ravel().astype(int)   # (40, 33, 4000), (40,)


def band_epochs(x, lo, hi):
    """返回 (试验, 通道, 2000 点):想象期 0-4 s。x: (40, 33, 4000),一次试验 = 2 s 提示 + 4 s 想象 + 2 s 休息。"""
    seg = x[:, EEG_IDX, 200:3000]                        # 想象开始前 800 点 + 想象期 2000 点
    bn, an = iirnotch(50, NOTCH_Q, FS)                   # 原作者 NotchFilter.m: Q=6;本文此前用 30。NOTCH_Q 环境变量可切换
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    y = lfilter(b, a, lfilter(bn, an, seg, axis=-1), axis=-1)
    return y[:, :, 800:]


def cov_norm(e):
    c = np.einsum('ict,idt->icd', e, e)
    return c / np.trace(c, axis1=1, axis2=2)[:, None, None]


def cov_raw(e):
    return np.einsum('ict,idt->icd', e, e) / e.shape[-1]


def csp_fit(e, y, m=2):
    c = cov_norm(e)
    c1, c2 = c[y == 1].mean(0), c[y == 2].mean(0)
    d, u = np.linalg.eigh(c1 + c2)
    p = (u / np.sqrt(d)).T
    d1, u1 = np.linalg.eigh(p @ c1 @ p.T)
    w = (u1[:, np.argsort(d1)[::-1]]).T @ p
    return np.vstack([w[:m], w[-m:]])


def csp_feat(e, w):
    z = np.einsum('fc,ict->ift', w, e)
    v = z.var(-1)
    return np.log(v / v.sum(1, keepdims=True))


def m_csp_lda(ep, y, tr, te):
    e = ep[(8, 30)]
    w = csp_fit(e[tr], y[tr])
    return LDA().fit(csp_feat(e[tr], w), y[tr]).predict(csp_feat(e[te], w))


def m_fbcsp_svm(ep, y, tr, te):
    ftr, fte = [], []
    for b in FB_BANDS:
        e = ep[b]
        w = csp_fit(e[tr], y[tr])
        ftr.append(csp_feat(e[tr], w))
        fte.append(csp_feat(e[te], w))
    return SVC(kernel='linear').fit(np.hstack(ftr), y[tr]).predict(np.hstack(fte))


def m_ts_lda(ep, y, tr, te):
    c = COV[(8, 30)]
    ts = TangentSpace().fit(c[tr])
    return LDA(solver='lsqr', shrinkage='auto').fit(ts.transform(c[tr]), y[tr]).predict(ts.transform(c[te]))


def fgmdm_pred(c, y, tr, te):
    return FgMDM().fit(c[tr], y[tr]).predict(c[te])


def m_fgmdm(ep, y, tr, te):
    return fgmdm_pred(COV[(8, 30)], y, tr, te)


def m_twfb_inner(ep, y, tr, te):
    """只用训练集做 3 折内部交叉验证选频带,再在测试集上评一次。"""
    best, best_b = -1, TW_BANDS[0]
    for b in TW_BANDS:
        c = COV[b]
        accs = []
        for itr, ival in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, y[tr]):
            accs.append(np.mean(fgmdm_pred(c, y, tr[itr], tr[ival]) == y[tr[ival]]))
        if np.mean(accs) > best:
            best, best_b = np.mean(accs), b
    return fgmdm_pred(COV[best_b], y, tr, te)


METHODS = {'csp_lda': m_csp_lda, 'fbcsp_svm': m_fbcsp_svm, 'ts_lda': m_ts_lda,
           'fgmdm': m_fgmdm, 'twfb_inner': m_twfb_inner}


def run_subject(path, n_rep, seed):
    warnings.filterwarnings('ignore')
    x, y = load_subject(path)
    bands = sorted(set([(8, 30)] + FB_BANDS + TW_BANDS))
    ep = {b: band_epochs(x, *b) for b in bands}
    COV.clear()
    COV.update({b: cov_raw(ep[b]) for b in bands})
    rng = np.random.default_rng(seed)
    res = {k: [] for k in list(METHODS) + ['twfb_max']}
    res.update({k + '_perm': [] for k in list(METHODS) + ['twfb_max']})
    for _ in range(n_rep):
        for tag, yy in (('', y), ('_perm', rng.permutation(y))):
            perm = rng.permutation(40)
            tr, te = perm[:24], perm[24:]
            for name, fn in METHODS.items():
                try:
                    res[name + tag].append(float(np.mean(fn(ep, yy, tr, te) == yy[te]) * 100))
                except Exception:
                    res[name + tag].append(np.nan)
            # 原作者做法:每个频带重新随机划分一次,取 8 个测试准确率里的最大值
            accs = []
            for b in TW_BANDS:
                p2 = rng.permutation(40)
                tr2, te2 = p2[:24], p2[24:]
                try:
                    accs.append(np.mean(fgmdm_pred(COV[b], yy, tr2, te2) == yy[te2]) * 100)
                except Exception:
                    accs.append(np.nan)
            res['twfb_max' + tag].append(float(np.nanmax(accs)))
    return {k: float(np.nanmean(v)) for k, v in res.items()}


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA, 'sub-*', '*.mat')))
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    def one(i, f):
        r = run_subject(f, n_rep, seed=1000 + i)
        r['sub'] = os.path.basename(os.path.dirname(f))
        print(r['sub'], ' '.join(f'{k}={r[k]:.1f}' for k in ['csp_lda', 'fbcsp_svm', 'ts_lda', 'fgmdm', 'twfb_inner', 'twfb_max', 'twfb_max_perm']), flush=True)
        return r

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(i, f) for i, f in enumerate(files))
    keys = [k for k in rows[0] if k != 'sub']
    summ = {k: {'mean': float(np.mean([r[k] for r in rows])), 'sd': float(np.std([r[k] for r in rows], ddof=1))} for k in keys}
    with open(os.path.join(OUT, f'per_subject_rep{n_rep}{SUFFIX}.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    with open(os.path.join(OUT, f'summary_rep{n_rep}{SUFFIX}.json'), 'w') as fh:
        json.dump({'n_subjects': len(rows), 'n_rep': n_rep, 'summary': summ}, fh, indent=1, ensure_ascii=False)
    print('\n=== 汇总 (%d 人, 每人重复 %d 次) ===' % (len(rows), n_rep))
    for k in keys:
        print(f'{k:16s} {summ[k]["mean"]:6.2f} ± {summ[k]["sd"]:.2f}')


if __name__ == '__main__':
    main()
