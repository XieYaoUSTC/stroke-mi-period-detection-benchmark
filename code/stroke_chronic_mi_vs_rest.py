#!/usr/bin/env python3
"""慢性卒中 HS Stroke(14 人 × 约 20 次会话):「在不在想」检测与跨会话稳定性。

每次试验 17 s = 准备 9 s + 想象 4 s + 反馈 4 s(256 Hz,11 导,双乳突参考)。
功率曲线(1,060 试次实测)显示准备期头 2 s 功率最高、想象期 10–13 s 最低,故:
  静息窗 = 准备期 0.5–2.0 s;想象窗 = 10.0–11.5 s 与 11.5–13.0 s(与急性卒中数据同为 1.5 s 窗)
特征:C3、C4 的 8–30 Hz 对数功率;另给 11 导 CSP。
逐会话:rule(固定规则,不训练)/ lda(2 特征,会话内 5 折)/ csp_lda(11 导,会话内 5 折)的 AUC
跨会话(真正的部署口径):
  rule_fixed   在第 1 次会话上定一个阈值(规则分数的最佳平衡点),原样用到后面所有会话 → 逐会话平衡准确率
  lda_fixed    在第 1 次会话训练 LDA,原样用到后面所有会话 → 逐会话 AUC
  lda_retrain  每次会话重训(会话内 5 折)→ 对照
同一会话内重复的试次按内容去重。
"""
import glob
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from scipy.signal import butter, iirnotch, lfilter
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'stroke_chronic_hs')
OUT = os.path.join(ROOT, 'results', 'stroke_chronic_mi_vs_rest')
FS = 256
C3, C4 = 3, 7                      # FC3,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP4
WIN = int(1.5 * FS)
REST0, MI0, MI1 = int(0.5 * FS), int(10.0 * FS), int(11.5 * FS)


def filt(x, lo=8, hi=30):
    bn, an = iirnotch(50, 30, FS)
    b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    pad = FS
    xp = np.concatenate([x[:, :, pad:0:-1], x], axis=-1)
    return lfilter(b, a, lfilter(bn, an, xp, axis=-1), axis=-1)[:, :, pad:]


def load_session(sdir):
    trials, seen = [], set()
    for f in sorted(glob.glob(os.path.join(sdir, 'eeg', '*run-*_eeg.mat'))):
        d = sio.loadmat(f, squeeze_me=True)
        x = np.transpose(d['EEGdata'], (2, 0, 1)).astype(float)          # (试次, 11, 4352)
        for tr in x:
            h = hashlib.md5(np.round(tr[:, :512] * 1e6, 3).tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            trials.append(tr)
    if not trials:
        return None
    x = np.stack(trials)
    x = x[~np.isnan(x).any(axis=(1, 2))]
    x = x - x.mean(1, keepdims=True)                                     # 平均参考(11 导)
    return x


def windows(xf):
    mi1 = xf[:, :, MI0:MI0 + WIN]
    mi2 = xf[:, :, MI1:MI1 + WIN]
    rest = xf[:, :, REST0:REST0 + WIN]
    X = np.concatenate([mi1, mi2, rest], 0)
    n = xf.shape[0]
    y = np.r_[np.ones(2 * n), np.zeros(n)].astype(int)
    grp = np.r_[np.arange(n), np.arange(n), np.arange(n)]
    return X, y, grp


def feats(X):
    return np.c_[np.log(X[:, C3, :].var(-1)), np.log(X[:, C4, :].var(-1))]


def rule_score(f, ref):
    return -(f - ref).min(1)


def csp_fit(e, y, m=2):
    c = np.einsum('ict,idt->icd', e, e)
    c = c / np.trace(c, axis1=1, axis2=2)[:, None, None]
    c1, c2 = c[y == 1].mean(0), c[y == 0].mean(0)
    c1 = c1 + 1e-6 * np.trace(c1) / c1.shape[0] * np.eye(c1.shape[0])     # 轻微正则,防奇异
    c2 = c2 + 1e-6 * np.trace(c2) / c2.shape[0] * np.eye(c2.shape[0])
    d, u = np.linalg.eigh(c1 + c2)
    p = (u / np.sqrt(d)).T
    d1, u1 = np.linalg.eigh(p @ c1 @ p.T)
    w = (u1[:, np.argsort(d1)[::-1]]).T @ p
    return np.vstack([w[:m], w[-m:]])


def csp_feat(e, w):
    v = np.einsum('fc,ict->ift', w, e).var(-1)
    return np.log(v / v.sum(1, keepdims=True))



def trial_folds(grp, n_splits=5, seed=0):
    """按试次分折:同一试次的全部窗(两个想象窗+一个静息窗)放同一折。
    修正 2026-10-04:此前用 StratifiedKFold(...).split(f, grp % 5) 只是按 grp%5 分层,同一试次的窗会跨折(98/120 测试窗泄漏)。"""
    trials = np.unique(grp)
    perm = np.random.default_rng(seed).permutation(trials)
    fold_of = {t: i % n_splits for i, t in enumerate(perm)}
    fold = np.array([fold_of[t] for t in grp])
    for k in range(n_splits):
        te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
        assert not (set(grp[tr]) & set(grp[te])), '训练/测试试次有交集'
        yield tr, te

def cv_auc(fn_fit_pred, X, y, grp):
    pred = np.zeros(len(y))
    for tr, te in trial_folds(grp):
        pred[te] = fn_fit_pred(tr, te)
    return float(roc_auc_score(y, pred))


def run_subject(sub):
    sdirs = sorted(glob.glob(os.path.join(DATA, sub, 'ses-*')))
    sessions = []
    for sdir in sdirs:
        x = load_session(sdir)
        if x is None or x.shape[0] < 10:
            continue
        X, y, grp = windows(filt(x))
        f = feats(X)
        sessions.append({'ses': os.path.basename(sdir), 'X': X, 'y': y, 'grp': grp, 'f': f, 'n_trials': x.shape[0]})
    if not sessions:
        return sub, None
    rows = []
    s0 = sessions[0]
    ref0 = np.median(s0['f'][s0['y'] == 0], 0)
    sc0 = rule_score(s0['f'], ref0)
    fpr, tpr, thr = roc_curve(s0['y'], sc0)
    thr0 = thr[np.argmax(tpr - fpr)]                                      # 第 1 次会话上的最佳平衡阈值
    lda0 = LDA().fit(s0['f'], s0['y'])
    for s in sessions:
        f, y, X, grp = s['f'], s['y'], s['X'], s['grp']
        r = {'ses': s['ses'], 'n_trials': s['n_trials']}
        ref = np.median(f[y == 0], 0)
        r['rule_auc'] = float(roc_auc_score(y, rule_score(f, ref)))
        pred_r = np.zeros(len(y))                                             # 样本外参考:只用训练折参考窗算中位数(2026-10-05 加)
        for tr, te in trial_folds(grp):
            pred_r[te] = rule_score(f[te], np.median(f[tr][y[tr] == 0], 0))
        r['rule_oos_auc'] = float(roc_auc_score(y, pred_r))
        r['lda_auc'] = cv_auc(lambda tr, te: LDA().fit(f[tr], y[tr]).decision_function(f[te]), f, y, grp)
        def csp_fp(tr, te):
            w = csp_fit(X[tr], y[tr])
            return LDA().fit(csp_feat(X[tr], w), y[tr]).decision_function(csp_feat(X[te], w))
        try:
            r['csp_lda_auc'] = cv_auc(csp_fp, X, y, grp)
        except Exception:
            r['csp_lda_auc'] = float('nan')
        # 部署口径:第 1 次会话定下的东西原样用
        sc = rule_score(f, ref0)                                           # 参考值也沿用第 1 次会话
        pred = (sc >= thr0).astype(int)
        sens = float((pred[y == 1] == 1).mean())
        spec = float((pred[y == 0] == 0).mean())
        r['rule_fixed_bacc'] = (sens + spec) / 2
        r['rule_fixed_auc'] = float(roc_auc_score(y, sc))
        r['lda_fixed_auc'] = float(roc_auc_score(y, lda0.decision_function(f)))
        # 部署口径二:阈值沿用第 1 次会话,但静息参考水平每次会话重标(用该会话的静息窗中位数)
        sc2 = rule_score(f, ref)
        pred2 = (sc2 >= thr0).astype(int)
        r['rule_recal_bacc'] = (float((pred2[y == 1] == 1).mean()) + float((pred2[y == 0] == 0).mean())) / 2
        rows.append(r)
    out = {'sessions': rows}
    for k in ['rule_auc', 'rule_oos_auc', 'lda_auc', 'csp_lda_auc', 'rule_fixed_bacc', 'rule_recal_bacc', 'rule_fixed_auc', 'lda_fixed_auc']:
        v = np.array([r[k] for r in rows], float)
        out[k + '_mean'] = float(np.nanmean(v))
        out[k + '_sd'] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
        later = v[1:] if len(v) > 1 else v
        out[k + '_later_mean'] = float(np.nanmean(later))                       # 第 2 次会话起(部署口径看这个)
    v = np.array([r['csp_lda_auc'] for r in rows], float)
    ok = ~np.isnan(v)
    out['trend_rho'] = float(spearmanr(np.arange(len(v))[ok], v[ok]).correlation) if ok.sum() > 3 else float('nan')
    print(sub, f"会话 {len(rows)}  rule {out['rule_auc_mean']:.3f}  lda {out['lda_auc_mean']:.3f}  csp {out['csp_lda_auc_mean']:.3f}  | 固定规则(2 次起) 平衡准确率 {out['rule_fixed_bacc_later_mean']:.3f} 重标参考 {out['rule_recal_bacc_later_mean']:.3f} AUC {out['rule_fixed_auc_later_mean']:.3f}  固定LDA {out['lda_fixed_auc_later_mean']:.3f}  趋势ρ {out['trend_rho']:+.2f}", flush=True)
    return sub, out


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(DATA, 'sub-*')) if os.path.isdir(d))
    res = dict(Parallel(n_jobs=4)(delayed(run_subject)(s) for s in subs))
    res = {k: v for k, v in res.items() if v}
    part = pd.read_csv(os.path.join(DATA, 'participants.tsv'), sep='\t').set_index('participant_id')
    summ = {'n_subjects': len(res)}
    for k in ['rule_auc', 'rule_oos_auc', 'lda_auc', 'csp_lda_auc', 'rule_fixed_bacc', 'rule_recal_bacc', 'rule_fixed_auc', 'lda_fixed_auc']:
        v = np.array([res[s][k + '_mean'] for s in res])
        summ[k] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'n_gt_0.7': int((v > 0.7).sum()), 'n_gt_0.8': int((v > 0.8).sum())}
        vl = np.array([res[s][k + '_later_mean'] for s in res])
        summ[k]['later_mean'] = float(vl.mean())
        summ[k]['within_subject_session_sd'] = float(np.mean([res[s][k + '_sd'] for s in res]))
    # 与 FMA 的关系(n=14,功效低,只报相关)
    auc = np.array([res[s]['csp_lda_auc_mean'] for s in res])
    pre = np.array([part.loc[s, 'fma_pre'] for s in res], float)
    gain = np.array([part.loc[s, 'fma_post'] - part.loc[s, 'fma_pre'] for s in res], float)
    summ['fma'] = {'rho_auc_vs_fma_pre': float(spearmanr(auc, pre).correlation), 'p': float(spearmanr(auc, pre).pvalue),
                   'rho_auc_vs_fma_gain': float(spearmanr(auc, gain).correlation), 'p_gain': float(spearmanr(auc, gain).pvalue)}
    summ['trend_rho_mean'] = float(np.nanmean([res[s]['trend_rho'] for s in res]))
    json.dump({'per_subject': res, 'summary': summ}, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 慢性卒中 %d 人 ===' % len(res))
    for k in ['rule_auc', 'rule_oos_auc', 'lda_auc', 'csp_lda_auc', 'rule_fixed_bacc', 'rule_recal_bacc', 'rule_fixed_auc', 'lda_fixed_auc']:
        s = summ[k]
        print(f"{k:16s} 人均 {s['mean']:.3f} ± {s['sd']:.3f}  >0.7: {s['n_gt_0.7']}/{len(res)}  >0.8: {s['n_gt_0.8']}/{len(res)}  第2次起 {s['later_mean']:.3f}  会话间SD {s['within_subject_session_sd']:.3f}")
    print('AUC 与 FMA 基线 ρ=%.2f (p=%.2f);与 FMA 增量 ρ=%.2f (p=%.2f);会话序号趋势 ρ 均值 %.2f' % (summ['fma']['rho_auc_vs_fma_pre'], summ['fma']['p'], summ['fma']['rho_auc_vs_fma_gain'], summ['fma']['p_gain'], summ['trend_rho_mean']))


if __name__ == '__main__':
    main()
