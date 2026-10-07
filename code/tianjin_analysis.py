#!/usr/bin/env python3
"""天津两个纵向卒中数据集:把论文二的流水线搬过去(2026-10-06)。

子命令:
  sf   第六指数据集(24 人;原始 .set 250 Hz 64 导含 HEOR/HEOL/VEOU/VEOL/ECG;事件 1=静息段起 2=患手 MI 起 3=第六指 MI 起;三时点 pre/post/follow)
  ll   下肢数据集(27 人;PEEG 预处理版 250 Hz 60 导平均参考;试次内 1/7 注视 2/8 文字 3–6/9–12 任务;MI 试次 vs 空闲试次;时点 pre/post/follow + IES/SES)
每个数据集、每人、每时点:
  想象窗 vs 参考窗的 固定规则(样本外参考;C3/C4;下肢另给 Cz)、2 特征 LDA、全导 CSP+LDA 的逐人 AUC(按试次分折)
  远端 9 导规则、C3/C4 减远端规则、LDA(远端)/LDA(远端+中央)增量、相对特征 LDA
  第六指另给 EOG(水平/垂直)1–8 与 8–30 Hz 规则、患侧对侧/同侧单导
跨时点(pre→post、pre→follow):pre 定参考+阈值(Youden / 参考分位数 80%)固定;前 10 试次前缀重标;整场离线重标;pre 训练 LDA 固定;掉幅
输出:results/tianjin/{sf,ll}.json
"""
import os, sys, glob, json, re
import numpy as np
import mne
from scipy.signal import butter, iirnotch, lfilter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stroke_mi_vs_rest as A            # trial_folds
import stroke_chronic_mi_vs_rest as C    # 带轻微正则的 csp_fit / csp_feat
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'tianjin'); os.makedirs(OUT, exist_ok=True)
FS = 250; WIN = int(1.5 * FS); REST_Q = 80; K = 10; KS = [5, 10, 20]
mne.set_log_level('ERROR')

def filt(x, lo, hi, fs=FS):
    bn, an = iirnotch(50, 30, fs); b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return lfilter(b, a, lfilter(bn, an, x, axis=-1), axis=-1)

def logp(sig): return np.log(sig.var(-1) + 1e-30)
def rule(f, ref, agg='min'):
    d = f - ref; return -(d.min(1) if agg == 'min' else d.mean(1))
def bacc(y, pred): return float((pred[y == 1].mean() + (1 - pred[y == 0]).mean()) / 2)

def oos_rule_auc(f, y, grp, agg='min'):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp): pred[te] = rule(f[te], np.median(f[tr][y[tr] == 0], 0), agg)
    return float(roc_auc_score(y, pred))
def cv_lda_auc(f, y, grp):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp): pred[te] = LDA().fit(f[tr], y[tr]).decision_function(f[te])
    return float(roc_auc_score(y, pred))
def _foldmean(scores_by_fold):
    v = [roc_auc_score(yt, st) for yt, st in scores_by_fold if len(set(yt)) == 2]; return float(np.mean(v)) if v else float('nan')

def oos_rule_auc_both(f, y, grp, agg='min'):
    """返回 (拼接 AUC, 折内 AUC 均值)"""
    pred = np.zeros(len(y)); per = []
    for tr, te in A.trial_folds(grp):
        pred[te] = rule(f[te], np.median(f[tr][y[tr] == 0], 0), agg); per.append((y[te], pred[te]))
    return float(roc_auc_score(y, pred)), _foldmean(per)
def cv_lda_auc_both(f, y, grp):
    pred = np.zeros(len(y)); per = []
    for tr, te in A.trial_folds(grp):
        pred[te] = LDA().fit(f[tr], y[tr]).decision_function(f[te]); per.append((y[te], pred[te]))
    return float(roc_auc_score(y, pred)), _foldmean(per)
def cv_csp_auc_both(X, y, grp):
    pred = np.zeros(len(y)); per = []
    for tr, te in A.trial_folds(grp):
        w = C.csp_fit(X[tr], y[tr]); pred[te] = LDA().fit(C.csp_feat(X[tr], w), y[tr]).decision_function(C.csp_feat(X[te], w)); per.append((y[te], pred[te]))
    return float(roc_auc_score(y, pred)), _foldmean(per)

def cv_riem_auc(X, y, grp, kind='ts_lr', both=False):
    """黎曼几何基线(2026-10-06 加,强基线):OAS 协方差 → 切空间+逻辑回归(ts_lr)或最小距离到均值(mdm);同一试次分折。"""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros(len(y)); cov = Covariances(estimator='oas').transform(X.astype(float)); per = []
    for tr, te in A.trial_folds(grp):
        if kind == 'ts_lr':
            clf = make_pipeline(TangentSpace(metric='riemann'), StandardScaler(), LogisticRegression(max_iter=1000)).fit(cov[tr], y[tr]); pred[te] = clf.decision_function(cov[te])   # 标准化后 LR 收敛快(未标准化时单折 >100 s)
        else:
            clf = MDM(metric='riemann').fit(cov[tr], y[tr]); pr = clf.predict_proba(cov[te]); pred[te] = pr[:, list(clf.classes_).index(1)]
        per.append((y[te], pred[te]))
    return (float(roc_auc_score(y, pred)), _foldmean(per)) if both else float(roc_auc_score(y, pred))

def riem_fixed_auc(Xtr, ytr, Xte, yte, idx_tr, idx_te):
    """首次评估时点训练的切空间+LR 全导分类器,原样用于后续时点(通道按名称取交集,不重标、不重训)。返回 AUC。"""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    common = [n for n in idx_tr if n in idx_te]
    a = Xtr[:, [idx_tr[n] for n in common]]; b = Xte[:, [idx_te[n] for n in common]]
    ok_a = a.std(-1).mean(0) > 0; ok_b = b.std(-1).mean(0) > 0; ok = ok_a & ok_b; a, b = a[:, ok], b[:, ok]
    ca = Covariances(estimator='oas').transform(a.astype(float)); cb = Covariances(estimator='oas').transform(b.astype(float))
    clf = make_pipeline(TangentSpace(metric='riemann'), StandardScaler(), LogisticRegression(max_iter=1000)).fit(ca, ytr)
    return float(roc_auc_score(yte, clf.decision_function(cb)))

def cv_csp_auc(X, y, grp):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp):
        w = C.csp_fit(X[tr], y[tr]); pred[te] = LDA().fit(C.csp_feat(X[tr], w), y[tr]).decision_function(C.csp_feat(X[te], w))
    return float(roc_auc_score(y, pred))

def cut(sig, onsets, offs, fs=FS, others=(), t0=0.0):
    """sig: (ch, T);onsets: 事件时刻(s);offs: 窗起点列表(s);t0: 本 run 在整块内的时间偏移(s),使窗时刻跨 run 单调。
    返回 (窗, ch, WIN)、窗所属事件编号、窗起始绝对时刻(s),以及 others 中各信号在同一位置切出的窗。含 NaN 的窗整窗丢弃。"""
    W, G, T, O = [], [], [], [[] for _ in others]
    for i, t in enumerate(onsets):
        for o in offs:
            s0 = int(round((t + o) * fs))
            if s0 + WIN > sig.shape[1]: continue
            w = sig[:, s0:s0 + WIN]; ws = [x[:, s0:s0 + WIN] for x in others]
            if not np.isfinite(w).all() or any((not np.isfinite(x).all()) for x in ws): continue
            W.append(w); G.append(i); T.append(t0 + t + o)
            for k, x in enumerate(ws): O[k].append(x)
    return (np.stack(W), np.array(G), np.array(T)) + tuple(np.stack(o) if o else np.zeros((0, x.shape[0], WIN)) for o, x in zip(O, others))

def drop_flat(X, idx):
    """剔除整块内方差为零(断线)的通道,返回新 X 与新 idx;记录被剔通道名。"""
    sd = X.std(-1).mean(0); keep = np.where(sd > 0)[0]; dropped = [n for n, i in idx.items() if i not in set(keep)]
    newidx = {n: int(np.where(keep == i)[0][0]) for n, i in idx.items() if i in set(keep)}
    return X[:, keep, :], newidx, dropped, keep

def within_block(X, y, grp, idx, eog=None, side_contra=None, side_ipsi=None, bands=None, riem=True):
    """一个(人,时点)上的全部会话内指标。idx: 通道名→下标;X: 8–30 Hz 已滤波的 EEG 窗。"""
    r = {}
    names_contra = [n for n, i in idx.items() if i == side_contra]; names_ipsi = [n for n, i in idx.items() if i == side_ipsi]
    X, idx, dropped, keep = drop_flat(X, idx); r['dropped_channels'] = ','.join(dropped)
    if side_contra is not None: side_contra = idx.get(names_contra[0]); side_ipsi = idx.get(names_ipsi[0])
    lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1)
    c34 = lp[:, [idx['C3'], idx['C4']]]
    far_names = [n for n in FAR if n in idx] or [n for n in FAR_FALLBACK if n in idx]; far = lp[:, [idx[n] for n in far_names]].mean(1, keepdims=True); r['far_set'] = ','.join(far_names)
    r['rule_c34'], r['rule_c34_foldmean'] = oos_rule_auc_both(c34, y, grp); r['lda_c34'], r['lda_c34_foldmean'] = cv_lda_auc_both(c34, y, grp); r['csp_all'], r['csp_all_foldmean'] = cv_csp_auc_both(X, y, grp)
    if 'Cz' in idx: r['rule_cz'] = oos_rule_auc(lp[:, [idx['Cz']]], y, grp)
    m7 = [idx[n] for n in ['FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'Cz'] if n in idx]; r['rule_m7'] = oos_rule_auc(lp[:, m7], y, grp, 'mean'); r['lda_m7'] = cv_lda_auc(lp[:, m7], y, grp)
    r['rule_far'] = oos_rule_auc(far, y, grp); r['rule_c34_minus_far'] = oos_rule_auc(c34 - far, y, grp)
    r['lda_far'] = cv_lda_auc(far, y, grp); r['lda_far_c34'] = cv_lda_auc(np.c_[far, c34], y, grp); r['lda_rel_c34'] = cv_lda_auc(c34 - far, y, grp)
    r['rule_global'] = oos_rule_auc(lp.mean(1, keepdims=True), y, grp); r['lda_all_abs'] = cv_lda_auc(lp, y, grp); r['lda_all_rel'] = cv_lda_auc(lp - lp.mean(1, keepdims=True), y, grp)
    if side_contra is not None:
        r['rule_contra'] = oos_rule_auc(lp[:, [side_contra]], y, grp); r['rule_ipsi'] = oos_rule_auc(lp[:, [side_ipsi]], y, grp)
    if eog is not None:
        for band, E in eog.items():
            fe = np.stack([logp(E[:, c]) for c in range(E.shape[1])], 1); r[f'eog_rule_{band}'] = oos_rule_auc(fe, y, grp)
    if riem:                                                                      # 黎曼基线(全导 8–30 Hz)
        r['riem_ts_lr'], r['riem_ts_lr_foldmean'] = cv_riem_auc(X, y, grp, 'ts_lr', both=True); r['riem_mdm'] = cv_riem_auc(X, y, grp, 'mdm')
        cen = [idx[n] for n in CENTRAL9 if n in idx]; far9 = [idx[n] for n in far_names]                     # 同一黎曼管线、等通道数:中央 9 vs 远端 9
        if len(cen) >= 6 and len(far9) >= 6:
            k = min(len(cen), len(far9)); r['riem_central'] = cv_riem_auc(X[:, cen[:k]], y, grp, 'ts_lr'); r['riem_distal'] = cv_riem_auc(X[:, far9[:k]], y, grp, 'ts_lr'); r['riem_subset_nch'] = int(k)
    if bands is not None:                                                         # 频带拆分 + 枕区 alpha 对照(对照「视觉 alpha」)
        occ = [idx[n] for n in ['O1', 'O2', 'Oz'] if n in idx]
        lpb = {b: np.stack([logp(Xb[:, keep][:, c]) for c in range(len(keep))], 1) for b, Xb in bands.items()}
        for b in lpb:
            r[f'rule_c34_{b}'] = oos_rule_auc(lpb[b][:, [idx['C3'], idx['C4']]], y, grp); r[f'lda_c34_{b}'] = cv_lda_auc(lpb[b][:, [idx['C3'], idx['C4']]], y, grp)
            if occ: r[f'rule_occ_{b}'] = oos_rule_auc(lpb[b][:, occ].mean(1, keepdims=True), y, grp)
        if occ and 'mu' in lpb and 'beta' in lpb:
            oa = lpb['mu'][:, occ].mean(1, keepdims=True); cb = lpb['beta'][:, [idx['C3'], idx['C4']]]
            r['lda_occ_mu'] = cv_lda_auc(oa, y, grp); r['lda_occ_mu_c34_beta'] = cv_lda_auc(np.c_[oa, cb], y, grp)      # 中央 beta 在枕区 alpha 之上的增量
    return r

def deploy(pre, later):
    """pre/later: dict(f=(窗,2) 对数功率, y, grp(物理试次), t(窗起始时刻 s))。
    pre 定参考与阈值 → later:固定 / 时间前缀重标 / 整场离线 / pre 训练 LDA 固定。
    时间前缀:取 later 中按时间最早的 K 个参考试次的参考窗估参考;校准结束时刻 = 这些窗中最晚者的结束;
    **评价集 = 起始时刻晚于校准结束的全部窗(两类)**;固定、前缀、离线三方案都在同一评价集上算。"""
    def sens_spec(yy, pred): return float(pred[yy == 1].mean()), float((1 - pred[yy == 0]).mean())
    f0, y0 = pre['f'], pre['y']; ref0 = np.median(f0[y0 == 0], 0); sc0 = rule(f0, ref0)
    fpr, tpr, thr = roc_curve(y0, sc0); thr_y = float(thr[np.argmax(tpr - fpr)]); thr_q = float(np.percentile(sc0[y0 == 0], REST_Q))
    lda0 = LDA().fit(f0, y0); d0 = lda0.decision_function(f0); fpr, tpr, thr = roc_curve(y0, d0); thr_l = float(thr[np.argmax(tpr - fpr)])
    f, y, grp, t = later['f'], later['y'], later['grp'], later['t']; out = {'n_pre_windows': int(len(y0)), 'n_pre_ref_windows': int((y0 == 0).sum())}
    # 参考试次按首个参考窗时刻排序,取最早 K 个
    for tn, th in [('youden', thr_y), ('restq', thr_q)]:
        pfull = (rule(f, ref0) >= th).astype(int); out[f'fixed_{tn}_bacc'] = bacc(y, pfull); out[f'fixed_{tn}_sens'], out[f'fixed_{tn}_spec'] = sens_spec(y, pfull)                        # 全场(全部窗)
        out[f'offline_{tn}_bacc'] = bacc(y, (rule(f, np.median(f[y == 0], 0)) >= th).astype(int))
    ref_sorted = sorted(set(grp[y == 0]), key=lambda g: t[(grp == g) & (y == 0)].min())
    # 公共后缀评价集:以最大 k 的校准结束时刻为界,各 k 与固定/离线都在其上评价
    Kmax = max(KS); cal_max = np.isin(grp, ref_sorted[:Kmax]) & (y == 0); t_common = float(t[cal_max].max() + WIN / FS) if cal_max.any() and len(ref_sorted) >= Kmax else np.inf
    common = t >= t_common; out['common_suffix_start_s'] = t_common; out['n_common_mi'] = int((y[common] == 1).sum()); out['n_common_ref'] = int((y[common] == 0).sum())
    for Kk in KS:                                                                 # 2026-10-06:k=5/10/20 与慢性口径一致;k=10 为主
        ref_trials = ref_sorted[:Kk]
        calib = np.isin(grp, ref_trials) & (y == 0)
        calib_end = float(t[calib].max() + WIN / FS) if calib.any() else np.inf
        test = t >= calib_end
        if Kk == K: out['calib_ref_windows'] = int(calib.sum()); out['calib_end_s'] = calib_end; out['n_test_mi'] = int((y[test] == 1).sum()); out['n_test_ref'] = int((y[test] == 0).sum())
        out[f'pro{Kk}_calib_end_s'] = calib_end; out[f'pro{Kk}_n_test_mi'] = int((y[test] == 1).sum()); out[f'pro{Kk}_n_test_ref'] = int((y[test] == 0).sum())
        for tn, th in [('youden', thr_y), ('restq', thr_q)]:
            if calib.sum() >= 3 and (y[test] == 1).sum() >= 5 and (y[test] == 0).sum() >= 5 and len(ref_trials) == Kk:
                refk = np.median(f[calib], 0); yt = y[test]
                pk = (rule(f[test], refk) >= th).astype(int); out[f'pro{Kk}_{tn}_bacc'] = bacc(yt, pk); out[f'pro{Kk}_{tn}_sens'], out[f'pro{Kk}_{tn}_spec'] = sens_spec(yt, pk)
                pf = (rule(f[test], ref0) >= th).astype(int); out[f'pro{Kk}_{tn}_fixed_on_test'] = bacc(yt, pf); out[f'pro{Kk}_{tn}_fixed_sens'], out[f'pro{Kk}_{tn}_fixed_spec'] = sens_spec(yt, pf)
                out[f'pro{Kk}_{tn}_offline_on_test'] = bacc(yt, (rule(f[test], np.median(f[y == 0], 0)) >= th).astype(int))
                if (y[common] == 1).sum() >= 5 and (y[common] == 0).sum() >= 5:                      # 公共后缀
                    yc = y[common]; out[f'pro{Kk}_{tn}_bacc_common'] = bacc(yc, (rule(f[common], refk) >= th).astype(int))
                    if Kk == KS[0]: out[f'fixed_{tn}_bacc_common'] = bacc(yc, (rule(f[common], ref0) >= th).astype(int)); out[f'offline_{tn}_bacc_common'] = bacc(yc, (rule(f[common], np.median(f[y == 0], 0)) >= th).astype(int))
            else:
                out[f'pro{Kk}_{tn}_bacc'] = None
    out['rule_fixed_auc'] = float(roc_auc_score(y, rule(f, ref0))); out['lda_fixed_auc'] = float(roc_auc_score(y, lda0.decision_function(f)))
    out['lda_fixed_bacc'] = bacc(y, (lda0.decision_function(f) >= thr_l).astype(int))
    out['rule_within_auc'] = oos_rule_auc(f, y, grp); out['lda_within_auc'] = cv_lda_auc(f, y, grp)
    out['lda_drop'] = out['lda_within_auc'] - out['lda_fixed_auc']; out['rule_drop'] = out['rule_within_auc'] - out['rule_fixed_auc']
    return out

FAR = ['Fp1', 'Fp2', 'F7', 'F8', 'P7', 'P8', 'Oz', 'O1', 'O2']
CENTRAL9 = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']
FAR_FALLBACK = ['AF3', 'AF4', 'F5', 'F6', 'P5', 'P6', 'PO5', 'PO6', 'POz']       # 下肢 PEEG 只剩 40 导时的外周集合

# ====================================================================== 第六指
def sf_trials(on, desc):
    """按原文定义重建物理试次:每个静息事件(1)与其后紧随的 MI 事件(2 患手 / 3 第六指)构成一个 14 s 试次。返回 [(t_rest, t_mi, para)]"""
    ev = sorted(zip(on, desc), key=lambda x: x[0]); trials = []; cur = None
    for t, d in ev:
        if d == '1': cur = t
        elif d in ('2', '3') and cur is not None: trials.append((cur, t, int(d))); cur = None
    return trials

def sf():
    base = os.path.join(ROOT, 'data', 'stroke_tj_sixthfinger', 'sourcedata')
    part = {l.split('\t')[0]: l.strip().split('\t') for l in open(os.path.join(ROOT, 'data', 'stroke_tj_sixthfinger', 'participants.tsv')).readlines()[1:]}
    rest_offs = [1.0, 4.0] if os.environ.get('SF_REST', 'both') == 'both' else [4.0, 5.5]      # 静息窗起点:both=1.0–2.5 与 4.0–5.5;late=4.0–5.5 与 5.5–7.0(均在 7 s 绿色注视期内)
    mi_offs = [1.0, 2.5]
    res = {}
    for sub in sorted(x for x in os.listdir(base) if x.startswith('sub-') and (not os.environ.get('SF_SUBS') or x in os.environ['SF_SUBS'].split(','))):
        side = part[sub][3]; contra_name, ipsi_name = ('C4', 'C3') if side == 'L' else ('C3', 'C4')
        res[sub] = {'side': side}; blocks = {}
        for tp in ['pre', 'post', 'follow']:
            files = sorted(glob.glob(os.path.join(base, sub, tp, '*_original.set')))
            if not files: continue
            Xs, E1s, E2s, ys, gs, ts, paras, MUs, BEs, goff, t0 = [], [], [], [], [], [], [], [], [], 0, 0.0
            for f in files:
                raw = mne.io.read_raw_eeglab(f, preload=True, uint16_codec='latin1')
                if raw.info['sfreq'] != FS: raw.resample(FS)
                names = raw.ch_names; eeg_idx = [i for i, n in enumerate(names) if n not in ('ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL')]
                eog_idx = [names.index(n) for n in ('HEOR', 'HEOL', 'VEOU', 'VEOL') if n in names]
                data = raw.get_data() * 1e6; x = filt(data[eeg_idx], 8, 30); e1 = filt(data[eog_idx], 1, 8); e2 = filt(data[eog_idx], 8, 30)
                xmu = filt(data[eeg_idx], 8, 13); xbe = filt(data[eeg_idx], 13, 30)
                idx = {n: i for i, n in enumerate([names[i] for i in eeg_idx])}
                trials = sf_trials(raw.annotations.onset, raw.annotations.description)
                if not trials: continue
                t_rest = np.array([tr[0] for tr in trials]); t_mi = np.array([tr[1] for tr in trials]); para = np.array([tr[2] for tr in trials])
                Wr, Gr, Tr, E1r, E2r, Mr, Br = cut(x, t_rest, rest_offs, others=(e1, e2, xmu, xbe), t0=t0); Wm, Gm, Tm, E1m, E2m, Mm, Bm = cut(x, t_mi, mi_offs, others=(e1, e2, xmu, xbe), t0=t0)
                Xs += [Wr, Wm]; E1s += [E1r, E1m]; E2s += [E2r, E2m]; ys += [np.zeros(len(Wr), int), para[Gm]]; gs += [Gr + goff, Gm + goff]; ts += [Tr, Tm]; MUs += [Mr, Mm]; BEs += [Br, Bm]
                paras += [para[Gr], para[Gm]]                     # 每个窗所属物理试次的范式(静息窗也带其后 MI 的范式)
                goff += len(trials); t0 += raw.times[-1] + 1.0
            if not Xs: continue
            blocks[tp] = dict(X=np.concatenate(Xs), E1=np.concatenate(E1s), E2=np.concatenate(E2s), y_all=np.concatenate(ys), grp=np.concatenate(gs), t=np.concatenate(ts), para=np.concatenate(paras), MU=np.concatenate(MUs), BE=np.concatenate(BEs), idx=idx, n_trials=goff)
        for tp, B in blocks.items():
            r = {'n_trials': int(B['n_trials']), 'n_trials_AH': int(len(set(B['grp'][B['para'] == 2]))), 'n_trials_SF': int(len(set(B['grp'][B['para'] == 3])))}
            for lab, name in [(2, 'AH'), (3, 'SF')]:
                keep = (B['y_all'] == 0) | (B['y_all'] == lab)          # 参考 = 全部试次的静息窗;正类 = 该范式的 MI 窗;同一物理试次的静息窗与 MI 窗同组
                X, y, grp, t = B['X'][keep], (B['y_all'][keep] == lab).astype(int), B['grp'][keep], B['t'][keep]
                if (y == 1).sum() < 10 or (y == 0).sum() < 10: continue
                eog = {'1-8': B['E1'][keep], '8-30': B['E2'][keep]} if B['E1'].shape[1] > 0 else None
                r[name] = within_block(X, y, grp, B['idx'], eog=eog, side_contra=B['idx'][contra_name], side_ipsi=B['idx'][ipsi_name], bands={'mu': B['MU'][keep], 'beta': B['BE'][keep]})
                lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1)
                r[name]['_f'] = lp[:, [B['idx']['C3'], B['idx']['C4']]].tolist(); r[name]['_y'] = y.tolist(); r[name]['_grp'] = grp.tolist(); r[name]['_t'] = t.tolist()
            res[sub][tp] = r
            print(sub, tp, {k: (round(v['rule_c34'], 3), round(v['lda_c34'], 3), round(v['csp_all'], 3), round(v['rule_far'], 3), round(v['eog_rule_1-8'], 3)) for k, v in r.items() if k in ('AH', 'SF')}, flush=True)
        for name in ['AH', 'SF']:
            if 'pre' in res[sub] and name in res[sub]['pre']:
                P = res[sub]['pre'][name]; pre = {'f': np.array(P['_f']), 'y': np.array(P['_y']), 'grp': np.array(P['_grp']), 't': np.array(P['_t'])}
                for tp in ['post', 'follow']:
                    if tp in res[sub] and name in res[sub][tp]:
                        Q = res[sub][tp][name]; later = {'f': np.array(Q['_f']), 'y': np.array(Q['_y']), 'grp': np.array(Q['_grp']), 't': np.array(Q['_t'])}
                        res[sub][f'deploy_{name}_pre_to_{tp}'] = deploy(pre, later)
                        try:                                                             # 跨时点黎曼(全导,首次评估训练→后续时点原样用)
                            lab = 2 if name == 'AH' else 3; kp = (blocks['pre']['y_all'] == 0) | (blocks['pre']['y_all'] == lab); kl = (blocks[tp]['y_all'] == 0) | (blocks[tp]['y_all'] == lab)
                            res[sub][f'deploy_{name}_pre_to_{tp}']['riem_fixed_auc'] = riem_fixed_auc(blocks['pre']['X'][kp], (blocks['pre']['y_all'][kp] == lab).astype(int), blocks[tp]['X'][kl], (blocks[tp]['y_all'][kl] == lab).astype(int), blocks['pre']['idx'], blocks[tp]['idx'])
                            res[sub][f'deploy_{name}_pre_to_{tp}']['riem_within_auc'] = res[sub][tp][name]['riem_ts_lr']
                        except Exception as e: res[sub][f'deploy_{name}_pre_to_{tp}']['riem_error'] = str(e)
        for tp in ['pre', 'post', 'follow']:
            if tp in res[sub]:
                for name in ['AH', 'SF']:
                    if name in res[sub][tp]:
                        for k in ['_f', '_y', '_grp', '_t']: res[sub][tp][name].pop(k, None)
    json.dump(res, open(os.path.join(OUT, os.environ.get('OUT_NAME') or ('sf.json' if os.environ.get('SF_REST', 'both') == 'both' else 'sf_restlate.json')), 'w'), indent=1, ensure_ascii=False)
    summarize('sf', res)

# ====================================================================== 下肢
def ll_sessions(sub):
    """{(time, task): [files]}(从文件名解析;LL_SRC=raw 用 sourcedata/*_ori.set,否则 PEEG)"""
    pattern = os.path.join(ROOT, 'data', 'stroke_tj_lowerlimb', 'sourcedata', sub, 'ses-*', '*_ori.set') if os.environ.get('LL_SRC', 'peeg') == 'raw' else os.path.join(ROOT, 'data', 'stroke_tj_lowerlimb', 'PEEG', sub, 'ses-*', 'eeg', '*.set')
    out = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r'sub-\d+_([A-Za-z-]+)_run', os.path.basename(f)); tag = m.group(1) if m else 'unknown'
        key = (tag, 'MI') if tag in ('pre', 'post', 'follow') else ('pre', tag.replace('MI-', ''))
        out.setdefault(key, []).append(f)
    return out

def ll_trials(on, desc):
    """物理试次:MI 试次 = 注视(1)→文字(2)→任务(3);空闲试次 = 7→8→9。返回 [(t_fix, t_task, is_mi)]"""
    ev = sorted(zip(on, desc), key=lambda x: x[0]); trials = []; cur = None
    for t, d in ev:
        if d in ('1', '7'): cur = (t, d == '1')
        elif d in ('3', '9') and cur is not None and ((d == '3') == cur[1]): trials.append((cur[0], t, cur[1])); cur = None
    return trials

def ll():
    base = os.path.join(ROOT, 'data', 'stroke_tj_lowerlimb', 'sourcedata' if os.environ.get('LL_SRC', 'peeg') == 'raw' else 'PEEG')
    part = {l.split('\t')[0]: l.strip().split('\t') for l in open(os.path.join(ROOT, 'data', 'stroke_tj_lowerlimb', 'participants.tsv')).readlines()[1:]}
    subs = [x for x in sorted(os.listdir(base)) if x.startswith('sub-') and (not os.environ.get('LL_SUBS') or x in os.environ['LL_SUBS'].split(','))]
    res = {}
    for sub in subs:
        res[sub] = {'side': part.get(sub, [None] * 6)[5]}; blocks = {}
        for (tp, task), files in ll_sessions(sub).items():
            if task != 'MI': continue
            Xs, Fx, E1s, E2s, ys, gs, ts, runs, mixed, MUs, BEs, goff, t0 = [], [], [], [], [], [], [], [], [], [], [], 0, 0.0
            for ri, f in enumerate(files):
                raw = mne.io.read_raw_eeglab(f, preload=True, uint16_codec='latin1')
                if raw.info['sfreq'] != FS: raw.resample(FS)
                names_all = raw.ch_names; data_all = raw.get_data() * 1e6
                eeg_i = [i for i, n in enumerate(names_all) if n not in ('ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL')]; eog_i = [names_all.index(n) for n in ('HEOR', 'HEOL', 'VEOU', 'VEOL') if n in names_all]
                names = [names_all[i] for i in eeg_i]; x = filt(data_all[eeg_i], 8, 30); idx = {n: i for i, n in enumerate(names)}; xmu = filt(data_all[eeg_i], 8, 13); xbe = filt(data_all[eeg_i], 13, 30)
                has_eog = len(eog_i) > 0; e1 = filt(data_all[eog_i], 1, 8) if has_eog else np.zeros((0, x.shape[1])); e2 = filt(data_all[eog_i], 8, 30) if has_eog else np.zeros((0, x.shape[1]))
                trials = ll_trials(raw.annotations.onset, raw.annotations.description)
                if not trials: continue
                t_fix = np.array([tr[0] for tr in trials]); t_task = np.array([tr[1] for tr in trials]); is_mi = np.array([tr[2] for tr in trials])
                run_mixed = bool(is_mi.any() and (~is_mi).any())
                W, G, T, E1, E2, MU, BE = cut(x, t_task, [0.5, 2.0], others=(e1, e2, xmu, xbe), t0=t0)          # 任务期窗(MI 与空闲)
                Wf, Gf, Tf = cut(x, t_fix[is_mi], [0.5], t0=t0)                                  # MI 试次的注视期窗(与该 MI 试次同组)
                mi_trial_ids = np.where(is_mi)[0]
                Xs.append(W); E1s.append(E1); E2s.append(E2); ys.append(is_mi[G].astype(int)); gs.append(G + goff); ts.append(T); runs.append(np.full(len(W), ri)); mixed.append(np.full(len(W), run_mixed)); MUs.append(MU); BEs.append(BE)
                Fx.append((Wf, mi_trial_ids[Gf] + goff, Tf))
                goff += len(trials); t0 += raw.times[-1] + 1.0
            if not Xs: continue
            blocks[tp] = dict(X=np.concatenate(Xs), E1=np.concatenate(E1s), E2=np.concatenate(E2s), y=np.concatenate(ys), grp=np.concatenate(gs), t=np.concatenate(ts), run=np.concatenate(runs), mixed=np.concatenate(mixed), MU=np.concatenate(MUs), BE=np.concatenate(BEs), idx=idx,
                              fix=(np.concatenate([a[0] for a in Fx]), np.concatenate([a[1] for a in Fx]), np.concatenate([a[2] for a in Fx])) if Fx and sum(len(a[0]) for a in Fx) else None, n_runs=len(files), n_trials=goff)
        for tp, B in blocks.items():
            X, y, grp, t = B['X'], B['y'], B['grp'], B['t']
            if (y == 1).sum() < 10 or (y == 0).sum() < 10: continue
            r = {'n_windows': {'idle': int((y == 0).sum()), 'MI': int((y == 1).sum())}, 'n_runs': int(B['n_runs']), 'n_mixed_runs': int(len(set(B['run'][B['mixed']]))), 'n_trials': int(B['n_trials'])}
            eog = {'1-8': B['E1'], '8-30': B['E2']} if B['E1'].shape[1] > 0 else None
            r['task_vs_idle'] = within_block(X, y, grp, B['idx'], eog=eog, bands={'mu': B['MU'], 'beta': B['BE']})
            m = B['mixed']
            if m.any() and (y[m] == 1).sum() >= 10 and (y[m] == 0).sum() >= 10:                  # 敏感性:只用两类混合的 run
                r['task_vs_idle_mixed_runs'] = within_block(X[m], y[m], grp[m], B['idx'], eog=({'1-8': B['E1'][m], '8-30': B['E2'][m]} if eog else None), riem=False); r['task_vs_idle_mixed_runs']['n_windows'] = {'idle': int((y[m] == 0).sum()), 'MI': int((y[m] == 1).sum())}
            if B['fix'] is not None:                                                             # 对照设计:MI 任务窗 vs 同一 MI 试次的注视期窗(同组)
                Xf, gf, tf = B['fix']; sel = y == 1
                X2 = np.concatenate([X[sel], Xf]); y2 = np.r_[np.ones(sel.sum()), np.zeros(len(Xf))].astype(int); g2 = np.r_[grp[sel], gf]
                r['task_vs_fixation'] = within_block(X2, y2, g2, B['idx'])
            lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1); r['_f'] = lp[:, [B['idx']['Cz'], B['idx']['CPz' if 'CPz' in B['idx'] else 'FCz']]].tolist(); r['_y'] = y.tolist(); r['_grp'] = grp.tolist(); r['_t'] = t.tolist()
            res[sub][tp] = r
            print(sub, tp, 'task_vs_idle', {k: round(v, 3) for k, v in r['task_vs_idle'].items() if k in ('rule_c34', 'rule_cz', 'lda_c34', 'csp_all', 'rule_far', 'rule_c34_minus_far')}, '| mixed runs', r['n_mixed_runs'], '/', r['n_runs'], '| vs_fix rule_cz', round(r['task_vs_fixation']['rule_cz'], 3) if 'task_vs_fixation' in r else None, flush=True)
        if 'pre' in res[sub]:
            P = res[sub]['pre']; pre = {'f': np.array(P['_f']), 'y': np.array(P['_y']), 'grp': np.array(P['_grp']), 't': np.array(P['_t'])}
            for tp in ['post', 'follow']:
                if tp in res[sub]:
                    Q = res[sub][tp]; later = {'f': np.array(Q['_f']), 'y': np.array(Q['_y']), 'grp': np.array(Q['_grp']), 't': np.array(Q['_t'])}
                    res[sub][f'deploy_cz_pre_to_{tp}'] = deploy(pre, later)
                    try:
                        res[sub][f'deploy_cz_pre_to_{tp}']['riem_fixed_auc'] = riem_fixed_auc(blocks['pre']['X'], blocks['pre']['y'], blocks[tp]['X'], blocks[tp]['y'], blocks['pre']['idx'], blocks[tp]['idx'])
                        res[sub][f'deploy_cz_pre_to_{tp}']['riem_within_auc'] = res[sub][tp]['task_vs_idle']['riem_ts_lr']
                    except Exception as e: res[sub][f'deploy_cz_pre_to_{tp}']['riem_error'] = str(e)
        for tp in ['pre', 'post', 'follow']:
            if tp in res[sub]:
                for k in ['_f', '_y', '_grp', '_t']: res[sub][tp].pop(k, None)
    json.dump(res, open(os.path.join(OUT, os.environ.get('OUT_NAME') or ('ll_raw.json' if os.environ.get('LL_SRC', 'peeg') == 'raw' else 'll.json')), 'w'), indent=1, ensure_ascii=False)
    summarize('ll', res)

def summarize(tag, res):
    print(f'\n===== {tag} 汇总 =====')
    import collections
    agg = collections.defaultdict(list)
    for sub, d in res.items():
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            if isinstance(v3, (int, float)) and not k3.startswith('_'): agg[f'{k}/{k2}/{k3}'].append(v3)
                    elif isinstance(v2, (int, float)) and not k2.startswith('_'): agg[f'{k}/{k2}'].append(v2)
    for k in sorted(agg):
        v = np.array(agg[k]); print(f'  {k:48s} n={len(v):2d} 均值 {v.mean():.3f}' + (f'  >0.7:{int((v > 0.7).sum())}' if 'auc' in k or 'rule' in k or 'lda' in k or 'csp' in k else ''))

if __name__ == '__main__':
    for cmd in sys.argv[1:]: globals()[cmd]()
