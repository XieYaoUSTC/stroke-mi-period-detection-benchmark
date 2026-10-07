#!/usr/bin/env python3
"""论文二 · 投稿前强化分析(2026-10-06,用户批准项 1–5):
  acute      急性 50 人(主口径=提示期参考 REST_MODE=cue):黎曼基线(切空间+LR / MDM)、频带拆分(C3/C4 mu / beta)、枕区 alpha 对照、
             按想象手的偏侧化(对侧单导 vs 同侧单导的阶段检测;左右手判别 AUC)
  chronic    慢性 14 人(CAR,逐会话均值):黎曼基线、频带拆分(11 导无枕区)、按想象腕的偏侧化
  splithalf  锁定方案的对半验证:在一半患者上按均值 AUC 选参考窗候选,在另一半上报结果(急性 break/cue/late;患手 early+late/late;慢性 3 个候选由本脚本算)
天津两数据集的黎曼/频带/k=5/20 在 tianjin_analysis.py 内(同一分折),此处不重复。
输出 results/paper2_strengthen/{acute,chronic,splithalf}.json
"""
import glob
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('REST_MODE', 'cue')
import stroke_mi_vs_rest as A
import stroke_chronic_mi_vs_rest as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'paper2_strengthen'); os.makedirs(OUT, exist_ok=True)
ACUTE_OCC = [26, 27, 28]            # Oz O1 O2(29 导顺序见 rule_montage_variants.py)
ACUTE_CENTRAL8 = [7, 8, 9, 12, 13, 14, 17, 18]   # FCz FC3 FC4 Cz C3 C4 CP3 CP4
ACUTE_DISTAL8 = [0, 1, 5, 6, 24, 25, 27, 28]     # Fp1 Fp2 F7 F8 T5 T6 O1 O2


def logp(sig): return np.log(sig.var(-1) + 1e-30)
def rule(f, ref): return -(f - ref).min(1)


def oos_rule_auc(f, y, grp):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp): pred[te] = rule(f[te], np.median(f[tr][y[tr] == 0], 0))
    return float(roc_auc_score(y, pred))


def cv_lda_auc(f, y, grp):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp): pred[te] = LDA().fit(f[tr], y[tr]).decision_function(f[te])
    return float(roc_auc_score(y, pred))


def _foldmean(per):
    v = [roc_auc_score(yt, st) for yt, st in per if len(set(yt)) == 2]; return float(np.mean(v)) if v else float('nan')


def cv_riem_auc(X, y, grp, kind='ts_lr', both=False):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM
    pred = np.zeros(len(y)); cov = Covariances(estimator='oas').transform(X.astype(float)); per = []
    for tr, te in A.trial_folds(grp):
        if kind == 'ts_lr':
            clf = make_pipeline(TangentSpace(metric='riemann'), StandardScaler(), LogisticRegression(max_iter=1000)).fit(cov[tr], y[tr]); pred[te] = clf.decision_function(cov[te])   # 标准化后 LR 收敛快(未标准化时单折 >100 s)
        else:
            clf = MDM(metric='riemann').fit(cov[tr], y[tr]); pred[te] = clf.predict_proba(cov[te])[:, list(clf.classes_).index(1)]
        per.append((y[te], pred[te]))
    return (float(roc_auc_score(y, pred)), _foldmean(per)) if both else float(roc_auc_score(y, pred))


def oos_rule_auc_both(f, y, grp):
    pred = np.zeros(len(y)); per = []
    for tr, te in A.trial_folds(grp): pred[te] = rule(f[te], np.median(f[tr][y[tr] == 0], 0)); per.append((y[te], pred[te]))
    return float(roc_auc_score(y, pred)), _foldmean(per)


def lateral(lp, y, grp, hand, c3, c4):
    """按想象手的偏侧化。hand: 每窗所属试次的手(1=左 2=右;静息窗也带其试次的手)。
    contra/ipsi 阶段检测:左手试次用 C4/C3,右手试次用 C3/C4(单导规则,样本外参考);
    lr_auc:左右手判别——想象窗上 (C3 下降 − C4 下降) 的样本外分数,右手想象 C3 下降更多 → AUC>0.5 为对侧占优;lr_lda:同特征 LDA。"""
    r = {}
    contra = np.where(hand == 1, lp[:, c4], lp[:, c3]); ipsi = np.where(hand == 1, lp[:, c3], lp[:, c4])
    r['rule_contra_hand'] = oos_rule_auc(contra[:, None], y, grp); r['rule_ipsi_hand'] = oos_rule_auc(ipsi[:, None], y, grp)
    mi = y == 1; f = lp[:, [c3, c4]]
    ymi = (hand[mi] == 2).astype(int); gmi = grp[mi]; fmi = f[mi]
    if len(set(ymi)) < 2 or min((ymi == 0).sum(), (ymi == 1).sum()) < 5: return r
    pred = -(fmi[:, 0] - fmi[:, 1])                                           # 固定分数:C3 功率低于 C4 越多 → 越像右手;不含任何拟合,参考项对 AUC 无影响
    r['lr_rule_auc'] = float(roc_auc_score(ymi, pred))
    pred = np.zeros(mi.sum())
    for tr, te in A.trial_folds(gmi): pred[te] = LDA().fit(fmi[tr], ymi[tr]).decision_function(fmi[te])
    r['lr_lda_auc'] = float(roc_auc_score(ymi, pred))
    return r


# ------------------------------------------------------------------ 急性
def acute():
    import csv
    rows = []
    for fpath in sorted(glob.glob(os.path.join(A.DATA, 'sub-*', '*.mat'))):
        sub = os.path.basename(os.path.dirname(fpath)); m = sio.loadmat(fpath)['eeg'][0, 0]
        raw = m['rawdata'].astype(float)[:, A.EEG_IDX, :]; lab = m['label'].ravel().astype(int)      # 1/2 = 左/右手(数据集说明)
        r = {'sub': sub}
        X, y, grp = A.windows(A.filt(raw, 8, 30)); hand = lab[grp]
        lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1)
        r['rule_c34'], r['rule_c34_foldmean'] = oos_rule_auc_both(lp[:, [A.C3, A.C4]], y, grp)
        r['riem_ts_lr'], r['riem_ts_lr_foldmean'] = cv_riem_auc(X, y, grp, 'ts_lr', both=True); r['riem_mdm'] = cv_riem_auc(X, y, grp, 'mdm')
        r['riem_central'] = cv_riem_auc(X[:, ACUTE_CENTRAL8], y, grp, 'ts_lr'); r['riem_distal'] = cv_riem_auc(X[:, ACUTE_DISTAL8], y, grp, 'ts_lr')
        r.update(lateral(lp, y, grp, hand, A.C3, A.C4))
        lpb = {}
        for band, (lo, hi) in {'mu': (8, 13), 'beta': (13, 30)}.items():
            Xb, _, _ = A.windows(A.filt(raw, lo, hi)); lpb[band] = np.stack([logp(Xb[:, c]) for c in range(Xb.shape[1])], 1)
            r[f'rule_c34_{band}'] = oos_rule_auc(lpb[band][:, [A.C3, A.C4]], y, grp); r[f'lda_c34_{band}'] = cv_lda_auc(lpb[band][:, [A.C3, A.C4]], y, grp)
            r[f'rule_occ_{band}'] = oos_rule_auc(lpb[band][:, ACUTE_OCC].mean(1, keepdims=True), y, grp)
        oa = lpb['mu'][:, ACUTE_OCC].mean(1, keepdims=True); cb = lpb['beta'][:, [A.C3, A.C4]]
        r['lda_occ_mu'] = cv_lda_auc(oa, y, grp); r['lda_occ_mu_c34_beta'] = cv_lda_auc(np.c_[oa, cb], y, grp)
        rows.append(r); print(sub, ' '.join(f'{k}={v:.3f}' for k, v in r.items() if isinstance(v, float)), flush=True)
    json.dump({'rest_mode': os.environ['REST_MODE'], 'rows': rows, 'summary': summarize(rows)}, open(os.path.join(OUT, f"acute_{os.environ['REST_MODE']}.json"), 'w'), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ 急性:眼电回归后的黎曼基线(强分类器的增益是否来自眼电)
def acute_eogreg():
    """逐试次最小二乘回归去眼电(0.5–40 Hz 宽带后,同 paper2_followups.controls2),再 8–30 Hz;与「只多一道 0.5–40 Hz 滤波、不回归」的对照配平。
    报告 rule_c34 与 riem_ts_lr 在三种处理下的 AUC:原始 / 仅滤波 / 滤波+回归。REST_MODE 由环境变量决定。"""
    rows = []; PAD = float(os.environ.get('EOG_PAD', '1.0'))
    for fpath in sorted(glob.glob(os.path.join(A.DATA, 'sub-*', '*.mat'))):
        sub = os.path.basename(os.path.dirname(fpath)); m = sio.loadmat(fpath)['eeg'][0, 0]; raw_all = m['rawdata'].astype(float)
        eog_raw = raw_all[:, [30, 31], :]; eeg_raw = raw_all[:, A.EEG_IDX, :]; r = {'sub': sub}
        e_b = filt_pad(eog_raw, 0.5, 40, PAD); x_b = filt_pad(eeg_raw, 0.5, 40, PAD); clean = np.empty_like(x_b)
        for t in range(x_b.shape[0]):
            E = np.c_[e_b[t].T, np.ones(e_b.shape[-1])]; beta = np.linalg.lstsq(E, x_b[t].T, rcond=None)[0]; clean[t] = (x_b[t].T - E @ beta).T
        for tag, src in [('raw', eeg_raw), ('filt', x_b), ('eogreg', clean)]:
            X, y, grp = A.windows(A.filt(src, 8, 30)); lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1)
            r[f'rule_c34_{tag}'] = oos_rule_auc(lp[:, [A.C3, A.C4]], y, grp); r[f'riem_ts_lr_{tag}'] = cv_riem_auc(X, y, grp, 'ts_lr')
        rows.append(r); print(sub, ' '.join(f'{k}={v:.3f}' for k, v in r.items() if isinstance(v, float)), flush=True)
    json.dump({'rest_mode': os.environ['REST_MODE'], 'rows': rows, 'summary': summarize(rows)}, open(os.path.join(OUT, f"acute_eogreg_{os.environ['REST_MODE']}" + ('' if PAD == 1.0 else f'_pad{PAD:g}') + '.json'), 'w'), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ 急性:滤波瞬态假说的针对性检验(六审第 80 条)
def filt_pad(x, lo, hi, pad_s, zero_phase=False):
    """与 A.filt 相同的陷波+带通,但镜像补段长度可调;zero_phase=True 用 filtfilt(无因果瞬态)。"""
    from scipy.signal import butter, iirnotch, lfilter, filtfilt
    bn, an = iirnotch(50, 30, A.FS); b, a = butter(2, [lo / (A.FS / 2), hi / (A.FS / 2)], btype='band'); pad = int(pad_s * A.FS)
    xp = np.concatenate([x[:, :, pad:0:-1], x], axis=-1) if pad > 0 else x
    f = (lambda b_, a_, z: filtfilt(b_, a_, z, axis=-1)) if zero_phase else (lambda b_, a_, z: lfilter(b_, a_, z, axis=-1))
    y = f(b, a, f(bn, an, xp)); return y[:, :, pad:] if pad > 0 else y


def acute_transient():
    """提示期参考下,「多一道 0.5–40 Hz 滤波」对黎曼 AUC 的影响是否来自试次起点的因果滤波瞬态:
    变体 = 补段 1 s(原对照)/ 3 s / 零相位 filtfilt / 以及把窗起点推后(提示期 0.5–2.0 s)。都在同一分折上算规则与黎曼。REST_MODE=cue。"""
    rows = []
    for fpath in sorted(glob.glob(os.path.join(A.DATA, 'sub-*', '*.mat'))):
        sub = os.path.basename(os.path.dirname(fpath)); m = sio.loadmat(fpath)['eeg'][0, 0]; eeg_raw = m['rawdata'].astype(float)[:, A.EEG_IDX, :]; r = {'sub': sub}
        variants = {'raw': eeg_raw, 'pre_pad1': filt_pad(eeg_raw, 0.5, 40, 1.0), 'pre_pad3': filt_pad(eeg_raw, 0.5, 40, 3.0), 'pre_zerophase': filt_pad(eeg_raw, 0.5, 40, 1.0, zero_phase=True)}
        for tag, src in variants.items():
            X, y, grp = A.windows(A.filt(src, 8, 30)); lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1)
            r[f'rule_c34_{tag}'] = oos_rule_auc(lp[:, [A.C3, A.C4]], y, grp); r[f'riem_ts_lr_{tag}'] = cv_riem_auc(X, y, grp, 'ts_lr')
        rows.append(r); print(sub, ' '.join(f'{k}={v:.3f}' for k, v in r.items() if isinstance(v, float)), flush=True)
    json.dump({'rest_mode': os.environ['REST_MODE'], 'rows': rows, 'summary': summarize(rows)}, open(os.path.join(OUT, f"acute_transient_{os.environ['REST_MODE']}.json"), 'w'), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ 慢性
def load_session_lab(sdir):
    trials, labs, seen = [], [], set()
    for f in sorted(glob.glob(os.path.join(sdir, 'eeg', '*run-*_eeg.mat'))):
        d = sio.loadmat(f, squeeze_me=True)
        x = np.transpose(d['EEGdata'], (2, 0, 1)).astype(float); lb = np.atleast_1d(d['EEGdatalabel']).astype(int)
        for tr, l in zip(x, lb):
            h = hashlib.md5(np.round(tr[:, :512] * 1e6, 3).tobytes()).hexdigest()
            if h in seen: continue
            seen.add(h); trials.append(tr); labs.append(l)
    if not trials: return None, None
    x = np.stack(trials); lb = np.array(labs); ok = ~np.isnan(x).any(axis=(1, 2)); x, lb = x[ok], lb[ok]
    return x - x.mean(1, keepdims=True), lb


def chronic():
    rows = []
    for sdir_sub in sorted(glob.glob(os.path.join(C.DATA, 'sub-*'))):
        sub = os.path.basename(sdir_sub); per = []
        for sdir in sorted(glob.glob(os.path.join(sdir_sub, 'ses-*'))):
            x, lb = load_session_lab(sdir)
            if x is None or x.shape[0] < 10: continue
            X, y, grp = C.windows(C.filt(x)); hand = lb[grp]
            lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1); r = {'ses': os.path.basename(sdir)}
            r['rule_c34'], r['rule_c34_foldmean'] = oos_rule_auc_both(lp[:, [C.C3, C.C4]], y, grp)
            r['riem_ts_lr'], r['riem_ts_lr_foldmean'] = cv_riem_auc(X, y, grp, 'ts_lr', both=True); r['riem_mdm'] = cv_riem_auc(X, y, grp, 'mdm')
            r.update(lateral(lp, y, grp, hand, C.C3, C.C4))
            for band, (lo, hi) in {'mu': (8, 13), 'beta': (13, 30)}.items():
                Xb, _, _ = C.windows(C.filt(x, lo, hi)); lpb = np.stack([logp(Xb[:, c]) for c in range(Xb.shape[1])], 1)
                r[f'rule_c34_{band}'] = oos_rule_auc(lpb[:, [C.C3, C.C4]], y, grp); r[f'lda_c34_{band}'] = cv_lda_auc(lpb[:, [C.C3, C.C4]], y, grp)
            per.append(r)
        if not per: continue
        keys = [k for k in per[0] if k != 'ses']
        r = {'sub': sub, 'n_sessions': len(per)}
        for k in keys:
            v = [p[k] for p in per if k in p]; r[k] = float(np.mean(v)) if v else None
        rows.append(r); print(sub, len(per), ' '.join(f'{k}={v:.3f}' for k, v in r.items() if isinstance(v, float)), flush=True)
    json.dump({'rows': rows, 'summary': summarize(rows)}, open(os.path.join(OUT, 'chronic.json'), 'w'), indent=1, ensure_ascii=False)


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]; s = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        s[k] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else None, 'n': int(len(v)), 'n_gt_0.7': int((v > 0.7).sum())}
    return s


# ------------------------------------------------------------------ 对半验证
def chronic_window_candidates():
    """慢性:参考窗候选(准备期 9 s 内三个起点 0.5 / 3.0 / 6.0 s),想象窗固定;每人逐会话均值的样本外规则 AUC。"""
    import importlib
    out = {}
    for name, rest0 in [('prep0.5', 0.5), ('prep3.0', 3.0), ('prep6.0', 6.0)]:
        C.REST0 = int(rest0 * C.FS); vals = {}
        for sdir_sub in sorted(glob.glob(os.path.join(C.DATA, 'sub-*'))):
            sub = os.path.basename(sdir_sub); per = []
            for sdir in sorted(glob.glob(os.path.join(sdir_sub, 'ses-*'))):
                x, _ = load_session_lab(sdir)
                if x is None or x.shape[0] < 10: continue
                X, y, grp = C.windows(C.filt(x)); per.append(oos_rule_auc(C.feats(X), y, grp))
            if per: vals[sub] = float(np.mean(per))
        out[name] = vals; print(name, round(np.mean(list(vals.values())), 3), flush=True)
    C.REST0 = int(0.5 * C.FS)
    return out


def splithalf():
    """在随机一半患者上选「均值 AUC 最高」的候选,在另一半上报该候选的 AUC;与主口径在同一半上的 AUC 比较;200 次随机划分。"""
    rng = np.random.default_rng(0); res = {}
    cand = {}
    # 急性:三种参考窗(已有逐人结果)
    acute = {}
    for mode in ['break', 'cue', 'late']:
        S = json.load(open(os.path.join(ROOT, 'results', 'stroke_mi_vs_rest', f'summary_{mode}.json')))
        acute[mode] = {r['sub']: r['rule_oos_auc_8-30'] for r in S['rows']}
    cand['acute'] = (acute, 'cue')
    # 患手:早+晚两窗 vs 晚两窗(已有逐人结果,pre 时点 AH)
    sf = {}
    for name, fn in [('early+late', 'sf.json'), ('late', 'sf_restlate.json')]:
        S = json.load(open(os.path.join(ROOT, 'results', 'tianjin', fn)))
        sf[name] = {s: S[s]['pre']['AH']['rule_c34'] for s in S if 'pre' in S[s] and 'AH' in S[s]['pre']}
    cand['paretic_hand'] = (sf, 'late')
    # 慢性:三个候选现算
    cand['chronic'] = (chronic_window_candidates(), 'prep0.5')
    for ds, (cands, main) in cand.items():
        subs = sorted(set.intersection(*[set(v) for v in cands.values()])); n = len(subs); M = {k: np.array([v[s] for s in subs]) for k, v in cands.items()}
        sel_B, main_B, same, picks = [], [], 0, {k: 0 for k in cands}
        for _ in range(200):
            perm = rng.permutation(n); a, b = perm[: n // 2], perm[n // 2:]
            best = max(cands, key=lambda k: M[k][a].mean()); picks[best] += 1
            sel_B.append(M[best][b].mean()); main_B.append(M[main][b].mean()); same += best == main
        res[ds] = {'n': n, 'candidates': {k: float(M[k].mean()) for k in cands}, 'main': main, 'pick_rate': {k: picks[k] / 200 for k in cands},
                   'selected_on_A_eval_on_B_mean': float(np.mean(sel_B)), 'main_eval_on_B_mean': float(np.mean(main_B)), 'optimism_mean': float(np.mean(np.array(sel_B) - np.array(main_B))),
                   'all_patients_main': float(M[main].mean()), 'all_patients_best_in_sample': float(max(M[k].mean() for k in cands))}
        print(ds, json.dumps(res[ds], ensure_ascii=False), flush=True)
    json.dump(res, open(os.path.join(OUT, 'splithalf.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    for cmd in sys.argv[1:]:
        globals()[cmd]()
