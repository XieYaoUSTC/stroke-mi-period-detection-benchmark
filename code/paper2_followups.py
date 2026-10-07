#!/usr/bin/env python3
"""论文二 · 补充实验(2026-10-04)。

子命令(可多个):
  prospective  慢性:前瞻重标(只用会话最前 k=5/10/20 个试次的静息窗估参考,在其余试次上评价)+ 无标签分位数阈值(首会话静息分数的 80 百分位)
               + 首会话 LDA 对照 + 整场离线重标对照(同一评价集)+ LDA 与规则的跨会话掉幅交互
  rawref       慢性:不做 11 导平均参考(原始双乳突参考)的会话内三法与部署口径
  predict      慢性:前 1/2/3 次会话的规则 AUC 能否预测后续会话均值(Spearman、绝对误差、>0.7 判定一致)
  paretic      急性:患手试次(20)单独 vs 健手试次(20):规则(oos)/LDA/CSP,主口径提示期窗
  matched      急性:与多导规则同输入的训练对照(7 导对数功率 LDA;Laplacian 2 导 LDA)
  controls     急性:EOG 两导规则(1–8 Hz 与 8–30 Hz)、非运动区(Fp1/Fp2、O1/O2)规则、幅度剔窗(C3/C4 峰峰 >100 µV)后的规则
输出:results/paper2_followups/<子命令>.json
用法:.venv/bin/python code/paper2_followups.py prospective rawref predict paretic matched controls
"""
import os, sys, glob, json, csv, hashlib
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('REST_MODE', 'cue')
import stroke_mi_vs_rest as A
import stroke_chronic_mi_vs_rest as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'paper2_followups'); os.makedirs(OUT, exist_ok=True)
REST_Q = 80            # 无标签阈值:首会话静息分数的百分位(事先定,不调)
KS = [5, 10, 20]       # 前瞻重标用的试次数

def logp(sig): return np.log(sig.var(-1))
def two_ch_feats(X, c3, c4): return np.c_[logp(X[:, c3]), logp(X[:, c4])]
def rule(f, ref): return -(f - ref).min(1)
def bacc(y, pred): return float((pred[y == 1].mean() + (1 - pred[y == 0]).mean()) / 2)
def sens_spec(y, pred): return float(pred[y == 1].mean()), float((1 - pred[y == 0]).mean())

def oos_rule_auc(f, y, grp):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp):
        pred[te] = rule(f[te], np.median(f[tr][y[tr] == 0], 0))
    return float(roc_auc_score(y, pred))

def cv_lda_auc(f, y, grp):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp):
        pred[te] = LDA().fit(f[tr], y[tr]).decision_function(f[te])
    return float(roc_auc_score(y, pred))

def cv_csp_auc(X, y, grp):
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp):
        w = A.csp_fit(X[tr], y[tr]); pred[te] = LDA().fit(A.csp_feat(X[tr], w), y[tr]).decision_function(A.csp_feat(X[te], w))
    return float(roc_auc_score(y, pred))

def cv_csp_auc_C(X, y, grp):
    """与慢性主脚本相同的 CSP 实现(带轻微正则),用于原始参考对照,保证参考方式是唯一变量。"""
    pred = np.zeros(len(y))
    for tr, te in A.trial_folds(grp):
        w = C.csp_fit(X[tr], y[tr]); pred[te] = LDA().fit(C.csp_feat(X[tr], w), y[tr]).decision_function(C.csp_feat(X[te], w))
    return float(roc_auc_score(y, pred))

def summarize(rows, keys):
    return {k: {'mean': float(np.mean([r[k] for r in rows])), 'sd': float(np.std([r[k] for r in rows], ddof=1)),
                'median': float(np.median([r[k] for r in rows])), 'n': len(rows),
                'n_gt_0.7': int(sum(r[k] > 0.7 for r in rows))} for k in keys if all(k in r and r[k] is not None for r in rows)}

# ------------------------------------------------------------------ 慢性:会话加载(可选 CAR)
def load_session_ref(sdir, car=True):
    trials, seen = [], set()
    for f in sorted(glob.glob(os.path.join(sdir, 'eeg', '*run-*_eeg.mat'))):
        d = sio.loadmat(f, squeeze_me=True)
        x = np.transpose(d['EEGdata'], (2, 0, 1)).astype(float)
        for tr in x:
            h = hashlib.md5(np.round(tr[:, :512] * 1e6, 3).tobytes()).hexdigest()
            if h in seen: continue
            seen.add(h); trials.append(tr)
    if not trials: return None
    x = np.stack(trials); x = x[~np.isnan(x).any(axis=(1, 2))]
    if car: x = x - x.mean(1, keepdims=True)
    return x

def chronic_sessions(sub, car=True):
    out = []
    for sdir in sorted(glob.glob(os.path.join(C.DATA, sub, 'ses-*'))):
        x = load_session_ref(sdir, car)
        if x is None or x.shape[0] < 10: continue
        X, y, grp = C.windows(C.filt(x))
        out.append({'ses': os.path.basename(sdir), 'X': X, 'y': y, 'grp': grp, 'f': C.feats(X), 'n': x.shape[0]})
    return out

def subjects_chronic():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(C.DATA, 'sub-*')))

# ------------------------------------------------------------------ prospective
def prospective():
    per = {}
    for sub in subjects_chronic():
        S = chronic_sessions(sub, car=True)
        if len(S) < 2: continue
        s0 = S[0]; ref0 = np.median(s0['f'][s0['y'] == 0], 0); sc0 = rule(s0['f'], ref0)
        fpr, tpr, thr = roc_curve(s0['y'], sc0); thr_youden = float(thr[np.argmax(tpr - fpr)])          # 有标签
        thr_rest = float(np.percentile(sc0[s0['y'] == 0], REST_Q))                                       # 无标签:只用静息窗
        lda0 = LDA().fit(s0['f'], s0['y'])
        # 首会话 LDA 的工作点:同样用首会话 Youden(有标签)
        d0 = lda0.decision_function(s0['f']); fpr, tpr, thr = roc_curve(s0['y'], d0); thr_lda = float(thr[np.argmax(tpr - fpr)])
        rows = []
        for s in S[1:]:
            f, y, grp = s['f'], s['y'], s['grp']
            r = {'ses': s['ses'], 'n_trials': s['n']}
            for tname, th in [('youden', thr_youden), ('restq', thr_rest)]:
                # 固定参考(首会话)
                pred = (rule(f, ref0) >= th).astype(int); r[f'fixed_{tname}_bacc'] = bacc(y, pred); r[f'fixed_{tname}_sens'], r[f'fixed_{tname}_spec'] = sens_spec(y, pred)
                # 整场离线重标(对照;与此前主稿一致)
                pred = (rule(f, np.median(f[y == 0], 0)) >= th).astype(int); r[f'offline_{tname}_bacc'] = bacc(y, pred)
                # 前瞻:前 k 个试次的静息窗估参考,评价其余试次
                for k in KS:
                    calib = grp < k; test = ~calib
                    if test.sum() < 10 or (y[calib] == 0).sum() < 3: r[f'pro{k}_{tname}_bacc'] = None; continue
                    refk = np.median(f[calib & (y == 0)], 0)
                    pred = (rule(f[test], refk) >= th).astype(int); yt = y[test]
                    r[f'pro{k}_{tname}_bacc'] = bacc(yt, pred); r[f'pro{k}_{tname}_sens'], r[f'pro{k}_{tname}_spec'] = sens_spec(yt, pred)
                    # 同一评价集上的固定参考与整场重标,保证可比
                    r[f'pro{k}_{tname}_fixed_on_test'] = bacc(yt, (rule(f[test], ref0) >= th).astype(int))
                    r[f'pro{k}_{tname}_offline_on_test'] = bacc(yt, (rule(f[test], np.median(f[y == 0], 0)) >= th).astype(int))
            # 首会话 LDA 固定
            pred = (lda0.decision_function(f) >= thr_lda).astype(int); r['lda_fixed_bacc'] = bacc(y, pred); r['lda_fixed_auc'] = float(roc_auc_score(y, lda0.decision_function(f)))
            r['rule_fixed_auc'] = float(roc_auc_score(y, rule(f, ref0)))
            r['rule_within_auc'] = oos_rule_auc(f, y, grp); r['lda_within_auc'] = cv_lda_auc(f, y, grp)
            rows.append(r)
        keys = [k for k in rows[0] if k not in ('ses', 'n_trials')]
        agg = {}
        for k in keys:
            v = [r[k] for r in rows if r.get(k) is not None]
            agg[k] = float(np.mean(v)) if v else None
        agg['n_sessions_later'] = len(rows)
        # 跨会话掉幅:会话内 − 首会话固定(越大掉得越多)
        agg['lda_drop'] = agg['lda_within_auc'] - agg['lda_fixed_auc']; agg['rule_drop'] = agg['rule_within_auc'] - agg['rule_fixed_auc']
        per[sub] = {'summary': agg, 'sessions': rows}
        print(sub, f"固定(youden) {agg['fixed_youden_bacc']:.3f} 整场 {agg['offline_youden_bacc']:.3f} 前瞻5/10/20 {agg.get('pro5_youden_bacc') or float('nan'):.3f}/{agg.get('pro10_youden_bacc') or float('nan'):.3f}/{agg.get('pro20_youden_bacc') or float('nan'):.3f} | 无标签阈值 固定 {agg['fixed_restq_bacc']:.3f} 前瞻10 {agg.get('pro10_restq_bacc') or float('nan'):.3f} | LDA固定 {agg['lda_fixed_bacc']:.3f} 掉幅 lda {agg['lda_drop']:+.3f} rule {agg['rule_drop']:+.3f}", flush=True)
    keys = [k for k in next(iter(per.values()))['summary'] if k != 'n_sessions_later']
    summ = {k: {'mean': float(np.mean([per[s]['summary'][k] for s in per if per[s]['summary'][k] is not None])),
                'median': float(np.median([per[s]['summary'][k] for s in per if per[s]['summary'][k] is not None])),
                'n': int(sum(per[s]['summary'][k] is not None for s in per))} for k in keys}
    json.dump({'per_subject': per, 'summary': summ, 'REST_Q': REST_Q, 'KS': KS}, open(os.path.join(OUT, 'prospective.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 慢性 14 人 · 第 2 次会话起 ===')
    for k in ['fixed_youden_bacc', 'offline_youden_bacc', 'pro5_youden_bacc', 'pro10_youden_bacc', 'pro20_youden_bacc', 'pro10_youden_fixed_on_test', 'pro10_youden_offline_on_test',
              'fixed_restq_bacc', 'offline_restq_bacc', 'pro5_restq_bacc', 'pro10_restq_bacc', 'pro20_restq_bacc', 'pro10_restq_sens', 'pro10_restq_spec',
              'lda_fixed_bacc', 'lda_fixed_auc', 'rule_fixed_auc', 'lda_within_auc', 'rule_within_auc', 'lda_drop', 'rule_drop']:
        print(f"  {k:30s} {summ[k]['mean']:.3f} (中位 {summ[k]['median']:.3f}, n={summ[k]['n']})")

# ------------------------------------------------------------------ rawref
def rawref():
    per = {}
    for sub in subjects_chronic():
        S = chronic_sessions(sub, car=False)
        if len(S) < 2: continue
        s0 = S[0]; ref0 = np.median(s0['f'][s0['y'] == 0], 0); sc0 = rule(s0['f'], ref0)
        fpr, tpr, thr = roc_curve(s0['y'], sc0); thr0 = float(thr[np.argmax(tpr - fpr)])
        lda0 = LDA().fit(s0['f'], s0['y'])
        within = {'rule': [], 'lda': [], 'csp': []}; later = {'fixed_bacc': [], 'recal_bacc': [], 'fixed_auc': [], 'lda_fixed_auc': []}
        for i, s in enumerate(S):
            f, y, X, grp = s['f'], s['y'], s['X'], s['grp']
            within['rule'].append(oos_rule_auc(f, y, grp)); within['lda'].append(cv_lda_auc(f, y, grp)); within['csp'].append(cv_csp_auc_C(X, y, grp))
            if i == 0: continue
            later['fixed_bacc'].append(bacc(y, (rule(f, ref0) >= thr0).astype(int))); later['recal_bacc'].append(bacc(y, (rule(f, np.median(f[y == 0], 0)) >= thr0).astype(int)))
            later['fixed_auc'].append(float(roc_auc_score(y, rule(f, ref0)))); later['lda_fixed_auc'].append(float(roc_auc_score(y, lda0.decision_function(f))))
        per[sub] = {**{f'within_{k}': float(np.mean(v)) for k, v in within.items()}, **{k: float(np.mean(v)) for k, v in later.items()}, 'n_ses': len(S)}
        print(sub, {k: round(v, 3) for k, v in per[sub].items()}, flush=True)
    keys = [k for k in next(iter(per.values())) if k != 'n_ses']
    summ = summarize(list(per.values()), keys)
    json.dump({'per_subject': per, 'summary': summ, 'note': '无 11 导平均参考(原始双乳突参考)'}, open(os.path.join(OUT, 'rawref.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 慢性 · 原始参考 ===')
    for k, s in summ.items(): print(f"  {k:16s} {s['mean']:.3f} (>0.7: {s['n_gt_0.7']}/14)")

# ------------------------------------------------------------------ predict
def predict():
    """前 m 次会话能否预测后续:预测目标统一为第 4 次会话起的均值(与 m 无关);两两会话差先在患者内平均再跨患者平均;ρ 给患者 bootstrap 95% 区间。"""
    from scipy import stats
    d = json.load(open(os.path.join(ROOT, 'results/stroke_chronic_mi_vs_rest/summary.json')))['per_subject']
    subs = sorted(d); res = {}; rng = np.random.default_rng(0)
    def boot_rho(a, b, nb=5000):
        rs = []
        for _ in range(nb):
            idx = rng.integers(0, len(a), len(a)); rs.append(stats.spearmanr(a[idx], b[idx])[0])
        rs = np.array(rs); rs = rs[~np.isnan(rs)]
        return [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
    for key in ['rule_oos_auc', 'rule_auc', 'csp_lda_auc']:
        if key not in d[subs[0]]['sessions'][0]: continue
        res[key] = {}
        target = np.array([np.mean([s[key] for s in d[sub]['sessions'][3:]]) for sub in subs])          # 固定目标:第 4 次起
        for m in [1, 2, 3]:
            early = np.array([np.mean([s[key] for s in d[sub]['sessions'][:m]]) for sub in subs])
            rho, p = stats.spearmanr(early, target)
            res[key][f'first{m}'] = {'spearman_rho': float(rho), 'p': float(p), 'rho_ci95': boot_rho(early, target), 'mean_abs_err': float(np.abs(early - target).mean()),
                                     'agree_gt_0.7': int(((early > 0.7) == (target > 0.7)).sum()), 'early_gt_0.7': int((early > 0.7).sum()), 'target_gt_0.7': int((target > 0.7).sum()), 'n': len(subs)}
        per_pat = [np.mean([abs(a[key] - b[key]) for i, a in enumerate(d[sub]['sessions']) for b in d[sub]['sessions'][i + 1:]]) for sub in subs]
        res[key]['pairwise_session_abs_diff_patient_mean'] = float(np.mean(per_pat)); res[key]['pairwise_session_abs_diff_patient_median'] = float(np.median(per_pat))
        h1 = np.array([np.mean([s[key] for s in d[sub]['sessions'][:len(d[sub]['sessions']) // 2]]) for sub in subs]); h2 = np.array([np.mean([s[key] for s in d[sub]['sessions'][len(d[sub]['sessions']) // 2:]]) for sub in subs])
        rho, p = stats.spearmanr(h1, h2); res[key]['split_half'] = {'spearman_rho': float(rho), 'p': float(p), 'rho_ci95': boot_rho(h1, h2)}
        # 单次会话 AUC 的方差分解:患者间方差 vs 患者内会话间方差(以单次会话为观测)
        allv = [(i, s[key]) for i, sub in enumerate(subs) for s in d[sub]['sessions']]
        means = {i: np.mean([v for j, v in allv if j == i]) for i in range(len(subs))}
        within = np.mean([np.var([v for j, v in allv if j == i], ddof=1) for i in range(len(subs))]); between = np.var(list(means.values()), ddof=1)
        res[key]['variance_within_patient'] = float(within); res[key]['variance_between_patients'] = float(between); res[key]['icc_like_single_session'] = float(between / (between + within))
    json.dump(res, open(os.path.join(OUT, 'predict.json'), 'w'), indent=1, ensure_ascii=False)
    for key in res:
        print(key)
        for m in ['first1', 'first2', 'first3']:
            r = res[key][m]; print(f"  {m}: rho={r['spearman_rho']:.2f} [{r['rho_ci95'][0]:.2f},{r['rho_ci95'][1]:.2f}] p={r['p']:.3f} MAE={r['mean_abs_err']:.3f} agree {r['agree_gt_0.7']}/14")
        print(f"  患者内两两会话差 均值 {res[key]['pairwise_session_abs_diff_patient_mean']:.3f} 中位 {res[key]['pairwise_session_abs_diff_patient_median']:.3f}; 分半 ρ={res[key]['split_half']['spearman_rho']:.2f}; 单次会话 ICC 样 {res[key]['icc_like_single_session']:.2f}")

# ------------------------------------------------------------------ 急性公共
def acute_subjects():
    part = {r['Participant_ID']: r for r in csv.DictReader(open(os.path.join(ROOT, 'data/stroke_acute/participants.tsv')), delimiter='\t')}
    for fpath in sorted(glob.glob(os.path.join(A.DATA, 'sub-*', '*.mat'))):
        sub = os.path.basename(os.path.dirname(fpath)); m = sio.loadmat(fpath)['eeg'][0, 0]
        yield sub, m['rawdata'].astype(float), m['label'].ravel().astype(int), part[sub]

def paretic():
    rows = []
    for sub, raw, lab, p in acute_subjects():
        x = raw[:, A.EEG_IDX, :]; X, y, grp = A.windows(A.filt(x, 8, 30))
        side = 1 if p['ParalysisSide'] == 'left' else 2                  # 标签 1=左手 2=右手
        trial_lab = lab[grp]
        r = {'sub': sub, 'side': p['ParalysisSide']}
        for name, keep in [('paretic', trial_lab == side), ('nonparetic', trial_lab != side)]:
            Xk, yk, gk = X[keep], y[keep], grp[keep]
            fk = two_ch_feats(Xk, A.C3, A.C4)
            r[f'{name}_rule_oos'] = oos_rule_auc(fk, yk, gk); r[f'{name}_lda'] = cv_lda_auc(fk, yk, gk); r[f'{name}_csp'] = cv_csp_auc(Xk, yk, gk)
            # 只看患侧对侧导联(左瘫→右半球 C4;右瘫→C3)单导规则
            contra = A.C4 if side == 1 else A.C3; ipsi = A.C3 if side == 1 else A.C4
            for lname, ch in [('contra', contra), ('ipsi', ipsi)]:
                f1 = logp(Xk[:, ch])[:, None]
                pred = np.zeros(len(yk))
                for tr, te in A.trial_folds(gk): pred[te] = -(f1[te] - np.median(f1[tr][yk[tr] == 0], 0)).ravel()
                r[f'{name}_rule_{lname}'] = float(roc_auc_score(yk, pred))
        rows.append(r); print(sub, p['ParalysisSide'], ' '.join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)
    keys = [k for k in rows[0] if k not in ('sub', 'side')]
    summ = summarize(rows, keys)
    json.dump({'rows': rows, 'summary': summ}, open(os.path.join(OUT, 'paretic.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 急性 50 人 · 患手 vs 健手(每侧 20 试次) ===')
    for k, s in summ.items(): print(f"  {k:24s} {s['mean']:.3f} ± {s['sd']:.3f}  >0.7: {s['n_gt_0.7']}/50")

def matched():
    from rule_montage_variants import variant_feats, MONT_A
    rows = []
    for sub, raw, lab, p in acute_subjects():
        x = raw[:, A.EEG_IDX, :]; X, y, grp = A.windows(A.filt(x, 8, 30))
        r = {'sub': sub}
        f2 = two_ch_feats(X, A.C3, A.C4); r['lda_2ch'] = cv_lda_auc(f2, y, grp)
        f7, _ = variant_feats(X, 'rule_m7', MONT_A); r['lda_7ch'] = cv_lda_auc(f7, y, grp)
        flap, _ = variant_feats(X, 'rule_lap', MONT_A); r['lda_lap2'] = cv_lda_auc(flap, y, grp)
        f29 = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1); r['lda_29ch_power'] = cv_lda_auc(f29, y, grp)
        r['csp_29ch'] = cv_csp_auc(X, y, grp)
        # 只用运动区 7 导做 CSP
        r['csp_7ch'] = cv_csp_auc(X[:, MONT_A['m7'], :], y, grp)
        rows.append(r); print(sub, ' '.join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)
    keys = [k for k in rows[0] if k != 'sub']
    summ = summarize(rows, keys)
    json.dump({'rows': rows, 'summary': summ}, open(os.path.join(OUT, 'matched.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 急性 50 人 · 同输入的训练对照(提示期窗) ===')
    for k, s in summ.items(): print(f"  {k:16s} {s['mean']:.3f} ± {s['sd']:.3f}  >0.7: {s['n_gt_0.7']}/50  >0.8: {sum(r[k] > 0.8 for r in rows)}")

def controls():
    rows = []
    for sub, raw, lab, p in acute_subjects():
        r = {'sub': sub}
        # EOG(原始 33 导里的 30、31 = HEOL、HEOR)
        eog = raw[:, [30, 31], :]
        for band, (lo, hi) in {'1-8': (1, 8), '8-30': (8, 30)}.items():
            Xe, y, grp = A.windows(A.filt(eog, lo, hi)); fe = two_ch_feats(Xe, 0, 1)
            r[f'eog_rule_{band}'] = oos_rule_auc(fe, y, grp); r[f'eog_lda_{band}'] = cv_lda_auc(fe, y, grp)
        x = raw[:, A.EEG_IDX, :]; X, y, grp = A.windows(A.filt(x, 8, 30))
        for name, (a, b) in {'fp': (0, 1), 'occ': (27, 28), 'c34': (A.C3, A.C4)}.items():
            r[f'{name}_rule'] = oos_rule_auc(two_ch_feats(X, a, b), y, grp)
        # 幅度剔窗:C3/C4 峰峰值 >100 µV 的窗剔除
        ptp = np.ptp(X[:, [A.C3, A.C4], :], -1).max(1); keep = ptp <= 100
        r['n_windows_kept'] = int(keep.sum())
        if (y[keep] == 1).sum() >= 10 and (y[keep] == 0).sum() >= 10:
            r['c34_rule_clean'] = oos_rule_auc(two_ch_feats(X[keep], A.C3, A.C4), y[keep], grp[keep])
        else:
            r['c34_rule_clean'] = None
        rows.append(r); print(sub, ' '.join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), f"kept={r['n_windows_kept']}", flush=True)
    keys = [k for k in rows[0] if k not in ('sub', 'n_windows_kept')]
    summ = summarize(rows, keys)
    summ['n_windows_kept'] = {'mean': float(np.mean([r['n_windows_kept'] for r in rows])), 'min': int(min(r['n_windows_kept'] for r in rows))}
    json.dump({'rows': rows, 'summary': summ}, open(os.path.join(OUT, 'controls.json'), 'w'), indent=1, ensure_ascii=False)
    print('\n=== 急性 50 人 · 对照(提示期窗) ===')
    for k, s in summ.items():
        if 'mean' in s and 'sd' in s: print(f"  {k:18s} {s['mean']:.3f} ± {s['sd']:.3f}  >0.7: {s['n_gt_0.7']}/{s['n']}")
    print('  剔窗后保留窗数 均值', summ['n_windows_kept'])



# ------------------------------------------------------------------ controls2:眼电/全局成分的来源分解(2026-10-04 晚加,对照实验发现 EOG 1–8 Hz 规则 AUC 0.98 之后)
def controls2():
    """对急性数据,分别在提示期窗与休息期窗下:
    eog_reg_c34   先用两导 EOG 对每导 EEG 做逐试次线性回归去眼电,再算 C3/C4 规则
    global_rule   29 导对数功率均值的规则(全局成分有多大)
    motor_rel     C3/C4 相对枕区(O1/O2)的对数功率差的规则(去掉全局 alpha 变化后剩多少)
    motor_rel_g   C3/C4 相对 29 导均值的规则
    并重复 eog_rule_1-8 / eog_rule_8-30 / fp / occ 在休息期窗下的结果。"""
    import importlib
    res = {}
    for mode in ['cue', 'break']:
        os.environ['REST_MODE'] = mode; importlib.reload(A)
        rows = []
        for sub, raw, lab, p in acute_subjects():
            r = {'sub': sub}
            eog_raw = raw[:, [30, 31], :]; eeg_raw = raw[:, A.EEG_IDX, :]
            # 逐试次最小二乘回归去眼电(宽带,0.5–40 Hz 后做;用因果滤波保持一致)
            e_b = A.filt(eog_raw, 0.5, 40); x_b = A.filt(eeg_raw, 0.5, 40)
            clean = np.empty_like(x_b)
            for t in range(x_b.shape[0]):
                E = np.c_[e_b[t].T, np.ones(e_b.shape[-1])]           # (时间, 3)
                beta = np.linalg.lstsq(E, x_b[t].T, rcond=None)[0]     # (3, 29)
                clean[t] = (x_b[t].T - E @ beta).T
            Xc, y, grp = A.windows(A.filt(clean, 8, 30)); r['eog_reg_c34'] = oos_rule_auc(two_ch_feats(Xc, A.C3, A.C4), y, grp)
            Xf, yf, gf = A.windows(A.filt(x_b, 8, 30)); r['c34_filt_only'] = oos_rule_auc(two_ch_feats(Xf, A.C3, A.C4), yf, gf)   # 同样多一道 0.5–40 Hz 滤波但不回归的对照
            X, y, grp = A.windows(A.filt(eeg_raw, 8, 30))
            lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1)         # (窗, 29)
            r['c34_rule'] = oos_rule_auc(lp[:, [A.C3, A.C4]], y, grp)
            r['global_rule'] = oos_rule_auc(lp.mean(1, keepdims=True), y, grp)
            occ = lp[:, [27, 28]].mean(1, keepdims=True); r['motor_rel_occ'] = oos_rule_auc(lp[:, [A.C3, A.C4]] - occ, y, grp)
            r['motor_rel_global'] = oos_rule_auc(lp[:, [A.C3, A.C4]] - lp.mean(1, keepdims=True), y, grp)
            r['occ_rule'] = oos_rule_auc(lp[:, [27, 28]], y, grp); r['fp_rule'] = oos_rule_auc(lp[:, [0, 1]], y, grp)
            for band, (lo, hi) in {'1-8': (1, 8), '8-30': (8, 30)}.items():
                Xe, ye, ge = A.windows(A.filt(eog_raw, lo, hi)); r[f'eog_rule_{band}'] = oos_rule_auc(two_ch_feats(Xe, 0, 1), ye, ge)
            rows.append(r); print(mode, sub, ' '.join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)
        keys = [k for k in rows[0] if k != 'sub']; res[mode] = {'rows': rows, 'summary': summarize(rows, keys)}
        print(f'\n=== 急性 50 人 · 来源分解 · 静息窗={mode} ===')
        for k, s in res[mode]['summary'].items(): print(f"  {k:18s} {s['mean']:.3f} ± {s['sd']:.3f}  >0.7: {s['n_gt_0.7']}/50")
    json.dump(res, open(os.path.join(OUT, 'controls2.json'), 'w'), indent=1, ensure_ascii=False)


# ------------------------------------------------------------------ farref:远端参照与中央增量信息
FAR = [0, 1, 5, 6, 24, 25, 26, 27, 28]           # Fp1 Fp2 F7 F8 T5 T6 Oz O1 O2:不含中央/额中央/顶区
def farref():
    """急性(提示期窗与休息期窗):
    rule_c34_minus_far   C3/C4 对数功率减远端 9 导均值的固定规则
    lda_far / lda_far_c34 / lda_c34   远端均值单特征 LDA、远端+C3/C4 三特征 LDA、C3/C4 两特征 LDA(增量信息)
    lda_rel_c34          (C3−远端, C4−远端) 两特征 LDA(学方向与权重的相对特征)
    lda_29_rel           29 导对数功率减窗内均值后的 LDA(去掉绝对尺度,与 CSP 的归一化可比)
    rule_lap             C3/C4 Laplacian 规则(复核)"""
    import importlib
    from rule_montage_variants import variant_feats, MONT_A
    res = {}
    for mode in ['cue', 'break']:
        os.environ['REST_MODE'] = mode; importlib.reload(A)
        rows = []
        for sub, raw, lab, p in acute_subjects():
            X, y, grp = A.windows(A.filt(raw[:, A.EEG_IDX, :], 8, 30))
            lp = np.stack([logp(X[:, c]) for c in range(X.shape[1])], 1)
            far = lp[:, FAR].mean(1, keepdims=True); c34 = lp[:, [A.C3, A.C4]]
            r = {'sub': sub}
            r['rule_c34'] = oos_rule_auc(c34, y, grp); r['rule_c34_minus_far'] = oos_rule_auc(c34 - far, y, grp); r['rule_far'] = oos_rule_auc(far, y, grp)
            r['lda_far'] = cv_lda_auc(far, y, grp); r['lda_c34'] = cv_lda_auc(c34, y, grp); r['lda_far_c34'] = cv_lda_auc(np.c_[far, c34], y, grp)
            r['lda_rel_c34'] = cv_lda_auc(c34 - far, y, grp); r['lda_29_rel'] = cv_lda_auc(lp - lp.mean(1, keepdims=True), y, grp); r['lda_29_abs'] = cv_lda_auc(lp, y, grp)
            fl, _ = variant_feats(X, 'rule_lap', MONT_A); r['rule_lap'] = oos_rule_auc(fl, y, grp); r['lda_lap'] = cv_lda_auc(fl, y, grp)
            rows.append(r); print(mode, sub, ' '.join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)
        keys = [k for k in rows[0] if k != 'sub']; res[mode] = {'rows': rows, 'summary': summarize(rows, keys), 'FAR_channels': FAR}
        print(f'\n=== 急性 50 人 · 远端参照与增量信息 · 参考窗={mode} ===')
        for k, s_ in res[mode]['summary'].items(): print(f"  {k:20s} {s_['mean']:.3f} ± {s_['sd']:.3f}  >0.7: {s_['n_gt_0.7']}/50")
    json.dump(res, open(os.path.join(OUT, 'farref.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    for cmd in sys.argv[1:]:
        print(f'\n##### {cmd}', flush=True); globals()[cmd]()
