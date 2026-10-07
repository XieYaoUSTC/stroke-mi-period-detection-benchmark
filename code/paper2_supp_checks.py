#!/usr/bin/env python3
"""S1 逐折 AUC 敏感性(急性提示期窗,三法)+ 幅值异常受试者核查。输出 results/paper2_followups/supp_checks.json"""
import os, sys, glob, json
import numpy as np, scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ['REST_MODE'] = 'cue'
import stroke_mi_vs_rest as A
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, 'results', 'paper2_followups')
rows = []; amp = []
for fpath in sorted(glob.glob(os.path.join(A.DATA, 'sub-*', '*.mat'))):
    sub = os.path.basename(os.path.dirname(fpath)); raw = sio.loadmat(fpath)['eeg'][0, 0]['rawdata'].astype(float)
    x = raw[:, A.EEG_IDX, :]; X, y, grp = A.windows(A.filt(x, 8, 30)); f = np.c_[np.log(X[:, A.C3].var(-1)), np.log(X[:, A.C4].var(-1))]
    r = {'sub': sub}
    for name in ['rule', 'lda', 'csp']:
        fold_auc, pred = [], np.zeros(len(y))
        for tr, te in A.trial_folds(grp):
            if name == 'rule': p = -(f[te] - np.median(f[tr][y[tr] == 0], 0)).min(1)
            elif name == 'lda': p = LDA().fit(f[tr], y[tr]).decision_function(f[te])
            else: w = A.csp_fit(X[tr], y[tr]); p = LDA().fit(A.csp_feat(X[tr], w), y[tr]).decision_function(A.csp_feat(X[te], w))
            pred[te] = p; fold_auc.append(roc_auc_score(y[te], p))
        r[f'{name}_concat'] = float(roc_auc_score(y, pred)); r[f'{name}_foldmean'] = float(np.mean(fold_auc))
    rows.append(r)
    # 幅值:原始(未滤波)C3/C4 逐试次峰峰中位数,以及 8–30 Hz 窗峰峰中位数
    amp.append({'sub': sub, 'raw_ptp_median_uV': float(np.median(np.ptp(x[:, [A.C3, A.C4], :], -1))), 'band_ptp_median_uV': float(np.median(np.ptp(X[:, [A.C3, A.C4], :], -1))), 'raw_std_median': float(np.median(x[:, [A.C3, A.C4], :].std(-1)))})
    print(sub, {k: round(v, 3) for k, v in r.items() if k != 'sub'}, 'raw ptp', round(amp[-1]['raw_ptp_median_uV'], 1), flush=True)
summ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'sub'}
rank = {}
for name in ['rule', 'lda', 'csp']: rank[name] = (summ[f'{name}_concat'], summ[f'{name}_foldmean'])
json.dump({'foldwise': {'rows': rows, 'summary': summ}, 'amplitude': amp}, open(os.path.join(OUT, 'supp_checks.json'), 'w'), indent=1, ensure_ascii=False)
print('\n逐折 vs 拼接 AUC(均值):', {k: round(v, 3) for k, v in summ.items()})
a = sorted(amp, key=lambda d: -d['raw_ptp_median_uV']); print('幅值最大的 5 人(原始 C3/C4 逐试次峰峰中位数 µV):', [(d['sub'], round(d['raw_ptp_median_uV'])) for d in a[:5]], ' 中位人:', round(np.median([d['raw_ptp_median_uV'] for d in amp])))
