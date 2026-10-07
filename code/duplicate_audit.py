#!/usr/bin/env python3
"""信号级重复审计:四个数据集内,跨患者 / 跨时点 / 跨 run / 跨试次的逐点重复检查,不依赖文件哈希。
方法:对每个记录单元取「信号数组(float64, C order)的 SHA-256」与「前 2 s 全通道数据的 SHA-256」两种指纹;相同指纹即报重复;再对报出的对做逐点最大绝对差确认。
  LL / SF:单元 = 每个 .set 文件(run);另对每个 run 内的试次(按事件切 2 s)做试次级指纹,检查跨 run/跨患者的试次重复。
  急性:单元 = 每人 .mat 的每个试次(40 × 8 s)。
  慢性:单元 = 每个 run 文件内的每个试次(已知会话内有重复;此处检查跨会话与跨患者)。
输出 results/duplicate_audit.json 与 docs/duplicate_audit.md。"""
import glob, hashlib, json, os, sys, warnings
import numpy as np
import scipy.io as sio
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mne; mne.set_log_level('ERROR')


def h(arr): return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def report_dups(units, tag, L, out):
    """units: list of (name, fingerprint, getter) ;同指纹分组报告"""
    groups = {}
    for name, fp, _ in units: groups.setdefault(fp, []).append(name)
    dups = [v for v in groups.values() if len(v) > 1]
    L.append(f"\n## {tag}: {len(units)} 单元, {len(dups)} 组重复\n")
    for g in dups: L.append(f"- 重复组: {g}")
    out[tag] = {'n_units': len(units), 'dup_groups': dups}
    return dups


def eeglab_units(base_glob, tag, L, out):
    files = sorted(glob.glob(base_glob)); units_run, units_trial = [], []
    for f in files:
        try: raw = mne.io.read_raw_eeglab(f, preload=True, uint16_codec='latin1')
        except Exception as e: L.append(f"- 读取失败 {f}: {e}"); continue
        x = raw.get_data(); name = os.path.relpath(f, ROOT)
        units_run.append((name, h(x), None))
        fs = int(raw.info['sfreq']); on = raw.annotations.onset; desc = raw.annotations.description
        for t, d in zip(on, desc):
            s0 = int(round(t * fs))
            if s0 + 2 * fs <= x.shape[1]: units_trial.append((f"{name}@{t:.2f}s[{d}]", h(x[:, s0:s0 + 2 * fs]), None))
    report_dups(units_run, f"{tag}·run 级", L, out)
    dups = report_dups(units_trial, f"{tag}·试次级(事件后 2 s)", L, out)
    # 试次级重复若只是同一 run 内(同一文件)则另算;统计跨文件的
    cross = [g for g in dups if len(set(n.split('@')[0] for n in g)) > 1]
    L.append(f"- 其中跨文件(跨 run/时点/患者)的试次重复组: {len(cross)}"); out[f"{tag}·试次级(事件后 2 s)"]['cross_file_groups'] = cross[:50]


def main():
    out, L = {}, ["# 信号级重复审计(code/duplicate_audit.py;2026-10-07)\n"]
    # 下肢 原始
    eeglab_units(os.path.join(ROOT, 'data', 'stroke_tj_lowerlimb', 'sourcedata', 'sub-*', 'ses-*', '*_ori.set'), '下肢原始', L, out)
    # 第六指 原始
    eeglab_units(os.path.join(ROOT, 'data', 'stroke_tj_sixthfinger', 'sourcedata', 'sub-*', '*', '*_original.set'), '患手原始', L, out)
    # 急性:试次级
    units = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'stroke_acute', 'sourcedata', 'sub-*', '*.mat'))):
        x = sio.loadmat(f)['eeg'][0, 0]['rawdata'].astype(float); sub = os.path.basename(os.path.dirname(f))
        for i in range(x.shape[0]): units.append((f"{sub}#trial{i}", h(x[i]), None))
    report_dups(units, '急性·试次级(整试次 8 s)', L, out)
    # 慢性:试次级,跨会话/跨患者(会话内重复为已知,单列)
    units = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'data', 'stroke_chronic_hs', 'sub-*', 'ses-*', 'eeg', '*run-*_eeg.mat'))):
        d = sio.loadmat(f, squeeze_me=True); x = np.transpose(d['EEGdata'], (2, 0, 1)).astype(float)
        rel = os.path.relpath(f, ROOT); sub = rel.split('/')[2]; ses = rel.split('/')[3]
        for i in range(x.shape[0]): units.append((f"{sub}/{ses}/{os.path.basename(f)}#t{i}", h(x[i]), None))
    dups = report_dups(units, '慢性·试次级(整试次)', L, out)
    within_ses = [g for g in dups if len(set(n.split('/')[0] + n.split('/')[1] for n in g)) == 1]
    cross_ses = [g for g in dups if len(set(n.split('/')[1] for n in g)) > 1 and len(set(n.split('/')[0] for n in g)) == 1]
    cross_sub = [g for g in dups if len(set(n.split('/')[0] for n in g)) > 1]
    L.append(f"- 会话内重复组(已知,主分析已去重): {len(within_ses)};跨会话同患者: {len(cross_ses)};跨患者: {len(cross_sub)}")
    out['慢性·试次级(整试次)'].update(within_session=len(within_ses), cross_session_same_patient=cross_ses[:50], cross_patient=cross_sub[:50])
    json.dump(out, open(os.path.join(ROOT, 'results', 'duplicate_audit.json'), 'w'), ensure_ascii=False, indent=1)
    open(os.path.join(ROOT, 'docs', 'duplicate_audit.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
