"""
Data pipeline: preprocessing, feature extraction, labels and the fixed event-level split.
====================================================================
This module documents exactly how the feature cache consumed by ``train.py``
is built from the raw waveform archive (Diting 2.0). The shipped package
contains the reduced label/split cache (``data/event_split_cache.npz``);
building the full feature cache additionally requires the raw ``.h5`` archive
and the fixed event split file.

Arrays in the full feature cache (1.15 GB, not shipped; the shipped file
``data/event_split_cache.npz`` keeps only ``event_id``, ``labels3``,
``tr_idx``/``va_idx``/``te_idx``, ``pol`` and ``pol_mask``):
  pw4 (N,1000,4)    3 normalized 10-s waveforms + 1 log-amplitude channel
  rf    (N,15)      raw waveform features          df (N,1) log10 distance
  sf    (N,5)       spectral features
  labels3 (N,)      0 natural / 1-2-4 non-natural subtypes / else excluded
  mags  (N,)        event magnitude
  pol, pol_mask     first-motion polarity label and its availability mask
  ql, rl            waveform-quality / hazard-risk label columns (unused in this study)
  event_id (N,)     per-sample event identifier
  tr_idx, va_idx, te_idx   fixed event-level split indices
"""
import os, json
import numpy as np
from scipy import signal as sig
import h5py

SR = 100
P_LEN = 1000
N_CH = 3
N_CH_ENC = 4
N_RAW_FEAT = 15
N_DIST_FEAT = 1
N_SPEC_FEAT = 5
T3 = {0: 0, 1: 1, 2: 2, 3: -1, 4: 1, 5: -1, 9: 0}

# ============ Preprocessing ============
def preprocess_to_10s(wf3_full, pg_idx_abs):
    start = max(0, int(pg_idx_abs) - 50)
    end = start + P_LEN
    if end > wf3_full.shape[1]:
        end = wf3_full.shape[1]
        start = max(0, end - P_LEN)
    seg = wf3_full[:, start:end].copy()
    if seg.shape[1] < P_LEN:
        padded = np.zeros((3, P_LEN), dtype=np.float32)
        padded[:, :seg.shape[1]] = seg
        seg = padded
    out_norm = np.zeros_like(seg)
    out_raw = np.zeros_like(seg)
    for ch in range(3):
        w = sig.detrend(seg[ch])
        nyq = SR / 2.0
        b, a = sig.butter(4, [1.0/nyq, 20.0/nyq], btype='band')
        w = sig.filtfilt(b, a, w)
        out_raw[ch] = w
        mx = np.max(np.abs(w))
        if mx > 1e-10:
            w = w / mx
        out_norm[ch] = (w + 1.0) / 2.0
    return out_norm, out_raw

def extract_raw_features(raw_wf):
    features = []
    for ch in range(3):
        w = raw_wf[ch]
        w_abs = np.abs(w)
        features.extend([
            np.log10(np.max(w_abs) + 1e-10),
            np.log10(np.sqrt(np.mean(w**2)) + 1e-10),
            np.log10(np.sum(w**2) + 1e-10),
            np.log10(np.std(w) + 1e-10),
            np.log10((np.max(w) - np.min(w)) + 1e-10),
        ])
    return np.array(features, dtype=np.float32)

def compute_spectral_features(wz):
    w = wz * np.hanning(len(wz))
    sp = np.abs(np.fft.rfft(w)) ** 2
    fr = np.fft.rfftfreq(len(wz), 1.0 / SR)
    total = sp.sum() + 1e-10
    m_dom = (fr >= 1) & (fr <= 25)
    domf = fr[m_dom][np.argmax(sp[m_dom])] if m_dom.any() else 0.0
    e1 = sp[(fr >= 1) & (fr <= 5)].sum() / total
    e2 = sp[(fr > 5) & (fr <= 10)].sum() / total
    e3 = sp[(fr > 10) & (fr <= 20)].sum() / total
    e4 = sp[(fr > 20) & (fr <= 50)].sum() / total
    return np.array([domf, np.log10(e1 + 1e-10), np.log10(e2 + 1e-10),
                     np.log10(e3 + 1e-10), np.log10(e4 + 1e-10)], dtype=np.float32)

def compute_quality_label(norm_wf):
    """Legacy 3-class waveform-quality label; kept for cache compatibility, not used by the paper's model."""
    noise_seg = norm_wf[0, :200]
    signal_seg = norm_wf[0, 200:]
    noise_rms = np.sqrt(np.mean((noise_seg - 0.5)**2)) + 1e-10
    signal_rms = np.sqrt(np.mean((signal_seg - 0.5)**2))
    snr = signal_rms / noise_rms
    x = np.arange(len(norm_wf[0]))
    bd = abs(np.polyfit(x, norm_wf[0], 1)[0]) * 1000
    bd = min(bd, 0.1)
    std_w = np.std(norm_wf[0]) + 1e-10
    amr = np.sum(np.abs(np.diff(norm_wf[0])) > 3*std_w) / len(norm_wf[0])
    amr = min(amr, 1.0)
    pi = 1.0 if snr > 1.5 else 0.0
    q = 0.4/(1.0 + np.exp(-(snr - 3))) + 0.2*(1.0 - bd/0.1) + 0.2*(1.0 - amr) + 0.2*pi
    if q >= 0.7: return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif q >= 0.35: return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else: return np.array([0.0, 0.0, 1.0], dtype=np.float32)

def compute_risk_label(mag, dist):
    """Legacy 3-class hazard-risk label; kept for cache compatibility, not used by the paper's model."""
    m_est = mag if mag > 0 else 0
    r = max(dist, 1.0)
    s = 0 if r < 20 else 1
    r_score = max(m_est - 1.0, 0) * (1.0 / (1.0 + np.exp(-(15 - r) / 8))) * (1 + 0.5 * (1 - s))
    pl = 1.0 / (1.0 + np.exp(r_score - 2))
    pm = (np.exp(r_score - 2) - np.exp(r_score - 5)) / ((1 + np.exp(r_score - 2)) * (1 + np.exp(r_score - 5)) + 1e-10)
    pm = max(0, min(pm, 1 - pl))
    ph = max(0, 1 - pl - pm)
    t = pl + pm + ph
    if t > 0: pl /= t; pm /= t; ph /= t
    idx = np.argmax([pl, pm, ph])
    lb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lb[idx] = 1.0
    return lb

def generate_event_id(lat, lon, precision=3):
    return int(round(lat, precision) * 10**precision * 10000 + round(lon, precision) * 10**precision)

def check_zero_variance(wf3_full, pg_idx_abs):
    start = max(0, int(pg_idx_abs) - 50)
    end = start + P_LEN
    if end > wf3_full.shape[1]:
        end = wf3_full.shape[1]
        start = max(0, end - P_LEN)
    seg = wf3_full[:, start:end]
    return np.all(np.std(seg, axis=1) < 1e-10)

def load_data(path='merge/merge_enhanced_v2.h5'):  # path to the raw Diting 2.0 archive; adjust to your copy
    print(f"\n[1] Loading {path} ...")
    with h5py.File(path, 'r') as f:
        gn = 'samples' if 'samples' in f else list(f.keys())[0]
        data = f[gn][:]
    ev = data[:, 18000].astype(int)
    valid = np.array([T3.get(e, -1) >= 0 for e in ev])
    labels3 = np.array([T3[e] for e in ev[valid]])
    wz = data[:, 0:6000][valid].astype(np.float32)
    wn = data[:, 6000:12000][valid].astype(np.float32)
    we = data[:, 12000:18000][valid].astype(np.float32)
    wf3_full = np.stack([wz, wn, we], axis=1)
    mags = data[:, 18001][valid].astype(np.float32)
    lons = data[:, 18002][valid].astype(np.float32)
    lats = data[:, 18003][valid].astype(np.float32)
    dists = data[:, 18004][valid].astype(np.float32)
    pgidx = np.nan_to_num(data[:, 18006][valid].astype(np.float32), nan=100.0)
    orig_pol = data[:, 18010][valid].astype(np.int64)
    event_id = np.array([generate_event_id(lats[i], lons[i]) for i in range(len(lats))], dtype=np.int64)
    print(f"  Total valid: {len(labels3)}, Unique events: {len(np.unique(event_id))}")

    print(f"\n[2] Data cleaning ...")
    clean = np.ones(len(labels3), dtype=bool)
    clean[np.isnan(mags)] = False
    clean[np.isnan(dists)] = False
    zv = np.zeros(len(labels3), dtype=bool)
    for i in range(len(labels3)):
        if check_zero_variance(wf3_full[i], pgidx[i]):
            zv[i] = True
    clean[zv] = False
    labels3, wf3_full = labels3[clean], wf3_full[clean]
    mags, dists, pgidx = mags[clean], dists[clean], pgidx[clean]
    orig_pol, event_id = orig_pol[clean], event_id[clean]
    print(f"  Remaining: {len(labels3)}")

    # binary task labels (natural=0, non-natural=1)
    labels = (labels3 > 0).astype(np.int64)
    print(f"  binary: natural={np.sum(labels==0)}, non-natural={np.sum(labels==1)}")

    idx = np.arange(len(labels))
    print(f"  Preprocessing (4ch + spectral)...")
    pw_norm = np.zeros((len(idx), P_LEN, N_CH_ENC), dtype=np.float32)
    raw_feat = np.zeros((len(idx), N_RAW_FEAT), dtype=np.float32)
    dist_feat = np.zeros((len(idx), N_DIST_FEAT), dtype=np.float32)
    spec_feat = np.zeros((len(idx), N_SPEC_FEAT), dtype=np.float32)
    pol = np.zeros(len(idx), dtype=np.int64)
    pol_mask = np.zeros(len(idx), dtype=np.float32)
    ql = np.zeros((len(idx), 3), dtype=np.float32)
    rl = np.zeros((len(idx), 3), dtype=np.float32)
    for i, ii in enumerate(idx):
        norm_seg, raw_seg = preprocess_to_10s(wf3_full[ii], pgidx[ii])
        pw_norm[i, :, :3] = norm_seg.T
        log_amp = np.log10(np.abs(raw_seg).max(axis=0) + 1e-10)
        pw_norm[i, :, 3] = (np.clip(log_amp, -10.0, 10.0) + 10.0) / 20.0
        raw_feat[i] = extract_raw_features(raw_seg)
        spec_feat[i] = compute_spectral_features(raw_seg[0])
        dist_feat[i] = np.log10(dists[ii] + 1.0)
        if labels3[ii] == 0:
            pol[i] = orig_pol[ii]
            pol_mask[i] = 1.0
        ql[i] = compute_quality_label(norm_seg)
        rl[i] = compute_risk_label(mags[ii], dists[ii])
        if (i + 1) % 20000 == 0:
            print(f"    {i+1}/{len(idx)}")
    return {'pw': pw_norm, 'raw_feat': raw_feat, 'dist_feat': dist_feat, 'spec_feat': spec_feat,
            'labels': labels, 'mags': mags[idx], 'pol': pol, 'pol_mask': pol_mask, 'ql': ql, 'rl': rl,
            'event_id': event_id[idx]}

def get_event_split(data, split_file='event_split.json'):
    event_id, labels = data['event_id'], data['labels']
    if os.path.exists(split_file):
        print(f"\n[3] Loading fixed event split from {split_file} ...")
        with open(split_file) as f:
            sp = json.load(f)
        tr, va, te = set(sp['train_events']), set(sp['val_events']), set(sp['test_events'])
    else:
        raise RuntimeError(f"missing {split_file}")
    tr_idx = np.where(np.isin(event_id, list(tr)))[0]
    va_idx = np.where(np.isin(event_id, list(va)))[0]
    te_idx = np.where(np.isin(event_id, list(te)))[0]
    print(f"  Samples: Train={len(tr_idx)}, Val={len(va_idx)}, Test={len(te_idx)}")
    return tr_idx, va_idx, te_idx
