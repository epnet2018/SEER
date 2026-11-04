# -*- coding: utf-8 -*-
"""
Training of the paper's multi-task model (shared residual self-attention encoder,
station-level attention aggregation, event-level type + magnitude heads, auxiliary
station-level type head with loss weight W_ST, station-level polarity head).
====================================================================
Protocol: fixed event-level split (training / validation / test), trained from
scratch, noise augmentation + EMA + polarity floor; the validation set selects the
model (event-level macro-F1 with a polarity-accuracy floor) and the decision
thresholds; the test set is evaluated in full.
Total loss = W_POL*Focal(polarity) + W_TYPE*Focal(type) + W_MAG*SmoothL1(magnitude)
           + W_ST*Focal(station_type, station_label),
where station_label = event label broadcast to its member stations.
Usage: python train.py <seed> [W_ST]          (default W_ST = 0.30)
Outputs (written to <package>/data/predictions/):
    auxhead_w{W_ST*100:03d}_seed{seed:02d}.npz          cached predictions (consumed by evaluate.py)
    model_auxhead_w{W_ST*100:03d}_seed{seed:02d}.pt
    auxhead_w{W_ST*100:03d}_seed{seed:02d}_train_summary.json
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

warnings.filterwarnings('ignore')
_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)
sys.path.insert(0, _DIR)

from model import SeismicModel, enable_mc_dropout

DEVICE = torch.device('cuda:0')
PACKAGE_ROOT = os.path.dirname(os.path.dirname(_DIR))
CACHE = os.path.join(PACKAGE_ROOT, 'data', 'full_feature_cache.npz')
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
# W_ST: auxiliary station-head loss weight; the paper's scan uses 0.15 / 0.30 / 0.50 (Table 2)
W_ST = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
TAG = f'auxhead_w{int(round(W_ST * 100)):03d}_seed{SEED:02d}'
OUT_DIR = os.path.join(PACKAGE_ROOT, 'data', 'predictions')
BATCH_EVENTS = 48
EPOCHS = 250
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
WARMUP_EPOCHS = 10
PATIENCE = 35
EMA_DECAY = 0.999
POL_FLOOR = 0.85
W_POL, W_TYPE, W_MAG = 0.35, 0.50, 0.40
K_MAX = 12
STATION_DROP = 0.15
NOISE = (0.005, 0.02)
MC_T = 20

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def prf(y, p, pos):
    tp = int(((p == pos) & (y == pos)).sum()); fp = int(((p == pos) & (y != pos)).sum())
    fn = int(((p != pos) & (y == pos)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1, tp, fp, fn


def optimize_threshold(logits, labels, grid=np.arange(-1.5, 1.51, 0.02), target=0.90):
    best_any, feasible = None, []
    for b in grid:
        p = ((logits[:, 1] + b) > logits[:, 0]).astype(int)
        p1, r1, f1, *_ = prf(labels, p, 1)
        p0, r0, f0, *_ = prf(labels, p, 0)
        macro = (f0 + f1) / 2
        if p1 >= target and r1 >= target and p0 >= target and r0 >= target:
            feasible.append({'b': float(b), 'macroF1': macro})
        if best_any is None or macro > best_any['macroF1']:
            best_any = {'b': float(b), 'macroF1': macro}
    chosen = max(feasible, key=lambda r: r['macroF1']) if feasible else best_any
    return chosen, len(feasible), best_any


class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            focal = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        else:
            focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


class EventDataset(Dataset):
    """Each item = one event (returns its station-sample indices, randomly subsampled during training)"""
    def __init__(self, ev_samples, train):
        self.eids = sorted(ev_samples.keys())
        self.ev = ev_samples
        self.train = train
    def __len__(self):
        return len(self.eids)
    def __getitem__(self, i):
        idxs = np.asarray(self.ev[self.eids[i]])
        if self.train:
            if len(idxs) > K_MAX:
                idxs = np.random.choice(idxs, K_MAX, replace=False)
            if STATION_DROP > 0 and len(idxs) >= 3:
                keep = np.random.rand(len(idxs)) > STATION_DROP
                if keep.sum() >= 2:
                    idxs = idxs[keep]
        return np.asarray(idxs)


def make_collate(D, labels, mags):
    def collate(batch):
        ev_idx, pos_idx = [], []
        type_y, mag_y = [], []
        for b, idxs in enumerate(batch):
            type_y.append(labels[idxs[0]])
            mag_y.append(mags[idxs[0]])
            for p, i in enumerate(idxs):
                ev_idx.append(b); pos_idx.append(p)
        idx_all = np.concatenate(batch)
        Kmax = max(len(x) for x in batch)
        return (
            torch.FloatTensor(D['pw4'][idx_all]),
            torch.FloatTensor(D['rf_n'][idx_all]),
            torch.FloatTensor(D['df_n'][idx_all]),
            torch.FloatTensor(D['sf_n'][idx_all]),
            torch.LongTensor(type_y), torch.FloatTensor(mag_y),
            torch.LongTensor(D['pol'][idx_all]), torch.FloatTensor(D['pol_mask'][idx_all]),
            torch.LongTensor(ev_idx), torch.LongTensor(pos_idx),
            torch.LongTensor([len(x) for x in batch]), Kmax)
    return collate


def forward_batch(model, batch, train_mode):
    pw, rf, df, sf, ty, mg, py, pm, ev_idx, pos_idx, counts, Kmax = [
        b.to(DEVICE, non_blocking=True) if not isinstance(b, int) else b for b in batch]
    x = pw.permute(0, 2, 1)
    if train_mode:
        sig = float(np.random.uniform(*NOISE))
        x = torch.cat([x[:, :3, :] + torch.randn_like(x[:, :3, :]) * sig,
                       x[:, 3:, :]], dim=1)
    pol_out, type_out, mag_out, st_type_out, alpha = model(
        x, rf, df, sf, ev_idx, pos_idx, len(counts), Kmax)
    return pol_out, type_out, mag_out, st_type_out, alpha, ty, mg, py, pm, ev_idx


@torch.no_grad()
def infer_events(model, ev_samples, D, mc_passes=0):
    """Event-level inference (all stations, no subsampling). MC Dropout enabled when mc_passes>0"""
    model.eval()
    if mc_passes > 0:
        enable_mc_dropout(model)
    eids = sorted(ev_samples.keys())
    out_type, out_mag, out_pol, out_st_type = [], [], [], []
    out_type_mc = [] if mc_passes > 0 else None
    out_mag_mc = [] if mc_passes > 0 else None
    station_eids = []
    B = 48
    for s in range(0, len(eids), B):
        chunk = [ev_samples[e] for e in eids[s:s + B]]
        idx_all = np.concatenate(chunk)
        ev_idx, pos_idx = [], []
        for b, idxs in enumerate(chunk):
            for p in range(len(idxs)):
                ev_idx.append(b); pos_idx.append(p)
        ev_idx = torch.LongTensor(ev_idx).to(DEVICE)
        pos_idx = torch.LongTensor(pos_idx).to(DEVICE)
        Kmax = max(len(c) for c in chunk)
        x = torch.FloatTensor(D['pw4'][idx_all]).permute(0, 2, 1).to(DEVICE)
        rf = torch.FloatTensor(D['rf_n'][idx_all]).to(DEVICE)
        df = torch.FloatTensor(D['df_n'][idx_all]).to(DEVICE)
        sf = torch.FloatTensor(D['sf_n'][idx_all]).to(DEVICE)
        if mc_passes > 0:
            probs_t, mags_t = [], []
            for _ in range(mc_passes):
                pol_o, type_o, mag_o, _, _ = model(x, rf, df, sf, ev_idx, pos_idx, len(chunk), Kmax)
                probs_t.append(F.softmax(type_o, 1).cpu().numpy())
                mags_t.append(mag_o.cpu().numpy())
            out_type_mc.append(np.stack(probs_t, 0))
            out_mag_mc.append(np.stack(mags_t, 0))
            pol_o, type_o, mag_o, st_type_o, _ = model(x, rf, df, sf, ev_idx, pos_idx, len(chunk), Kmax)
        else:
            pol_o, type_o, mag_o, st_type_o, _ = model(x, rf, df, sf, ev_idx, pos_idx, len(chunk), Kmax)
        out_type.append(type_o.cpu().numpy())
        out_mag.append(mag_o.cpu().numpy())
        out_pol.append(pol_o.cpu().numpy())
        out_st_type.append(st_type_o.cpu().numpy())
        station_eids.append(np.repeat(np.arange(s, s + len(chunk)),
                                       [len(c) for c in chunk]))
    res = {'eids': np.array(eids), 'type_logits': np.concatenate(out_type),
           'mag': np.concatenate(out_mag), 'pol_logits': np.concatenate(out_pol),
           'st_type_logits': np.concatenate(out_st_type),
           'station_event_row': np.concatenate(station_eids),
           'station_idx': np.concatenate([np.asarray(ev_samples[e]) for e in eids])}
    if mc_passes > 0:
        res['type_probs_mc'] = np.concatenate(out_type_mc, axis=1)
        res['mag_mc'] = np.concatenate(out_mag_mc, axis=1)
    return res


@torch.no_grad()
def infer_stations(model, idx_array, D):
    """Station-level independent inference (K=1). Returns station-head output + K=1 event-head output (for ablation)"""
    model.eval()
    out_st, out_k1, out_pol = [], [], []
    B = 256
    for s in range(0, len(idx_array), B):
        ii = idx_array[s:s + B]
        n = len(ii)
        ev_idx = torch.arange(n, device=DEVICE)
        pos_idx = torch.zeros(n, dtype=torch.long, device=DEVICE)
        x = torch.FloatTensor(D['pw4'][ii]).permute(0, 2, 1).to(DEVICE)
        rf = torch.FloatTensor(D['rf_n'][ii]).to(DEVICE)
        df = torch.FloatTensor(D['df_n'][ii]).to(DEVICE)
        sf = torch.FloatTensor(D['sf_n'][ii]).to(DEVICE)
        pol_o, type_o, mag_o, st_type_o, _ = model(x, rf, df, sf, ev_idx, pos_idx, n, 1)
        out_st.append(st_type_o.cpu().numpy())
        out_k1.append(type_o.cpu().numpy())
        out_pol.append(pol_o.cpu().numpy())
    return {'st_type_logits': np.concatenate(out_st), 'k1_type_logits': np.concatenate(out_k1),
            'pol_logits': np.concatenate(out_pol)}


def main():
    t0 = time.time()
    print(f"[1] Loading cache (seed={SEED}) ...")
    c = np.load(CACHE, allow_pickle=True)
    data = {k: c[k] for k in ['labels3', 'mags', 'event_id', 'tr_idx', 'va_idx', 'te_idx',
                              'pw4', 'rf', 'df', 'sf', 'pol', 'pol_mask']}
    tr, va, te = data['tr_idx'], data['va_idx'], data['te_idx']
    labels = (data['labels3'] > 0).astype(np.int64)

    mn = data['rf'][tr].min(0); rg = data['rf'][tr].max(0) - mn; rg[rg < 1e-10] = 1.0
    data['rf_n'] = (data['rf'] - mn) / rg
    dmn = data['df'][tr].min(); drg = data['df'][tr].max() - dmn
    drg = drg if drg > 0 else 1.0
    data['df_n'] = (data['df'] - dmn) / drg
    smn = data['sf'][tr].min(0); srg = data['sf'][tr].max(0) - smn; srg[srg < 1e-10] = 1.0
    data['sf_n'] = (data['sf'] - smn) / srg
    norm_params = {'raw_feat_min': mn.tolist(), 'raw_feat_range': rg.tolist(),
                   'dist_min': float(dmn), 'dist_range': float(drg),
                   'spec_min': smn.tolist(), 'spec_range': srg.tolist()}

    eid = data['event_id']
    def group(idx):
        d = {}
        for i in idx:
            d.setdefault(int(eid[i]), []).append(int(i))
        return d
    ev_tr, ev_va, ev_te = group(tr), group(va), group(te)
    print(f"  #events: Tr={len(ev_tr)}, Va={len(ev_va)}, Te={len(ev_te)}")
    print(f"  #samples: Tr={len(tr)}, Va={len(va)}, Te={len(te)}")

    ev_label = {e: labels[v[0]] for e, v in ev_tr.items()}
    ev_y_tr = np.array([ev_label[e] for e in sorted(ev_tr.keys())])
    cc = np.bincount(ev_y_tr)
    print(f"  train event classes: natural={cc[0]}, non-natural={cc[1]}")
    inv = 1.0 / cc.astype(np.float64)
    alpha_norm = inv / inv.mean()
    sampler = WeightedRandomSampler((1.0 / cc)[ev_y_tr], len(ev_y_tr), replacement=True)

    collate = make_collate(data, labels, data['mags'])
    train_loader = DataLoader(EventDataset(ev_tr, True), batch_size=BATCH_EVENTS,
                              sampler=sampler, collate_fn=collate, num_workers=0)

    model = SeismicModel().to(DEVICE)
    print(f"[2] parameters {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model, EMA_DECAY)
    ema_model = SeismicModel().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warm = LambdaLR(optimizer, lambda e: (e + 1) / WARMUP_EPOCHS if e < WARMUP_EPOCHS else 1.0)
    cos = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.01)
    crit_pol = FocalLossMultiClass(alpha=None, gamma=2.0)
    crit_type = FocalLossMultiClass(alpha=torch.FloatTensor(alpha_norm).to(DEVICE), gamma=2.0)
    crit_mag = nn.SmoothL1Loss(beta=0.5)

    print(f"[3] Training (W_ST={W_ST}, event-batch={BATCH_EVENTS}, K_max={K_MAX}, "
          f"station-drop={STATION_DROP}, patience={PATIENCE}) ...")
    best_sel, best_epoch, best_state = -1.0, 0, None
    no_improve, history = 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl, nb = 0.0, 0
        t_ep = time.time()
        for batch in train_loader:
            pol_o, type_o, mag_o, st_type_o, _, ty, mg, py, pm, ev_idx = forward_batch(model, batch, True)
            is_eq = pm > 0.5
            pol_loss = crit_pol(pol_o[is_eq], py[is_eq]) if is_eq.sum() > 0 else torch.tensor(0.0, device=DEVICE)
            st_y = ty[ev_idx]  # station label = event label broadcast
            loss = (W_POL * pol_loss + W_TYPE * crit_type(type_o, ty)
                    + W_MAG * crit_mag(mag_o, mg) + W_ST * crit_type(st_type_o, st_y))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ema.update(model)
            tl += loss.item(); nb += 1
            if nb % 100 == 0:
                print(f"    [ep{epoch} batch {nb}/{len(train_loader)}] {time.time()-t_ep:.0f}s", flush=True)
        if epoch <= WARMUP_EPOCHS: warm.step()
        else: cos.step()

        # val (EMA weights): event-level selection + station-level F1 monitoring
        ema_model.load_state_dict(ema.shadow)
        ov = infer_events(ema_model, ev_va, data)
        yv_ev = np.array([labels[ev_va[e][0]] for e in ov['eids']])
        ch, n_feas, _ = optimize_threshold(ov['type_logits'], yv_ev, target=0.90)
        sel = ch['macroF1']
        va_station_order = ov['station_idx']
        pm_v = data['pol_mask'][va_station_order] > 0.5
        pol_pred = ov['pol_logits'].argmax(1)
        pol_acc = float((pol_pred[pm_v] == data['pol'][va_station_order][pm_v]).mean()) if pm_v.sum() else 0.0
        mag_mae = float(np.mean(np.abs(ov['mag'] - data['mags'][[ev_va[e][0] for e in ov['eids']]])))
        # station-level F1 (monitoring only, not used for selection)
        ch_st, _, _ = optimize_threshold(ov['st_type_logits'], labels[va_station_order], target=0.90)
        history.append({'epoch': epoch, 'train_loss': tl / max(nb, 1), 'val_sel_macroF1': sel,
                        'val_feasible_090': n_feas, 'val_b': ch['b'],
                        'val_pol_acc': pol_acc, 'val_mag_mae': mag_mae,
                        'val_station_macroF1': ch_st['macroF1']})
        if epoch % 5 == 0 or epoch <= 3:
            print(f"  Ep{epoch:3d} loss={tl/max(nb,1):.4f} | val-event-macroF1={sel:.4f} "
                  f"feas0.9={n_feas} b={ch['b']:+.2f} pol={pol_acc:.4f} mae={mag_mae:.4f} "
                  f"stF1={ch_st['macroF1']:.4f} | {time.time()-t0:.0f}s", flush=True)
        if sel > best_sel and pol_acc >= POL_FLOOR:
            best_sel, best_epoch = sel, epoch
            best_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop: best=Ep{best_epoch} sel={best_sel:.4f}")
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in ema.shadow.items()}
        best_epoch = -1
    model.load_state_dict(best_state); model.to(DEVICE)
    print(f"\nTraining done: best_epoch={best_epoch}, val event-level macroF1={best_sel:.4f}")

    # ===== Final evaluation =====
    ov = infer_events(model, ev_va, data)
    yv_ev = np.array([labels[ev_va[e][0]] for e in ov['eids']])
    ch, n_feas, _ = optimize_threshold(ov['type_logits'], yv_ev, target=0.90)
    b = ch['b']
    print(f"val event threshold: b={b:+.2f}, 0.9 feasible points={n_feas}")
    va_st_order = ov['station_idx']
    ch_st, n_feas_st, _ = optimize_threshold(ov['st_type_logits'], labels[va_st_order], target=0.90)
    b_st = ch_st['b']
    print(f"val station threshold: b={b_st:+.2f} (feasible points={n_feas_st}, macroF1={ch_st['macroF1']:.4f})")

    print("[4] test inference: event-level + station-independent + MC Dropout ...")
    ot = infer_events(model, ev_te, data)
    ot_mc = infer_events(model, ev_te, data, mc_passes=MC_T)
    ot_st = infer_stations(model, te, data)

    yt_ev = np.array([labels[ev_te[e][0]] for e in ot['eids']])
    mag_ev_true = np.array([data['mags'][ev_te[e][0]] for e in ot['eids']])

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, f'{TAG}.npz'),
        val_eids=ov['eids'], val_type_logits=ov['type_logits'],
        val_pol_logits=ov['pol_logits'], val_mag=ov['mag'],
        val_st_type_logits=ov['st_type_logits'], val_station_idx=ov['station_idx'],
        val_labels=yv_ev, val_b=float(b), val_b_station=float(b_st),
        test_eids=ot['eids'], test_type_logits=ot['type_logits'],
        test_pol_logits=ot['pol_logits'], test_mag=ot['mag'],
        test_labels=yt_ev, test_mag_true=mag_ev_true,
        test_type_probs_mc=ot_mc['type_probs_mc'], test_mag_mc=ot_mc['mag_mc'],
        test_station_type_logits=ot_st['st_type_logits'],
        test_station_type_logits_k1=ot_st['k1_type_logits'],
        test_station_pol_logits=ot_st['pol_logits'],
        test_idx=te,
        station_event_row=ot['station_event_row'],
    )
    torch.save({'model_state': best_state, 'norm_params': norm_params,
                'best_epoch': best_epoch, 'best_bias': float(b),
                'best_bias_station': float(b_st),
                'alpha': alpha_norm.tolist(), 'history': history},
               os.path.join(OUT_DIR, f'model_{TAG}.pt'))
    with open(os.path.join(OUT_DIR, f'{TAG}_train_summary.json'), 'w') as f:
        json.dump({'seed': SEED, 'best_epoch': best_epoch, 'best_val_macroF1': best_sel,
                   'val_b': float(b), 'val_b_station': float(b_st),
                   'val_feasible_090': n_feas, 'W_ST': W_ST,
                   'history': history}, f, indent=2, default=float)
    print(f"Saved {TAG}.npz, model_{TAG}.pt | total time {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
