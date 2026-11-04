# -*- coding: utf-8 -*-
"""
Ablation training for Tables 6-7 of the paper. Identical protocol to train.py
(fixed event-level split, noise augmentation + EMA, validation-set model and
threshold selection); only the task configuration changes:
====================================================================
Usage: python train_ablation.py <mode> <seed>
  mode = 'w0'     : auxiliary station-level head present but without station-level
                    supervision (W_ST = 0); the remaining tasks stay supervised
  mode = 'single' : single-task baseline: only the event-level type head is supervised
Model selection on the validation set: mode 'w0' uses event-level macro-F1 with the
polarity-accuracy floor (same as train.py); mode 'single' has no polarity task, so
macro-F1 only. Test set evaluated in full; the prediction-cache layout matches
train.py, so the outputs can be fed directly to evaluate_ablation.py.
Outputs (written to <package>/data/predictions/):
    auxhead_w0_seed{seed:02d}.npz       (mode w0)
    single_task_seed{seed:02d}.npz      (mode single)
plus model_*.pt and *_train_summary.json.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)
sys.path.insert(0, _DIR)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'w0'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
assert MODE in ('w0', 'single'), MODE

# reuse all components of train.py (constants / losses / dataset / inference / threshold search)
sys.argv = ['train.py', str(SEED), '0.0']   # W_ST only feeds train.py's output TAG
import train as T
from train import (
    DEVICE, CACHE, BATCH_EVENTS, EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP, WARMUP_EPOCHS,
    PATIENCE, EMA_DECAY, POL_FLOOR, W_POL, W_TYPE, W_MAG, K_MAX, STATION_DROP, NOISE, MC_T,
    FocalLossMultiClass, EMA, EventDataset, make_collate, forward_batch,
    infer_events, infer_stations, optimize_threshold, prf,
)
from model import SeismicModel, enable_mc_dropout

TAG = (f'auxhead_w0_seed{SEED:02d}' if MODE == 'w0' else f'single_task_seed{SEED:02d}')
OUT_DIR = T.OUT_DIR
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    t0 = time.time()
    print(f"[1] Loading cache (mode={MODE}, seed={SEED}) ...")
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

    ev_label = {e: labels[v[0]] for e, v in ev_tr.items()}
    ev_y_tr = np.array([ev_label[e] for e in sorted(ev_tr.keys())])
    cc = np.bincount(ev_y_tr)
    inv = 1.0 / cc.astype(np.float64)
    alpha_norm = inv / inv.mean()
    sampler = torch.utils.data.WeightedRandomSampler((1.0 / cc)[ev_y_tr], len(ev_y_tr), replacement=True)

    collate = make_collate(data, labels, data['mags'])
    train_loader = torch.utils.data.DataLoader(EventDataset(ev_tr, True), batch_size=BATCH_EVENTS,
                              sampler=sampler, collate_fn=collate, num_workers=0)

    model = SeismicModel().to(DEVICE)
    print(f"[2] parameters {sum(p.numel() for p in model.parameters()):,} | mode={MODE}")
    ema = EMA(model, EMA_DECAY)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / WARMUP_EPOCHS if e < WARMUP_EPOCHS else 1.0)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.01)
    crit_pol = FocalLossMultiClass(alpha=None, gamma=2.0)
    crit_type = FocalLossMultiClass(alpha=torch.FloatTensor(alpha_norm).to(DEVICE), gamma=2.0)
    crit_mag = nn.SmoothL1Loss(beta=0.5)

    use_pol = (MODE == 'w0')     # single: type task only
    use_mag = (MODE == 'w0')
    use_st = (MODE == 'w0')      # w0: no station-head supervision (ablation target); single: same
    w_st = 0.0 if (MODE in ('w0', 'single')) else T.W_ST

    print(f"[3] Training (mode={MODE}, event-batch={BATCH_EVENTS}, K_max={K_MAX}, "
          f"pol={use_pol}, mag={use_mag}, st={use_st}) ...")
    best_sel, best_epoch, best_state = -1.0, 0, None
    no_improve, history = 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl, nb = 0.0, 0
        t_ep = time.time()
        for batch in train_loader:
            pol_o, type_o, mag_o, st_type_o, _, ty, mg, py, pm, ev_idx = forward_batch(model, batch, True)
            loss = torch.zeros((), device=DEVICE)
            if use_pol:
                is_eq = pm > 0.5
                pol_loss = crit_pol(pol_o[is_eq], py[is_eq]) if is_eq.sum() > 0 else torch.tensor(0.0, device=DEVICE)
                loss = loss + W_POL * pol_loss
            loss = loss + W_TYPE * crit_type(type_o, ty)
            if use_mag:
                loss = loss + W_MAG * crit_mag(mag_o, mg)
            if use_st:
                st_y = ty[ev_idx]
                loss = loss + w_st * crit_type(st_type_o, st_y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ema.update(model)
            tl += loss.item(); nb += 1
            if nb % 100 == 0:
                print(f"    [ep{epoch} batch {nb}/{len(train_loader)}] {time.time()-t_ep:.0f}s", flush=True)
        if epoch <= WARMUP_EPOCHS: warm.step()
        else: cos.step()

        # val (EMA weights)
        m_eval = SeismicModel().to(DEVICE)
        m_eval.load_state_dict(ema.shadow)
        ov = infer_events(m_eval, ev_va, data)
        yv_ev = np.array([labels[ev_va[e][0]] for e in ov['eids']])
        ch, n_feas, _ = optimize_threshold(ov['type_logits'], yv_ev, target=0.90)
        sel = ch['macroF1']
        va_station_order = ov['station_idx']
        pm_v = data['pol_mask'][va_station_order] > 0.5
        pol_pred = ov['pol_logits'].argmax(1)
        pol_acc = float((pol_pred[pm_v] == data['pol'][va_station_order][pm_v]).mean()) if pm_v.sum() else 0.0
        mag_mae = float(np.mean(np.abs(ov['mag'] - data['mags'][[ev_va[e][0] for e in ov['eids']]])))
        ch_st, _, _ = optimize_threshold(ov['st_type_logits'], labels[va_station_order], target=0.90)
        history.append({'epoch': epoch, 'train_loss': tl / max(nb, 1), 'val_sel_macroF1': sel,
                        'val_feasible_090': n_feas, 'val_b': ch['b'],
                        'val_pol_acc': pol_acc, 'val_mag_mae': mag_mae,
                        'val_station_macroF1': ch_st['macroF1']})
        del m_eval
        if epoch % 5 == 0 or epoch <= 3:
            print(f"  Ep{epoch:3d} loss={tl/max(nb,1):.4f} | val-event-macroF1={sel:.4f} "
                  f"b={ch['b']:+.2f} pol={pol_acc:.4f} mae={mag_mae:.4f} "
                  f"stF1={ch_st['macroF1']:.4f} | {time.time()-t0:.0f}s", flush=True)
        # model selection: w0 uses macroF1 + polarity floor (same as train.py); single: macroF1 only
        cond = sel > best_sel and (pol_acc >= POL_FLOOR if use_pol else True)
        if cond:
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

    ov = infer_events(model, ev_va, data)
    yv_ev = np.array([labels[ev_va[e][0]] for e in ov['eids']])
    ch, n_feas, _ = optimize_threshold(ov['type_logits'], yv_ev, target=0.90)
    b = ch['b']
    va_st_order = ov['station_idx']
    ch_st, n_feas_st, _ = optimize_threshold(ov['st_type_logits'], labels[va_st_order], target=0.90)
    b_st = ch_st['b']
    print(f"val event threshold: b={b:+.2f} | val station threshold: b={b_st:+.2f}")

    print("[4] Test inference: event-level + station-independent + MC Dropout ...")
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
                'best_bias_station': float(b_st), 'mode': MODE,
                'alpha': alpha_norm.tolist(), 'history': history},
               os.path.join(OUT_DIR, f'model_{TAG}.pt'))
    with open(os.path.join(OUT_DIR, f'{TAG}_train_summary.json'), 'w') as f:
        json.dump({'mode': MODE, 'seed': SEED, 'best_epoch': best_epoch,
                   'best_val_macroF1': best_sel, 'val_b': float(b),
                   'val_b_station': float(b_st), 'val_feasible_090': n_feas,
                   'history': history}, f, indent=2, default=float)
    print(f"Saved {TAG}.npz, model_{TAG}.pt | total time {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
