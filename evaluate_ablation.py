# -*- coding: utf-8 -*-
"""
Tables 6-7: controlled comparison and ablation ladder (NumPy only; no GPU, no PyTorch).
====================================================================
Settings (manuscript terminology):
  * 'Attention only'                  - station-attention model without the auxiliary
                                         station-level head (archived single-model
                                         evaluation, data/archived/
                                         attention_only_seed42.json, under that run's own
                                         validation-set thresholds)
  * 'Auxiliary head, W_ST = 0.30'     - the paper's model, single model (seed 42)
  * 'Auxiliary head, W_ST = 0'        - same architecture, auxiliary head kept but
                                         station-level supervision removed
  * 'Auxiliary head, single-task'     - only the event-level type head is supervised
  * 'Multi-seed ensemble'             - logit averaging of the three W_ST = 0.30 models
                                         (seeds 42/7/2024)
All rows except 'Attention only' are recomputed under one unified protocol (identical
threshold search and metric definitions to evaluate.py): decision bias searched on the
validation set (grid -1.5..1.51, step 0.02, target P/R >= 0.90 for both classes), test
set evaluated in full; sample level = event decision broadcast to member stations;
station F1 = station-independent convention.
Outputs: results/ablation_results.json
"""
import os, json, warnings
import numpy as np

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE = 'data/event_split_cache.npz'
PRE = 'data/predictions/'
ARCH_ATTENTION_ONLY = 'data/archived/attention_only_seed42.json'

MULTI_SEED = [PRE + 'auxhead_w030_seed42.npz', PRE + 'auxhead_w030_seed07.npz',
              PRE + 'auxhead_w030_seed2024.npz']


def prf(y, p, pos):
    tp = int(((p == pos) & (y == pos)).sum()); fp = int(((p == pos) & (y != pos)).sum())
    fn = int(((p != pos) & (y == pos)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {'P': pr, 'R': rc, 'F1': f1, 'FP': fp, 'FN': fn, 'TP': tp}


def pred_from_logits(logits, b):
    return ((logits[:, 1] + b) > logits[:, 0]).astype(int)


def search_threshold(logits, labels, grid=np.arange(-1.5, 1.51, 0.02), target=0.90):
    best_any, feasible = None, []
    for b in grid:
        p = pred_from_logits(logits, b)
        r1 = prf(labels, p, 1); r0 = prf(labels, p, 0)
        macro = (r0['F1'] + r1['F1']) / 2
        if r1['P'] >= target and r1['R'] >= target and r0['P'] >= target and r0['R'] >= target:
            feasible.append({'b': float(b), 'macroF1': macro})
        if best_any is None or macro > best_any['macroF1']:
            best_any = {'b': float(b), 'macroF1': macro}
    return (max(feasible, key=lambda r: r['macroF1']) if feasible else best_any), len(feasible)


def build_context():
    c = np.load(CACHE, allow_pickle=True)
    event_id = c['event_id']; labels = (c['labels3'] > 0).astype(int)
    te = c['te_idx']; va = c['va_idx']
    ue, counts = np.unique(event_id[te], return_counts=True)
    size_map = dict(zip(ue.tolist(), counts.tolist()))
    ev_size = np.array([size_map[int(e)] for e in event_id[te]])

    def group(idx):
        d = {}
        for i in idx:
            d.setdefault(int(event_id[i]), []).append(int(i))
        return d
    ev_va, ev_te = group(va), group(te)
    yv = np.array([labels[ev_va[e][0]] for e in sorted(ev_va.keys())])
    yt = np.array([labels[ev_te[e][0]] for e in sorted(ev_te.keys())])
    return {'c': c, 'labels': labels, 'te': te,
            'single': ev_size == 1, 'multi': ev_size >= 2,
            'ev_size': ev_size,
            'yv': yv, 'yt': yt}


def evaluate_npz(z, name, ctx):
    """Unified protocol (identical to evaluate.py's ensemble/single conventions)"""
    c, labels, te = ctx['c'], ctx['labels'], ctx['te']
    event_id = c['event_id']
    ch_ev, _ = search_threshold(z['val_type_logits'], ctx['yv'], target=0.90)
    p_ev = pred_from_logits(z['test_type_logits'], ch_ev['b'])
    ev_nn = prf(ctx['yt'], p_ev, 1)
    eid_to_row = {int(e): r for r, e in enumerate(z['test_eids'])}
    ev_row = np.array([eid_to_row[int(event_id[i])] for i in te])
    st_bc = p_ev[ev_row]
    sm_nn = prf(labels[te], st_bc, 1)
    va_st = z['val_station_idx']
    ch_st, _ = search_threshold(z['val_st_type_logits'], labels[va_st], target=0.90)
    p_st = pred_from_logits(z['test_station_type_logits'], ch_st['b'])
    st_nn = prf(labels[te], p_st, 1)
    ev_size = ctx['ev_size']
    single = ev_size == 1; multi = ev_size >= 2
    dec = {'single': prf(labels[te][single], p_st[single], 1),
           'multi': prf(labels[te][multi], p_st[multi], 1)}
    mag_mae = float(np.mean(np.abs(z['test_mag'] - z['test_mag_true'])))
    soi = z['test_idx']
    pm = c['pol_mask'][soi] > 0.5
    pol_acc = float((z['test_station_pol_logits'].argmax(1)[pm] == c['pol'][soi][pm]).mean())
    return {'setting': name, 'source': 'recomputed under the unified protocol from cached predictions',
            'b_event': float(ch_ev['b']), 'b_station': float(ch_st['b']),
            'event_nn': ev_nn, 'sample_nn': sm_nn, 'station_nn': st_nn,
            'decomp': dec, 'mag_mae_event': mag_mae, 'polarity_acc': pol_acc,
            'fp_event': ev_nn['FP']}


def main():
    ctx = build_context()
    rows = []

    # ---- 1. Attention only: archived single-model evaluation ----
    j = json.load(open(ARCH_ATTENTION_ONLY, encoding='utf-8'))
    d = j['model']
    rows.append({
        'setting': 'Attention only',
        'source': 'archived single-model evaluation (that run\'s validation-set thresholds)',
        'event_nn': d['event_level']['nn'], 'sample_nn': d['sample_broadcast']['nn'],
        'station_nn': d['station_independent']['nn'], 'decomp': None,
        'mag_mae_event': float(d['mag_mae_event']),
        'polarity_acc': float(d['polarity_acc']),
        'fp_event': int(d['event_level']['nn']['FP']),
        'b_event': None, 'b_station': None,
    })

    # ---- 2-4. unified-protocol rows ----
    for name, path in [('Auxiliary head, W_ST = 0.30', PRE + 'auxhead_w030_seed42.npz'),
                       ('Auxiliary head, W_ST = 0', PRE + 'auxhead_w0_seed42.npz'),
                       ('Auxiliary head, single-task', PRE + 'single_task_seed42.npz')]:
        rows.append(evaluate_npz(np.load(path), name, ctx))

    # ---- 5. Multi-seed ensemble ----
    P = [np.load(p) for p in MULTI_SEED]
    ens_va = np.mean([p['val_type_logits'] for p in P], axis=0)
    ens_te = np.mean([p['test_type_logits'] for p in P], axis=0)
    ch_ev, _ = search_threshold(ens_va, ctx['yv'], target=0.90)
    p_ev = pred_from_logits(ens_te, ch_ev['b'])
    ev_nn = prf(ctx['yt'], p_ev, 1)
    eid_to_row = {int(e): r for r, e in enumerate(P[0]['test_eids'])}
    ev_row = np.array([eid_to_row[int(ctx['c']['event_id'][i])] for i in ctx['te']])
    st_bc = p_ev[ev_row]
    sm_nn = prf(ctx['labels'][ctx['te']], st_bc, 1)
    ens_va_st = np.mean([p['val_st_type_logits'] for p in P], axis=0)
    ens_te_st = np.mean([p['test_station_type_logits'] for p in P], axis=0)
    va_st = P[0]['val_station_idx']
    ch_st, _ = search_threshold(ens_va_st, ctx['labels'][va_st], target=0.90)
    p_st = pred_from_logits(ens_te_st, ch_st['b'])
    st_nn = prf(ctx['labels'][ctx['te']], p_st, 1)
    ev_size = ctx['ev_size']
    single = ev_size == 1; multi = ev_size >= 2
    dec = {'single': prf(ctx['labels'][ctx['te']][single], p_st[single], 1),
           'multi': prf(ctx['labels'][ctx['te']][multi], p_st[multi], 1)}
    mag_mae = float(np.mean(np.abs(np.mean([p['test_mag'] for p in P], axis=0)
                                   - P[0]['test_mag_true'])))
    # polarity: ensemble the polarity logits after dropping members with val accuracy < 0.85
    c = ctx['c']
    pol_ok = []
    for p in P:
        vi = p['val_station_idx']; pmv = c['pol_mask'][vi] > 0.5
        pol_ok.append(float((p['val_pol_logits'].argmax(1)[pmv] == c['pol'][vi][pmv]).mean()))
    keep = [i for i, a in enumerate(pol_ok) if a >= 0.85] or list(range(len(P)))
    ens_pol = np.mean([P[i]['test_station_pol_logits'] for i in keep], axis=0)
    soi = P[0]['test_idx']
    pms = c['pol_mask'][soi] > 0.5
    pol_acc = float((ens_pol.argmax(1)[pms] == c['pol'][soi][pms]).mean())
    rows.append({'setting': 'Multi-seed ensemble',
                 'source': 'recomputed under the unified protocol (3-model logit averaging)',
                 'b_event': float(ch_ev['b']), 'b_station': float(ch_st['b']),
                 'event_nn': ev_nn, 'sample_nn': sm_nn, 'station_nn': st_nn,
                 'decomp': dec, 'mag_mae_event': mag_mae, 'polarity_acc': pol_acc,
                 'fp_event': ev_nn['FP']})

    # ---- Table 7 layout ----
    t7 = []
    for r in rows:
        t7.append({'setting': r['setting'],
                   'event_F1': r['event_nn']['F1'], 'sample_F1': r['sample_nn']['F1'],
                   'station_F1': r['station_nn']['F1'], 'fp': r['fp_event'],
                   'mae': r['mag_mae_event'], 'polarity': r['polarity_acc'],
                   'source': r['source'],
                   'event_P': r['event_nn']['P'], 'event_R': r['event_nn']['R'],
                   'b_event': r.get('b_event'), 'b_station': r.get('b_station'),
                   'decomp': r.get('decomp')})

    # ---- Table 6 layout (controlled comparison) ----
    att, aux, ens = rows[0], rows[1], rows[4]
    t6 = {'settings': ['Attention-only single model', 'Auxiliary-head single model',
                       'Multi-seed ensemble'],
          'rows': {
              'Event-level F1': [att['event_nn']['F1'], aux['event_nn']['F1'], ens['event_nn']['F1']],
              'Event-level P':  [att['event_nn']['P'], aux['event_nn']['P'], ens['event_nn']['P']],
              'Event-level R':  [att['event_nn']['R'], aux['event_nn']['R'], ens['event_nn']['R']],
              'Event-level FP': [att['fp_event'], aux['fp_event'], ens['fp_event']],
              'Sample-level F1': [att['sample_nn']['F1'], aux['sample_nn']['F1'], ens['sample_nn']['F1']],
              'Station-independent F1': [att['station_nn']['F1'], aux['station_nn']['F1'],
                                         ens['station_nn']['F1']],
              'Polarity acc': [att['polarity_acc'], aux['polarity_acc'], ens['polarity_acc']],
              'Event magnitude MAE': [att['mag_mae_event'], aux['mag_mae_event'],
                                      ens['mag_mae_event']],
          }}

    out = {
        'table7_ablation_ladder': t7,
        'table6_controlled_comparison': t6,
        'protocol': ('unified protocol: decision bias searched on the validation set '
                     '(grid -1.5..1.51, step 0.02, target P/R >= 0.90 for both classes), '
                     'test set (8464 records) evaluated in full; sample level = event '
                     'decision broadcast to member stations; station F1 = '
                     'station-independent convention'),
        'note': ("event-level/sample-level/station-independent are all non-natural F1; MAE is the "
                 "event-level magnitude MAE; polarity is the accuracy computed over the test "
                 "stations where polarity labels are available. Station F1 denotes the "
                 "station-independent convention; 'Attention only' has no auxiliary head; "
                 "W_ST = 0 keeps the auxiliary head but removes station-level supervision; "
                 "'single-task' supervises the type head only."),
    }
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_results.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)

    print("Ablation ladder (Table 7):")
    for r in t7:
        pol = '-' if r['polarity'] is None else f"{r['polarity']:.4f}"
        print(f"  {r['setting']:<30s} event={r['event_F1']:.4f} sample={r['sample_F1']:.4f} "
              f"station={r['station_F1']:.4f} FP={r['fp']} MAE={r['mae']:.4f} pol={pol}")
    print('Saved: results/ablation_results.json')


if __name__ == '__main__':
    main()
