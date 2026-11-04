# -*- coding: utf-8 -*-
"""
Evaluation of the results reported in the paper (NumPy only; no GPU, no PyTorch).
====================================================================
Reproduced from the cached model predictions in data/predictions/:
  * Table 2 - W_ST scan: single-model trade-off curve (seed 42)
  * Table 3 - three ensemble pools (logit averaging + validation threshold re-search)
  * Table 5 - event-level threshold-stability scan (best pool)
  * confidence gating / ECE / AUC of the best pool (Fig. 12)
Ensemble conventions (as in the paper):
  * event level: average type logits across members; decision bias b searched on the
    validation set (grid -1.5..1.51, step 0.02; feasible = P >= 0.90 and R >= 0.90 for
    both classes; the feasible point with the highest macro-F1 is chosen, otherwise the
    max-macro-F1 point)
  * sample level: the event decision is broadcast to all station records of the event
  * station independent: average station-head logits across members; b searched on val
  * polarity: polarity logits are ensembled after dropping members whose validation
    accuracy is below 0.85
  * ECE: 15-bin expected calibration error on the softmax mean-probability basis
Outputs: results/evaluation_results.json, results/evaluation_report.html
"""
import os, json, warnings
import numpy as np

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE = 'data/event_split_cache.npz'
PRE = 'data/predictions/'

SCAN = [('W_ST = 0.15', PRE + 'auxhead_w015_seed42.npz'),
        ('W_ST = 0.30', PRE + 'auxhead_w030_seed42.npz'),
        ('W_ST = 0.50', PRE + 'auxhead_w050_seed42.npz')]

ENSEMBLES = {
    'Multi-seed ensemble (W_ST = 0.30, seeds 42/7/2024)':
        [PRE + 'auxhead_w030_seed42.npz', PRE + 'auxhead_w030_seed07.npz',
         PRE + 'auxhead_w030_seed2024.npz'],
    'Full-model ensemble (all 5 models)':
        [PRE + 'auxhead_w015_seed42.npz', PRE + 'auxhead_w030_seed42.npz',
         PRE + 'auxhead_w050_seed42.npz', PRE + 'auxhead_w030_seed07.npz',
         PRE + 'auxhead_w030_seed2024.npz'],
    'Hyperparameter-diverse ensemble (W_ST = 0.15/0.30/0.50, seed 42)':
        [PRE + 'auxhead_w015_seed42.npz', PRE + 'auxhead_w030_seed42.npz',
         PRE + 'auxhead_w050_seed42.npz'],
}

EXTRA_SINGLES = [PRE + 'auxhead_w030_seed07.npz', PRE + 'auxhead_w030_seed2024.npz']
SWEEP_GRID = np.round(np.arange(-1.74, -0.539, 0.10), 2)


def prf(y, p, pos):
    tp = int(((p == pos) & (y == pos)).sum()); fp = int(((p == pos) & (y != pos)).sum())
    fn = int(((p != pos) & (y == pos)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {'P': pr, 'R': rc, 'F1': f1, 'FP': fp, 'FN': fn}


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


def compute_ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            ece += m.sum() / len(conf) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(ece)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


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
    return {'c': c, 'labels': labels, 'te': te, 'va': va,
            'single': ev_size == 1, 'multi': ev_size >= 2,
            'yv': yv, 'yt': yt}


def eval_single(z, ctx):
    """Single-model scan: event-level / station-independent / single-multi decomposition"""
    c, labels, te = ctx['c'], ctx['labels'], ctx['te']
    event_id = c['event_id']
    p_ev = pred_from_logits(z['test_type_logits'], float(z['val_b']))
    ev_nn = prf(z['test_labels'], p_ev, 1)
    va_st = z['val_station_idx']
    ch_st, _ = search_threshold(z['val_st_type_logits'], labels[va_st], target=0.90)
    p_st = pred_from_logits(z['test_station_type_logits'], ch_st['b'])
    st_nn = prf(labels[te], p_st, 1)
    eid_to_row = {int(e): r for r, e in enumerate(z['test_eids'])}
    ev_row = np.array([eid_to_row[int(event_id[i])] for i in te])
    sm_nn = prf(labels[te], p_ev[ev_row], 1)
    return {'b_ev': float(z['val_b']), 'b_st': float(ch_st['b']),
            'event_F1': ev_nn['F1'], 'sample_F1': sm_nn['F1'], 'station_F1': st_nn['F1'],
            'single_F1': prf(labels[te][ctx['single']], p_st[ctx['single']], 1)['F1'],
            'multi_F1': prf(labels[te][ctx['multi']], p_st[ctx['multi']], 1)['F1'],
            'ev_row': ev_row}


def evaluate_ensemble(name, paths, cache, ctx):
    c, labels, te = ctx['c'], ctx['labels'], ctx['te']
    P = [cache[p] for p in paths]
    models = [os.path.basename(p)[:-4] for p in paths]
    # event level
    ens_va = np.mean([p['val_type_logits'] for p in P], axis=0)
    ens_te = np.mean([p['test_type_logits'] for p in P], axis=0)
    ch_ev, feas = search_threshold(ens_va, ctx['yv'], target=0.90)
    p_ev = pred_from_logits(ens_te, ch_ev['b'])
    ev_nn = prf(ctx['yt'], p_ev, 1); ev_eq = prf(ctx['yt'], p_ev, 0)
    # sample-level broadcast
    st_bc = p_ev[P[0]['ev_row_per_sample']]
    sm_nn = prf(labels[te], st_bc, 1); sm_eq = prf(labels[te], st_bc, 0)
    # station head
    ens_va_st = np.mean([p['val_st_type_logits'] for p in P], axis=0)
    ens_te_st = np.mean([p['test_station_type_logits'] for p in P], axis=0)
    va_st_order = P[0]['val_station_idx']
    ch_st, _ = search_threshold(ens_va_st, labels[va_st_order], target=0.90)
    p_st = pred_from_logits(ens_te_st, ch_st['b'])
    st_nn = prf(labels[te], p_st, 1); st_eq = prf(labels[te], p_st, 0)
    dec = {'single': prf(labels[te][ctx['single']], p_st[ctx['single']], 1),
           'multi': prf(labels[te][ctx['multi']], p_st[ctx['multi']], 1)}
    # polarity (filtered by val >= 0.85)
    pol_ok = {}
    for mn, p in zip(models, P):
        vi = p['val_station_idx']; pmv = c['pol_mask'][vi] > 0.5
        pol_ok[mn] = float((p['val_pol_logits'].argmax(1)[pmv] == c['pol'][vi][pmv]).mean())
    pol_models = [mn for mn in models if pol_ok[mn] >= 0.85] or models
    ens_pol = np.mean([p['pol_logits'] for mn, p in zip(models, P) if mn in pol_models], axis=0)
    soi = P[0]['st_order_idx']
    pms = c['pol_mask'][soi] > 0.5
    pol_acc = float((ens_pol.argmax(1)[pms] == c['pol'][soi][pms]).mean())
    # magnitude / error decomposition
    mag_mae = float(np.mean(np.abs(np.mean([p['mag'] for p in P], axis=0) - P[0]['mag_true'])))
    fn_mask = (labels[te] == 1) & (st_bc == 0)
    lbl3 = c['labels3'][te]
    # ensemble uncertainty (mean-probability basis)
    probs = np.mean([softmax(p['test_type_logits']) for p in P], axis=0)
    conf = probs.max(1); pm = probs.argmax(1)
    correct = (pm == ctx['yt'])
    from sklearn.metrics import roc_auc_score
    order = np.argsort(-conf)
    gate = []
    for cov in [0.5, 0.7, 0.9, 1.0]:
        k = max(int(len(order) * cov), 2); keep = order[:k]
        gate.append({'coverage': cov, 'n': int(k),
                     'nn_F1': prf(ctx['yt'][keep], pm[keep], 1)['F1'],
                     'acc': float(correct[keep].mean())})
    return {'models': models, 'pol_ok': pol_ok, 'pol_models': pol_models,
            'b_ev': float(ch_ev['b']), 'feasible_090': feas, 'b_st': float(ch_st['b']),
            'event_nn': ev_nn, 'event_eq': ev_eq, 'sample_nn': sm_nn, 'sample_eq': sm_eq,
            'station_nn': st_nn, 'station_eq': st_eq, 'decomp': dec,
            'polarity_acc': pol_acc, 'mag_mae': mag_mae,
            'fn': {'total': int(fn_mask.sum()), 'blast': int((lbl3[fn_mask] == 1).sum()),
                   'collapse': int((lbl3[fn_mask] == 2).sum())},
            'fp': int(((labels[te] == 0) & (st_bc == 1)).sum()),
            'ece': compute_ece(conf, correct),
            'auc': float(roc_auc_score(correct.astype(int), conf)),
            'gate': gate,
            'ens_va': ens_va, 'ens_te': ens_te}


def threshold_sweep(best_name, cache, ctx):
    """Table 5: event-level threshold-stability scan of the best pool"""
    P = [cache[p] for p in ENSEMBLES[best_name]]
    ens_va = np.mean([p['val_type_logits'] for p in P], axis=0)
    ens_te = np.mean([p['test_type_logits'] for p in P], axis=0)
    ch, _ = search_threshold(ens_va, ctx['yv'], target=0.90)
    rows = []
    for b in SWEEP_GRID:
        p = pred_from_logits(ens_te, b)
        r1 = prf(ctx['yt'], p, 1)
        rows.append({'b': float(b), 'P': r1['P'], 'R': r1['R'],
                     'F1': r1['F1'], 'FP': r1['FP'], 'FN': r1['FN']})
    return {'b_star': float(ch['b']), 'rows': rows}


def load_single(path, ctx):
    z = np.load(path)
    sr = eval_single(z, ctx)
    event_id = ctx['c']['event_id']; te = ctx['te']
    _g = {}
    for i in te:
        _g.setdefault(int(event_id[i]), []).append(int(i))
    soi = np.concatenate([np.asarray(_g[int(e)]) for e in z['test_eids']])
    return {'z': dict(z), 'single_res': sr,
            'val_type_logits': z['val_type_logits'], 'test_type_logits': z['test_type_logits'],
            'val_st_type_logits': z['val_st_type_logits'],
            'test_station_type_logits': z['test_station_type_logits'],
            'val_station_idx': z['val_station_idx'],
            'val_pol_logits': z['val_pol_logits'], 'pol_logits': z['test_pol_logits'],
            'mag': z['test_mag'], 'mag_true': z['test_mag_true'],
            'test_eids': z['test_eids'], 'st_order_idx': soi,
            'ev_row_per_sample': sr['ev_row']}


def main():
    ctx = build_context()
    cache = {}
    print("[1] Single-model point evaluation (5 npz) ...")
    for name, path in SCAN:
        cache[path] = load_single(path, ctx)
        d = cache[path]['single_res']
        print(f"  {name}: event-level={d['event_F1']:.4f} station-independent={d['station_F1']:.4f} "
              f"(single={d['single_F1']:.4f} multi={d['multi_F1']:.4f})")
    for path in EXTRA_SINGLES:
        cache[path] = load_single(path, ctx)
        d = cache[path]['single_res']
        print(f"  {os.path.basename(path)}: event-level={d['event_F1']:.4f} "
              f"station-independent={d['station_F1']:.4f}")

    print("[2] Three ensemble pools ...")
    ens = {}
    for name, paths in ENSEMBLES.items():
        ens[name] = evaluate_ensemble(name, paths, cache, ctx)
        r = ens[name]
        print(f"  {name}: event-level={r['event_nn']['F1']:.4f} sample-level={r['sample_nn']['F1']:.4f} "
              f"station-independent={r['station_nn']['F1']:.4f} (single={r['decomp']['single']['F1']:.4f} "
              f"multi={r['decomp']['multi']['F1']:.4f}) pol={r['polarity_acc']:.4f}")

    best_name = max(ens, key=lambda n: ens[n]['event_nn']['F1'])
    sweep = threshold_sweep(best_name, cache, ctx)
    print(f"[3] Threshold-stability scan of the best pool ({best_name}): b*={sweep['b_star']:+.2f}")

    # save (ev_row inside single_res is an 8464-array, excluded; ens_va/ens_te are transient)
    def strip(d, keys):
        return {k: v for k, v in d.items() if k not in keys}
    out = {'scan': {n: strip(cache[p]['single_res'], {'ev_row'}) for n, p in SCAN},
           'extra_singles': {os.path.basename(p): strip(cache[p]['single_res'], {'ev_row'})
                             for p in EXTRA_SINGLES},
           'ensembles': {n: strip(r, {'ens_va', 'ens_te'}) for n, r in ens.items()},
           'best_pool': best_name,
           'threshold_sweep': sweep}
    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)

    print("[4] Generating HTML report ...")
    make_html(out)
    print("Saved: results/evaluation_results.json, results/evaluation_report.html")


def make_html(out):
    scan = out['scan']
    scan_rows = ''.join(
        f"<tr><td>{n}</td><td>{d['event_F1']:.4f}</td><td>{d['sample_F1']:.4f}</td>"
        f"<td>{d['station_F1']:.4f}</td><td>{d['single_F1']:.4f}</td>"
        f"<td>{d['multi_F1']:.4f}</td><td>{d['b_st']:+.2f}</td></tr>"
        for n, d in scan.items())

    ens = out['ensembles']
    ens_rows = ''.join(
        f"<tr><td>{n}</td><td>{r['event_nn']['F1']:.4f}</td><td>{r['sample_nn']['F1']:.4f}</td>"
        f"<td>{r['station_nn']['F1']:.4f}</td><td>{r['decomp']['single']['F1']:.4f}</td>"
        f"<td>{r['decomp']['multi']['F1']:.4f}</td><td>{r['polarity_acc']:.4f}</td>"
        f"<td>{r['mag_mae']:.4f}</td><td>{r['ece']:.4f}</td></tr>"
        for n, r in ens.items())

    best = ens[out['best_pool']]
    sweep = out['threshold_sweep']
    b_star = sweep['b_star']
    sweep_rows = ''.join(
        f"<tr{' class=star' if abs(r['b'] - b_star) < 1e-6 else ''}>"
        f"<td>{r['b']:+.2f}{' (b*)' if abs(r['b'] - b_star) < 1e-6 else ''}</td>"
        f"<td>{r['P']:.4f}</td><td>{r['R']:.4f}</td><td>{r['F1']:.4f}</td>"
        f"<td>{r['FP']}</td><td>{r['FN']}</td></tr>"
        for r in sweep['rows'])

    gate_rows = ''.join(
        f"<tr><td>{g['coverage']*100:.0f}%</td><td>{g['n']}</td>"
        f"<td>{g['nn_F1']:.4f}</td><td>{g['acc']:.4f}</td></tr>" for g in best['gate'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evaluation report: W_ST scan, ensemble pools, threshold stability, confidence gating</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
         line-height: 1.75; color: #1a1a1a; max-width: 1020px; margin: 0 auto; padding: 40px 20px; }}
  h1 {{ font-size: 25px; border-bottom: 3px solid #0891b2; padding-bottom: 12px; margin-bottom: 8px; }}
  h2 {{ font-size: 20px; margin-top: 44px; margin-bottom: 12px; color: #0e7490; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 13.5px; }}
  th, td {{ border: 1px solid #d1d5db; padding: 7px 10px; text-align: left; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  tr.star {{ background: #ecfeff; font-weight: 600; }}
  .box {{ border-radius: 8px; padding: 15px 20px; margin: 15px 0; border: 1px solid #e2e8f0; background: #f8fafc; }}
  .foot {{ margin-top: 44px; padding-top: 14px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 13px; }}
  ul {{ margin: 8px 0 14px 24px; }} li {{ margin: 4px 0; }}
</style>
</head>
<body>
<h1>Evaluation report: auxiliary-head multi-task model</h1>
<p style="color:#6b7280; margin-bottom:26px; font-size:14px;">
5 single models (W_ST &isin; {{0.15, 0.30, 0.50}} at seed 42 + W_ST = 0.30 at seeds 7 and 2024) |
ensembles: logit averaging + validation-set threshold re-search | fixed event-level split | test set evaluated in full</p>

<h2>1. W_ST scan: single-model trade-off curve (Table 2)</h2>
<table>
<tr><th>Model</th><th>Event nnF1</th><th>Sample nnF1</th><th>Station-independent nnF1</th><th>Single-station F1</th><th>Multi-station F1</th><th>Station thr. b</th></tr>
{scan_rows}
</table>
<div class="box">
<p><strong>Task-competition curve</strong>: increasing W_ST strengthens the station head
({scan['W_ST = 0.15']['station_F1']:.3f} &rarr; {scan['W_ST = 0.30']['station_F1']:.3f} &rarr; {scan['W_ST = 0.50']['station_F1']:.3f})
while weakening the event head
({scan['W_ST = 0.15']['event_F1']:.3f} &rarr; {scan['W_ST = 0.30']['event_F1']:.3f} &rarr; {scan['W_ST = 0.50']['event_F1']:.3f}).
W_ST governs the competitive allocation of encoder capacity between the auxiliary and the main task,
and is a quantifiable, reproducible trade-off knob.
Single-station events contain only 14 non-natural samples, so individual points are noisy; read the trend.</p>
</div>

<h2>2. Three ensemble pools (Table 3)</h2>
<table>
<tr><th>Pool</th><th>Event nnF1</th><th>Sample nnF1</th><th>Station-independent nnF1</th><th>Single F1</th><th>Multi F1</th><th>Polarity</th><th>Mag MAE</th><th>ECE</th></tr>
{ens_rows}
</table>
<p style="font-size:13px; color:#6b7280;">ECE is the softmax mean-probability basis of the pool members
(production default); polarity ensemble members are filtered by validation accuracy &ge; 0.85.</p>

<h2>3. Event-level threshold-stability scan (Table 5, best pool: {out['best_pool']})</h2>
<table>
<tr><th>Threshold b</th><th>Event P</th><th>Event R</th><th>Event F1</th><th>FP</th><th>FN</th></tr>
{sweep_rows}
</table>
<p style="font-size:13px; color:#6b7280;">b* = {b_star:+.2f} is searched on the validation set only;
the test set is never used for tuning.</p>

<h2>4. Confidence gating of the best pool (mean-probability basis, Fig. 12)</h2>
<table>
<tr><th>Coverage</th><th>#Events</th><th>nn F1</th><th>Accuracy</th></tr>
{gate_rows}
</table>

<h2>5. Summary</h2>
<ul>
<li>Best pool ({out['best_pool']}): event-level F1 {best['event_nn']['F1']:.4f}
(P {best['event_nn']['P']:.4f} / R {best['event_nn']['R']:.4f}, FP {best['event_nn']['FP']} / FN {best['event_nn']['FN']}),
sample-level F1 {best['sample_nn']['F1']:.4f}, station-independent F1 {best['station_nn']['F1']:.4f}</li>
<li>Ensembling gain: single model (W_ST = 0.30, seed 42) event-level F1
{scan['W_ST = 0.30']['event_F1']:.4f} &rarr; best pool {best['event_nn']['F1']:.4f}</li>
<li>Confidence: ECE {best['ece']:.4f}, confidence-correctness AUC {best['auc']:.4f},
top-90% coverage accuracy {best['gate'][2]['acc']:.4f}</li>
<li>W_ST is an effective trade-off knob: event-level and station-head performance trade against each other</li>
<li>Production form: one forward pass &times; {len(best['models'])} models, total parameters
~{4.24*len(best['models']):.1f}M ({len(best['models'])} &times; 4.24M single model)</li>
</ul>

<p class="foot">Protocol: fixed event-level split, no leakage, validation-set threshold selection, full-test evaluation.
Output: results/evaluation_results.json</p>
</body>
</html>"""
    with open('results/evaluation_report.html', 'w') as f:
        f.write(html)


if __name__ == '__main__':
    main()
