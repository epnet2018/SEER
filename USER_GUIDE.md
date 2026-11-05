# User guide

This page describes the inputs, outputs, options and expected behaviour of the
reproduction package. Quick-start commands are in `README.md`; the dependency
list is in `requirements.txt`.

## 1. Evaluation scripts (no GPU required)

### `evaluate.py`

| item | description |
|---|---|
| Inputs | `data/predictions/*.npz` (cached logits/magnitudes of the five main models), `data/event_split_cache.npz` (anonymised labels and the fixed event-level split) |
| Outputs | `results/evaluation_results.json`, `results/evaluation_report.html` |
| Options | none (all protocol constants - threshold grid, coverage levels, ensemble pools - are fixed in the script header and match the manuscript) |
| Runtime | a few seconds on CPU; needs only `numpy` and one `scikit-learn` AUC call |
| Expected behaviour | prints the W_ST scan (Table 2), the three ensemble pools (Table 3) and the threshold-stability scan (Table 5); values match the manuscript exactly (event-level F1 0.8707 for the multi-seed ensemble) |

### `evaluate_ablation.py`

| item | description |
|---|---|
| Inputs | the same two data files, plus `data/predictions/auxhead_w0_seed42.npz` and `data/predictions/single_task_seed42.npz` |
| Outputs | `results/ablation_results.json` |
| Options | none |
| Runtime | a few seconds on CPU |
| Expected behaviour | prints the controlled comparison (Table 6) and the ablation ladder (Table 7); recomputes every row under one unified protocol except the archived "Attention only" row (see README) |

### Metric conventions encoded in the scripts

* **event-level**: one decision per seismic event (validation-selected threshold b*);
* **sample-level broadcast**: the event decision is broadcast to all station records of that event;
* **station-independent**: one decision per station record;
* **single- / multi-station decomposition**: station-independent results split by the number of stations that recorded the event;
* **confidence gating**: events ranked by mean ensemble probability; F1/accuracy of the retained subsets at coverages 0.5/0.7/0.9/1.0 (the gating analysis uses the probability-argmax decision rule, so its full-coverage F1 differs from the tuned-threshold value in Table 1 of the manuscript).

## 2. Training code (requires CUDA and the full feature cache)

### `src/train.py`

```bash
python src/train.py <seed> <W_ST>
# paper configuration:
python src/train.py 42 0.30
```

| item | description |
|---|---|
| Inputs | `data/full_feature_cache.npz` (1.15 GB, built from the public Diting 2.0 archive with `src/data_pipeline.py`) |
| Outputs | `data/predictions/auxhead_w0<W_ST>_seed<seed>.npz` (test/validation logits, magnitudes, polarity logits, event ids) |
| Options | `argv[1]` = random seed (paper uses 42 / 7 / 2024); `argv[2]` = auxiliary station-head loss weight W_ST (paper scans 0.15 / 0.30 / 0.50) |
| Expected behaviour | trains with EMA weights, early stopping (patience 35), validation-set model and threshold selection; roughly 250 epochs per seed on one GPU |

### `src/train_ablation.py`

```bash
python src/train_ablation.py <mode> <seed>
# modes used in the paper:
python src/train_ablation.py w0 42       # auxiliary head without station-level supervision (W_ST = 0)
python src/train_ablation.py single 42   # single-task (type head only)
```

| item | description |
|---|---|
| Inputs | same feature cache as `src/train.py` |
| Outputs | `data/predictions/auxhead_w0_seed<seed>.npz` or `data/predictions/single_task_seed<seed>.npz` |
| Options | `argv[1]` = mode (`w0` or `single`); `argv[2]` = random seed |
| Expected behaviour | identical training protocol to `src/train.py`; only the supervised heads differ |

## 3. Data files shipped in this repository

| file | contents |
|---|---|
| `data/event_split_cache.npz` | `event_id` (anonymous 0..30698), `labels3`, `tr_idx/va_idx/te_idx`, `pol`, `pol_mask` - 246 KB |
| `data/predictions/*.npz` | per-record cached model outputs: `test_type_logits`, `val_type_logits`, `test_mag`, `test_mag_true`, `test_pol_logits`, `val_pol_logits`, `test_eids`, `val_eids` |
| `data/archived/*.json` | archived official evaluations of the two single models (inputs of `evaluate_ablation.py`) |

No waveforms, raw features, epicentral distances or station identifiers are
included. All event identifiers are remapped to sequential anonymous indices;
event grouping and all metrics are preserved exactly.

## 4. Expected behaviour on a clean machine

```bash
git clone https://github.com/epnet2018/SEER
cd SEER
pip install -r requirements.txt
python evaluate.py
python evaluate_ablation.py
```

Both scripts must exit 0 and print the manuscript values; they also rewrite the
files in `results/`. If a value differs, please open an issue with the console
output and your `numpy` / `scikit-learn` versions.
