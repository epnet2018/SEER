# Reproduction package - station-level attention multi-task framework

This package reproduces the numerical results reported in the manuscript from the
**cached predictions** of the trained models. No retraining, no waveforms and no
station parameters are required: a plain Python 3 environment with NumPy (plus
scikit-learn for one AUC call) reproduces the paper's Tables 2, 3, 5, 6 and 7 in a
few seconds.

## Quick start

```bash
pip install -r requirements.txt
python evaluate.py            # Table 2 + Table 3 + Table 5 + confidence gating
python evaluate_ablation.py   # Table 6 + Table 7 (controlled comparison + ablation ladder)
```

Tested with Python 3.12, NumPy 2.4. No GPU, no PyTorch.

A detailed user guide (inputs, outputs, options, metric conventions and expected
behaviour of every script) is provided in [`USER_GUIDE.md`](USER_GUIDE.md).

### Expected output (exact paper values)

```
[2] Three ensemble pools ...
  Multi-seed ensemble (W_ST = 0.30, seeds 42/7/2024): event-level=0.8707 sample-level=0.9549 station-independent=0.8251 (single=0.3333 multi=0.8394) pol=0.8914
  Full-model ensemble (all 5 models): event-level=0.8621 sample-level=0.9474 station-independent=0.8330 (single=0.3288 multi=0.8468) pol=0.8873
  Hyperparameter-diverse ensemble (W_ST = 0.15/0.30/0.50, seed 42): event-level=0.8327 sample-level=0.9502 station-independent=0.8173 (single=0.2750 multi=0.8332) pol=0.8813
```

```
Ablation ladder (Table 7):
  Attention only                 event=0.7764 sample=0.9246 station=0.6886 FP=28 MAE=0.2502 pol=0.8974
  Auxiliary head, W_ST = 0.30    event=0.7915 sample=0.9208 station=0.7945 FP=25 MAE=0.2457 pol=0.8835
  Auxiliary head, W_ST = 0       event=0.8122 sample=0.9316 station=0.2640 FP=19 MAE=0.2404 pol=0.8802
  Auxiliary head, single-task    event=0.8115 sample=0.9365 station=0.0029 FP=28 MAE=1.8356 pol=0.5157
  Multi-seed ensemble            event=0.8707 sample=0.9549 station=0.8251 FP=14 MAE=0.2308 pol=0.8914
```

The scripts also write `results/evaluation_results.json`,
`results/evaluation_report.html` and `results/ablation_results.json`.

## What is in this folder

| path | role |
|------|------|
| `evaluate.py` | W_ST scan (Table 2), three ensemble pools (Table 3), threshold-stability scan (Table 5), confidence gating (Fig. 12) |
| `evaluate_ablation.py` | controlled comparison (Table 6) and ablation ladder (Table 7) |
| `data/event_split_cache.npz` | minimal anonymised metadata: `event_id`, `labels3`, `te_idx`, `va_idx`, `pol`, `pol_mask` (246 KB) |
| `data/predictions/*.npz` (7 files) | cached logits/magnitudes of the five main models plus the two ablation models |
| `data/predictions/auxhead_w030_seed42.npz` etc. | single models; the three `W_ST = 0.30` files (seeds 42/7/2024) form the multi-seed ensemble |
| `data/predictions/auxhead_w0_seed42.npz` | ablation: auxiliary head without station-level supervision |
| `data/predictions/single_task_seed42.npz` | ablation: single-task (type head only) |
| `data/archived/*.json` | archived official single-model evaluations ("Attention only", auxiliary head W_ST = 0.30); used as inputs by `evaluate_ablation.py` |
| `results/*.json`, `results/*.html` | generated outputs of the two evaluators |
| `src/` | the model and the training code (for full retraining; see below) |
| `src/data_pipeline.py` | preprocessing / feature extraction / labels / fixed event-level split (how the feature cache is built) |
| `src/building_blocks.py` | shared residual self-attention encoder and task decoders (Fig. 3) |
| `src/model.py` | the paper's model: encoder + station-level attention + event-level heads + auxiliary station-level type head + polarity head |
| `src/train.py` | training script (fixed event-level split, EMA, validation-set model/threshold selection) |
| `src/train_ablation.py` | ablation training (`w0` = W_ST = 0; `single` = type task only) |

## What the package reproduces

| manuscript item | how |
|---|---|
| Table 1 (headline metrics) | `evaluate.py` - the multi-seed ensemble row (0.8707 / 0.9549 / 0.8251, polarity 0.8914, MAE 0.2308, ECE 0.0102, AUC 0.9613, top-90% accuracy 0.9982) |
| Table 2 (W_ST scan) | `evaluate.py` - from the three `W_ST = 0.15/0.30/0.50` single models (seed 42) |
| Table 3 (three ensemble pools) | `evaluate.py` - logit averaging + validation-set threshold re-search |
| Table 4 (granularity decomposition) | `evaluate.py` - single/multi-station decomposition of the multi-seed ensemble |
| Table 5 (threshold-stability scan) | `evaluate.py` - sweep of the event-level decision threshold around b* = −1.14 |
| Table 6 (controlled comparison) | `evaluate_ablation.py` |
| Table 7 (ablation ladder) | `evaluate_ablation.py` - the "Attention only" row is the archived official single-model evaluation; the other rows are recomputed from the cached predictions under one unified protocol |
| Fig. 12 (confidence gating / calibration) | `evaluate.py` - gating curve, ECE and confidence-correctness AUC of each pool |
| Figs. 4-8 (bar/curve forms of Tables 2-7) | `evaluate.py` / `evaluate_ablation.py` - the same numbers, plotted |

Not reproducible from this package (they require the full 1.15 GB feature cache and
GPU training, or data that is not shipped):

* retraining the models from scratch - `src/train.py` / `src/train_ablation.py` are the
  exact scripts used; see "Full retraining" below;
* Table 8 (Grad-CAM energy ratios) - requires the trained model and full cache; the
  reported ratios are archived by the authors;
* distance-stratified stratification (Fig. 10d / Fig. 11b) and the attention-weight vs
  distance analysis (Fig. 13c) - raw features and attention weights are not shipped.

## Data-protection measures

The package contains **no waveforms, no raw features, no spectral features, no
epicentral distances and no station identifiers**:

* the 1.15 GB feature cache was reduced to six label/split arrays (246 KB);
* every event identifier - in the cache and in **all** prediction caches
  (training/val/test) - was remapped to a sequential anonymous index, with the event
  grouping preserved exactly (all per-event metrics are unchanged, as confirmed by
  the identical values above);
* only the models' cached predictions are shipped, not the training data.

Reproducing the numbers therefore requires no access to the Diting 2.0 waveforms.

## Full retraining (optional)

`src/` contains the exact model and training code used for the paper's experiments
(only comments/docstrings/console messages are localised to English; every line of
model code and every hyper-parameter is unchanged):

```bash
python src/train.py 42 0.30          # the paper's model, seed 42, W_ST = 0.30
python src/train.py 42 0.15          # W_ST scan points (0.15 / 0.30 / 0.50)
python src/train_ablation.py w0 42   # auxiliary head without station-level supervision
python src/train_ablation.py single 42   # single-task (type head only)
```

Retraining writes `data/predictions/auxhead_w030_seed42.npz` (and friends), which
can be fed directly to `evaluate.py`. It requires:

1. PyTorch with CUDA (`pip install torch`); the script targets `cuda:0`;
2. the complete feature cache (1.15 GB) placed at `data/full_feature_cache.npz` -
   build it from the raw Diting 2.0 archive with `src/data_pipeline.py` (adjust the
   archive path and the fixed event-split file to your copy);
3. roughly 250 epochs per seed on a GPU (early stopping with patience 35).

## Note on fidelity

All evaluators are plain NumPy; every metric, threshold rule and numeric value is
unchanged from the original experiments, so re-running them reproduces the published
results exactly, as shown by the identical values above.

## License

This code is released under the MIT License - see [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite the accompanying manuscript.
