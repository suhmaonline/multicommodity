# C-Garlic-Multi Replication Package

Self-contained code and data to reproduce **every** number in the manuscript
"Zero-Shot Time-Series Foundation Models versus Naive Persistence across Seven Chinese
Agricultural Price Series".

---

## Data provenance

- **Source:** Ministry of Agriculture and Rural Affairs key agricultural-products market
  information platform — `https://ncpscxx.moa.gov.cn`.
- **Definition:** for each trading day and commodity, the value is the simple mean of all
  reported wholesale-market prices — a **self-aggregated approximation** of the national
  wholesale average (CNY/kg). This is *not* an officially published national average; the
  convention is documented honestly in `data/processed/ncpscxx_provenance.json`.
- **Panel:** 7 storable commodities (garlic, ginger, onion, potato, white radish, green
  pepper, Chinese cabbage); 1,162 calendar days (2023-08-01 → 2026-10-05); 0–2 missing days
  per commodity, carried as non-observed and excluded from per-origin evaluation.
- **Integrity:** the original platform export (`data/raw/ncpscxx/`) and its SHA-256 are
  recorded in the provenance file so reviewers can verify the aggregation.

---

## Files

```
data/processed/agri_daily_prices_wide.csv     # 7-commodity panel (analysis-ready)
data/processed/ncpscxx_provenance.json        # source, 口径, SHA-256, exclusions
data/raw/ncpscxx/...csv                       # original platform export (verify SHA-256)
experiments/metrics.py                        # MAE/RMSE/MAPE + Diebold–Mariano (self-test on run)
experiments/download_chronos_ms.py            # fetch Chronos-T5-small weights (ModelScope mirror)
scripts/bench_multicommodity.py               # naive vs seasonal-naive vs Chronos (H=7,14)
scripts/seasonal_naive.py                     # seasonal-naive baseline MAPE
scripts/martingale_multicommodity.py          # ADF / KPSS / variance-ratio diagnostics
scripts/stats_multicommodity.py               # HAC-DM + TOST + cross-commodity joint tests
results/                                      # produced artifacts (also committed for transparency)
```

> **Model weights are NOT included** (large + publicly released). Fetch them with the
> provided script into `data/model_cache/`.

---

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# 1) fetch Chronos-T5-small weights (public; ~hundreds of MB) -> data/model_cache/
python experiments/download_chronos_ms.py

# 2) baselines + zero-shot Chronos benchmark -> results/results_multicommodity_small.json
python scripts/bench_multicommodity.py
#    quick smoke test:  python scripts/bench_multicommodity.py --max_origins 20

# 3) seasonal-naive baseline -> results/seasonal_naive.json
python scripts/seasonal_naive.py

# 4) martingale diagnostics -> results/martingale_multicommodity.{json,md}
python scripts/martingale_multicommodity.py

# 5) statistics (HAC-DM, TOST, joint) -> results/stats_multicommodity_full.{json,md}
python experiments/metrics.py     # optional: verify MAE/RMSE/MAPE/DM against hand values
python scripts/stats_multicommodity.py
```

`bench_multicommodity.py` runs Chronos **zero-shot on CPU** (20 samples/horizon). On the full
test set (~300 origins × 7 commodities × 2 horizons) it takes roughly one hour on CPU.

---

## Verified outputs

The committed `results/` files correspond exactly to the manuscript tables:
**14/14** commodity-horizon cells show Chronos worse than naive; directional binomial
p = 0.0001; Fisher-combined HAC-DM p < 0.0001; TOST equivalence holds in **1/14** cells only
(Green pepper, H=14).
