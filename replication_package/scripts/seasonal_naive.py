#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C 方案附加基线：Seasonal Naive (y_hat[t+H] = price[t+H-7])。
复用 bench_multicommodity.py 的 origin 选择逻辑（仅跳过 raw[t+H]/raw[t] 为 NaN 的 origin），
不运行 Chronos，故秒级完成。输出 seasonal_naive.json 供填表 1。"""
import os, csv, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(BASE, "data", "processed", "agri_daily_prices_wide.csv")
COMMODITY_EN = {"土豆": "potato", "大白菜": "Chinese cabbage", "大蒜": "garlic",
                "洋葱": "onion", "生姜": "ginger", "白萝卜": "white radish", "青椒": "green pepper"}
HORIZONS = [7, 14]
VAL_END = "2025-11-30"


def load_wide(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    dates = [r["date"] for r in rows]
    series = {}
    for c in COMMODITY_EN:
        vals = []
        for r in rows:
            v = r[c].strip()
            vals.append(float(v) if v not in ("", "NA", "NaN") else float("nan"))
        series[c] = np.array(vals, dtype=np.float64)
    return dates, series


def ffill(arr):
    out = arr.copy()
    mask = np.isnan(out)
    idx = np.where(~mask)[0]
    if len(idx) == 0:
        return out
    last = out[idx[0]]
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    first = out[idx[0]]
    for i in range(idx[0] - 1, -1, -1):
        out[i] = first
    return out


def main():
    dates, series = load_wide(WIDE)
    n = len(dates)
    val_idx = max(i for i, d in enumerate(dates) if d <= VAL_END)
    res = {"config": {"seasonal_naive": "y_hat[t+H] = price[t+H-7] (period=7)"},
           "commodities": {}}
    for cname, raw in series.items():
        filled = ffill(raw)
        cres = {}
        for H in HORIZONS:
            y, sn = [], []
            for t in range(val_idx + 1, n - H):
                if np.isnan(raw[t + H]):
                    continue
                if np.isnan(raw[t]):
                    continue
                y.append(float(raw[t + H]))
                sn.append(float(filled[t + H - 7]))
            y = np.array(y)
            sn = np.array(sn)
            mape_sn = float(np.mean(np.abs((y - sn) / y)) * 100)
            cres[f"h{H}"] = {"n": len(y), "mape_seasonal_naive_pct": round(mape_sn, 4)}
            print(f"  {cname} H={H}: n={len(y)} MAPE seasonal_naive={mape_sn:.3f}%")
        res["commodities"][cname] = cres
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seasonal_naive.json")
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("DONE", outp)


if __name__ == "__main__":
    main()
