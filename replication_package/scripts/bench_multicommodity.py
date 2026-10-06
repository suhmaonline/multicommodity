#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C 方案：多品种面板基准（Chronos 零样本 vs 朴素持久性）
  - 数据源：data/processed/agri_daily_prices_wide.csv（ncpscxx 7 品种日频自聚合近似均价）
  - 协议（与 B 对齐但修正点预测）：对每个品种，固定测试集（date > VAL_END）上做 H=7/14 直接多步预测
      * 目标：price[t+H]（H 步 ahead）
      * 朴素持久性：ŷ = price[t]（origin 处最后观测值）—— 标准 persistence
      * Chronos 点预测：context = price[t-CTX_LEN+1 .. t]，predict(H) 后取各样本第 H 步的均值
        （修正 B 中 "mean over all steps" 的含糊写法，明确为第 H 步点预测）
  - 缺失值：序列做前向/后向填充仅用于构造 context；target 与 naive 必须是真实观测值，否则跳过该 origin
  - 输出：per-commodity {y, naive, chronos} 数组 + MAPE，写入 results_multicommodity.json
纯 numpy/scipy/torch/chronos，需用含这些包的 venv 运行。
"""
import os
import csv
import json
import time
import argparse

import numpy as np
import torch
from chronos import ChronosPipeline
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(BASE, "data", "processed", "agri_daily_prices_wide.csv")
MODEL_DIR_SMALL = os.path.join(BASE, "data", "model_cache", "chronos-t5-small-ms")
MODEL_DIR_BASE = os.path.join(BASE, "data", "model_cache", "chronos-t5-base-ms")

CTX_LEN = 512
HORIZONS = [7, 14]
NUM_SAMPLES = 20
VAL_END = "2025-11-30"

# 中文名 -> 英文论文 label
COMMODITY_EN = {
    "土豆": "potato",
    "大白菜": "Chinese cabbage",
    "大蒜": "garlic",
    "洋葱": "onion",
    "生姜": "ginger",
    "白萝卜": "white radish",
    "青椒": "green pepper",
}


def load_wide(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    dates = [r["date"] for r in rows]
    series = {}
    for cname in COMMODITY_EN:
        vals = []
        for r in rows:
            v = r[cname].strip()
            vals.append(float(v) if v not in ("", "NA", "NaN") else float("nan"))
        series[cname] = np.array(vals, dtype=np.float64)
    return dates, series


def ffill(arr):
    """仅用于构造 context：前向填充，再后向填充（极少的前导缺失）。target/naive 不使用填充值。"""
    out = arr.copy()
    mask = np.isnan(out)
    idx = np.where(~mask)[0]
    if len(idx) == 0:
        return out
    # 前向
    last = out[idx[0]]
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    # 后向（前导）
    first = out[idx[0]]
    for i in range(idx[0] - 1, -1, -1):
        out[i] = first
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["small", "base"], default="small")
    ap.add_argument("--max_origins", type=int, default=0,
                    help="每品种最大测试 origin 数（0=全部）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = MODEL_DIR_BASE if args.model == "base" else MODEL_DIR_SMALL
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_multicommodity_{args.model}.json")

    dates, series = load_wide(WIDE)
    n = len(dates)
    val_idx = max(i for i, d in enumerate(dates) if d <= VAL_END)
    print(f"[INFO] 总天数 n={n}; VAL_END={VAL_END} -> val_idx={val_idx}; "
          f"测试窗口起点 {dates[val_idx+1]}")

    pipe = ChronosPipeline.from_pretrained(model_dir, device_map="cpu")
    print(f"[INFO] Chronos({args.model}) 已加载自 {model_dir}")

    def chronos_point(ctx_vals, H):
        ctx = torch.tensor(ctx_vals, dtype=torch.float32)
        fc = pipe.predict(inputs=ctx, prediction_length=H, num_samples=NUM_SAMPLES)
        arr = np.asarray(fc, dtype=np.float64)          # (num_samples, H)
        # 修正：第 H 步点预测 = 各样本第 H-1 列（0-indexed）的均值
        return float(arr[:, H - 1].mean())

    results = {"config": {
        "model": args.model, "model_dir": model_dir,
        "context_length": CTX_LEN, "horizons": HORIZONS,
        "num_samples": NUM_SAMPLES, "val_end": VAL_END,
        "naive_rule": "y_hat[t+H] = price[t]",
        "point_forecast": "mean over sampled trajectories of the H-th step",
        "data_source": "ncpscxx.moa.gov.cn (self-aggregated national wholesale daily avg)",
    }, "commodities": {}}

    for cname, raw in series.items():
        filled = ffill(raw)
        n_missing = int(np.isnan(raw).sum())
        en = COMMODITY_EN[cname]
        cres = {"n_days": n, "n_missing": n_missing, "test_origins": 0,
                "h7": None, "h14": None}
        print(f"\n=== {cname} ({en}) : n_missing={n_missing} ===")

        for H in HORIZONS:
            y, nv, ch = [], [], []
            for t in range(val_idx + 1, n - H):
                # target 必须是真实观测
                if np.isnan(raw[t + H]):
                    continue
                # naive 必须是真实观测
                if np.isnan(raw[t]):
                    continue
                # context 用填充后的序列（仅构造输入），但要求至少有足够长度
                ctx = filled[max(0, t - CTX_LEN + 1): t + 1]
                if len(ctx) < 1:
                    continue
                y.append(float(raw[t + H]))
                nv.append(float(raw[t]))
                ch.append(chronos_point(ctx, H))
                if args.max_origins and len(y) >= args.max_origins:
                    break
            y = np.array(y); nv = np.array(nv); ch = np.array(ch)
            mape_nv = float(np.mean(np.abs((y - nv) / y)) * 100)
            mape_ch = float(np.mean(np.abs((y - ch) / y)) * 100)
            cres["test_origins"] = max(cres["test_origins"], len(y))
            cres[f"h{H}"] = {
                "y": y.tolist(), "naive": nv.tolist(), "chronos": ch.tolist(),
                "mape_naive_pct": round(mape_nv, 4),
                "mape_chronos_pct": round(mape_ch, 4),
            }
            print(f"  H={H}: n={len(y)}  MAPE naive={mape_nv:.3f}%  "
                  f"chronos={mape_ch:.3f}%  (Δ={mape_ch-mape_nv:+.3f}pp)")
        results["commodities"][cname] = cres

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] -> {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[TIME] 总耗时 {time.time()-t0:.1f}s")
