#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C 方案：逐品种 martingale / 随机游走诊断（与 B 方法一致）
  对每个品种做 ADF（水平/收益）、KPSS（水平）、Lo-MacKinlay 方差比（q=2,4,8,16）。
  目的：形式化刻画各品种价格序列的"persistence-dominated"结构，解释为何零样本基础模型
        跨品种都难以击败朴素持久性。
  数据源：data/processed/agri_daily_prices_wide.csv（ncpscxx 7 品种）
"""
import os
import csv
import json
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDE = os.path.join(BASE, "data", "processed", "agri_daily_prices_wide.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "martingale_multicommodity.json")
REP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "martingale_multicommodity.md")

COMMODITY_EN = {
    "土豆": "potato", "大白菜": "Chinese cabbage", "大蒜": "garlic", "洋葱": "onion",
    "生姜": "ginger", "白萝卜": "white radish", "青椒": "green pepper",
}


def load():
    rows = list(csv.DictReader(open(WIDE, encoding="utf-8-sig")))
    out = {}
    for cname, en in COMMODITY_EN.items():
        vals = []
        for r in rows:
            v = r[cname].strip()
            vals.append(float(v) if v not in ("", "NA", "NaN") else float("nan"))
        arr = np.array(vals, dtype=np.float64)
        # 诊断用：ffill/bfill 处理 ≤2 个缺失（仅用于单位根/方差比估计，非预测）
        mask = np.isnan(arr)
        if mask.any():
            idx = np.where(~mask)[0]
            last = arr[idx[0]]; arr = arr.copy()
            for i in range(len(arr)):
                arr[i] = last if np.isnan(arr[i]) else (last := arr[i])
            first = arr[idx[0]]
            for i in range(idx[0] - 1, -1, -1):
                arr[i] = first
        out[cname] = arr
    return out


def lo_macKinlay_vr(r, q):
    n = len(r); mu = r.mean()
    sigma2_1 = np.sum((r - mu) ** 2) / n
    m = n - q + 1
    acc = 0.0
    for k in range(q, n):
        s = np.sum(r[k - q + 1:k + 1]) - q * mu
        acc += s * s
    sigma2_q = acc / (m * q)
    vr = sigma2_q / sigma2_1 if sigma2_1 > 0 else float("nan")
    phi = np.sqrt((2.0 * (2 * q - 1) * (q - 1)) / (3.0 * q))
    z = (vr - 1.0) * np.sqrt(n) / phi if phi > 0 else float("nan")
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(vr), float(z), float(p)


def classify(adf_lvl_p, adf_ret_p, kpss_p, vr_list):
    """诚实的异质分类：基于 ADF(水平)/KPSS/方差比三件套。
    - 单位根性：adf_lvl_p>0.05 且 kpss_p<0.05 -> 单位根/持久性主导
    - 自相关方向：VR(q)<1 显著 -> 均值回复；VR(q)>1 显著 -> 趋势(动量)
    """
    unit_root = (adf_lvl_p > 0.05)
    kpss_nonstat = (kpss_p < 0.05)
    vr_meanrev = all((vr < 1 and p < 0.05) for vr, _, p in vr_list)
    vr_trend = all((vr > 1 and p < 0.05) for vr, _, p in vr_list)
    if unit_root and kpss_nonstat:
        base = "unit-root / persistence-dominated (I(1))"
    elif kpss_nonstat:
        base = "weakly non-stationary (borderline)"
    elif (kpss_p >= 0.05) and (adf_lvl_p < 0.05):
        base = "trend-stationary / weakly persistent"
    else:
        base = "mixed"
    if vr_meanrev:
        return base + "; mean-reverting at all horizons"
    if vr_trend:
        return base + "; trending (positive autocorrelation) at longer horizons"
    return base + "; mixed autocorrelation"


def main():
    series = load()
    out = {"commodities": {}}
    rep = ["# C 方案逐品种 martingale / 随机游走诊断\n"]
    rep.append("| 品种 | ADF水平 p | ADF收益 p | KPSS水平 p | VR(2) | VR(8) | VR(16) | 分类 |")
    rep.append("|---|---|---|---|---|---|---|---|")
    for cname, price in series.items():
        en = COMMODITY_EN[cname]
        logp = np.log(price); ret = np.diff(logp)
        adf_lvl = adfuller(logp, autolag="AIC")
        adf_ret = adfuller(ret, autolag="AIC")
        kp = kpss(logp, regression="c", nlags="auto")
        vr = {str(q): lo_macKinlay_vr(ret, q) for q in [2, 4, 8, 16]}
        cls = classify(adf_lvl[1], adf_ret[1], kp[1], [vr[str(q)] for q in [2, 4, 8, 16]])
        out["commodities"][cname] = {
            "en": en, "n": int(len(price)), "n_missing_filled": int(np.isnan(price).sum()),
            "adf_log_price_level": {"stat": float(adf_lvl[0]), "p": float(adf_lvl[1])},
            "adf_log_return": {"stat": float(adf_ret[0]), "p": float(adf_ret[1])},
            "kpss_log_price_level": {"stat": float(kp[0]), "p": float(kp[1])},
            "variance_ratio": {q: {"vr": v[0], "z": v[1], "p": v[2]} for q, v in vr.items()},
            "classification": cls,
            "return_stats": {
                "mean_daily": float(ret.mean()), "std_daily": float(ret.std()),
                "skew": float(stats.skew(ret)), "kurtosis": float(stats.kurtosis(ret)),
                "abs99_pct": float(np.percentile(np.abs(ret), 99)),
            },
        }
        rep.append(
            f"| {cname}({en}) | {adf_lvl[1]:.4f} | {adf_ret[1]:.4f} | {kp[1]:.4f} | "
            f"{vr['2'][0]:.3f} | {vr['8'][0]:.3f} | {vr['16'][0]:.3f} | {cls} |")
        print(f"[OK] {cname}: {cls}")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(REP, "w", encoding="utf-8").write("\n".join(rep))
    print(f"\n[DONE] {OUT}\n[DONE] {REP}")


if __name__ == "__main__":
    main()
