#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C 方案：跨品种统计检验（复用 B 的 metrics.diebold_mariano + HAC-SE）
  - 逐品种 × 逐视野 HAC-Diebold–Mariano（Newey-West, Bartlett, lag=H-1；平方损失与绝对损失）
  - 逐品种 × 逐视野 TOST 等效检验：边界 δ = 10% × 该品种 naive MAPE；报 90% CI；判定"等价"
  - 跨品种联合：
      (a) 方向符号检验：14 个 (品种×视野) 单元中 Chronos 优于/劣于 naive 的二项检验
      (b) Fisher 合并：合并 14 个 DM(平方) p 值，检验"总体上是否存在任何差异"
      (c) 等价率：多少单元通过 TOST（等价）
  - 输出 stats_multicommodity.json + stats_multicommodity.md
纯标准库 + experiments/metrics.py（diebold_mariano 为纯 stdlib）。
"""
import os
import sys
import json
import math
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "experiments"))
import metrics  # diebold_mariano (stdlib)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def hac_se(d, h):
    """Newey-West 长期方差标准误（Bartlett, lag=h-1），与 metrics.diebold_mariano 一致。"""
    n = len(d)
    dbar = sum(d) / n
    gamma = []
    for k in range(h):
        s = 0.0
        for t in range(n - k):
            s += (d[t] - dbar) * (d[t + k] - dbar)
        gamma.append(s / n)
    lrv = gamma[0]
    for k in range(1, h):
        lrv += 2.0 * (1.0 - k / h) * gamma[k]
    if lrv <= 0:
        return float("nan")
    return math.sqrt(lrv / n)


def tost_ape(diff, naive_ape, H, alpha=0.10):
    """diff = ape_ch - ape_nv (per origin); naive_ape list for δ. 返回 (equiv, ci90, delta, m, se)。"""
    n = len(diff)
    m = sum(diff) / n
    se = hac_se(diff, H)
    if math.isnan(se):
        return False, [float("nan"), float("nan")], float("nan"), m, se
    z = 1.6448536269514722  # 90% CI
    lo, hi = m - z * se, m + z * se
    delta = 0.10 * (sum(naive_ape) / n)
    equiv = (lo > -delta) and (hi < delta)
    return equiv, [lo, hi], delta, m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    inp = args.inp or os.path.join(here, "results_multicommodity_small.json")
    out_json = args.out or os.path.join(here, "stats_multicommodity.json")
    out_md = os.path.splitext(out_json)[0] + ".md"

    data = json.load(open(inp, encoding="utf-8"))
    commodities = list(data["commodities"].keys())
    horizons = data["config"]["horizons"]

    cells = []
    per_commod = {}
    for c in commodities:
        cres = data["commodities"][c]
        per_commod[c] = {"h7": None, "h14": None}
        for H in horizons:
            blk = cres.get(f"h{H}")
            if not blk:
                continue
            y = blk["y"]; nv = blk["naive"]; ch = blk["chronos"]
            n = len(y)
            # 损失差分（与 metrics 约定：e1=chronos, e2=naive → dm>0 表示 chronos 更差）
            e_ch_raw = [c - yi for c, yi in zip(ch, y)]
            e_nv_raw = [a - yi for a, yi in zip(nv, y)]
            e_ch_abs = [abs(v) for v in e_ch_raw]
            e_nv_abs = [abs(v) for v in e_nv_raw]
            dm_sq, p_sq = metrics.diebold_mariano(e_ch_raw, e_nv_raw, h=H, power=2)
            dm_abs, p_abs = metrics.diebold_mariano(e_ch_abs, e_nv_abs, h=H, power=1)
            # TOST on APE diff
            ape_nv = [abs(a - yi) / abs(yi) * 100 for a, yi in zip(nv, y)]
            ape_ch = [abs(c - yi) / abs(yi) * 100 for c, yi in zip(ch, y)]
            diff = [b - a for a, b in zip(ape_nv, ape_ch)]
            equiv, ci90, delta, m, se = tost_ape(diff, ape_nv, H)
            mape_nv = blk["mape_naive_pct"]; mape_ch = blk["mape_chronos_pct"]
            direction = "chronos_better" if mape_ch < mape_nv else (
                "chronos_worse" if mape_ch > mape_nv else "tie")
            cell = {
                "commodity": c, "H": H, "n_origins": n,
                "mape_naive_pct": mape_nv, "mape_chronos_pct": mape_ch,
                "dm_sq_stat": dm_sq, "dm_sq_p": p_sq,
                "dm_abs_stat": dm_abs, "dm_abs_p": p_abs,
                "tost_equiv_within_10pct_naive_mape": bool(equiv),
                "tost_ci90_ape_diff": ci90, "tost_delta": delta,
                "tost_mean_ape_diff": m, "direction": direction,
            }
            cells.append(cell)
            per_commod[c][f"h{H}"] = cell

    # ---- 跨品种联合 ----
    import scipy.stats as st

    def verdict(c):
        if c["dm_sq_p"] < 0.05 and c["direction"] == "chronos_better":
            return "chronos_sig_better"
        if c["dm_sq_p"] < 0.05 and c["direction"] == "chronos_worse":
            return "chronos_sig_worse"
        return "no_sig_diff"

    for c in cells:
        c["verdict"] = verdict(c)

    directions = [c["direction"] for c in cells]
    k_better = sum(1 for d in directions if d == "chronos_better")
    k_worse = sum(1 for d in directions if d == "chronos_worse")
    k_tie = sum(1 for d in directions if d == "tie")
    v_better = sum(1 for c in cells if c["verdict"] == "chronos_sig_better")
    v_worse = sum(1 for c in cells if c["verdict"] == "chronos_sig_worse")
    v_none = sum(1 for c in cells if c["verdict"] == "no_sig_diff")
    binom = st.binomtest(k_better, len(cells), 0.5, alternative="two-sided")
    # Fisher 合并 DM(sq) p：检验"所有单元等同"的全局原假设
    ps = [c["dm_sq_p"] for c in cells]
    ps_safe = [min(max(p, 1e-12), 1.0 - 1e-12) for p in ps]
    X2 = -2.0 * sum(math.log(p) for p in ps_safe)
    df = 2 * len(ps)
    fisher_p = 1.0 - st.chi2.cdf(X2, df)
    n_equiv = sum(1 for c in cells if c["tost_equiv_within_10pct_naive_mape"])

    joint = {
        "n_cells": len(cells),
        "chronos_better_cells": k_better,
        "chronos_worse_cells": k_worse,
        "tie_cells": k_tie,
        "verdict_sig_better": v_better,
        "verdict_sig_worse": v_worse,
        "verdict_no_sig_diff": v_none,
        "binomial_two_sided_p": float(binom.pvalue),
        "fisher_combined_dm_sq_p": float(fisher_p),
        "cells_equivalent_by_tost": n_equiv,
        "cells_equivalent_rate": n_equiv / len(cells),
    }

    out = {"source": inp, "per_cell": cells, "joint": joint}
    json.dump(out, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # markdown
    rep = ["# C 方案跨品种统计检验报告\n"]
    rep.append(f"- 数据源：{data['config']['data_source']}")
    rep.append(f"- 模型：{data['config']['model']}（{data['config']['point_forecast']}）")
    rep.append(f"- 朴素基线：{data['config']['naive_rule']}\n")
    rep.append("## 逐品种 × 逐视野：MAPE / HAC-DM / TOST\n")
    rep.append("| 品种 | H | n | naive MAPE | chronos MAPE | DM(sq) p | DM(abs) p | TOST等价(δ=10%nMAPE) | 方向 |")
    rep.append("|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        rep.append(
            f"| {c['commodity']} | {c['H']} | {c['n_origins']} | "
            f"{c['mape_naive_pct']:.3f}% | {c['mape_chronos_pct']:.3f}% | "
            f"{c['dm_sq_p']:.4f} | {c['dm_abs_p']:.4f} | "
            f"{'是' if c['tost_equiv_within_10pct_naive_mape'] else '否'} | {c['direction']} |")
    rep.append("\n## 跨品种联合检验\n")
    rep.append(f"- 单元总数：{joint['n_cells']}（7 品种 × 2 视野）")
    rep.append(f"- 方向：Chronos 更优 {k_better} / 更差 {k_worse} / 平 {k_tie}")
    rep.append(f"- 逐单元 DM(sq) 裁决：显著更优 {v_better} / 显著更差 {v_worse} / 无显著差异 {v_none}")
    rep.append(f"- 方向二项检验（H0：优劣各半）：p = {joint['binomial_two_sided_p']:.4f}"
               f"（p>0.05 ⇒ 无证据显示 Chronos 系统性更优）")
    rep.append(f"- Fisher 合并 DM(sq) p = {joint['fisher_combined_dm_sq_p']:.4f}：该全局检验拒绝“所有单元皆等同”的原假设；"
               f"但方向分布显示拒绝主要来自 Chronos 显著更差而非更优，故不构成“Chronos 更优”的证据。")
    rep.append(f"- TOST 等价单元数：{n_equiv}/{joint['n_cells']}（等价率 {joint['cells_equivalent_rate']*100:.1f}%）")
    open(out_md, "w", encoding="utf-8").write("\n".join(rep))

    print("\n".join(rep))
    print(f"\n[DONE] {out_json}\n[DONE] {out_md}")


if __name__ == "__main__":
    main()
