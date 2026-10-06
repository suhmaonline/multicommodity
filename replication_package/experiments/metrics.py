#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R003 指标校验模块：MAE / RMSE / MAPE / Diebold-Mariano 检验
纯标准库（math），无需 numpy/scipy，可在基础 Python 直接运行。

自检：python experiments/metrics.py   （用一组手算可验证的合成数校验正确性）
"""
import math


def _norm_cdf(x):
    """标准正态分布 CDF，用 math.erf 实现。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mae(y_true, y_pred):
    """平均绝对误差。"""
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    """均方根误差。"""
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mape(y_true, y_pred):
    """平均绝对百分比误差（%）。跳过真实值为 0 的样本（避免除零）。"""
    valid = [(a, b) for a, b in zip(y_true, y_pred) if a != 0]
    if not valid:
        return float("nan")
    return 100.0 * sum(abs(a - b) / abs(a) for a, b in valid) / len(valid)


def diebold_mariano(e1, e2, h=1, power=2):
    """
    Diebold-Mariano 检验（双侧，原假设：两模型预测精度无显著差异）。

    e1: 模型 A 的预测误差序列；e2: 模型 B 的预测误差序列（等长）
    h:   预测 horizon（>1 时误差自相关，用 Newey-West 长期方差修正）
    power: 损失幂次，2 = MSE 损失，1 = MAE 损失

    返回 (dm_stat, p_value)。dm_stat > 0 表示 A 误差更大（A 更差）；
    若损失差分序列方差为 0（退化），返回 (nan, nan)。
    """
    assert len(e1) == len(e2) and len(e1) > 1
    d = [a ** power - b ** power for a, b in zip(e1, e2)]
    n = len(d)
    dbar = sum(d) / n

    # 长期方差：Bartlett 核 + 滞后 h-1（Newey-West）
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
        return float("nan"), float("nan")  # 损失差分恒定，方差为 0

    dm = dbar / math.sqrt(lrv / n)
    p = 2.0 * (1.0 - _norm_cdf(abs(dm)))
    return dm, p


def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def self_test():
    ok = True

    # --- MAE / RMSE / MAPE：手算值 ---
    y = [1.0, 2.0, 3.0]
    p = [2.0, 2.0, 2.0]
    # MAE = (1 + 0 + 1) / 3 = 2/3
    # RMSE = sqrt((1 + 0 + 1)/3) = sqrt(2/3)
    assert _approx(mae(y, p), 2.0 / 3.0), "MAE 手算不符"
    assert _approx(rmse(y, p), math.sqrt(2.0 / 3.0)), "RMSE 手算不符"

    # MAPE = (10/100 + 10/200)/2 = (0.1+0.05)/2 = 7.5%
    y2 = [100.0, 200.0]
    p2 = [110.0, 190.0]
    assert _approx(mape(y2, p2), 7.5), "MAPE 手算不符"

    # --- DM 检验 ---
    # A 处处优于 B（e1 < e2 逐点）→ dbar < 0 → dm < 0，p 小
    e1 = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    e2 = [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    dm, p = diebold_mariano(e1, e2, h=1)
    assert dm < 0 and math.isfinite(dm), f"DM 应为负，实际 {dm}"
    assert math.isfinite(p) and 0 <= p <= 1, f"p 值异常 {p}"

    # 两模型误差完全一致 → 损失差分恒 0 → 方差 0 → nan（退化，应安全返回）
    dm2, p2v = diebold_mariano([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], h=1)
    assert math.isnan(dm2) and math.isnan(p2v), "恒等误差应返回 nan"

    # 多步 horizon 也应给出有限值
    dm3, p3 = diebold_mariano(e1, e2, h=3)
    assert math.isfinite(dm3) and math.isfinite(p3), "h=3 时 DM 应为有限值"

    print("[PASS] MAE/RMSE/MAPE/DM 全部通过手算校验")
    return ok


if __name__ == "__main__":
    self_test()
