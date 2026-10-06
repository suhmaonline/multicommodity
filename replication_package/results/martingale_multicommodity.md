# C 方案逐品种 martingale / 随机游走诊断

| 品种 | ADF水平 p | ADF收益 p | KPSS水平 p | VR(2) | VR(8) | VR(16) | 分类 |
|---|---|---|---|---|---|---|---|
| 土豆(potato) | 0.0155 | 0.0000 | 0.0100 | 0.686 | 0.478 | 0.607 | weakly non-stationary (borderline); mean-reverting at all horizons |
| 大白菜(Chinese cabbage) | 0.0133 | 0.0000 | 0.0365 | 0.960 | 1.726 | 2.711 | weakly non-stationary (borderline); mixed autocorrelation |
| 大蒜(garlic) | 0.5184 | 0.0036 | 0.0100 | 0.702 | 0.399 | 0.519 | unit-root / persistence-dominated (I(1)); mean-reverting at all horizons |
| 洋葱(onion) | 0.1146 | 0.0000 | 0.0100 | 0.634 | 0.308 | 0.329 | unit-root / persistence-dominated (I(1)); mean-reverting at all horizons |
| 生姜(ginger) | 0.3410 | 0.0005 | 0.0100 | 0.705 | 0.587 | 0.844 | unit-root / persistence-dominated (I(1)); mixed autocorrelation |
| 白萝卜(white radish) | 0.0321 | 0.0000 | 0.1000 | 0.825 | 1.023 | 1.528 | trend-stationary / weakly persistent; mixed autocorrelation |
| 青椒(green pepper) | 0.0279 | 0.0000 | 0.0946 | 1.076 | 2.407 | 3.825 | trend-stationary / weakly persistent; trending (positive autocorrelation) at longer horizons |