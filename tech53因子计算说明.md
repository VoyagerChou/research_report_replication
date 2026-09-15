# 53 个技术因子计算说明

实现见 `input_data_update/build_tech53.py`，产物 `data/derived/tech53.dat` (T×J×53)。
因子名单与研报附录表 2 逐字一致。窗口约定：1m=21、3m=63、6m=126、1y=252、2y=504 个交易日；
滚动统计要求窗内有效样本 ≥80%，否则输出 NaN。

## 基础序列

`CLOSE` 后复权收盘价 ｜ `VOL` 成交量 ｜ `R` 日收益（停牌置 NaN）｜ `AMT` 成交额（元）
｜ `VWAP` 后复权成交均价 ｜ `TURN` 换手率（%）｜ `SPEC` Barra 特质收益（小数）

## 动量族

| 因子 | 计算 |
|---|---|
| `mom_3m` `mom_6m` `mom_1y` `mom_2y` `reverse_1m` | `CLOSE_t / CLOSE_{t−n} − 1`，n = 63/126/252/504/21 |
| `mom_1y_1m` | `(1+mom_1y) / (1+reverse_1m) − 1` |
| `specific_mom1` `specific_mom6` `specific_mom12` | 特质收益 n 日对数累计 `expm1(Σ log1p(SPEC))`，n = 21/126/252 |
| `up_list` `down_list` | 当日收益全市场排名进前 80 名记 1，对最近 20 日作半衰期 10 日的指数加权 |
| `ideal_reversal` | 最近 21 日按当日成交额排序，成交额最大 10 日收益和 − 最小 10 日收益和 |

## 行为族

| 因子 | 计算 |
|---|---|
| `clo_5d_60d` `vwap_5d_60d` | `MA5 / MA60`，分别取 `CLOSE` 与 `VWAP` |
| `close_max_div_min_1m` `_3m` `_6m` | `max(CLOSE, n) / min(CLOSE, n)`，n = 21/63/126 |
| `return_max_1m` | `max(R, 21)` |
| `return_std_1w` `_1m` `_3m` `_6m` `_12m` | `std(R, n)`，n = 5/21/63/126/252 |

## 情绪族

| 因子 | 计算 |
|---|---|
| `amt_1m_3m` | `MA21(AMT) / MA63(AMT)` |
| `ln_volume_mean_1m` `_3m` `_6m` `_12m` | `log( MA_n(VOL) )`，n = 21/63/126/252 |
| `ln_volume_std_1m` `_3m` `_6m` `_12m` | `log( std_n(VOL) )` |
| `volume_1m_div_12m` `volume_1m_minus_12m` | `MA21(VOL)` 与 `MA252(VOL)` 的比值 / 差值 |
| `volume_std_1m_div_12m` | `std21(VOL) / std252(VOL)` |
| `swap_1m` `swap_3m` `swap_1y` | `log( MA_n(TURN) )`，n = 21/63/252 |
| `turnover_mean_1m` `_3m` `_6m` | `MA_n(TURN)`，n = 21/63/126 |
| `turnover_std_1m` `_3m` `_6m` | `std_n(TURN)` |
| `turnover_stdrate_1m` `_3m` `_6m` | `std_n(TURN) / MA_n(TURN)` |
| `illiq` | `MA21( |R| / (AMT / 1e8) )`，Amihud 非流动性 |
| `corr_close_turnover` | `CLOSE` 与 `TURN` 的 20 日滚动相关系数 |
| `ivr` | `std60(SPEC) / std60(R)` |
| `duvol` | `std60(R \| R>0) / std60(R \| R<0)`，各腿最少 12 个样本 |
| `ncskew` | 特质收益 60 日负偏度 `−(m₃ − 3m₁m₂ + 2m₁³) / var^1.5` |
| `spread_bias` | 见下 |

**`spread_bias`**：逐月末以过去 252 日收益计算全市场相关矩阵，为每只股票取相关性最高的
10 只等权构成特征组合（次月沿用）；个股与特征组合各取 60 日对数累计收益作差，再作 60 日
滚动 z 标准化。参与相关矩阵要求过去 252 日交易日占比 ≥90%。

## 后处理

±inf 置 NaN，上市前与退市后置 NaN。逐交易日在全市场有值截面上计算均值与标准差存入
`tech53_stats.npz`，模型消费时作 `(x − μ_t) / σ_t`，NaN 置 0（落在截面均值位）。

## 研报未披露、由本项目确定的细则

研报附录约半数因子的"计算方式"列为因子名复读，以下细则无字面依据：

- 全部滚动窗口的最小有效样本比例（取 80%）
- `up_list` / `down_list` 的排名阈值（前 80 名）与衰减方式（20 日、半衰 10 日）
- `ideal_reversal` 的"每笔成交"以**成交额**代替笔数（无逐笔数据）
- `illiq` 分母取成交额（亿元）；`corr_close_turnover` 窗口取 20 日；`ivr` 以特质波动比总波动
- `ncskew` 以特质收益而非原始收益计算
- `spread_bias` 的相关矩阵刷新频率（逐月末）与 90% 覆盖门槛
- 标准化基准取**全市场**有值截面（非池内，避免窗口内幸存者倾斜），且不作截断
- `vwap_5d_60d` 使用后复权 vwap

