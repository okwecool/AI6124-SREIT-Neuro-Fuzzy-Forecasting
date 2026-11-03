# SRT / STI / SORA 对齐数据（2026-10-28 抓取；同日补齐 SRT 缺价）

本目录是当前 LSTM → 三维特征 → 两条模糊规则 → 双局部专家方案的**初版数据准备**，尚未训练模型。课件 *AI6124 - week 8 - Neural in Neuro-Fuzzy Systems.pdf* 第 28–29 页以 Yahoo Finance 和 `yfinance` 下载新加坡金融行情为示例；本项目使用 [Yahoo Finance 的 SRT.SI](https://finance.yahoo.com/quote/SRT.SI/) 与 [^STI](https://sg.finance.yahoo.com/quote/%5ESTI/history/) 日行情，以及 [MAS 官方 Domestic Interest Rates 下载表](https://eservices.mas.gov.sg/statistics/dir/)中的 SORA 日利率。原始响应和 SHA-256 见 `raw/` 与 `source_manifest.json`。行情使用 Yahoo 的公开 chart 响应，因此重新下载可能随供应方修订而变化；已保存的原始响应是本次实验的固定快照。

## 文件

| 文件 | 用途 |
| --- | --- |
| `raw/yahoo_*.json`、`raw/mas_sora_*.csv` | 原样保存的三个来源响应 |
| `raw/srt_2025-10-24_sgtrades_excerpt.csv` | 2025-10-24 SRT 的另一来源报价摘录及交叉核对链接；不改动 Yahoo 原始响应 |
| `processed/srt_yahoo_normalized.csv`、`sti_yahoo_normalized.csv`、`sora_mas_normalized.csv` | 各来源统一列名后的表；SORA 同时保留 **value date** 与 **publication date** |
| `processed/srt_repaired_normalized.csv` | 用有来源记录的报价补齐 SRT 当日收盘、最高、最低、成交量；开盘价仍为空 |
| `processed/srt_dividends_yahoo.csv` | Yahoo 返回的 9 次 SRT 现金分派事件，仅用于候选总收益率标签及审计 |
| `processed/srt_daily_aligned.csv` | 以 SRT 日期为主键的完整日表；2025-10-24 的补值及来源标志在此表中 |
| `processed/window_index.csv` | 每个合格 20 日窗口的起止日期、预测日、目标日及标签 |
| `processed/srt_windows_unscaled.npz` | 未标准化的 `X`、两个候选 `y`、日期和特征名，可直接供 Python 读取 |
| `processed/fold_assignments.csv` | 三个扩展窗口验证折及一个最终保留测试期的窗口编号 |
| `processed/audit.json` | 行数、缺失值、窗口数及各折数量 |

## 对齐与预测时点

- 预测在 SRT 交易日 `t` 的**收盘价已可取得之后**形成；目标日为源数据表中的下一条 SRT 交易日 `t+1`。
- STI 使用截至 `t` 的最近一个已知收盘价，并记录该收盘价的原日期。本次 SRT 日期与 STI 日期全部重合，未发生 STI 向前填充。
- MAS 表明确给出 SORA 对应日和公布日。按 [MAS 的发布说明](https://www.mas.gov.sg/monetary-policy/SORA)，对应日利率在**下一新加坡工作日 9:00**公布；仅把 `publication_date ≤ t` 的最新利率并入 `t` 的输入。例如 2026-09-30 的输入最多使用当日公布、对应 2026-09-29 的利率。
- Yahoo 的 `adjclose` 是回溯调整后的诊断列，**没有放入模型输入**。初版主要标签为 `log(raw_close[t+1] / raw_close[t])`，用于预测实际收盘价。另导出 `log((raw_close[t+1] + dividend[t+1]) / raw_close[t])` 作为分派后总收益率候选；最终采用哪个标签，须在实验设计中固定，并验证分派完整性。未来目标日的分派金额绝不可作为当日输入。

## 特征与窗口

`X` 的形状为 **`(1182, 20, 6)`**，每个窗口按时间升序放入 20 个 SRT 交易日。六个未缩放特征顺序为：

1. SRT 原始收盘价对数日收益率；
2. SRT 过去 5 个交易日的对数动量；
3. SRT 过去 20 个对数日收益率的总体标准差（未年化）；
4. STI 截至当日可知收盘价在相邻 SRT 交易日之间的对数变化；
5. 当时最新已公布的 SORA，单位为**百分比**；
6. 最新已公布 SORA 与前 20 条已公布记录之差，单位为**百分点**。

数据消融按特征名选择：`仅 SRT` 使用 1–3；`+SORA` 再加入 5–6；`+STI` 再加入 4。**这里没有对全样本拟合标准化器。**每一折只在该折训练样本上计算均值／方差，再将同一变换用于验证或测试；同一训练折内也应按同一规则缩放目标。

## 初步质量检查

| 项目 | 结果 |
| --- | ---: |
| SRT 原始日期行 / Yahoo 有效收盘价 / 补值后有效收盘价 | 1,222 / 1,221 / 1,222 |
| STI 原始日期行 | 1,255（从 2021-10-01 起） |
| MAS SORA 原始日记录 | 1,255（对应日 2021-10-01 至 2026-09-30） |
| 可用的 20 日输入＋下一日标签窗口 | 1,182 |
| 最早／最晚预测日 | 2022-01-12 / 2026-09-29 |
| Yahoo SRT 分派事件 | 9 |

Yahoo 的 SRT 在 **2025-10-24** 有日期行但无 OHLC／收盘价，STI 同日正常。现用 [SGTrades 的该日历史报价](https://sgtrades.com/stock/SRT.SI/price/5/)补入收盘价 **S$0.801**、最低 **S$0.799**、最高 **S$0.803**、成交量 **152,293**；[The Business Times 2025-10-25/26 行情页](https://cassette.sphdigital.com.sg/attachments/businesstimes/8a637227de781466acd26a639d8b03ba076d4518f9994382c7ce2c82f92a25ae)明确标注交易日为 2025-10-24，列出最后成交价 80.1 分、日区间 79.9–80.3 分、约 15.2 万成交量，独立核对了补值。**开盘价未找到可核对来源，仍为空**，使用次日开盘价模拟交易时必须单独处理这一天。Yahoo 原始 JSON 和其标准化表保持不变；处理脚本只在建模副本应用该报价，并把来源和文件 SHA-256 记入 `audit.json`。这恢复了先前因单日缺价排除的 41 个窗口，未做价格插值。

9 条分派事件的日期与半年分派节奏一致；已用 [SGX 的 2024-07 公告](https://links.sgx.com/1.0.0/corporate-announcements/NOYZQ20FW51IVXSN/809400_Announcement%20Dividend%20Distribution%202H2024%20240711.pdf)、[2025-07 公告](https://links.sgx.com/1.0.0/corporate-announcements/NSYA6FGDMPV1FXCJ/851495_Announcement%20Dividend%20Distribution%202H2025%20250710.pdf)及 [2026-07 公告](https://links.sgx.com/FileOpen/Announcement%20Dividend%20Distribution%202H2026%20260710.ashx?App=Announcement&FileID=896004)抽查了三次金额与除息日。余下事件在正式采用总收益率标签或回测现金分派前仍需核对。ETF 价格与它追踪的指数点位不同，不应混称。

## 时间划分

折的筛选依据是**目标日**，这样边界上产生于上月最后交易日、预测下月首交易日的窗口会进入下月验证期。

| 折 | 训练目标日截止 | 评估目标日 | 训练 / 评估窗口 |
| --- | --- | --- | ---: |
| validation_1 | 2024-03-31 | 2024-04 至 2024-09 | 553 / 126 |
| validation_2 | 2024-09-30 | 2024-10 至 2025-03 | 679 / 124 |
| validation_3 | 2025-03-31 | 2025-04 至 2025-09 | 803 / 128 |
| final_holdout | 2025-09-30 | 2025-10 至 2026-09 | 931 / 251 |

最终测试期只在确定模型与超参数后评估。以上日期和标签基于本次快照，若改变补值或目标定义，应重新运行预处理并重审数量。

## 复现

使用 Python、`pandas`、`numpy` 和 `lxml`。下载需要联网；离线预处理只读取 `raw/` 中的已保存文件。

```powershell
python data/download_sources.py
python data/prepare_dataset.py
```
