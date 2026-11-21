# AI6124 Project Status and Next Steps: Ryan–Destin Alignment Draft

**Date:** 4 November 2026
**Purpose:** Internal discussion draft, not the final report or slides. **Version status:** The first version's final test is frozen and complete; the second version's optimization experiments are development-only. This document separates measured results, explicit course guidance, and our interpretation.

## 1. Executive summary

We have implemented a reproducible **LSTM → three-dimensional latent features → two Gaussian fuzzy rules → two local experts** experiment for a Singapore REIT ETF. It includes ordinary-model benchmarks, chronological validation, a frozen final test, and a trading comparison. The original fuzzy model has **not demonstrated a material improvement**. A later development experiment found a more substantial forecasting improvement from distribution information announced before prediction: RMSE fell **2.76%** against the zero-return baseline. That improvement comes from new, observable information; the fuzzy network still does not beat the resulting stronger baseline. We therefore have a complete experiment and a qualified optimization lead, but cannot claim a validated fuzzy-model or trading improvement.

## 2. Research question, data, and models

- **Target:** The next trading day's **raw close-to-close log return**, `log(C_{t+1}/C_t)`, for the SGD counter `SRT.SI` of the CSOP iEdge S-REIT Leaders Index ETF. We reconstruct the next close as `C_t exp(ŷ_t)`. SRT is a tradable ETF, **not** the level of its underlying index; the target is not a buy/sell label.
- **Original inputs:** Twenty SRT trading days × six features observable by the forecast close: SRT daily return, five-day momentum, trailing 20-day volatility, the available STI return, and the latest published SORA level and change. There are **1,182** valid windows.
- **Proposed model:** A one-layer LSTM with 32 hidden units produces `h`, projected to `z∈R³`. Two Gaussian rules assign weights `α₁,α₂` to two affine local predictions `e₁,e₂`; the forecast is `ŷ=α₁e₁+α₂e₂`. The fuzzy layer receives **three learned values per forecast**. It does not alter the LSTM's internal input or forget gates.
- **Comparators:** Constant zero return/unchanged price (`Z0`), ordinary `LSTM32`, a single head on the same three-dimensional representation (`LSTM3`), and a non-Gaussian two-expert gate (`Gate2`). Version 2 adds training-mean and Ridge references. These controls help separate the effects of compression, a second expert, and Gaussian gating.
- **Additional sources:** Three development-period SRT distributions documented in SGX manager announcements and a fixed `CLR.SI` daily snapshot with 2,243 observations. CLR is another pure Singapore REIT ETF; its earlier observations cannot simply be appended to SRT's price history.

## 3. Completed work and test method

| Area | Completed work |
| --- | --- |
| Data and audit | Saved source snapshots, preprocessing, source hashes, and aligned SRT/STI/MAS SORA data. SORA uses its actual publication date. A sourced quote repairs the missing SRT close on 24 October 2025; that day's open remains missing. Prices, returns, missingness, and distributions have been audited. |
| Chronology | Three expanding development folds train on 553/679/803 windows and evaluate on 126/124/128 later targets: **378 distinct validation dates**. A further **251 dates** from October 2025 through September 2026 were used once as the frozen first-version final test. |
| Training | Input and target scalers are fitted only to each run's training portion. The last 15% of the training period selects the stopping epoch; the model is restarted with the same seed and refitted on all outer-training rows. Seeds are 11/23/37, with shared AdamW/MSE settings and run manifests. Version 1 completed 48 core development configurations, nine `Gate2` configurations, and 12 frozen final fits. |
| Forecast evaluation | Matched-date raw-return RMSE/MAE, reconstructed-price error, and directional accuracy, reported by fold and seed. Fuzzy weights, expert outputs, local error, and mixture identities are checked. |
| Trading application | A forecast is made after close `t`; trades occur at the next observed open. Terminal compounded wealth is compared with cash, always holding the ETF, and past-only momentum. The primary cost is **10 bps per side**, with 0/20 bps and distribution-ledger sensitivities. A missing open triggers the declared no-trade-and-carry simulation, without looking ahead. |
| Artifacts | Code, tests, manifests, predictions, diagnostics, an executed Notebook, and an earlier results deck exist. Version 2 is **not yet** in the earlier handoff ZIP. The full **31-test suite passes**, as does verification of the frozen Version 1 outputs. |

The first version's final test has already been examined. All later Version 2 trials use development folds and cannot treat those 251 days as a fresh blind test.

## 4. Measured outcomes

The following RMSE values concern the **same one-day raw-return target**; neural results average three seeds. Compare methods within each column, since the development and final periods are different market periods.

| Version 1 method | 378-day development RMSE | Frozen 251-day final RMSE | Final MAE |
| --- | ---: | ---: | ---: |
| `Z0` | **0.009431** | 0.006377 | **0.004547** |
| `LSTM32` | 0.009624 | 0.006384 | 0.004651 |
| `LSTM3` | 0.009473 | 0.006355 | 0.004593 |
| `Gate2` | 0.009456 | 0.006353 | 0.004582 |
| `Fuzzy2` | 0.009489 | **0.006349** | 0.004595 |

`Fuzzy2` beats `Z0` by about **0.44%** in final-period RMSE but loses in development and in final MAE. Its margin over `Gate2` and `LSTM3` is tiny. Under the primary trading policy, mean terminal wealth is approximately **0.999** for `Fuzzy2` versus **1.000** for cash; two seeds never trade. Rule weights stay near 0.5 and do not support stable “strong-market/recession” local-expert labels. Thus the 0.44% result is **not a convincing optimization**, a profitable trading result, or evidence of interpretable market regimes.

Additional, recorded Version 2 development experiments found:

| Experiment | Outcome and interpretation |
| --- | --- |
| Simple references and local expert loss | Training-mean RMSE is 0.009444; initial-grid Ridge is 0.009956; `Fuzzy2` with a fixed `λ=0.1` local loss is 0.009505. The loss reduces expert-output cancellation but does not improve the mixture or establish stable specialization. |
| Stronger Ridge, OHLCV, and a five-day target | Stronger Ridge and separate price-decomposition or volume feature groups do not beat `Z0`. For a separately defined five-trading-day task, Ridge is about 7.06% worse than that task's zero-return baseline; five-day RMSE is not comparable numerically with one-day RMSE. |
| **Announced distributions (`DivSchedule`)** | A deterministic adjustment using distribution amounts and ex-dates announced before the forecast produces development RMSE **0.009170** and MAE **0.006148**: **2.76%/2.26%** better than `Z0`, with RMSE gains in 2/3 folds. Only **three of 378** evaluation targets are ex-dividend dates. This is an **information-driven development candidate**, without independent confirmation. |
| Residual fuzzy model and CLR | Training `Fuzzy2` on the return remaining after the distribution adjustment, then adding the announced adjustment back, yields RMSE **0.009232**, still **0.67% worse** than `DivSchedule`. Adding same-day CLR return to a Ridge residual model improves on its six-input Ridge counterpart by **1.24%**, reaching RMSE **0.009197**, but remains **0.29% worse** than `DivSchedule`. The distribution gain cannot be attributed to the fuzzy mechanism. |

## 5. Instructor guidance and our interpretation

“Explicit guidance” below reflects our review of the local Week 8 slides and the opening of the Week 11 lecture. “Our interpretation” is a project judgement, **not** a prediction of the instructor's grade. See the [course-requirements review](design_review_2026-11-03_zh.md) and [experimental protocol](experiment_protocol_lstm_fuzzy_local_experts.md).

| Explicit course guidance | Current fit |
| --- | --- |
| Define a financial time-series problem and a hybrid/neuro-fuzzy method; compare with standard AI/ML models | The LSTM+TSK architecture and ordinary LSTM, compressed-head, ordinary-gate, and no-change controls are defined and tested. |
| Inspect the data; test on later observations; optimize parameters only on earlier training data | Version 1 uses chronological folds, train-only scaling and stopping, and a configuration frozen before final evaluation. Version 2 remains development exploration because that final period has already been seen. |
| Predict price changes rather than raw non-stationary price levels; reconstruct price and compare with “tomorrow equals today” | The target is next-day log return, with reconstructed-price metrics and `Z0`. |
| Compare terminal compounded wealth with no-prediction/simple strategies | Matched-rule cash, always-hold, momentum, and model comparisons are complete; no practical trading improvement is established. |
| Discuss literature, parameter choices, strengths/limits, other applications, and theoretical questions; deliver a reproducible Notebook and results slides | Code and drafts exist, but final writing and presentation need joint review. The local course material records separate NTULearn submissions by both members by **18 November 2026, 23:59 Singapore time**; confirm against the latest course-platform notice. |

**Our interpretation:** The instructor likely expects a controlled “standard model → proposed improvement” comparison, a defensible mechanism, and an explanation of when it helps, rather than a complex architecture or one attractive score. The slides do **not** say that a negative result automatically fails the project, and we cannot predict the grade. Our methods and evaluation are aligned with the course; the stronger goal of a repeatable, useful **fuzzy-specific** gain has not been met. The proposed phrase *interpretable ... under different market regimes* also overstates what the learned rules currently show, so the final title should wait for evidence of semantic regimes.

## 6. Decisions and next work for Ryan and Destin

1. **Agree on success criteria.** If the goal is forecasting-pipeline improvement over `Z0`, the announced-distribution approach is a development candidate. If the goal is improvement specifically from the fuzzy network, it must beat `DivSchedule`, `LSTM3`, and `Gate2` given the same information. It does not yet do so.
2. **Choose one bounded new experiment.** First assess whether CLR's longer history can support a separately defined REIT target or auxiliary pretraining task. Freeze matched non-fuzzy controls, information timing, and a genuinely unused evaluation period before training. CLR tracks a different index and cannot be spliced onto SRT. If SRT remains the target, give every comparator the same announced-distribution information; avoid broad searches over loss weights, rule counts, or trading thresholds.
3. **Keep the evidence boundary clear.** The original 251-day final period has been viewed and cannot be reused as a blind test for Version 2. A new candidate needs independent data/time for confirmation. Any trading claim also needs matched execution timing and costs; lower forecast RMSE alone does not establish profit.
4. **Coordinate deliverables.** Ryan can continue source verification, timing rules, code, and controlled experiments. Destin can work in parallel on method and literature checks, the course rubric, and the Notebook/figures/slides narrative. Both should check key tables, limitations, and submission instructions. The existing Notebook and deck describe the frozen first version and **do not incorporate Version 2**; update the presentation only when the research conclusion is stable.

**Key records:** [frozen final results](final_results_2026-11-04.md), [Version 2 first round](v2_first_round_results_2026-11-04_zh.md), [Version 2 follow-up](v2_followup_results_2026-11-04_zh.md), [development plan](development_plan.md), and [Destin handoff notes](README_DESTIN_HANDOFF.md). This draft summarizes saved evidence; it does not replace the original protocol or frozen-test conclusion.
