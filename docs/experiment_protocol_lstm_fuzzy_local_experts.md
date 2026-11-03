# Experimental Protocol — LSTM-Encoded Fuzzy Local Experts for SRT

**Version:** 1.2 review response, 3 November 2026

**Team:** Group 1 — Zou Runze (Ryan) and Ngeow Shan Yong Destin  
**Status:** The team reports that the revised architecture has instructor approval. This protocol fixes the experiment before model training; deviations must be logged with a reason and date.  
**Primary question:** Does a compact LSTM encoder with a three-dimensional Gaussian fuzzy mixture of two local experts forecast the next SRT trading day's raw closing-price return better than comparable LSTM prediction heads?

## 1. Scope and claims

The forecast target is the **SGD trading counter of the CSOP iEdge S-REIT Leaders Index ETF, `SRT.SI`**. SRT is a tradable ETF; its close is not the level of the index it tracks. The main outcome is next-trading-day **raw close-to-close log return**, not a buy/sell label:

```text
y_t = log(C_(t+1) / C_t),       predicted C_(t+1) = C_t exp(ŷ_t)
```

The model sees information available after SRT closes on trading day `t`. It cannot trade at that already observed close. Forecast accuracy is the primary model comparison; the course's trading application additionally asks whether adding forecasts improves terminal compounded wealth under a fixed, timed policy. This is a testable question, not a guaranteed outcome. No claim of “bull” or “crash” experts is made before examining the learned rules. In particular, SRT began trading after the 2020 COVID crash.

The architecture follows the team's approved **LSTM → learned 3D features → fuzzy rules → local return experts** design. It does **not** put fuzzy adjustments inside LSTM input or forget gates. It is a simplified adaptation of the cited neuro-fuzzy forecasting paper, not a reproduction of its attention or ARIX components.

## 2. Frozen data contract

Use the saved 28 October 2026 snapshot in `data/`; do not silently refresh Yahoo or MAS series during an experiment. `data/verify_dataset.py` must pass before fitting. Preserve `data/raw/`, `data/source_manifest.json`, and `data/processed/audit.json` with the submission. Any source correction requires rerunning preprocessing, rechecking all counts, and a new protocol version.

| Item | Fixed definition |
| --- | --- |
| Input file | `data/processed/srt_windows_unscaled.npz` |
| Index and splits | `data/processed/window_index.csv` and `data/processed/fold_assignments.csv`; join by `window_id` |
| Tensor | `X.shape = (1182, 20, 6)`; 20 consecutive SRT trading days, oldest to newest |
| Primary label | `y_raw`, the next SRT raw closing-price log return; `y_total_return_candidate` is **not** used for primary training or selection |
| SRT-only input | columns 0–2: daily log return, 5-day log momentum, trailing 20-return population standard deviation |
| SRT + SORA | columns 0–2 and 4–5: latest published SORA in percent and change over 20 published observations in percentage points |
| SRT + SORA + STI | columns 0–5, including STI as-of log return in column 3; **primary feature group** |
| Sample | 1,182 valid forecast windows; earliest forecast date 2022-01-12, latest 2026-09-29 |

The SORA value used for a forecast must have `publication_date ≤ forecast_date`; it is not selected by its earlier value date alone. STI is the most recent available close as of the forecast. All rolling SRT features end at `t`. The 2025-10-24 SRT close, high, low and volume were repaired from a documented second source; its **open remains missing**. The repaired close is valid for the forecasting experiment. Do not impute that open in the trading exercise.

Fit every input-column standardizer on the **training portion of that run only**, using all time steps of those training windows. Use the same mean and standard deviation on validation/test windows. Fit a separate target mean and standard deviation on the run's training labels; train learned models on standardized `y_raw` and invert predictions before reporting metrics. Replace a zero fitted standard deviation with 1. Never fit transforms on outer validation or final test data. Repeated rows caused by overlapping training windows may enter the scaler; apply this identical rule to every learned model.

Before fitting, inspect the **pretest data only**: plot raw SRT close beside daily log return, compare rolling mean and standard deviation, show the return distribution and zero-return frequency, and mark missing-price dates. First differences/log returns may be closer to stationary than price levels but can still have changing volatility; do not claim a formal stationarity result from a plot. Keep the final holdout out of diagnostic-driven design choices.

## 3. Chronological evaluation

Select rows by **target date**, as already encoded in `fold_assignments.csv`. A validation window may contain past input dates from the prior period, but its label is strictly after every training label in that fold.

| Fold | Training target dates end | Evaluation target dates | Train / evaluation windows |
| --- | --- | --- | ---: |
| `validation_1` | 2024-03-31 | 2024-04 through 2024-09 | 553 / 126 |
| `validation_2` | 2024-09-30 | 2024-10 through 2025-03 | 679 / 124 |
| `validation_3` | 2025-03-31 | 2025-04 through 2025-09 | 803 / 128 |
| `final_holdout` | 2025-09-30 | 2025-10 through 2026-09 | 931 / 251 |

These are **three expanding, forward-time validation folds**, not randomly shuffled cross-validation. The 251-row final holdout is opened only after architectures, training settings, metrics, and source comparisons are fixed. There is no test-set tuning or repeated “try another model after seeing test RMSE.”

For each outer validation fold and each random seed:

1. Reserve the last `ceil(0.15 × training-window count)` training rows, ordered by target date, as an **inner stopping set**. Fit inner-run scalers on the preceding rows only.
2. Train for at most 150 epochs; keep the epoch with the lowest inner-set standardized-label MSE. Stop after 20 epochs without an improvement of at least `1e-4`. Record the best epoch, loss and wall time.
3. Restart the model from the same seed. Refit scalers on **all** outer-training rows, train for exactly the recorded best number of epochs, and predict the untouched outer validation rows. Outer validation is never used for early stopping.

Use seeds `11, 23, 37` for all primary learned-model comparisons. Fix library versions, set the Python/NumPy/PyTorch random seeds, and record CPU/GPU, device, and training times. Minibatches may be shuffled **within** training rows; no validation or test row may enter a gradient update.

For final testing, take the median of the nine best-epoch counts obtained for a given model configuration across its three folds and three seeds, rounded to an integer. Train that configuration from scratch on all 931 pretest rows with each of the same three seeds and a scaler fitted on those 931 rows. Evaluate each final model **once** on the same 251 test rows. Freeze and save the predictions before viewing any test summary.

## 4. Models and training settings

The core experiment uses four heads/references on the **full six-feature group**:

| ID | Method | Purpose |
| --- | --- | --- |
| `Z0` | Constant `ŷ=0` (unchanged close) | No-training price/return reference |
| `LSTM32` | One-layer, 32-hidden-unit LSTM → affine scalar return head | Course-relevant ordinary LSTM benchmark |
| `LSTM3` | Same LSTM → `tanh` 3D projection → affine scalar return head | Controls for the narrow learned representation |
| `Fuzzy2` | Same LSTM and 3D projection → two Gaussian rules → two affine return experts | Proposed hybrid method |

`LSTM32`, `LSTM3`, and `Fuzzy2` receive identical windows and use the same optimizer, folds, seeds, epoch policy, preprocessing and loss. Only their prediction heads differ. We will describe parameter counts and training time so that any gain is not presented as a free improvement.

For `Fuzzy2`, with `z=tanh(W_z h+b_z) ∈ [-1,1]^3`:

```text
d_k(z) = sum_j ((z_j - μ_kj) / s_kj)^2
α(z)   = softmax(-0.5 [d_1(z), d_2(z)])
e_k(z) = a_k·z + c_k
ŷ      = α_1 e_1 + α_2 e_2
```

The fuzzy layer receives **three learned numbers per forecast**, not a flattened 20 × 6 window. `μ_k` and `s_k` are trained jointly with the encoder and experts. Parameterize each scale as `s = 0.1 + 1.9 sigmoid(ρ)` so it remains in `(0.1,2.0)`, initialized at `1.0`. Initialize the two centers at `(-0.25,0,0)` and `(0.25,0,0)` and allow them to train. This resolves an inconsistency in the previous materials: **the primary protocol uses distinct fixed initial centers, not K-means initialization**. No auxiliary rule-balancing or center-separation loss is part of the primary model. Compute memberships with `softmax` for stability. The equations above operate in standardized-return units during training. For plots in raw-return units, transform each expert output with the training target mean and standard deviation before combining it with its rule weight. Rule names are merely `Expert 1` and `Expert 2` until analysis supports more specific labels.

The explicit latent-space Takagi–Sugeno rule is **`IF z_j belongs to Gaussian set A_kj for j=1,2,3, THEN y_k=a_k·z+c_k`**. The three one-dimensional antecedent memberships are multiplied, giving raw rule strength `m_k=exp(-d_k/2)` and `log m_k=-d_k/2`; softmax normalizes these strengths to `α_k`, and `Σ_k α_k e_k` aggregates the consequents. Each Gaussian peaks at one and has convex alpha-cuts, but unconstrained centers do not define an ordered linguistic partition over learned `z`; global raw-membership coverage is not guaranteed. Normalized `Σα_k=1` does **not** prove coverage: all raw `m_k` can be tiny far from both centers. The source paper uses separate low-dimensional representations, ARIX local models, winner-takes-all local optimization, and clustering/separation/balance losses; our single-`z` affine-expert, global-MSE version omits these mechanisms, so it cannot inherit the paper's local-specialization or semantic-interpretability claims. [Source paper, Methods II-B/II-C](https://arxiv.org/html/2510.00960v1).

Common fixed settings: AdamW (`learning rate 0.001`, `weight decay 0.0001`), batch size 64, standardized-label MSE loss, gradient-norm clipping at 1.0, at most 150 epochs, and the stopping policy above. Initialize learned layers with the framework's seeded default except for the specified fuzzy centers/scales. If a run produces nonfinite loss, log it and rerun **all learned models under the same affected setting** at learning rate `0.0003`; never silently repair only the proposed model. Save the actual settings in a machine-readable run manifest.

## 5. Predeclared comparisons

**Primary comparison:** `Fuzzy2` versus `LSTM32`, `LSTM3`, and `Z0`, all using six features. The primary ranking metric is **raw-return RMSE** on the concatenation of the three nonoverlapping outer-validation periods, averaged across the three seeds. All model choices and final reporting remain visible even if `Fuzzy2` loses.

**Input-source ablation:** Train `Fuzzy2` with (a) SRT-only 3 columns, (b) SRT + SORA 5 columns, and (c) all six columns. Keep the same 378 outer-validation target dates, architecture, seeds and training policy. This tests whether the added rate and broad-market sources help. The six-feature group remains the declared primary model regardless of the ablation result; an alternative source group can be shown as a secondary finding but is not substituted post hoc as the main result.

**Parameter sensitivity:** Compare `K=1`, `K=2`, and `K=3` fuzzy rules on the six-feature group over the three development folds with seed 11. `K=1` has one affine expert and is functionally the `LSTM3` head; `K=3` uses initial first-coordinate centers `-0.25, 0, +0.25` and the same scale parameterization. This is a diagnostic of capacity and rule collapse, **not** a test-set model search.

**Mechanism attribution boundary:** `Fuzzy2` differs from `LSTM3` by adding both a gate and a second expert. If it wins, the primary comparison supports the **two-rule TSK head as a whole**, not a unique benefit of Gaussian fuzziness. With two rules, `α_1=sigmoid((d_2-d_1)/2)`; equal diagonal scales make this logit affine in `z`, so some Gaussian gates overlap ordinary learned softmax gates. A priority secondary control, if resources permit **before final testing**, is `Gate2`: the same encoder, 3D projection and two affine experts, with a `3→2 tanh→2 softmax` gate trained under the same loss, folds and seeds. Its gate has 14 scalar parameters versus 12 Gaussian center/scale parameters; report complete parameter counts and nine extra development runs. This compares the Gaussian constraint with a similarly small flexible gate, not with every possible non-fuzzy mixture. If `Gate2` is not run, make no Gaussian-specific superiority claim. It is outside the fixed 48-run core plan and must be declared before any test result is viewed.

Do not add HSI, another REIT ticker, dividends as inputs, attention, or trading-optimized expert policies to the primary run. Such changes form a named extension with separate validation and course documentation.

## 6. Forecast metrics and reporting

Calculate metrics **after inverse scaling** on raw log returns and raw SGD closes:

- Primary: return RMSE. Also show return MAE and the paired RMSE/MAE difference versus `LSTM32` on identical target dates.
- Price reconstruction: `Ĉ_(t+1)=C_t exp(ŷ_t)`, then price MAE and RMSE in SGD. Use the close at forecast date `t` only to reconstruct; it is known at forecast time.
- The `Z0` no-change reference has `ŷ_t=0` and therefore `Ĉ_(t+1)=C_t`. This is the log-return equivalent of adding a predicted price difference to today's close; compare every model with `Z0` on identical dates, not only with other neural networks.
- Direction: sign accuracy on observations whose true return is nonzero, with the number excluded because of a zero true return and a majority-direction baseline. The current dataset has 102 zero-return windows out of 1,182 (8.6%); do not treat them as automatically correct directional forecasts.
- Report each fold and each seed, then the mean and range across seeds; also report one pooled 378-target development score per seed. The final table reports 251 common holdout targets and three seeded runs. Do not interpret overlapping training windows or three seeds as independent market replications or claim statistical significance from a small metric difference alone.

Save `window_id`, `forecast_date`, `target_date`, true return, predicted return, reconstructed price, model ID, source group, fold and seed for every forecast. A simple plot should show actual/predicted returns through time and a fold-wise error comparison; avoid a price-level plot as the only evidence, because adjacent ETF closes are highly similar.

**Export contract v2, implemented:** identify a run by `(model_id, source_group, rule_count, fold, seed)` and a forecast row by that tuple plus `window_id`. New prediction files require `model_id,source_group,rule_count,fold,seed,window_id,y_pred`; use `rule_count=0` for non-fuzzy models, `2` for `Fuzzy2`, and `3` for `Fuzzy3`. Preserve or explicitly migrate the existing six-column `Z0` CSV with `rule_count=0`; the evaluation module now uses the full seven-column key and explicitly migrates legacy Z0 only. The fuzzy case table uses the full forecast key plus `z1,z2,z3,y_mix_raw`; the fuzzy **rule-long table** uses the full key plus `rule_id,alpha,expert_raw,log_membership,distance2`, with exactly `rule_count` distinct rule rows per forecast. Validate key uniqueness, complete window IDs, finite values, nonnegative weights summing to one, and `y_mix_raw = Σ_k alpha_k × expert_raw_k = y_pred` within numerical tolerance. Do not merge ablation diagnostics on `(fold,seed,window_id)` alone.

## 7. Fuzzy interpretation checks

For each `FuzzyK` prediction save the three latent values `z`, per-rule squared distance `d_k`, raw `log m_k=-d_k/2`, normalized weight `α_k`, raw-return expert output `e_k`, and the mixed prediction. Report:

1. Mean and distribution of the **larger** rule weight, rule-weight entropy, and fraction of days with `max(α)>0.8` (rule-use strength without depending on rule labels).
2. Absolute difference `|e_1-e_2|` and each expert's contribution `α_k e_k`. Similar experts or nearly constant 0.5/0.5 weights indicate little useful specialization.
3. **Local error rather than weight alone:** for each rule `k`, define its high-activation set from the upper quartile of that rule's weights on the relevant **training** predictions; apply the frozen cut to validation/test predictions and also report the fixed `α_k>0.8` count. On the *same* high-activation rows compare raw-return MAE of expert `k`, the other expert and the mixture, and report sample count. A local-specialist claim requires the activated expert to beat the other expert in its own region in a reasonably repeated pattern across folds/seeds; no minimum sample count is silently imposed, so small regions are marked inconclusive.
4. **Coverage beyond normalized weights:** report quantiles of `max_k log m_k` and the fraction below the fifth percentile of that quantity on training predictions. Low raw support indicates that a unit-sum `α` may still be choosing between poorly matching rules. Report learned centers/scales and whether the rules overlap or collapse.
5. Rule weights versus **past-only** observed SRT momentum and trailing volatility, with low/middle/high bins cut at the training-period terciles, plus representative dates. Align or explicitly relabel rules when comparing seeds because their numbering can swap. Report forecast errors by these observed-volatility bins and state whether patterns repeat across folds; a plot alone does not establish a causal regime mechanism.

These checks support a precise claim about **which expert contributed to a prediction**. The LSTM features feed the fuzzy antecedents and affine consequents directly, and all components are trained jointly; the system is not two independently trained forecasters averaged at the end. Nevertheless, Gaussian rules over learned latent coordinates are not automatically human-readable linguistic rules. The affine Takagi–Sugeno consequents also trade some linguistic simplicity for flexibility. If activation does not align consistently with past-observed conditions, describe the result as a traceable latent-feature mixture rather than an interpretable market-regime model. The checks do not establish that a rule is a “crash specialist.”

## 8. Trading application and compounded-wealth comparison

The prediction is available **after close `t`**. The fixed primary policy takes `position_t=1` when the model's inverse-scaled raw-return forecast `ŷ_t>0`, otherwise `0` (cash). Set the position at the **next SRT open** (`t+1`) and hold it to the following open. A model-free momentum control uses the already available past-only signal `m_t=srt_momentum_5(t)/5` in the **same long/cash policy**. Also report always-long buy-and-hold and always-cash on identical eligible intervals. `Z0` remains the separate no-change **forecast** baseline; under the strict `>0` policy it is always-cash, so it is not the only no-prediction trading comparison.

For each strategy, start each evaluation fold with wealth `V_0=1` and cash. With the net interval return after position-change costs, calculate `V_{i+1}=V_i(1+r_i^net)`. Plot the full wealth path and report **terminal wealth multiple `V_T`** and cumulative price return `V_T−1`, as well as maximum drawdown, turnover and trade count. For development-period threshold selection, compound the three fold terminal multiples with a cash reset at each fold boundary; show each fold separately too. A better forecast RMSE does not automatically imply greater `V_T`. The central trading contrast is whether the forecast-based policy increases final wealth over the model-free momentum policy and fixed buy-and-hold/cash references on the same dates and cost assumptions. If it does not, report the shortfall rather than changing the test-period rule.

The `ŷ_t>0` policy is the predeclared, **untuned primary trading result**. As a secondary sensitivity, allow only a long-entry threshold `θ∈{0, 0.001, 0.002, 0.003}` in raw daily log-return units: `position_t=1{signal_t>θ}`. Apply the same grid to model forecasts and the daily-scaled momentum signal. Select one `θ` per model class by the highest mean, across seeds `11,23,37`, of the sum of the three development-fold log terminal wealth multiples at the **10-bps** cost assumption; select the momentum threshold by the same three-fold objective without seeds. Break ties by the smaller `θ`. These development validation periods are pretest selection data, so their tuned wealth is descriptive rather than an unbiased out-of-sample profit claim. Save the selected thresholds before final testing, use the same threshold for all final seeds of a model class, and never retune after viewing the holdout. No other trading parameter is optimized.

**Causal trade and cost ledger.** Use one-way proportional transaction costs of **0, 10 and 20 basis points** of traded notional, with 10 bps primary; these are simulation assumptions, not claims about broker fees or spread. At each **observed** open, first value existing cash and shares at that open, then apply only the latest signal known before that open. On cash-to-long entry, buy shares with available cash after the entry fee; on long-to-cash exit, sell shares and deduct the exit fee. If the position stays unchanged, do not charge a fee. The planned end of each validation/test period is known in advance: liquidate at its last observed open and charge that exit fee, but never borrow a price from the next period. This boundary liquidation is distinct from the unplanned missing-quote event. `V_T` is post-cost terminal cash; report annualized return only with its calendar-day convention stated. Log each observed-open signal time, requested/actual position, shares, cash, costs and wealth, then verify self-financing accounting.

**Missing-open rule, fixed before final testing.** First seek an independently verifiable 2025-10-24 SRT open, record source/units and checksum, and rerun the data audit if it is repaired. Do not infer the open from the close. If a verified open is unavailable at the freeze point, the primary price-only trading report uses a predeclared **no-trade-and-carry** data-availability simulation: at an unobserved open, make no trade and retain the previous shares and cash; do not liquidate at the previous open because the next quote will be missing. At the next observed open, value the carried position using the price ratio between the two observed opens and apply the most recent forecast made before that open. This policy never consults a future `valid` flag to decide an earlier trade. Apply the same observed-open calendar to all strategies; always-long remains continuously invested between the observed opens. Label the result a **constrained-data simulation**, since an actual SRT open existed but is unavailable in the saved data. A common-complete-interval segmented analysis may be shown as a sensitivity, explicitly labelled segmented and not called continuous buy-and-hold. The causal observed-open ledger replaces the earlier future-validity implementation; the baseline portfolio CSV has been regenerated with the new ledger.

**ETF distributions.** Before making a total-wealth claim, verify each of the nine known SRT distribution events against issuer or exchange records: ex-date, amount per share, payment date and currency. A share held **before the ex-date** earns a receivable on the ex-date; add that receivable to marked wealth, transfer it to cash on the payment date without double counting, and use cash for future purchases only when a policy-directed entry occurs (no automatic dividend reinvestment). Freeze these bookkeeping rules before viewing final portfolio results. Keep the raw-price forecast target unchanged. If the events cannot all be verified, report only **price-only, dividend-excluded** strategy wealth and explicitly state that its ranking may differ from actual ETF total wealth.

**Forecast/trade horizon diagnostic.** The label `y_t=log(C_{t+1}/C_t)` includes price movement that may occur before the trade can enter at `O_{t+1}`, whereas the executable interval is `O_{t+1}→O_{t+2}`. On development dates, compare **all three** of true close-to-close return, predicted close-to-close return and realized open-to-open trading return using aligned plots and correlations. These are descriptive diagnostics, not an extra test-selection target. A correct return prediction can fail to profit if the rise arrives in the pre-entry overnight gap; neither a low RMSE nor a poor trade result alone identifies the cause. If development evidence shows severe mismatch, a separately named open-to-open-target extension may be specified, trained and frozen **before** final testing; it does not replace the primary close-return experiment or retroactively change its claims.

## 9. Execution, ownership, and deliverables

| Owner | Main handoff |
| --- | --- |
| Ryan | Freeze and verify data; resolve the missing-open and distribution-source decisions before final evaluation; own split/scaling utility, zero baseline, model-free momentum control, causal wealth ledger, figures, data-source and timing explanation |
| Destin | Implement `LSTM32`, `LSTM3`, `Fuzzy2`; expose `forward(X) → ŷ` and, for fuzzy, `{z, α, expert_predictions, raw log memberships, distances}`; own training loop, checkpoints, seed/run manifest and version-2 case/rule diagnostics |
| Joint | Run one-fold smoke test, inspect sample predictions, review leakage and diagnostic-key checks, freeze development-selected trading thresholds and source-accounting rules, interpret results, write literature/paper comparison and final Notebook/slides |

Minimum model interface: a model accepts a float tensor `(batch,20,F)` with `F∈{3,5,6}` and returns a standardized-return prediction `(batch,)`. The evaluation layer inverse-transforms it to raw log return. For `FuzzyK`, also return `z` `(batch,3)` and weights, expert returns, log memberships and squared distances, each `(batch,K)`. Keep model outputs and train-only scalers with each checkpoint. A smoke test checks shapes, finite values, `α≥0`, `Σ_k α_k≈1`, and exact agreement between the returned mixture and `Σ_k α_k e_k`; the version-2 export checks in §6 must pass before parallel ablations.

The final submission is one reproducible **Jupyter Notebook** containing code, data and method explanation, a chronological and taxonomic literature review, benchmark and fold results, parameter and source ablations, rule-coverage/local-error analysis, matched trading wealth paths, limitations, future **theoretical** questions and other plausible application domains, plus concise results slides. Discuss the source paper's reported accuracy/interpretability pattern against this project's observed pattern, while avoiding direct numeric RMSE ranking across unrelated datasets and horizons. The course PDF states a **23:59 Singapore time, 18 November 2026** submission deadline for both students in NTULearn. The old proposal slides remain a historical record; final slides must match the implemented method and measured results.

## 10. Decision log and change control

| Decision | Protocol setting | Status |
| --- | --- | --- |
| Revised LSTM→fuzzy→experts architecture | Team reports instructor approval | Confirmed by Ryan on 28 October 2026; keep the approval record. This does not by itself establish approval of every later training/evaluation detail. |
| Primary target | `y_raw` next-close log return | Fixed |
| Primary inputs | SRT + published SORA + STI, six columns | Fixed; ablations are secondary |
| Fuzzy initialization and optimizer | Distinct small centers, bounded scales, AdamW; no K-means or balance/separation loss in the primary run | Fixed; historical proposal and old design prose differ, so final slides/Notebook must follow this protocol |
| Primary metric and holdout | Pooled forward-validation return RMSE; one final 251-window test | Fixed |
| Trading ledger and data gap | Untuned `θ=0` plus pretest-selected threshold sensitivity; self-financing next-open ledger; verify missing open or use causal no-trade-and-carry simulation | Causal ledger implemented and synthetic future-missing-open test passes; final open remains unavailable |
| ETF distributions | Verify nine events and book pre-ex-date entitlement, ex-date receivable and payment-date cash; otherwise price-only claim | Nine issuer events source-checked; causal gross-distribution ledger implemented; final path still constrained by missing open |
| Diagnostic export | Version-2 full run key and per-rule long table, including raw membership support and local errors | Full-key export and validator implemented; local-error and regime analyses remain Phase 3 work |
| Gaussian mechanism claim | `Fuzzy2` versus linear heads tests the whole TSK head; optional `Gate2` is predeclared for a narrower gate comparison | No Gaussian-specific superiority claim unless the control is actually run |

If a necessary repair changes a fixed row, record the old/new value, reason, approver, date, and rerun IDs. Finish selection using the three development folds before opening the final holdout. Protocol changes after seeing test outcomes must be labeled **post hoc** and cannot replace the preregistered result.

## Source materials

- Local course PDF: `AI6124 - week 8 - Neural in Neuro-Fuzzy Systems.pdf`, especially pages 17, 20–23 and 31–39.
- Week 11 *Neuro-Fuzzy Systems and POPFNN* lecture, opening project guidance on chronological testing, return targets and multiplicative profits, plus the later Fuzzy LSTM discussion; checked against the team-supplied 3 November 2026 transcript. Version 1.1 incorporates this guidance before learned-model training.
- Current design: `lstm_fuzzy_local_experts_design_en.md` and `lstm_fuzzy_local_experts_design_zh.md`.
- Design audit and remediation priorities: `design_review_2026-11-03_zh.md`; version 1.2 set the design; Phases 1–2 implement the core model, export and ledger contracts.
- Data audit and source provenance: `data/README.md`, `data/processed/audit.json`, `data/source_manifest.json` and `data/verify_dataset.py`.
- Ožbot, Škrjanc and Štruc, [*A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting*](https://arxiv.org/html/2510.00960v1) (method inspiration; different task, horizon and architecture).
- MAS, [SORA publication policy](https://www.mas.gov.sg/monetary-policy/SORA) (the next-business-day 9:00 publication constraint used in the data alignment).
