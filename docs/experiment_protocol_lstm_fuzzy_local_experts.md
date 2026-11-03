# Experimental Protocol — LSTM-Encoded Fuzzy Local Experts for SRT

**Version:** 1.0, 28 October 2026  
**Team:** Group 1 — Zou Runze (Ryan) and Ngeow Shan Yong Destin  
**Status:** The team reports that the revised architecture has instructor approval. This protocol fixes the experiment before model training; deviations must be logged with a reason and date.  
**Primary question:** Does a compact LSTM encoder with a three-dimensional Gaussian fuzzy mixture of two local experts forecast the next SRT trading day's raw closing-price return better than comparable LSTM prediction heads?

## 1. Scope and claims

The forecast target is the **SGD trading counter of the CSOP iEdge S-REIT Leaders Index ETF, `SRT.SI`**. SRT is a tradable ETF; its close is not the level of the index it tracks. The main outcome is next-trading-day **raw close-to-close log return**, not a buy/sell label:

```text
y_t = log(C_(t+1) / C_t),       predicted C_(t+1) = C_t exp(ŷ_t)
```

The model sees information available after SRT closes on trading day `t`. It cannot trade at that already observed close. Forecast accuracy is the primary result; a separately timed portfolio exercise addresses the course's trading application. No claim of “bull” or “crash” experts is made before examining the learned rules. In particular, SRT began trading after the 2020 COVID crash.

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

Common fixed settings: AdamW (`learning rate 0.001`, `weight decay 0.0001`), batch size 64, standardized-label MSE loss, gradient-norm clipping at 1.0, at most 150 epochs, and the stopping policy above. Initialize learned layers with the framework's seeded default except for the specified fuzzy centers/scales. If a run produces nonfinite loss, log it and rerun **all learned models under the same affected setting** at learning rate `0.0003`; never silently repair only the proposed model. Save the actual settings in a machine-readable run manifest.

## 5. Predeclared comparisons

**Primary comparison:** `Fuzzy2` versus `LSTM32`, `LSTM3`, and `Z0`, all using six features. The primary ranking metric is **raw-return RMSE** on the concatenation of the three nonoverlapping outer-validation periods, averaged across the three seeds. All model choices and final reporting remain visible even if `Fuzzy2` loses.

**Input-source ablation:** Train `Fuzzy2` with (a) SRT-only 3 columns, (b) SRT + SORA 5 columns, and (c) all six columns. Keep the same 378 outer-validation target dates, architecture, seeds and training policy. This tests whether the added rate and broad-market sources help. The six-feature group remains the declared primary model regardless of the ablation result; an alternative source group can be shown as a secondary finding but is not substituted post hoc as the main result.

**Parameter sensitivity:** Compare `K=1`, `K=2`, and `K=3` fuzzy rules on the six-feature group over the three development folds with seed 11. `K=1` has one affine expert and is functionally the `LSTM3` head; `K=3` uses initial first-coordinate centers `-0.25, 0, +0.25` and the same scale parameterization. This is a diagnostic of capacity and rule collapse, **not** a test-set model search. If time allows, a same-size ordinary neural softmax gate with two affine experts can be added as a secondary control to separate “fuzzy Gaussian gate” from “any mixture gate.”

Do not add HSI, another REIT ticker, dividends as inputs, attention, or trading-optimized expert policies to the primary run. Such changes form a named extension with separate validation and course documentation.

## 6. Forecast metrics and reporting

Calculate metrics **after inverse scaling** on raw log returns and raw SGD closes:

- Primary: return RMSE. Also show return MAE and the paired RMSE/MAE difference versus `LSTM32` on identical target dates.
- Price reconstruction: `Ĉ_(t+1)=C_t exp(ŷ_t)`, then price MAE and RMSE in SGD. Use the close at forecast date `t` only to reconstruct; it is known at forecast time.
- Direction: sign accuracy on observations whose true return is nonzero, with the number excluded because of a zero true return and a majority-direction baseline. The current dataset has 102 zero-return windows out of 1,182 (8.6%); do not treat them as automatically correct directional forecasts.
- Report each fold and each seed, then the mean and range across seeds; also report one pooled 378-target development score per seed. The final table reports 251 common holdout targets and three seeded runs. Do not interpret overlapping training windows or three seeds as independent market replications or claim statistical significance from a small metric difference alone.

Save `window_id`, `forecast_date`, `target_date`, true return, predicted return, reconstructed price, model ID, source group, fold and seed for every forecast. A simple plot should show actual/predicted returns through time and a fold-wise error comparison; avoid a price-level plot as the only evidence, because adjacent ETF closes are highly similar.

## 7. Fuzzy interpretation checks

For each `Fuzzy2` prediction save the three latent values `z`, both rule weights `α`, both expert outputs `e`, and the mixed prediction. Report:

1. Mean and distribution of the **larger** rule weight, rule-weight entropy, and fraction of days with `max(α)>0.8` (rule-use strength without depending on rule labels).
2. Absolute difference `|e_1-e_2|` and each expert's contribution `α_k e_k`. Similar experts or nearly constant 0.5/0.5 weights indicate little useful specialization.
3. Rule weights versus **past-only** observed SRT momentum and trailing volatility, with thresholds defined from the training period, plus representative test dates. Align or explicitly relabel the two rules when comparing seeds because their numbering can swap.
4. Forecast errors by predeclared low/middle/high trailing-volatility bins. State whether any pattern repeats across seeds and folds; do not infer a causal regime mechanism from a plot.

These checks support a precise claim about **which expert contributed to a prediction**. They do not make the latent coordinates direct economic variables or establish that a rule is a “crash specialist.”

## 8. Trading application (secondary)

The prediction is available **after close `t`**. A tradable long/cash signal is `position=1` if `ŷ_t>0`, else `0`, with no threshold tuned on the holdout. Place the new position at the **next SRT open** (`t+1`). Measure each position over the next-open to following-open interval; compare on identical eligible intervals with always-long buy-and-hold and always-cash. This realized open-to-open return is intentionally different from the forecast's close-to-close label. Report that horizon mismatch and treat portfolio results as an application, not proof that close-return RMSE directly translates into profit.

Use one-way transaction-cost assumptions of **0, 10 and 20 basis points per position change** as a sensitivity analysis, with 10 bps primary. These are simulation assumptions, not a claim about actual broker fees or ETF spread. Report cumulative return, annualized return only with its trading-day convention stated, maximum drawdown, turnover and trade count. Use the same eligible dates and cost assumptions for every strategy. Exclude the last signal of each evaluation fold when closing its open-to-open trade would require a price outside that fold; this prevents the final development fold from reading an October 2025 holdout open. The final available open is 2026-09-30, so a signal needing a later exit open is also excluded. The missing **2025-10-24 open** means intervals needing that entry or exit open are excluded from *all* portfolio comparisons; mark the break in the equity curve, treat the omitted gap as cash for every strategy, and do not fill the open with the close. To claim cash total-return performance, verify all nine SRT distribution events against official records and credit distributions to shares held before the ex-date. Until then, label the output **price-only, dividend-excluded exploratory backtest**.

## 9. Execution, ownership, and deliverables

| Owner | Main handoff |
| --- | --- |
| Ryan | Freeze and verify data; own split/scaling utility, zero baseline, metrics, price/portfolio evaluation, figures, data-source and timing explanation |
| Destin | Implement `LSTM32`, `LSTM3`, `Fuzzy2`; expose `forward(X) → ŷ` and, for fuzzy, `{z, α, expert_predictions}`; own training loop, checkpoints, seed/run manifest and rule diagnostics |
| Joint | Run one-fold smoke test, inspect sample predictions, review leakage checks, interpret results, write literature comparison and final Notebook/slides |

Minimum handoff contract: a model accepts a float tensor `(batch,20,F)` with `F∈{3,5,6}` and returns a standardized-return prediction `(batch,)`. The evaluation layer inverse-transforms it to raw log return. The fuzzy model additionally returns arrays of shapes `(batch,3)`, `(batch,2)`, `(batch,2)` for `z`, weights and expert returns. Keep model outputs and train-only scalers with each checkpoint. A smoke test should check output shapes, finite predictions, `α≥0`, `α_1+α_2≈1`, and exact agreement between the returned mixture and `sum_k α_k e_k`.

The final submission is one reproducible **Jupyter Notebook** containing code, data and method explanation, literature review, benchmark and fold results, model-parameter discussion, interpretation, portfolio comparison, limitations/future work and plots, plus concise presentation slides. The course PDF states a **23:59 Singapore time, 18 November 2026** submission deadline for both students in NTULearn. Cite the course lecture and the paper; compare the paper at the level of method and task, not its numerical results on a different dataset as if directly comparable.

## 10. Decision log and change control

| Decision | Protocol setting | Status |
| --- | --- | --- |
| Revised LSTM→fuzzy→experts method | Team reports instructor approval | Confirmed by Ryan on 28 October 2026; keep the approval record with project materials |
| Primary target | `y_raw` next-close log return | Fixed |
| Primary inputs | SRT + published SORA + STI, six columns | Fixed; ablations are secondary |
| Fuzzy initialization | Distinct small centers, bounded scales; no K-means or balance penalty | Fixed here; update any conflicting slide text before final presentation |
| Primary metric and holdout | Pooled forward-validation return RMSE; one final 251-window test | Fixed |
| Portfolio interpretation | Next-open executable, price-only until all dividends checked | Fixed |

If a necessary repair changes a fixed row, record the old/new value, reason, approver, date, and rerun IDs. Finish selection using the three development folds before opening the final holdout. Protocol changes after seeing test outcomes must be labeled **post hoc** and cannot replace the preregistered result.

## Source materials

- Local course PDF: `AI6124 - week 8 - Neural in Neuro-Fuzzy Systems.pdf`, especially pages 17, 20–23 and 31–39.
- Current design: `lstm_fuzzy_local_experts_design_en.md` and `lstm_fuzzy_local_experts_design_zh.md`.
- Data audit and source provenance: `data/README.md`, `data/processed/audit.json`, `data/source_manifest.json` and `data/verify_dataset.py`.
- Ožbot, Škrjanc and Štruc, [*A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting*](https://arxiv.org/html/2510.00960v1) (method inspiration; different task, horizon and architecture).
- MAS, [SORA publication policy](https://www.mas.gov.sg/monetary-policy/SORA) (the next-business-day 9:00 publication constraint used in the data alignment).
