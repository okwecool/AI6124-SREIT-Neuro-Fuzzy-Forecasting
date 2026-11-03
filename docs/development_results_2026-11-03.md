# Development-period results — 3 November 2026

**Scope.** This is a pretest development analysis, not the reserved final evaluation. It uses the three chronological validation folds (378 distinct target days), seeds 11/23/37 for the primary heads, the frozen six-feature source group, raw next-close log returns, and protocol v1.2. All data-source and rule-count checks use the same dates. The final 251-target holdout has not been scored or used to choose a method. Machine-readable tables, run manifests, full predictions, fuzzy cases/rules and plots are under `output/development/` and `output/runs/development/`.

## Prediction comparison

| Method | Seeds | Mean pooled return RMSE |
| --- | ---: | ---: |
| Zero-return/no-change `Z0` | Deterministic | 0.009431 |
| Non-Gaussian two-expert `Gate2` | 3 | 0.009456 |
| Narrow latent linear `LSTM3` | 3 | 0.009473 |
| Proposed Gaussian `Fuzzy2` | 3 | 0.009489 |
| Ordinary `LSTM32` | 3 | 0.009624 |
| Three-rule `Fuzzy3` | 1 diagnostic seed | 0.009617 |

The means of each learned model pool the three nonoverlapping validation periods **within each seed**, then average the three seed scores. `Fuzzy3` is a seed-11 capacity diagnostic and should not be ranked as if it had three seeds. `Z0` needs no seed. The proposed `Fuzzy2` is lower than `LSTM32` on mean RMSE by about 0.000135 (1.4% of the latter), but its paired seed-11 RMSE is **higher** than `LSTM32`; the advantage is not consistent across all seeds. `Fuzzy2` is slightly worse than both `LSTM3` and `Gate2` on mean RMSE, and all learned heads are worse than `Z0`. Differences among the compact heads are small relative to daily return variability; these results do not establish a reliable gain from Gaussian gating.

The source ablation for `Fuzzy2` gives mean RMSE 0.009485 with SRT only, 0.009479 with SRT+SORA, and 0.009489 with SRT+SORA+STI. The margins are too small to claim a useful extra source. The six-feature specification remains the declared primary model; no group is substituted after seeing the validation results. With seed 11, the rule-count check gives `K=1` 0.009492 (`LSTM3`, functionally identical to one affine TSK expert), `K=2` 0.009478 and `K=3` 0.009617. This is a capacity diagnostic, not a rule-count choice based on the final holdout.

## Mechanism and interpretation

The full `Fuzzy2` runs have mean larger-rule weight about **0.533**; none of their validation days has a maximum weight above **0.8**. The two experts therefore usually contribute at similar weights. A high-activation region defined by each rule's **training** 75th-percentile weight contains about 30 validation cases on average, but the activated expert has lower error than the other expert in only **9 of 18** fold/seed/rule checks. These results do not support calling the experts distinct bull, crash or rate-regime specialists. The defensible description is a **traceable learned-feature mixture** whose rule weights and expert contributions can be inspected.

The fraction of validation cases with maximum raw log membership below the corresponding **training** fifth percentile is about 5.1% on average, but reaches 15.6–20.3% for two seeds in the latest development fold. This support check complements normalized weights; it does not rescue a semantic market-regime claim. Rule 1's high-versus-low past-momentum weight difference is positive in all nine primary runs but averages only about **0.006**; its economic meaning remains uncertain. No first-coordinate center-order reversal occurs, yet the learned latent coordinates themselves can change across seeds, so rule IDs have not been treated as semantically aligned. The tables `fuzzy_coverage.csv`, `fuzzy_local_errors.csv`, `fuzzy_market_bins.csv` and `fuzzy_training_defined_cuts.json` preserve the per-run evidence and past-only volatility/momentum cuts. `Gate2` has two additional trainable parameters versus `Fuzzy2` (5,241 versus 5,239 in the six-feature setting) and a lower mean RMSE; thus no Gaussian-specific superiority claim is warranted.

## Trading and forecast horizon

With the **untuned** `ŷ>0` policy, 10 bps per side, fold-end cash resets and price-only wealth, the mean three-fold compounded multiples are `Fuzzy2` **0.975**, `LSTM3` 0.974, `Gate2` 0.963 and `LSTM32` 0.957. The model-free five-day momentum control is **1.050**, always-long is **1.020**, and always-cash is 1.000 on matched fold calendars. Gross-distribution accounting gives `Fuzzy2` 0.975, momentum 1.112 and always-long 1.109; it uses issuer-checked gross SGD amounts and is not an investor-specific after-tax return. The model policies do not improve wealth over the model-free controls in these development folds. See `wealth_compounded_by_seed.csv` for seed variation and the 0/20-bps price-only sensitivities; the paths and per-open cash/share/cost records are also saved.

The separate threshold grid `{0, 0.001, 0.002, 0.003}` was evaluated at 10 bps **only on the development folds**. The frozen selections in `selected_policy_thresholds.json` are `Fuzzy2=0.002`, `Gate2=0.001`, `LSTM3=0.002`, `LSTM32=0.003`, and momentum `=0`. The Fuzzy2 selected threshold yields an objective of zero log wealth, consistent with largely staying in cash. These selected results are development selection evidence, not unbiased profit estimates; the untuned `θ=0` comparison remains primary.

The three-return check aligns true next-close return, predicted next-close return and the next-open to following-open realized return. Across the nine primary `Fuzzy2` fold/seed groups, the mean within-run correlations are approximately **−0.038** for prediction versus true close return, **−0.058** for prediction versus executable return, and **0.704** for true close return versus executable return. Correlations of nearly flat predictions can be unstable. The substantial true-close/executable association suggests the weak trading result is not explained solely by a forecast/trade horizon mismatch; the predictions themselves show little directional relationship on these dates. This is a descriptive inference, not a causal decomposition. No trade-aligned target extension is adopted after this analysis.

## Compute and freeze decisions

The 48 core development runs plus nine predeclared `Gate2` controls made **57 run configurations and 114 inner/refit fits** on CPU. Logged fitting time sums to about **82 seconds** for inner fits and **21 seconds** for outer refits, excluding loading, export validation and analysis. The six-feature parameter counts are 5,153 (`LSTM32`), 5,223 (`LSTM3`), 5,239 (`Fuzzy2`), 5,241 (`Gate2`) and 5,249 (`Fuzzy3`).

For the reserved final test, retain the approved full six-feature primary comparison, all three seeds, the zero-return baseline, `Gate2` as a predeclared control, `θ=0` primary trading policy, the separately labelled selected-threshold sensitivity, 0/10/20-bps price-only costs, the issuer-checked gross-distribution ledger and the no-trade-and-carry rule for the unresolved 2025-10-24 opening quote. The median-nine-epoch rule fixes final refit lengths at **1** (`LSTM32`), **6** (`LSTM3`), **3** (`Fuzzy2`) and **4** (`Gate2`), giving **12** final learned fits including the optional control. These decisions and dataset/code hashes are written to `output/development/final_configuration_pretest.json`. Do not rename learned rules as economic regimes or claim Gaussian-specific superiority. The final test must be opened once after the team reviews this development analysis and the frozen configuration.
