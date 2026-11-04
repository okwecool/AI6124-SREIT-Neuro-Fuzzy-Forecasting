# Frozen final-holdout results — 4 November 2026

**Scope and provenance.** The 251 SRT target dates run from 1 October 2025 through 30 September 2026, after 931 pretest training windows. The Phase 3 freeze in `output/development/final_configuration_pretest.json` was committed as `d7808a3` before the final run. Its data, fold, training-source and model-source SHA-256 values passed preflight. The four six-feature heads, seeds 11/23/37, fixed refit lengths (1/6/3/4 epochs for `LSTM32`/`LSTM3`/`Fuzzy2`/`Gate2`), raw next-close log-return target, `Z0` baseline, trading policies and costs were unchanged. The saved 12 fit manifests, 3,263 prediction rows, 753 fuzzy cases and 1,506 rule rows pass the full-key and mixture checks. Final evidence is in `output/final/` and `output/runs/final/`.

The initial execution completed **all 12 fits and saved their predictions**, then stopped on a control-table column-name collision during analysis. `execution_started.json` records that first execution. The fixed analysis was completed **from the identical saved prediction files**, with no new fit or forecast; `execution_complete.json` records the input-file hashes and recovery reason. The wealth figures were refreshed from the saved ledgers to include the control paths; `figure_refresh.json` records that presentation-only change. These events should remain visible in an audit.

## Forecasting

| Method | Seeds | Final RMSE, mean | Final MAE, mean |
| --- | ---: | ---: | ---: |
| `Fuzzy2` | 3 | **0.006349** | 0.004595 |
| `Gate2` | 3 | 0.006353 | 0.004582 |
| `LSTM3` | 3 | 0.006355 | 0.004593 |
| Zero-return `Z0` | Deterministic | 0.006377 | **0.004547** |
| `LSTM32` | 3 | 0.006384 | 0.004651 |

`Fuzzy2` has the smallest final RMSE, 0.000028 (about **0.44%**) below `Z0` and 0.000035 (about 0.55%) below `LSTM32`. It improves RMSE over `LSTM32` for all three paired seeds, but its margin against `LSTM3` and `Gate2` changes sign in seed 37. `Z0` has the smallest MAE. The 251 consecutive days are one test period, and the three seeds share those days; neither the tiny RMSE differences nor the seed count proves broad superiority. Direction accuracy for `Fuzzy2` is 49.0% over the 227 nonzero-return days, below the training-majority-direction control's 50.7%. The exported `Z0` directional score of zero is a mechanical sign comparison for a zero forecast, not a useful directional benchmark. See `forecast_by_seed.csv`, `forecast_paired_fuzzy2.csv` and `forecast_rmse.png`.

## Causal trading result

The **primary** long/cash policy buys at the next observed open if predicted close return is positive, charges 10 bps on each side, and liquidates at the declared test boundary. The single unobserved SRT open on 24 October 2025 uses the frozen no-trade-and-carry rule. Price-only final wealth multiples are:

| Policy | Mean final wealth | One-way trades (mean) |
| --- | ---: | ---: |
| Always cash | **1.000** | 0 |
| `Fuzzy2`, `θ=0` | 0.999 | 1.3 |
| `LSTM3`, `θ=0` | 0.984 | 12.7 |
| `Gate2`, `θ=0` | 0.970 | 24.0 |
| `LSTM32`, `θ=0` | 0.963 | 13.3 |
| Past-only momentum | 0.924 | 38 |
| Always long | 0.878 | 2 |

The `Fuzzy2` seed-11 path ends at 0.9975 after four one-way trades; seeds 23 and 37 make **zero trades** and end at 1.000. Its apparent advantage over momentum and always-long mainly reflects avoiding exposure in a down period; it does **not** beat cash or demonstrate profitable market timing. At 0/20 bps, mean price-only `Fuzzy2` wealth is 1.0005/0.9978, showing the small edge before costs disappears with fees. Under the separately labelled gross-distribution ledger, `Fuzzy2` remains 0.999, momentum is 0.978 and always-long is 0.934. The ledger credits nine issuer-checked gross SGD distributions; it is a constrained-data simulation, not an investor-specific after-tax return.

The **secondary** development-selected thresholds are `Fuzzy2=.002`, `Gate2=.001`, `LSTM3=.002` and `LSTM32=.003`. All 12 final model-seed policies make zero trades and end at 1.000. This is a cash-only sensitivity result, not an improved trading result. The untuned `θ=0` analysis above remains primary. See `wealth_comparison.csv`, `wealth_summary_*`, `wealth_selected_threshold_10bps.csv` and the two `wealth_paths_*_10bps.png` figures.

## Fuzzy mechanism and horizon

Final `Fuzzy2` mean maximum rule weights range from **0.511 to 0.563** across seeds, with no case above 0.8. The fraction below each run's training fifth-percentile raw-membership support is zero. Training-defined 75th-percentile activation cuts yield widely varying test counts, including one rule with **zero** cases and another with only two. Local expert errors on those tiny subsets cannot establish specialization. Rule weights and contributions are traceable, but these data do not support bull/crash/rate labels or Gaussian-specific superiority over `Gate2`. See `fuzzy_local_checks.csv` and the saved case/rule tables.

The 248-date horizon diagnostic excludes the test boundary and dates without consecutive observed opens. Within the three `Fuzzy2` runs, predicted-close/true-close correlations are 0.151, 0.130 and 0.002; predicted-close/executable-open correlations are 0.168, 0.152 and 0.041. True close and executable open returns correlate at 0.655 over the same dates. This is descriptive: the forecast and trading horizons differ, and the model's measured association is weak. No post-test target or policy adjustment was made.

## Course-project conclusion and limits

The frozen comparison demonstrates a small final RMSE advantage for the proposed fuzzy mixture over the ordinary LSTM and no-change baseline, while MAE and trading do not show a practical gain. The non-Gaussian `Gate2` is very close, and the fuzzy rules do not have validated economic semantics. Present this as a **modest, metric-dependent forecasting result with negative trading and interpretability findings**. Do not claim a profitable strategy, robust Gaussian gating advantage or market-regime experts. Further work could test longer histories, other REIT instruments, calibrated rule separation and an explicitly trade-aligned target under a new preregistered evaluation.
