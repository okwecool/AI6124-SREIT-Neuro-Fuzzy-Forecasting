# Group 1 handoff — SRT LSTM/fuzzy project

**Prepared by Ryan's side, 30 October 2026.** This file is the short implementation entry point for Destin. The approved research design is in `experiment_protocol_lstm_fuzzy_local_experts.md`; follow that document when a detail here is abbreviated.

The shareable archive is `output/AI6124_Group1_Early_Work_Handoff_2026-11-03.zip`. It already contains the `data/` folder; extract it into an empty directory before running the commands below.

## Ready now

- The fixed source snapshot and provenance are in `data/`. The dataset audit passes and yields `X=(1182,20,6)` with 1,182 next-day raw-return labels.
- The primary input is all six features: SRT returns/momentum/volatility, STI as-of return, and published SORA level/change. Source ablations use `srt`, `srt_sora`, `all` as defined in `src/project_eval.py`.
- The three expanding validation folds have 126, 124 and 128 evaluation targets. The final 251-target holdout is reserved until the comparison is frozen.
- `src/project_eval.py` supplies train-only scaling, fold IDs, prediction-file validation, forecast metrics, and a price-only next-open portfolio comparison. `tests/test_project_eval.py` checks chronology, leakage and the missing-open behavior.
- `output/zero_baseline_validation.csv`, `output/zero_baseline_validation_scores.csv` and `output/zero_baseline_validation_pooled.csv` contain the no-change development baseline. They are actual calculations, not trained-model claims.
- `docs/literature_review_draft_en.md` is an initial chronological/taxonomy review. `docs/AI6124_Group1_Project_Starter.ipynb` loads the audited data and runs the baseline; trained-model and result sections are awaiting outputs.

## First commands

Run from the project directory with a Python environment containing NumPy, pandas, matplotlib, nbformat and PyTorch for model training:

```powershell
python data/verify_dataset.py
python -m unittest discover -s tests -v
python src/tools/build_starter_notebook.py
python src/tools/verify_starter_notebook.py
```

The last command executes the Notebook's code cells in a regular Python process without opening the final holdout results. A Jupyter kernel can run the same Notebook interactively.

## Destin's model interface

```python
from src.project_eval import Dataset

data = Dataset()
inner = data.prepare("validation_1", "all", inner_stop=True)
outer = data.prepare("validation_1", "all", inner_stop=False)

# inner.x_train, inner.y_train: train a model and choose the best epoch
# inner.x_eval, inner.y_eval_raw: stopping set (scale y using inner.scaler)
# outer.x_train, outer.y_train: restart with the same seed; fit for best_epoch
# outer.x_eval: predict only after refit, then invert with outer.scaler.y_inverse
```

The PyTorch model accepts `(batch,20,F)` float tensors and produces `(batch,)` **standardized-return** predictions. Use `F=3,5,6` for the three source groups. The final output file must contain **raw log-return** predictions after `scaler.y_inverse`, not standardized values. For `Fuzzy2`, also expose `z` `(batch,3)`, rule weights `(batch,2)` and expert outputs `(batch,2)`; ensure weights are nonnegative and sum to one and the weighted experts exactly reproduce the mixed prediction.

Implement `LSTM32`, `LSTM3`, `Fuzzy2` with identical input and optimization settings. The two fuzzy centers start at `(-0.25,0,0)` and `(0.25,0,0)`; their scales start at 1 and remain in `(0.1,2.0)`. **Do not use K-means initialization or add balancing/separation losses in the primary run.** The old proposal slide mentions K-means; the approved experiment protocol settles that discrepancy and the final slides should be corrected.

## Prediction-file contract

Save `output/model_predictions_validation.csv` with one row for every evaluation `window_id` in each complete `(model_id,source_group,fold,seed)` run. The mandatory columns are:

| Column | Example | Meaning |
| --- | --- | --- |
| `model_id` | `Fuzzy2` | `LSTM32`, `LSTM3`, `Fuzzy2`, or named extension |
| `source_group` | `all` | `srt`, `srt_sora`, or `all` |
| `fold` | `validation_1` | Exact fold name from the saved assignments |
| `seed` | `11` | Use the protocol's `11,23,37` for main comparisons |
| `window_id` | `553` | Matches the row in `window_index.csv` and the NPZ array |
| `y_pred` | `0.0021` | **Raw** next-close log return |

Check the output with:

```powershell
python -m src.project_eval score --input output/model_predictions_validation.csv --out output/model_scores_validation.csv
python -m src.project_eval pooled --input output/model_predictions_validation.csv --out output/model_scores_validation_pooled.csv
python -m src.project_eval portfolio --input output/model_predictions_validation.csv --cost-bps 10 --out output/portfolio_validation_price_only.csv
```

The scoring command rejects missing or misplaced window IDs, duplicate keys, and nonfinite predictions. The final holdout uses the same schema with `fold=final_holdout`, but should be run only after development choices are frozen. Save its predictions in a separate file so they cannot be mixed into development selection.

For the fuzzy interpretation section, save `output/fuzzy_diagnostics_validation.csv`, keyed by `fold,seed,window_id`, with `z1,z2,z3,alpha1,alpha2,expert1_raw,expert2_raw`. The expert values should be transformed back to raw-return units using the run's target scaler. Save checkpoints, scaler parameters, best epochs, code/library versions, and a run manifest alongside these files.

## What remains for the final course submission

1. Train the three core learned models on the full six-feature group for three folds × three seeds; run `Fuzzy2` source ablations. Complete the one/two/three-rule diagnostic described in the protocol.
2. Inspect the validation metrics and rule diagnostics. Freeze model choices, then train on all 931 pretest rows and open the final 251-target test once.
3. Fill the Notebook's result sections with tables, plots, paired model comparisons, parameter effects, source ablations, rule behavior, portfolio comparison, limitations and future work. The literature draft needs result-linked discussion, not only citations.
4. Build concise final slides from the measured results. The proposal's **three-slide maximum applies to the proposal**, not the final presentation. Correct the K-means wording and do not label experts “bull/crash” unless diagnostics support it.
5. Confirm all nine SRT distribution events before claiming ETF **total** portfolio returns. Until then label the backtest *price-only, dividends excluded*. The 2025-10-24 open is unavailable and must not be filled with that day's close.
6. Both students must submit the final Notebook and slides in NTULearn by **18 November 2026, 23:59 Singapore time**, according to the course lecture. Coordinate the later peer-review duty separately.

## Suggested checkpoints

| By | Reviewable outcome |
| --- | --- |
| 4 Nov | One-fold smoke run for every model; prediction/diagnostic files pass shape and alignment checks |
| 8 Nov | All three development folds and seeds completed; source ablations and rule diagnostic underway |
| 11 Nov | Model choice frozen; final test evaluated once; plots and interpretation drafted |
| 15 Nov | Notebook and slides ready for joint review, with all limitations stated |
| 17 Nov | Fresh-environment rerun, file audit and each student's submission plan checked |

These are team planning checkpoints, not instructor deadlines. If Ryan becomes unavailable, Destin can continue from the frozen protocol, scripts and Notebook without redefining the data or metrics.
