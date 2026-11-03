# Group 1 handoff — SRT LSTM/fuzzy project

**Prepared by Ryan's side, 30 October 2026; updated 3 November 2026.** This file is the short implementation entry point for Destin. The approved architecture and revised evaluation design are in `experiment_protocol_lstm_fuzzy_local_experts.md` (v1.2); follow that document when a detail here is abbreviated. Review findings F1–F7 are recorded in `design_review_2026-11-03_zh.md`.

The shareable archive is `output/AI6124_Group1_Early_Work_Handoff_2026-11-03.zip`. It already contains the `data/` folder; extract it into an empty directory before running the commands below.

## Ready now

- The fixed source snapshot and provenance are in `data/`. The dataset audit passes and yields `X=(1182,20,6)` with 1,182 next-day raw-return labels.
- The primary input is all six features: SRT returns/momentum/volatility, STI as-of return, and published SORA level/change. Source ablations use `srt`, `srt_sora`, `all` as defined in `src/project_eval.py`.
- The three expanding validation folds have 126, 124 and 128 evaluation targets. The final 251-target holdout is reserved until the comparison is frozen.
- `src/project_eval.py` currently supplies train-only scaling, fold IDs, prediction-file validation, forecast metrics, and a **legacy** price-only next-open comparison. Its missing-quote exit uses future validity, so the existing portfolio CSV must not support an executable-wealth claim. The causal ledger, model-free momentum control, wealth paths, dividend accounting and version-2 export validator remain to implement. The six existing tests do not yet check these new rules.
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

The PyTorch model accepts `(batch,20,F)` float tensors and produces `(batch,)` **standardized-return** predictions. Use `F=3,5,6` for the three source groups. The final output file must contain **raw log-return** predictions after `scaler.y_inverse`, not standardized values. For `FuzzyK`, also expose `z` `(batch,3)` and weights, expert outputs, raw log memberships and squared distances `(batch,K)`; ensure weights are nonnegative and sum to one and the weighted experts reproduce the mixed prediction.

Implement `LSTM32`, `LSTM3`, `Fuzzy2` with identical input and optimization settings. The two fuzzy centers start at `(-0.25,0,0)` and `(0.25,0,0)`; their scales start at 1 and remain in `(0.1,2.0)`. **Do not use K-means initialization or add balancing/separation losses in the primary run.** The old proposal slide mentions K-means; the approved experiment protocol settles that discrepancy and the final slides should be corrected.

## Prediction-file contract v2 (implementation pending)

Before batch training, update the evaluator and Notebook builder to accept one row per evaluation `window_id` in each complete `(model_id,source_group,rule_count,fold,seed)` run. The **new** mandatory prediction columns are:

| Column | Example | Meaning |
| --- | --- | --- |
| `model_id` | `Fuzzy2` | `LSTM32`, `LSTM3`, `Fuzzy2`, or named extension |
| `source_group` | `all` | `srt`, `srt_sora`, or `all` |
| `rule_count` | `2` | Use `0` for non-fuzzy models; `2` for `Fuzzy2`; `3` for `Fuzzy3` |
| `fold` | `validation_1` | Exact fold name from the saved assignments |
| `seed` | `11` | Use the protocol's `11,23,37` for main comparisons |
| `window_id` | `553` | Matches the row in `window_index.csv` and the NPZ array |
| `y_pred` | `0.0021` | **Raw** next-close log return |

The present `src.project_eval` CLI still implements the old six-column grouping. Its **forecast-scoring** commands below are valid only for the existing single-configuration legacy files until the version-2 validator and grouping are implemented; do not send mixed ablation/K-rule outputs to them yet:

```powershell
python -m src.project_eval score --input output/model_predictions_validation.csv --out output/model_scores_validation.csv
python -m src.project_eval pooled --input output/model_predictions_validation.csv --out output/model_scores_validation_pooled.csv
```

The legacy `portfolio` CLI action and saved portfolio CSV are **deprecated for executable-wealth claims** until F1's causal ledger is implemented. After the version-2 migration, the scoring command must reject missing or misplaced window IDs, duplicate **full** keys, wrong rule counts, and nonfinite predictions. Migrate legacy `Z0` with `rule_count=0`. The final holdout uses the same new schema with `fold=final_holdout`, but should be run only after development choices are frozen. Save its predictions separately from development selection.

For fuzzy interpretation, save a case table keyed by the **full prediction key** with `z1,z2,z3,y_mix_raw`, and a rule-long table keyed by the full prediction key plus `rule_id` with `alpha,expert_raw,log_membership,distance2`. Each case must have exactly `rule_count` rule rows and `Σ alpha × expert_raw = y_mix_raw = y_pred`. Transform expert outputs back to raw-return units with the run's target scaler. Save checkpoints, scalers, best epochs, code/library versions and a run manifest alongside these files.

## What remains for the final course submission

1. **Before batch training:** implement the version-2 prediction/case/rule-long keys and validator, update the Notebook's export instructions, run the pretest data diagnostics, and check a minibatch's gradients and weighted-expert identity. Use AdamW and the fixed distinct fuzzy centers; do not silently follow the older proposal's K-means or auxiliary-loss wording.
2. Train the three core learned models on the full six-feature group for three folds × three seeds; run `Fuzzy2` source ablations and the one/two/three-rule diagnostic. The core schedule is **48 development runs with two fits each, plus nine final fits = 105 fits** before optional controls or retries.
3. **Before a wealth claim:** replace the future-validity exit in the legacy evaluator with the protocol's causal open ledger, seek the missing 2025-10-24 open, and verify the nine SRT distribution events. If the quote or distributions remain unverified, use the specified no-trade-and-carry/price-only labels and limitations. The existing portfolio CSV is not a completed executable backtest.
4. Inspect rule coverage and each expert's local error on training-defined high-activation regions; compare true/predicted close returns with executable open-to-open returns. Select the secondary trading threshold on the pretest development folds only, freeze all settings, then refit on 931 pretest rows and open the 251-target final test once.
5. Complete the Notebook with forecast and matched-date wealth results, a literature taxonomy and result-level comparison with the source paper, theory-focused future work and another plausible application. Create a new final-results deck; preserve the submitted three-slide proposal as a historical record and correct its method differences in the new deck.
6. Both students must submit the final Notebook and slides in NTULearn by **18 November 2026, 23:59 Singapore time**. Record the peer-review assignment by 23 November and the designated member's score submission by **27 November 2026, 23:59**.

## Suggested checkpoints

| By | Reviewable outcome |
| --- | --- |
| 4 Nov | One-fold smoke run for every model; prediction/diagnostic files pass shape and alignment checks |
| 8 Nov | All three development folds and seeds completed; source ablations and rule diagnostic underway |
| 11 Nov | Model choice frozen; final test evaluated once; plots and interpretation drafted |
| 15 Nov | Notebook and slides ready for joint review, with all limitations stated |
| 17 Nov | Fresh-environment rerun, file audit and each student's submission plan checked |

These are team planning checkpoints, not instructor deadlines. If Ryan becomes unavailable, Destin can continue from the frozen protocol, scripts and Notebook without redefining the data or metrics.
