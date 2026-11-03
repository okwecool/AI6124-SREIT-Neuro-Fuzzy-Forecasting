# Group 1 handoff — SRT LSTM/fuzzy project

**Prepared by Ryan's side, 30 October 2026; updated 3 November 2026.** This file is the short implementation entry point for Destin. The approved architecture and revised evaluation design are in `experiment_protocol_lstm_fuzzy_local_experts.md` (v1.2); follow that document when a detail here is abbreviated. Review findings F1–F7 are recorded in `design_review_2026-11-03_zh.md`.

The existing `output/AI6124_Group1_Early_Work_Handoff_2026-11-03.zip` is a **pre-Phase-1 snapshot**. Use this working tree for the model, validator, training runner, wealth ledger and development results. Rebuild the shareable archive after Ryan reviews and commits Phase 3; the older ZIP already contains the `data/` folder but does not include these new files.

## Ready now

- The fixed source snapshot and provenance are in `data/`. The dataset audit passes and yields `X=(1182,20,6)` with 1,182 next-day raw-return labels.
- The primary input is all six features: SRT returns/momentum/volatility, STI as-of return, and published SORA level/change. Source ablations use `srt`, `srt_sora`, `all` as defined in `src/project_eval.py`.
- The three expanding validation folds have 126, 124 and 128 evaluation targets. The final 251-target holdout is reserved until the comparison is frozen.
- `src/project_eval.py` supplies train-only scaling, fold IDs, seven-column prediction and fuzzy-diagnostic validation, forecast metrics and the causal observed-open portfolio entry point. `src/portfolio.py` implements no-trade-and-carry for a missing open, a model-free momentum control, price-only and gross-distribution wealth, and per-event cash/share/cost records. The previously saved legacy portfolio CSV has been regenerated with this code.
- `src/models.py` contains the shared LSTM encoder, `LSTM32`, `LSTM3`, `FuzzyK` and optional non-Gaussian `Gate2` heads. `src/export_contract.py` inverse-transforms fuzzy experts and creates the prediction/case/rule tables.
- `src/train.py` performs inner early stopping followed by same-seed outer refit on development folds only; it writes full-key predictions, fuzzy diagnostics, checkpoints with scalers and a run manifest. `src/tools/collect_runs.py` validates and combines complete runs and writes matched forecast and wealth tables.
- `output/zero_baseline_validation.csv`, `output/zero_baseline_validation_scores.csv` and `output/zero_baseline_validation_pooled.csv` contain the no-change development baseline. They are actual calculations, not trained-model claims.
- `output/development/` contains all three-fold development comparisons from **48 core runs plus nine `Gate2` controls**. `docs/development_results_2026-11-03.md` explains the qualified negative findings: the unchanged-price baseline has the lowest pooled RMSE; the fuzzy trading policy does not beat model-free controls; rule weights do not justify market-regime-specialist labels. `output/development/final_configuration_pretest.json` freezes the 12 final learned fits and policy choices **before** the reserved holdout is opened.
- `docs/literature_review_draft_en.md` is an initial chronological/taxonomy review. `docs/AI6124_Group1_Project_Starter.ipynb` loads development outputs but still needs final-test and course-conclusion sections. `docs/srt_distribution_source_audit.md` cross-checks all nine Yahoo ex-date records with issuer documents; the 2025-10-24 open remains unresolved.

## First commands

Run from the project directory with a Python environment containing NumPy, pandas, matplotlib, nbformat and PyTorch for model training:

```powershell
python data/verify_dataset.py
python -m unittest discover -s tests -v
python src/tools/build_starter_notebook.py
python src/tools/verify_starter_notebook.py
```

The last command executes the Notebook's code cells in a regular Python process without opening final-holdout results. A Jupyter kernel can run the same Notebook interactively. To reproduce the full development matrix and analysis:

```powershell
python -m src.tools.run_development_matrix --include-gate --device cpu --out output/runs/development
python -m src.tools.export_training_diagnostics --runs output/runs/development
python -m src.tools.collect_runs --runs output/runs/development --out output/development --wealth-mode price_only --cost-bps 10
python -m src.tools.collect_runs --runs output/runs/development --out output/development --wealth-mode price_only --cost-bps 0
python -m src.tools.collect_runs --runs output/runs/development --out output/development --wealth-mode price_only --cost-bps 20
python -m src.tools.collect_runs --runs output/runs/development --out output/development --wealth-mode gross_distribution --cost-bps 10
python -m src.tools.analyze_development --runs output/runs/development --aggregate output/development
```

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

Instantiate `LSTM32`, `LSTM3`, or `Fuzzy2` with `from src.models import build_model`; use identical input and optimization settings. The two fuzzy centers start at `(-0.25,0,0)` and `(0.25,0,0)`; their scales start at 1 and remain in `(0.1,2.0)`. **Do not use K-means initialization or add balancing/separation losses in the primary run.** The old proposal slide mentions K-means; the approved experiment protocol settles that discrepancy and the final slides should be corrected.

## Prediction-file contract v2 (implemented)

The evaluator and Notebook builder accept one row per evaluation `window_id` in each complete `(model_id,source_group,rule_count,fold,seed)` run. The mandatory prediction columns are:

| Column | Example | Meaning |
| --- | --- | --- |
| `model_id` | `Fuzzy2` | `LSTM32`, `LSTM3`, `Fuzzy2`, or named extension |
| `source_group` | `all` | `srt`, `srt_sora`, or `all` |
| `rule_count` | `2` | Use `0` for non-fuzzy models; `2` for `Fuzzy2`; `3` for `Fuzzy3` |
| `fold` | `validation_1` | Exact fold name from the saved assignments |
| `seed` | `11` | Use the protocol's `11,23,37` for main comparisons |
| `window_id` | `553` | Matches the row in `window_index.csv` and the NPZ array |
| `y_pred` | `0.0021` | **Raw** next-close log return |

The CLI scores mixed source groups and rule counts with the full run key. A historical six-column CSV is migrated only when every row is `Z0`; other legacy files must be converted explicitly:

```powershell
python -m src.project_eval score --input output/development/predictions.csv --out output/development/forecast_scores_cli.csv
python -m src.project_eval pooled --input output/development/predictions.csv --out output/development/forecast_pooled_cli.csv
python -m src.project_eval check-fuzzy --input output/development/predictions.csv --cases output/development/cases.csv --rules output/development/rules.csv
```

The `portfolio` CLI action now uses the causal observed-open ledger. `--wealth-mode gross_distribution` uses the nine source-checked events in `data/reference/srt_distributions_verified.csv`; `--ledger-out` writes every event and trade. The scorer rejects missing or misplaced window IDs, duplicate **full** keys, wrong rule counts and nonfinite predictions. `src.train` rejects `final_holdout`; that fold is reserved for a separate frozen Phase 4 entry point.

For fuzzy interpretation, save a case table keyed by the **full prediction key** with `z1,z2,z3,y_mix_raw`, and a rule-long table keyed by the full prediction key plus `rule_id` with `alpha,expert_raw,log_membership,distance2`. Each case must have exactly `rule_count` rule rows and `Σ alpha × expert_raw = y_mix_raw = y_pred`. Transform expert outputs back to raw-return units with the run's target scaler. Save checkpoints, scalers, best epochs, code/library versions and a run manifest alongside these files.

## What remains for the final course submission

1. **Phases 1–3 implemented:** the full development comparison, source/rule ablations, `Gate2` control, causal wealth paths, training-defined fuzzy diagnostics, horizon check and selected-threshold manifest are available. Review `docs/development_results_2026-11-03.md` and `output/development/final_configuration_pretest.json` before the Phase 3 Git commit and the once-only final evaluation.
2. **Before a final wealth claim:** the nine ex-date distribution records have issuer corroboration and the gross ledger is implemented. The 2025-10-24 open remains unverified. Apply the no-trade-and-carry rule and label any final path a constrained-data simulation; report price-only and gross-distribution wealth separately.
3. Refit the four frozen learned heads on 931 pretest rows with the fixed median epoch counts and seeds; open the 251-target holdout once. The **core** fit count is 105 across development and final phases; `Gate2` adds 18 development fits and three frozen final fits, for **126** fits in total.
4. Complete the Notebook with forecast and matched-date wealth results, a literature taxonomy and result-level comparison with the source paper, theory-focused future work and another plausible application. Create a new final-results deck; preserve the submitted three-slide proposal as a historical record and correct its method differences in the new deck.
5. Both students must submit the final Notebook and slides in NTULearn by **18 November 2026, 23:59 Singapore time**. Record the peer-review assignment by 23 November and the designated member's score submission by **27 November 2026, 23:59**.

## Suggested checkpoints

| By | Reviewable outcome |
| --- | --- |
| 3 Nov | One-fold smoke runs and all development folds completed; source and rule diagnostics reviewed in the Phase 3 report |
| 8 Nov | Phase 3 reviewed and committed; frozen final evaluation completed once |
| 11 Nov | Final plots and interpretation drafted |
| 15 Nov | Notebook and slides ready for joint review, with all limitations stated |
| 17 Nov | Fresh-environment rerun, file audit and each student's submission plan checked |

These are team planning checkpoints, not instructor deadlines. If Ryan becomes unavailable, Destin can continue from the frozen protocol, scripts and Notebook without redefining the data or metrics.
