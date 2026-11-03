# Code development plan — review findings F1–F7, 3 November 2026

This plan implements [experiment protocol v1.2](experiment_protocol_lstm_fuzzy_local_experts.md) and resolves the [design audit](design_review_2026-11-03_zh.md). It is a development sequence, not a record of completed model results. The core workload is **48 development runs × two fits + nine final fits = 105 fits**, before optional `Gate2`, pilots or failed-run repeats. The team will review each phase before its Git commit.

## Phase 0 — repository baseline (complete)

1. Archive superseded files and initialize Git on `main`.
2. Keep the saved SRT/STI/SORA snapshot, current protocol, Notebook starter, evaluation code and current proposal deck.
3. Verify the dataset audit, existing evaluation tests and Notebook code cells.
4. Show the file inventory and proposed initial commit. Ryan approved it; commit `c5243e9` established the baseline on `main`.

The design revision was committed as `796dc27` after Ryan's approval. The next phases remain subject to Ryan's review before each new commit.

## Phase 1 — model definitions and smoke test

- **F6 interface gate before batch runs:** upgrade prediction exports and evaluation to the seven-column run key with `rule_count`; migrate the existing `Z0` baseline explicitly. Define a fuzzy case table and a per-rule long table containing raw log membership, distance, weight and expert output. Add uniqueness, completeness, mixture-identity and cross-file key checks. Update the Notebook builder and Destin handoff; the current six-column scorer must not silently merge `Fuzzy2` source groups or `Fuzzy3` diagnostics.
- Add a **pretest-only** data-diagnostic section to the Notebook before interpreting training results: raw SRT close versus daily log return, rolling mean/standard deviation, return distribution, zero-return frequency and the missing-open date. Explain why log returns reduce the price-level trend without guaranteeing stationarity. Do not use final-holdout patterns to change the design.
- Add one reusable 32-unit, single-layer LSTM encoder and three heads: `LSTM32`, `LSTM3`, `Fuzzy2`. The fuzzy head uses the protocol's three-dimensional `tanh` projection, two diagonal Gaussian rules and two affine experts. No attention, fuzzy internal LSTM gate or K-means initialization is in the primary model.
- Use **AdamW** consistently with the protocol. Expose standardized-return predictions and, for `Fuzzy2`, latent features, per-rule weights, expert outputs, distances and raw log memberships. Document parameter counts and the explicit latent-space TSK rule.
- Run a single minibatch and one short development-fold training run to check finite loss, tensor shapes, gradients through the encoder, memberships and experts, and `ŷ = Σ αₖeₖ`. The short run is a code check, not an outcome claim or evidence of market-regime interpretability.
- **F1/F5 data gate:** seek an independently verifiable 2025-10-24 open and check the nine distribution events with issuer/exchange records. Record ex-date, amount, payment date, source and unresolved cases. Document the causal no-trade-and-carry fallback and dividend ledger before any final wealth claim; do not use a missing future quote to decide an earlier exit.
- Do not open final-holdout predictions or metrics.

Review evidence: version-2 sample exports and validation checks, pretest diagnostic plots, model file diff, shape/gradient checks, source-audit status, smoke-test log and proposed commit message.

**Implementation status, 3 November:** the full-key validator, model definitions, export helper, pretest Notebook diagnostics and two-epoch development-fold smoke check are in the working tree for review. All nine Yahoo distribution dates/amounts have SGX-hosted issuer corroboration in `docs/srt_distribution_source_audit.md`; the 2025-10-24 open remains unresolved. Phase 1 has not been committed. The causal portfolio ledger and complete training runner remain Phase 2 work.

## Phase 2 — reproducible training runner

- Use `src.project_eval.Dataset.prepare()` for train-only standardization and the saved fold IDs. Implement the inner 15% chronological stopping set, early stopping, then same-seed refit on the full outer train for the selected epoch count.
- Fix seeds `11, 23, 37`; record Python, NumPy, pandas and PyTorch versions, device, feature group, model ID, seed, best epoch, train/validation losses, timing and checkpoint path in a run manifest.
- Write raw-return prediction CSVs using the **version-2 seven-column contract** and fuzzy diagnostics in the case/rule-long tables. Reject incomplete or wrong-window exports before scoring; retain a documented migration path for old six-column `Z0` results.
- **F1 causal ledger:** replace `portfolio_scores` future-validity exits with an event-driven open ledger. Charge entry/exit fees on traded notional, allow only latest-before-open signals, and at a missing open carry shares/cash to the next observed quote with no trade. Distinguish the predeclared fold-end liquidation from an unknown data gap. Export per-open cash, shares, costs, position and wealth, plus terminal `V_T`; keep a separately labelled segmented sensitivity if needed. The current `zero_baseline_validation_portfolio.csv` must be regenerated and not treated as a complete executable path.
- **F5 conditional total-wealth ledger:** if all nine events are verified, track ex-date receivables and payment-date cash without double counting or automatic reinvestment. Otherwise keep all comparisons explicitly price-only. Test entitlement using the position held before the ex-date, including a switch at the ex-date open.
- Keep the untuned `θ=0` model signal, always-long/cash, and the past-only `srt_momentum_5/5` control. Test self-financing accounting, identical observed-open calendars, no earlier trade based on a later missing quote, and continuous always-long exposure across the missing-open gap.
- Keep the CPU path usable. GPU acceleration can be enabled after a local check, but the experiment must not depend on it.

Review evidence: one complete fold for all three learned models, run manifest, version-2 contract checks, a synthetic future-missing-open test, distribution-ledger checks or explicit price-only branch, sample wealth paths and proposed commit message.

**Implementation status, 3 November — awaiting review and Git commit:** `src/train.py` runs the inner stopping and same-seed outer refit; `src/tools/collect_runs.py` validates complete exports and produces forecast/wealth tables. One full `validation_1` run at seed 11 for `LSTM32`, `LSTM3`, `Fuzzy2` produced 378 forecasts, 126 fuzzy cases and 252 rule rows. The causal ledger supports price-only and issuer-audited gross-distribution wealth with a past-signal momentum control. Unit tests cover missing-open carry, boundary liquidation, distribution entitlement, transaction costs and deterministic refits. Pilot files are local under `output/runs/tmp/phase2_pilot/`; they are one-fold engineering evidence, not the complete Phase 3 comparison. The 2025-10-24 open remains unresolved and the final holdout remains sealed.

## Phase 3 — development comparisons

- Run `LSTM32`, `LSTM3` and `Fuzzy2` on all six features for three expanding folds and three seeds. Compare with the already computed `Z0` no-change baseline.
- Run `Fuzzy2` source ablations (`srt`, `srt_sora`, `all`) on the same dates; perform the predeclared one/two/three-rule diagnostic. Report fold-wise and pooled validation metrics, paired differences, parameter counts and runtime.
- **F3 interpretability:** inspect rule-weight distributions, raw log-membership coverage, learned centers/scales, overlap, expert disagreement and relationships with **past-only** market summaries. Freeze high-activation thresholds and volatility/momentum bins from training predictions/data. Compare each expert and the mixture's error on the same high-activation dates with counts across folds/seeds; distinguish traceable contribution from semantic rule meaning or proven local specialization.
- Compare the **untuned `θ=0`** wealth curves and terminal multiples for `Fuzzy2`, ordinary LSTM, model-free momentum, always-long and always-cash on identical eligible intervals. Report 10-bps primary costs, 0/20-bps sensitivities, drawdown and turnover. Add a development-only diagnostic comparing close-to-close predictions with the actual next-open to following-open trading return.
- **F2 horizon check:** align true close-to-close return, predicted close-to-close return and realized next-open to following-open return on development dates. If the mismatch warrants a trade-aligned target extension, define and freeze it as a separate experiment before touching final results; never substitute it for the primary return target.
- **F4 attribution:** treat `Fuzzy2` gains over `LSTM3` as gains of the whole TSK head. If resources permit and before the final test, run the specified `Gate2` with identical encoder/experts, folds and seeds (nine extra development runs) and report all parameter counts; otherwise omit Gaussian-specific superiority claims.
- For the secondary threshold sensitivity, evaluate only the protocol's fixed `θ` grid. Select each model class's threshold by the predeclared mean-across-seeds, sum-across-folds log terminal-wealth objective; select momentum's threshold on the same three folds. Record tie resolution and selected values in a frozen policy manifest. Because those folds select the threshold, label their tuned wealth as **development selection evidence**, not unbiased profit.

Review evidence: development-only forecast and wealth tables/plots, selected threshold manifest, rule coverage/local-error evidence, three-return horizon diagnostic, mechanism-attribution decision and proposed commit message.

## Phase 4 — frozen final evaluation

- Freeze model configurations, optional `Gate2` decision, threshold manifest, missing-open treatment, distribution bookkeeping and diagnostic definitions from development. Retrain each predeclared core model on the 931 pretest windows with the protocol's epoch rule; evaluate the 251-window final holdout once.
- Store final version-2 predictions, case/rule diagnostics and wealth paths separately from development outputs. Report both forecast metrics and the final `V_T` difference against model-free and fixed-position controls, including variation across seeds, the price-only or verified-distribution branch, missing-open handling and the close/open horizon mismatch. Report a failure to improve wealth without altering test rules.

Review evidence: frozen configuration record, final results, leakage check and proposed commit message.

## Phase 5 — course delivery

- Complete the single Jupyter Notebook with the Week 11 requirement-to-method mapping, pretest diagnostics, models, unchanged-price baseline, full version-2 outputs, validation/ablations, matched trading wealth, causal missing-open and distribution assumptions, local-error/coverage evidence, and negative-result interpretation if needed. Re-run from a clean environment with declared dependencies and one entry point.
- Expand the chronological/taxonomic literature review: add recent related work and application areas; compare this project's **observed pattern** with the source paper's task, architecture, local training and reported performance, without direct cross-dataset numeric RMSE ranking. Add explicit theoretical future questions about latent-rule semantics, local-vs-global losses and concept drift, plus a plausible non-financial application and test design.
- Create a **new final-results deck** from measured results; retain the submitted proposal as history. Correct old K-means, auxiliary-loss, Adam-versus-AdamW and fit-count wording, and state whether the trading result is price-only or verified total wealth.
- Both students check their NTULearn submission requirements and submit by **18 November 2026, 23:59 Singapore time**. Record the peer-review allocation by 23 November and the designated group member's peer-review score submission by **27 November 2026, 23:59**.

Review evidence: clean rerun, Notebook/slides preview, submission checklist and proposed commit message.
