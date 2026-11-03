# Code development plan — review draft, 3 November 2026

This plan implements the [fixed experiment protocol](experiment_protocol_lstm_fuzzy_local_experts.md). It is a development sequence, not a record of completed model results. The team will review each phase before its Git commit.

## Phase 0 — repository baseline (current phase)

1. Archive superseded files and initialize Git on `main`.
2. Keep the saved SRT/STI/SORA snapshot, current protocol, Notebook starter, evaluation code and current proposal deck.
3. Verify the dataset audit, existing evaluation tests and Notebook code cells.
4. Show the file inventory and proposed initial commit. **Wait for Ryan's approval before committing.**

Suggested commit after approval: `chore: establish reproducible project baseline`.

## Phase 1 — model definitions and smoke test

- Add one reusable 32-unit, single-layer LSTM encoder and three heads: `LSTM32`, `LSTM3`, `Fuzzy2`. The fuzzy head uses the protocol's three-dimensional `tanh` projection, two diagonal Gaussian rules and two affine experts. No attention, fuzzy internal LSTM gate or K-means initialization is in the primary model.
- Expose standardized-return predictions and, for `Fuzzy2`, latent features, two rule weights and two expert outputs. Document parameter counts.
- Run a single minibatch and one short development-fold training run to check finite loss, tensor shapes, gradient flow and `ŷ = Σ αₖeₖ`. The short run is a code check, not an outcome claim.
- Do not open final-holdout predictions or metrics.

Review evidence: model file diff, shape/gradient checks, smoke-test log and proposed commit message.

## Phase 2 — reproducible training runner

- Use `src.project_eval.Dataset.prepare()` for train-only standardization and the saved fold IDs. Implement the inner 15% chronological stopping set, early stopping, then same-seed refit on the full outer train for the selected epoch count.
- Fix seeds `11, 23, 37`; record Python, NumPy, pandas and PyTorch versions, device, feature group, model ID, seed, best epoch, train/validation losses, timing and checkpoint path in a run manifest.
- Write raw-return prediction CSVs using the existing six-column contract; store fuzzy diagnostics separately. Reject incomplete or wrong-window exports before scoring.
- Keep the CPU path usable. GPU acceleration can be enabled after a local check, but the experiment must not depend on it.

Review evidence: one complete fold for all three learned models, manifest, prediction-contract checks and proposed commit message.

## Phase 3 — development comparisons

- Run `LSTM32`, `LSTM3` and `Fuzzy2` on all six features for three expanding folds and three seeds. Compare with the already computed `Z0` no-change baseline.
- Run `Fuzzy2` source ablations (`srt`, `srt_sora`, `all`) on the same dates; perform the predeclared one/two/three-rule diagnostic. Report fold-wise and pooled validation metrics, paired differences, parameter counts and runtime.
- Inspect rule-weight distributions, expert disagreement and relationships with **past-only** market summaries. Do not assign economic names to latent dimensions or rules without evidence.
- Add a compact stationarity diagnostic to the Notebook: raw price versus log return, rolling mean/variance, and a clear explanation of why the model predicts log-price differences. This follows the week-10 lecture's warning about direct nonstationary price forecasting.
- Extend the trading comparison to include a **no-prediction strategy whose parameters are selected using training-period data only**. Keep the protocol's next-open execution, transaction-cost sensitivity and price-only label until all dividends are verified. This closes the project Part-2 gap found in the week-8/9 lectures. Choose the strategy and its small parameter search before inspecting final test performance, then amend the protocol accordingly.

Review evidence: development-only tables/plots, strategy specification and selected parameters, interpretation checks, protocol amendment if needed and proposed commit message.

## Phase 4 — frozen final evaluation

- Freeze the model and strategy configurations from development results. Retrain each predeclared model on the 931 pretest windows with the protocol's epoch rule; evaluate the 251-window final holdout once.
- Store final prediction/diagnostic files separately from development outputs. Report both successes and failures, including variation across seeds and the price-only trading limitations.

Review evidence: frozen configuration record, final results, leakage check and proposed commit message.

## Phase 5 — course delivery

- Complete the single Jupyter Notebook with method, chronological literature review, source audit, models, baselines, validation, ablations, trading comparison, interpretation, limitations and future work. Re-run all code cells in the intended submission environment.
- Update the final slides from measured results. The existing proposal deck is background material, not a final-results deck. Correct any old wording that implies K-means initialization or fuzzy modifications inside LSTM gates.
- Both students check their NTULearn submission requirements and submit by the course deadline. Coordinate the later peer-review task separately.

Review evidence: clean rerun, Notebook/slides preview, submission checklist and proposed commit message.
