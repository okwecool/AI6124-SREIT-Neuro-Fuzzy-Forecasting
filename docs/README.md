# AI6124 Group 1 — SRT Neuro-Fuzzy Forecasting

**Team:** Zou Runze (Ryan) and Ngeow Shan Yong Destin.  
**Current method:** a 20-day LSTM encoder, three learned latent features, two Gaussian fuzzy rules and two affine local return experts. The primary target is the next SRT.SI raw closing-price log return. This is the approved revised direction reported by the team; it does not modify LSTM's internal gates.

## Start here

- [Experimental protocol](experiment_protocol_lstm_fuzzy_local_experts.md): fixed data, model, training and evaluation decisions.
- [English design](lstm_fuzzy_local_experts_design_en.md) and [Chinese design](lstm_fuzzy_local_experts_design_zh.md): architecture explanation. The protocol takes precedence where older design prose differs.
- [Literature review draft](literature_review_draft_en.md): initial research context.
- [Dataset guide](../data/README.md): source provenance, publication-time alignment and quality audit.
- [Development plan](development_plan.md): implementation phases and review gates.
- [Destin handoff](README_DESTIN_HANDOFF.md) and [Notebook starter](AI6124_Group1_Project_Starter.ipynb): current course deliverable scaffolding.

The exact input snapshot is stored under `data/raw/`; the aligned daily data, unscaled windows and fold assignments are under `data/processed/`. These small files are included in the proposed initial commit for reproducibility. A second-source SRT closing price repairs 2025-10-24; that day's opening price is still missing and must not be filled with its close for trading evaluation.

The project root has no source files: `src/` holds the evaluation module and supporting tools, `tests/` holds its checks, and `docs/` holds the written material and course Notebook. Run the commands below from the project root.

## Read-only checks

Run from the project root in an environment with NumPy, pandas, matplotlib and nbformat:

```powershell
python data/verify_dataset.py
python -m unittest discover -s tests -v
python src/tools/verify_starter_notebook.py
```

The default feature array has shape `(1182, 20, 6)`. The existing code in `src/project_eval.py` handles chronological fold selection, train-only scaling, prediction-file checks and scoring. No trained LSTM or fuzzy model is present yet.

## Git workflow

The repository was initialized on 3 November 2026 after moving superseded files into `archive/pre_git_2026-11-03/`. That archive is intentionally local and ignored by Git; [the cleanup record](pre_git_cleanup.md) lists its contents. Generated ZIP bundles, local environments and model checkpoints are also ignored. The source dataset, protocol, code, Notebook starter and small baseline results are intended for version control.

At the end of each development phase, present the changed files, validation results and proposed commit message to Ryan. **Do not create a commit until Ryan has checked and explicitly approved that phase.** Once approved, commit only the reviewed changes. The final holdout must remain unused for model selection.
