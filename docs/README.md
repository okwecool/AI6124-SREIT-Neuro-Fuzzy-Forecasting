# AI6124 Group 1 — SRT Neuro-Fuzzy Forecasting

**Team:** Zou Runze (Ryan) and Ngeow Shan Yong Destin.  
**Current method:** a 20-day LSTM encoder, three learned latent features, two Gaussian fuzzy rules and two affine local return experts. The primary target is the next SRT.SI raw closing-price log return. This is the approved revised direction reported by the team; it does not modify LSTM's internal gates.

## Start here

- [Experimental protocol v1.2](experiment_protocol_lstm_fuzzy_local_experts.md): fixed data, model, training and evaluation decisions.
- [English design](lstm_fuzzy_local_experts_design_en.md) and [Chinese design](lstm_fuzzy_local_experts_design_zh.md): architecture explanation. The protocol takes precedence where older design prose differs.
- [Design review F1–F7](design_review_2026-11-03_zh.md): audit findings and the issues addressed by protocol v1.2.
- [Final-artifact review response](artifact_review_response_2026-11-04_zh.md): R1–R7 fixes and independent package checks.
- [开发现状与交接评估（中文）](development_status_handoff_report_2026-11-04_zh.md)：已完成内容、效果、课程要求与 Destin 接手事项。
- [冻结结果事后诊断（中文）](frozen_result_diagnosis_2026-11-04_zh.md)：预测收缩、专家抵消、早停与微小 RMSE 差距的证据；不修改已冻结的实验结论。
- [第二版改进假设清单（中文）](improvement_hypotheses_2026-11-04_zh.md)：最小实验、对照、晋级与放弃条件，以及独立验证边界；当前为实验前讨论稿。
- [第二版 A+B 实现方案（中文）](v2_implementation_plan_2026-11-04_zh.md)：H1 简单基线与 H3 局部损失的代码边界、训练契约和验收顺序；尚未执行。
- [Literature review and result comparison](literature_review_draft_en.md): method lineage, observed differences and future questions.
- [Dataset guide](../data/README.md): source provenance, publication-time alignment and quality audit.
- [Development plan](development_plan.md): implementation phases and review gates.
- [Development results](development_results_2026-11-03.md) and [frozen final results](final_results_2026-11-04.md): measured comparisons and qualified conclusions.
- [Destin handoff](README_DESTIN_HANDOFF.md) and [executed final Notebook](AI6124_Group1_Final_Project.ipynb): course deliverables and reproduction entry points. The separate eight-slide final deck is `../output/final_slides/AI6124_Group1_Final_Results_2026-11-04_v2.pptx`, and the shareable package is `../output/AI6124_Group1_Final_Handoff_2026-11-04.zip`.

The exact input snapshot is stored under `data/raw/`; the aligned daily data, unscaled windows and fold assignments are under `data/processed/`. These small files were included in the initial commit for reproducibility. A second-source SRT closing price repairs 2025-10-24; that day's opening price is still missing and must not be filled with its close for trading evaluation.

The project root has no source files: `src/` holds the evaluation module and supporting tools, `tests/` holds its checks, and `docs/` holds the written material and course Notebook. Run the commands below from the project root.

## Read-only checks

Run from the project root with Python 3.10 and the packages in [`requirements.txt`](../requirements.txt), including PyTorch. On this Windows machine, the verified interpreter is `E:\Environment\Anaconda3\envs\env_cuda\python.exe`; the default `python` on PATH lacks PyTorch. A fresh environment can be prepared with `python -m venv .venv` and `.venv\Scripts\python -m pip install -r requirements.txt`. CPU execution is supported. Select that same interpreter as the Jupyter kernel when opening the Notebook.

Read-only checks (replace `python` with the verified interpreter or your environment's Python):

```powershell
python data/verify_dataset.py
python -m unittest discover -s tests -v
python src/tools/verify_starter_notebook.py
python -m src.tools.verify_final_outputs
python -m src.tools.analyze_development --saved-only
```

The default feature array has shape `(1182, 20, 6)`. `src/project_eval.py` handles chronological folds, train-only scaling, full-key prediction checks and forecast scoring. `src/train.py` implements development-fold stopping and refit; `src/portfolio.py` records causal observed-open price and gross-distribution wealth. The three-fold development matrix, optional `Gate2` control and once-only frozen final evaluation are complete. The results reports explain the small forecasting differences and negative trading findings.

The full per-open final wealth ledgers and PyTorch checkpoints remain local reproducible outputs and are ignored by Git; the final predictions, manifests, summary tables, figures and execution record are intended for version control. The final runner refuses an existing result directory to prevent an accidental second test run.

## Git workflow

The repository was initialized on 3 November 2026 after moving superseded files into `archive/pre_git_2026-11-03/`. That archive is intentionally local and ignored by Git; [the cleanup record](pre_git_cleanup.md) lists its contents. Generated ZIP bundles, local environments and model checkpoints are also ignored. The source dataset, protocol, code, executed final Notebook and small result files are intended for version control.

At the end of each development phase, present the changed files, validation results and proposed commit message to Ryan. **Do not create a commit until Ryan has checked and explicitly approved that phase.** Once approved, commit only the reviewed changes. The final holdout must remain unused for model selection.
