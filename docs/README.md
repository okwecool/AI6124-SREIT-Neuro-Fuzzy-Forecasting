# AI6124 Group 1 — SRT Neuro-Fuzzy Forecasting

**Team:** Zou Runze (Ryan) and Ngeow Shan Yong Destin.  
**Current method:** a 20-day LSTM encoder, three learned latent features, two Gaussian fuzzy rules and two affine local return experts. The primary target is the next SRT.SI raw closing-price log return. This is the approved revised direction reported by the team; it does not modify LSTM's internal gates.

## Start here

- [V3—V5代码清理与归档说明（中文）](research_archive_2026-11-05_zh.md)：跨阶段归档范围、公共分析模块、换行哈希保护与只读核验入口；最新研究方向仍以模型优化指导为准。
- [当前模型优化指导与历史假设（中文）](improvement_hypotheses_2026-11-04_zh.md)：第0节为V5后更新的研究主线，含旧建议对照、失败原因证据、表征学习→分阶段专家训练→状态约束前件及配套时间验证；第1—6节保留历史H1—H6。方向一、二的具体实验结果见 V6、V7。
- [V5 固定原数据的模型优化实测（中文）](v5_model_only_results_2026-11-05_zh.md)：共享预测＋局部专家、正则化消融及45组运行。相对原Fuzzy2改善0.28%，仍未超过零收益参考；V4海外信息结果不作为模型优化成果。
- [Experimental protocol v1.2](experiment_protocol_lstm_fuzzy_local_experts.md): fixed data, model, training and evaluation decisions.
- [English design](lstm_fuzzy_local_experts_design_en.md) and [Chinese design](lstm_fuzzy_local_experts_design_zh.md): architecture explanation. The protocol takes precedence where older design prose differs.
- [Design review F1–F7](design_review_2026-11-03_zh.md): audit findings and the issues addressed by protocol v1.2.
- [Final-artifact review response](artifact_review_response_2026-11-04_zh.md): R1–R7 fixes and independent package checks.
- [开发现状与交接评估（中文）](development_status_handoff_report_2026-11-04_zh.md)：已完成内容、效果、课程要求与 Destin 接手事项。
- [冻结结果事后诊断（中文）](frozen_result_diagnosis_2026-11-04_zh.md)：预测收缩、专家抵消、早停与微小 RMSE 差距的证据；不修改已冻结的实验结论。
- [第二版 A+B 实现方案（中文）](v2_implementation_plan_2026-11-04_zh.md)：H1 简单基线与 H3 局部损失的预先约定、代码边界及验收顺序。
- [第二版第一轮 A+B 比较（中文）](v2_first_round_results_2026-11-04_zh.md)：H1/H3 开发折实测、机制诊断与停止决定；属于事后探索，不使用原最终测试重新选模。
- [第二版后续优化实验（中文）](v2_followup_results_2026-11-04_zh.md)：更强正则、OHLCV、五日目标与已公告分派的比较；分派信息改善预测基线，模糊残差模型未超过该基线。
- [第二版趋势与国债收益率筛选（中文）](v2_trend_rate_results_2026-11-04_zh.md)：50/200 日总收益均线差及 MAS 10 年期国债收益率的开发期同日期筛选与停止判断。
- [V6 表征预训练试验（中文）](v6_representation_results_2026-11-05_zh.md)：固定原20×6输入、目标和 Fuzzy2 结构，以训练期遮盖重建检验三维表示；辅助损失下降但主预测未改善。
- [V7 分阶段专家训练试验（中文）](v7_staged_experts_results_2026-11-05_zh.md)：共同表示先学、随后日期拟合 Gaussian／普通门控局部残差；Gaussian 权重分开但外层预测弱于同训练共同预测。
- [V8 历史波动状态监督试验（中文）](v8_regime_supervision_results_2026-11-05_zh.md)：原双规则加入训练期高低波动辅助监督，并重跑 V1 匹配对照；状态损失下降，预测改善仅约 0.008%。
- [V9 原六输入只读信号诊断（中文）](v9_signal_audit_results_2026-11-05_zh.md)：跨折核查六项特征与次日收益的关联、高低波动关系和已存双专家误差；无新模型训练。
- [V10 两个向前停止段选轮试验（中文）](v10_multistop_epoch_results_2026-11-05_zh.md)：固定原数据和模型，用两个时间停止段重选 epoch；确实更常选到晚轮，但三个匹配模型的外层误差均恶化。
- [11 月 5 日V2阶段进展审核（中文）](progress_review_and_next_round_2026-11-05_zh.md)：保存结果独立复算、当时的设计与交付缺口，以及历史公告残差实验建议；不作为当前执行队列。
- [V3 保留基础方案的实质优化探索（中文）](material_improvement_exploration_2026-11-05_zh.md)：神经模糊训练、日内目标及早期CLR迁移的负结果；当时建议的开盘前信息增强后来已执行V4，现已退出模型优化主线。
- [V4 开盘前信息增强实测（中文）](v4_preopen_results_2026-11-05_zh.md)：VNQ/SPY信息筛选、108组匹配神经模型、时间审计和固定交易诊断；区分有效信息收益与尚未建立的模糊机制优势。
- [Ryan–Destin 项目对齐稿（中文）](team_alignment_status_2026-11-04_zh.md) / [English](team_alignment_status_2026-11-04_en.md)：工作进度、模型与测试、课程要求、实测效果和下一步决策。
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
python -m src.tools.verify_research_archive
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
