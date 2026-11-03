# Pre-Git cleanup record — 3 November 2026

The following paths were moved without changing their contents to `archive/pre_git_2026-11-03/` before `git init`. The local archive is ignored by Git, so it does not enlarge the initial repository. It remains available for manual inspection or restoration.

| Original path or group | Reason |
| --- | --- |
| `fuzzy_gated_lstm_project_plan_en.md`, `fuzzy_gated_lstm_project_plan_zh.md`, `fuzzy_gated_lstm_design_en.pdf` | Superseded fuzzy-inside-LSTM-gates proposal |
| `output/AI6124_Fuzzy_Gated_LSTM_Proposal.pptx` | Superseded proposal deck |
| `output/SRT_STI_SORA_dataset_v1_2026-10-28.zip` | Superseded by the repaired v2 snapshot; underlying raw data remain in `data/` |
| `work/build_english_design.py`, `work/build_english_pdf.py` | Builders for the superseded design |
| `work/fuzzy_gate_flow_english.png`, `work/fuzzy_gated_lstm_page-1.png`, `work/fuzzy_gated_lstm_page-2.png` | Rendered legacy figures |
| `work/slides_build/` | Old slide-build scripts and render/inspection outputs; scripts use an obsolete local runtime path |
| `work/rendered_english_design/`, `.codex-finalizer/` | Intermediate render and slide-finalization artifacts |

`__pycache__/` was deleted because it contained only reproducible Python bytecode. No source data, aligned data, current model documents, Notebook, evaluation code, current proposal deck or baseline results were removed.

The current `output/SRT_STI_SORA_dataset_v2_2026-10-28.zip` remains available locally but is ignored in Git because its underlying files are included in the proposed initial commit. After Git initialization, the active Markdown documents were organized into the tracked `docs/` directory. Their links and builders were updated. The outdated 30 October handoff ZIP was moved to the local archive and replaced by `output/AI6124_Group1_Early_Work_Handoff_2026-11-03.zip`; both ZIPs remain ignored by Git. The duplicate week-8 lecture PDF was also moved to the archive after its SHA-256 hash matched the copy one directory above the project.

Before the initial commit, the active evaluation module moved to `src/project_eval.py`, its tests to `tests/test_project_eval.py`, the Notebook to `notebooks/AI6124_Group1_Project_Starter.ipynb`, and the project entry document to `docs/README.md`. Imports, build scripts, handoff commands and internal links were updated. The root retains only Git-related files alongside project directories.

The final directory pass consolidated that Notebook into `docs/AI6124_Group1_Project_Starter.ipynb` and the three active helper scripts into `src/tools/`. The now-empty `notebooks/` and `work/` directories were removed. This also changed the handoff-package paths and the documented run commands.
