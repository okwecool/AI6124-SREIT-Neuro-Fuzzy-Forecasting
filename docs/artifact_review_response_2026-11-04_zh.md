# 4 November artifact review: response for Ryan's inspection

本轮只修复交付、呈现及复现链路。开发期冻结配置、最终 12 次拟合所得预测、交易规则及已报告的主结论均未更换。上一阶段最终测试已提交为 `48b015e`；本轮修改尚待 Ryan 检查后提交。

| 审阅项 | 本轮处理 | 检查证据 |
| --- | --- | --- |
| R1 单一 Notebook | 新建 `AI6124_Group1_Final_Project.ipynb`；包含代码、已冻结结果、图、文献与结论，全部单元在 Jupyter kernel 执行并保存输出。旧 Starter 留作历史记录，不进入最终 ZIP。 | 17 个代码单元、45 个保存输出；Notebook 校验通过。 |
| R2 checkpoint 缺失 | 完整重建仅在 manifest、预测、checkpoint 及模糊诊断均齐全且源码哈希匹配时跳过；否则要求新的复现目录。另设 `--saved-only` 读取已保存导出，不依赖 checkpoint。 | 缺失 checkpoint 的单元测试通过；解压副本只读开发核验通过。 |
| R3 打包清单 | 新版 ZIP 包含数据、源码、测试、依赖、报告、最终 Notebook/deck、开发与最终导出及逐文件 SHA-256 manifest；排除本机 checkpoint 和缓存。 | 437 个文件、约 10.8 MB；ZIP CRC 与 SHA-256 核对通过；独立解压目录验证数据、开发、最终和 Notebook。 |
| R4 正常完成状态 | 最终 runner 的正常及恢复完成记录均有聚合导出哈希；校验器接受两种合法终态。 | 契约单元测试与既有恢复记录核验通过，未重训最终模型。 |
| R5 环境 | 增加 `requirements.txt`，README/交接说明写明本机验证过的 Python 3.10 解释器和 Jupyter kernel 选择。 | 本机环境 21 项测试通过；解压副本可从源码重跑一个两轮 Fuzzy2 开发折 smoke fit。 |
| R6 研究呈现 | 增补预测分布、仓位、预测对真实值、规则贡献、内层停止曲线及代表案例；文献章节写入与 Fuzzformer 的方法/结果对照、局限、理论问题和另一应用。 | 图表由冻结 CSV 与 manifest 生成；报告、Notebook 与 deck 均明确说明微小 RMSE 改善、现金基准及弱规则分工。 |
| R7 最终 deck 与状态 | 新建独立八页最终结果 PPTX，保留历史提案；修正分派账本、阶段状态与九次来源核验/最终两次除息日的表述。 | PPTX 结构、几何、表格检查及逐页预览通过。 |

独立解压副本执行的只读检查显示：数据形状 `(1182,20,6)`；开发期 7,182 条预测、3,780 个模糊 case、7,938 条规则；最终 3,263 条预测、753 个模糊 case、1,506 条规则，四种本机财富账本终值核验通过。一个两 epoch 的 smoke fit 仅在解压副本的 `output/runs/reproduction/` 下，不能视为新的最终测试结果。

**交付前仍需人工检查：** Ryan 与 Destin 审阅最终 Notebook、八页 slides 的措辞和课堂要求，并各自确认 NTULearn 上传的文件及截止时间。最终测试期间的 2025-10-24 开盘价仍缺失；交易结论必须继续标为使用预先声明的 no-trade-and-carry 规则的模拟。
