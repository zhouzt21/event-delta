# AGENTS.md

本仓库用于 RGB + Event 流式 VLM 的副线实验。所有后续 Codex 工作必须遵守以下约束：

1. 严格按顺序推进：先验证 state staleness，再验证 Event state delta，最后才接 Qwen3-VL-2B。
2. Event 必须与 Frozen、因果恒速外推和 RGB-derived motion 在明确预算下比较。
3. 查询时只能使用查询时间之前的 RGB 和 Event；GT boxes/IDs/flow 用作初始化或上界时，必须标为 oracle。
4. 不把“结构化状态交给 LLM”宣称为已经节省 vision encoder；这一点要等 feature/token propagation 实现后才验证。
5. 不提交数据、模型权重、实验产物、密钥或服务器绝对路径。
6. 不在未经用户确认时下载大数据、运行 GPU 训练、创建远端仓库或执行 `git push`。
7. 每次实验记录 Git commit、配置、数据范围、指标和延时。

现有阶段性结论与调研以 `docs/research.md` 为依据。不要重新调研，不要重新论证已被否定的“Event Count 是 semantic update gate”，不要预先搭建复杂框架或猜测 SEVD 文件格式。
