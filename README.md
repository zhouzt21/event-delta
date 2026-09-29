# event-delta

目标：验证低频 RGB 语义锚点与连续 Event 状态增量能否以较低成本改善流式 VLM 在任意查询时刻的状态与回答。当前仍处于数据约定验证阶段，不涉及 Qwen、VLM、feature/token propagation 或训练。

## R1/R1.1：SEVD 静态审计

确认目录、NPZ schema、frame ID 交集、时间戳数值尺度与因果时间轴待决项：

```bash
python tools/audit_sevd.py \
  --root "$SEVD_ROOT" \
  --recording fixed-3 \
  --recording fixed-4 \
  --metadata-path "$SEVD_METADATA_JSON" \
  --output artifacts/r1/audit.json
```

## R1.2：Event/Flow 时空约定自动校准

R1.2 使用相邻 RGB、Event、flow 与 RGB KITTI object ID，对 Event 时间归属和坐标 flip，以及 flow 的尺度、符号、时间配对和背景处理做候选比较。工具默认仅使用 CPU，只读取已解压目录，不扫描或复制 zip：

```bash
python tools/calibrate_sevd.py \
  --root "$SEVD_ROOT" \
  --metadata-path "$SEVD_METADATA_JSON" \
  --recording fixed-3 \
  --recording fixed-4 \
  --samples 120 \
  --seed 20260929 \
  --output-dir artifacts/r1_2
```

`--samples` 是在 recordings 间平衡分配的总样本数；每条 recording 内使用 seeded stratified sampling 均匀覆盖时间轴，并排除缺少 `i-1` 或 `i+1` RGB 的样本。

输出包括：

- `calibration.json`：采样 frame IDs、完整候选指标、逐帧/总体 flow 数值统计、候选状态、工具版本与未决假设；
- `event_candidate_scores.csv`、`flow_candidate_scores.csv`：所有候选的分项指标与排名；
- `event_alignment_<recording>.png`：Event 与前/后 RGB difference 的早/中/晚定性检查；
- `flow_calibration_<recording>.png`：带 colorbar 的 raw u/v、raw magnitude、去背景 magnitude 与最佳候选像素位移；
- `best_warp_<recording>.png`：source、target、forward-splatted source 与 residual；
- `summary.md`：确认项、歧义项、约定结论与 R2 readiness。

Event 的 integer timestamp unit `≈ ns` 只是存储数值尺度推断，不代表 1 GHz perception。一个 Event packet 约覆盖 100 ms，即 10 Hz。首个实验计划仍是 **1 Hz RGB anchor + 10 Hz Event delta**；中间 RGB 只作为 offline ground truth，不能作为查询时刻之前不可用的在线输入。

R1.2 只验证数据的时空约定，不验证 Event state delta 的核心研究假设，也不宣称结构化状态已经节省 vision encoder。KITTI boxes/IDs 与 flow 在这里属于 oracle 校准证据。
