# event-delta

目标：验证低频 RGB 语义锚点与连续 Event 状态增量能否以较低成本改善流式 VLM 在任意查询时刻的状态与回答。R1/R1.2 负责数据约定验证；R2a/R2b 只做 Qwen vision feature 与 GT-flow oracle propagation 的最小及扩展验证，不涉及训练、LLM generation/QA 或 Event encoder。

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

## R2a：Qwen visual feature + GT-flow oracle propagation smoke test

R2a 是 feature propagation 路线的第一项 kill test：在 1 Hz RGB anchor 上只运行一次 Qwen vision encoder，用已确认方向和尺度的 SEVD GT flow recurrently forward-splat visual feature，并与 stale anchor feature 一起对照离线 dense-RGB feature ground truth。

运行环境需要兼容 Qwen3-VL 的 `torch`、`transformers`，以及 `numpy`、Pillow、`matplotlib`。工具不会自动下载依赖或模型；`--local-files-only` 会强制只使用服务器已有缓存。

先对一张 RGB 做结构与 token-order probe：

```bash
mkdir -p artifacts/r2
python tools/probe_qwen_vision.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --image /path/to/one/rgb.png \
  --output artifacts/r2/probe.json \
  --local-files-only
```

probe 使用 BF16、SDPA、eval 和 inference mode，只运行 vision module。它通过实际模块结构和 hooks 记录 last vision block、merger 前后 feature、最终 visual tokens、grid/token 一致性和安装版本源码中的 token-order 证据。若源码不能可靠证明二维 grid 顺序，状态为 `blocked`，不会猜测 row-major 后继续。

运行 6-window oracle smoke test。`--windows` 始终表示所有 `--recording` 合计的窗口总数，不是每条 recording 的数量；这里总计 6，因此 fixed-3/fixed-4 各 3 个：

```bash
mkdir -p artifacts/r2_smoke
python tools/run_r2_oracle.py \
  --root "$SEVD_ROOT" \
  --recording fixed-3 \
  --recording fixed-4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --windows 6 \
  --anchor-stride 10 \
  --max-offset 9 \
  --seed 20260929 \
  --output-dir artifacts/r2_smoke \
  --device auto \
  --local-files-only
```

窗口预算尽量均分；不能整除时，按 CLI 中 `--recording` 的顺序把余数逐一分给前面的 recording，并在 `config.json` 中记录 `total_windows`、逐 recording 的 `windows_per_recording` 和分配规则。每条 recording 自动按 9-step window 的 GT-flow mean pixel activity 均匀覆盖 activity ranks；3 个窗口时对应 low/mid/high，更多窗口时记录精确 activity quantile，并额外归入 low/mid/high 三组。默认比较 post-merger feature；只有 probe 确认 pre-merger merge-blocked 顺序可逆时，才可显式使用 `--feature-stage pre_merger`。

逐窗口原始输出保留为 `per_offset_metrics.csv` 和 `window_summary.csv`。此外输出 `aggregate_metrics.csv`（window-macro、motion-token-count weighted micro，以及 overall、recording、activity group、offset 和交叉分组）、`bootstrap.json`、`probe.json`、`config.json`、`efficiency.json`、`summary.md`、`distance_vs_offset.png`、`recovery_vs_offset.png`、兼容用 `qualitative_features.png`，以及每条 recording 的 `qualitative_features_<recording>_high.png`。无 motion token 的窗口或分组记为 N/A，不按 motion failure 或 win=0 处理。其中 target RGB 只用于 offline dense-feature ground truth；stale/propagated 方法路径不会用 target feature 填补 splatting holes。

GT flow 是 oracle，不是最终 Event 方法。该 smoke test 只回答“避免重复 ViT 是否可能省计算”，不代表 Event encoder 的最终成本；它不运行 LLM generation/QA、不训练模型，也不实现 Event encoder、E-RAFT 或 EventFlash。只有 motion-token cosine distance 改善、offset 3/5/9 多数获胜、改善不只来自固定背景、coverage 足够且 warp 显著快于 vision encode 时，才标记为 `promising`。否则先检查 feature stage、grid ordering 和 warp validity；若三者正确，则记录 kill result 并停止传播路线。

### R2b：expanded-oracle validation

R2b 不改变 R2a 的 post-merger feature、GT-flow convention、motion threshold、因果输入或 efficiency 定义，只把独立采样单位扩展到 60 个窗口。正式服务器命令为：

```bash
mkdir -p artifacts/r2b_expanded
CUDA_VISIBLE_DEVICES=0 python tools/run_r2_oracle.py \
  --root "$SEVD_ROOT" \
  --recording fixed-3 \
  --recording fixed-4 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --windows 60 \
  --anchor-stride 10 \
  --max-offset 9 \
  --seed 20260929 \
  --feature-stage post_merger \
  --bootstrap-replicates 2000 \
  --profile-warmup 3 \
  --profile-repeats 10 \
  --device cuda \
  --output-dir artifacts/r2b_expanded \
  --local-files-only
```

Bootstrap 固定使用完整 window 作为重采样单位，并携带该 window 的全部 offsets；绝不把 token 或 window-offset row 当作独立样本。它分别报告 overall、fixed-3、fixed-4 的 window-macro 与 motion-token weighted micro cosine error reduction 的 point estimate、bootstrap mean/median 和 percentile 95% CI。

R2b 只有在保留 R2a 原有条件的同时，overall、fixed-3、fixed-4 的 macro 和 micro 95% CI 下界都大于 0，才判为 `promising`。任一分组没有 motion evidence 或 CI 跨过 0 均不算稳定复现；N/A 仍表示证据不足，不等同于 propagation failure。

### R2 troubleshooting

- `--device auto` 在 `torch.cuda.is_available()` 为 true 时选择 CUDA，否则明确选择 CPU；可用 `--device cuda` 强制要求 CUDA，CUDA 不可用时会在目录扫描和窗口采样前立即失败，不会静默回退。
- 启动时 `config.json` 和 stdout 会记录 selected device、CUDA availability 与 `CUDA_VISIBLE_DEVICES`；模型加载和首次 feature 提取后继续记录 model parameter、processor tensor 和 feature device，以及 feature stage/grid/shape。选择 CUDA 时任何核心 tensor 留在 CPU 都会立即报错。
- 首次运行到第一步 warp 前的长 CPU-only 区间来自两个串行阶段：low/mid/high window selection 会读取并解压全部 stride-aligned candidate window 的 flow NPZ；随后 `from_pretrained` 先反序列化模型，再由显式 `.to(selected_device)` 移动模型。目录扫描只枚举文件名，feature warp 尚未开始，单张图片 processor 也不是主要的全数据遍历。旧代码硬编码 CUDA，因此并没有有意的 CPU fallback，但缺少日志，无法从外部区分这些阶段；现在会分别显示 `resolving recordings`、`sampling windows`、`loading processor/model` 和各阶段耗时，并对实际 tensor device 做断言。
- 纯 ViT latency 不包含磁盘读取或 processor；CUDA 模式用 CUDA events，并在测量前后 synchronize。CPU 模式使用 `perf_counter`，不会调用 CUDA synchronize。

## R3a：单 transition E-RAFT/SEVD smoke test

R3a 只验证 Event-derived flow 的接口、方向与尺度，不比较 VLM accuracy，也不改动 R2 oracle 结果。工具动态导入固定 commit 的外部 E-RAFT checkout，严格加载服务器已有 `dsec.tar`，不下载数据、代码或 checkpoint。正式输入映射固定为 `Event_i`（old）+ `Event_{i+1}`（new），预测 `i -> i+1`，只用 `flow_i` 作为 oracle 评价。

```bash
conda activate eraft
CUDA_VISIBLE_DEVICES=0 python tools/smoke_eraft_sevd.py \
  --root "$SEVD_ROOT" \
  --recording fixed-3 \
  --source-frame-id "$SOURCE_FRAME_ID" \
  --eraft-root "$ERAFT_ROOT" \
  --checkpoint "$ERAFT_CHECKPOINT" \
  --bins 15 \
  --input-height 480 \
  --input-width 640 \
  --iterations 12 \
  --mode standard \
  --device cuda \
  --output-dir artifacts/r3a_eraft_smoke
```

成功完成接口运行后，输出 `config.json`、`environment.json`、`metrics.json`、`prediction.npz`、`qualitative.png` 和 `summary.md`。`prediction.npz` 的正式 `flow` 始终是正确 packet 顺序和正号的原生 1280×960 pixel displacement；负号及 swapped-event 只作诊断，绝不自动替换正式结果。

本 smoke 只有在 checkpoint strict load、输入输出 shape/finite/尺度正确、预测非全零、正号明显优于负号且正确顺序明显优于 swapped 时才标记 `passed`。否则写出 `blocked` 与原因；这表示接口、方向或单样本诊断尚未通过，不代表正式 E-RAFT 泛化结论。timestamp 单位、Event 区间归属、flow GT 解码仍沿用 R1.2/R3a audit 的既有边界，不在本轮重新论证。
