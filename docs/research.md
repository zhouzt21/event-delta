# RGB–Event 流式 VLM：阶段 0 调研、最小实验与停机边界

**日期：** 2026-09-28  
**目标：** 用一周内可结束的实验判断“低频 RGB 语义锚点 + 连续 Event 状态增量”是否值得继续；不把本项目扩成个人主线。

## 1. 当前进度的客观解读

现有工作已经完成了一次有价值的剪枝：在 SEVD fixed-3 / fixed-4、765 个独立定义的状态转移标签上，Event Count 与 Pixel Difference 的 exact-rate frontier、AP 和无标签持续运动区间的触发率几乎重合。因此已有证据支持：

1. **Event Count 是 motion / visual-change proxy。**
2. **没有证据支持 Event Count 是天然更好的 semantic update gate。**
3. 旧假设应该停止，不应继续通过换阈值、挑案例或增加 gate 网络“救结论”。

新假设比旧假设合理，但需要进一步收窄：

> 在昂贵 RGB 语义编码只以低频运行时，Event 是否能以较低增量成本，因果地修正当前结构化世界状态，并改善任意查询时刻的回答？

这里 Event 不决定“值不值得更新”，只提供“自上次 RGB 语义锚点后发生了什么”的证据。该假设符合传感器互补性，也与 LiFR-Seg 的传播范式一致；但它尚未被 VLM 层验证。

## 2. 论文和开源状态带来的判断

### 2.1 LiFR-Seg：机制先例成立，但不能直接等价为 VLM 贡献

[LiFR-Seg](https://arxiv.org/abs/2603.21115)定义了 Anytime Interframe Semantic Segmentation：只给一个过去 RGB 帧和随后到达的异步事件，在任意中间时刻输出语义分割。论文报告 DSEC 上 73.82 mIoU，与能看到目标 RGB 帧的高帧率上界只差 0.09 个百分点。其 Lite 版本使用 RAFT-small 和 1/16 光流，在 RTX 3090、440×640 输入上报告 73.49 mIoU、约 65.6 FPS。

这证明了“低频语义锚点 + Event delta”在**密集像素语义**上的可行性，也提示计算要按摊销成本衡量。它没有证明：

- Event 能维护对象身份、属性、关系和语言可查询状态；
- 传播后的状态能提升 VLM 回答；
- Event 分支比提高 RGB 频率或使用 RGB 光流更划算。

论文链接的 [LiFR-Seg GitHub](https://github.com/Candy-Crusher/LiFR-Seg) 在本次核查时没有可用源文件，因此不能作为本周直接复现的底座。

更关键的是，[LiFR v2](https://arxiv.org/abs/2609.25803)（2026-09-22）明确指出：纯事件传播在**新出现目标和解除遮挡区域**缺少有效 RGB 支撑，因此加入 Event-Guided Completion 与历史检索。这个限制会直接出现在我们的 Object Entry、遮挡和重识别问题中，必须单列指标，不能用总体平均掩盖。

### 2.2 EventFlash：适合作为后续设计参考，本周不适合作为原型

[EventFlash](https://github.com/XduSyL/EventFlash)面向 event-only MLLM，使用自适应时间窗聚合和稀疏密度引导注意力，论文报告相对 EventFlash-Zero 12.4× throughput，并支持 1,000 个时间 bin。它回答的是事件 token 的长序列压缩问题，与“encoder 之后的 token sparsification”相关。

代码核查结论：仓库仍标注 Stage-2 SFT 和完整代码 “coming soon”；当前文件还引用了未发布的 encoder、projector、token merge 和数据处理模块。因此它不能承担第一周的可运行基线。最合理用途是第二阶段借鉴其 token 密度统计和时间窗设计。

### 2.3 EventGPT：可运行的事件表征参考，但不是目标系统

[EventGPT](https://github.com/XduSyL/EventGPT)公开了推理代码、样例和 7B checkpoint。其事件输入被整理为少量 event frames，再经 CLIP 视觉塔和 LLM 推理。它可以用于检查“事件栅格化后能否被语言模型识别”，也可以复用数据处理思路。

限制是：它是 event-only、7B、短窗模型，没有低频 RGB anchor、因果状态维护或同预算系统比较。因此不建议把它接成主底座。

### 2.4 EventDrive：最接近的 2026 竞争工作，迫使创新点收窄

[EventDrive](https://arxiv.org/abs/2606.18242)已经提供 47.1 万个 event–frame–language 样本，覆盖 perception、understanding、prediction、planning，并用多时间尺度事件体、RVT、Event Q-Former 与 Qwen2.5-VL-7B 训练 RGB+Event VLM。公开训练配置使用 16 张 H20，说明完整复现不符合本项目的“快速探索”目标。其[数据和评测工具](https://huggingface.co/datasets/dylanorange/EventDrive)已公开，模型完整训练代码/权重不宜作为本周依赖。

因此不能再主张“首次把 RGB 和 Event 接入 VLM”或“Event 有助于高层理解”。仍可能成立的差异是：

- 因果、在线、任意查询时刻；
- 昂贵 RGB semantic encoder 低频运行；
- Event 用于 inter-anchor state maintenance，而非同步双模融合；
- 对比提升 RGB 采样率、RGB 光流、恒速外推和 codec delta；
- 报告准确率、视觉编码次数、token、P50/P95 延时和显存的 Pareto 前沿。

如果实验最后只是把事件图和 RGB 图拼给 Qwen，创新点会与 EventDrive 大量重叠，应停止论文化投入。

### 2.5 Streaming Video MLLM：State Staleness 问题合理，但基线要强

[TimeChat-Online](https://arxiv.org/abs/2504.17343)用 RGB token 差异删除 82.8% 视觉 token，同时保持 98% StreamingBench 性能，说明变化驱动的 RGB 压缩是必须比较的强基线。QueryStream 进一步把 query relevance 加入 token 保留和响应时机，说明 query-agnostic change 不能直接代表任务价值。

[StreamMind](https://arxiv.org/abs/2503.06220)在 A100 上报告 100 FPS 感知，并通过轻量前端和 cognition gate 控制昂贵 LLM 调用。这里的 “Event-Preserving” 指语义事件，并非事件相机，但它清楚地区分了高频轻量感知与高频昂贵认知。

[StreamingVLM](https://github.com/mit-han-lab/streaming-vlm)已有完整训练、推理和效率脚本，报告单 H100 最多 8 FPS；它适合后续接真实 streaming KV cache，不适合第一周，因为会把“Event delta 是否有效”和“长流记忆系统是否正确”两个风险绑在一起。

## 3. 研究假设分层

必须依次通过四层，前一层失败就不做后一层：

| 层级 | 假设 | 直接指标 | 失败含义 |
|---|---|---|---|
| H0 问题 | 低频 RGB anchor 会造成可测的状态陈旧 | 误差随 anchor age 增长 | 当前数据不适合，先换数据或问题 |
| H1 机制 | Event delta 能比冻结状态和廉价 RGB 外推更好地恢复当前状态 | box/center/zone/count/entry-exit | Event 没有带来独特 state delta |
| H2 下游 | 更准确的状态会提升 query-time QA | event-sensitive QA accuracy/F1 | 低层收益没有转成 VLM 价值 |
| H3 系统 | 收益在相同总成本下优于增加 RGB 语义频率 | accuracy–latency–encoder-call Pareto | 没有效率论文空间 |

已有另一条流式视频实验中，“只给过时观测”的回答准确率从 0 秒的 0.724 下降到 5 秒的 0.592、30 秒的 0.439。这支持 H0 在一般视频问答中存在，但不能代替本项目在 RGB–Event 数据上的 H0，也不能证明 H1。

## 4. 数据集选择

### 第一选择：SEVD Fixed

[SEVD](https://github.com/eventbasedvision/SEVD)提供 fixed 和 ego 两类仿真数据，包含 RGB、DVS、depth、GT optical flow、semantic/instance segmentation 和带 object ID 的 boxes。fixed 录制可以去掉相机自运动；仿真代码显示各视觉传感器使用同一分辨率、FOV 和位姿，并按相同 CARLA frame ID 入队，适合做严格机制诊断。

限制：RGB/标签主时钟约 10 Hz、场景是 CARLA、语言标注需自行构造。它只能支持 100 ms 级的有标签验证，不能据此声称微秒级 QA。

### 第二选择：FRED 或 MEVDT，用于真实数据方向验证

| 数据集 | 优点 | 限制 | 建议 |
|---|---|---|---|
| [FRED](https://arxiv.org/abs/2506.05163) | 7+ 小时，1280×720，RGB+原始 Event，时空对齐，bbox+track ID，检测/跟踪/轨迹预测 | 目标几乎都是小型无人机，语义窄；下载和预处理较重 | H1 通过后首选真实 RGB 验证 |
| [MEVDT](https://arxiv.org/abs/2407.20446) | 固定 DAVIS 传感器，63 段、24 Hz 标注、bbox+ID，规模小，适合快速复核 | APS 是灰度而非 RGB，规模与语义有限 | 最快的真实传感器复核 |
| [TUMTraf Event](https://arxiv.org/abs/2401.08474) | 固定路侧、真实 RGB+Event、4,111 对同步图像、50,496 boxes | 必须先确认是否保留连续 raw events 与精确时间戳 | 通过格式审计后再选 |
| EventDrive | 有语言任务和公开评测，规模大 | DSEC/M3ED 含 ego-motion；事件多为与帧配对的表示；完整训练昂贵 | 作为外部 VLM 语义评测，不作第一 kill test |

**选择规则：** 第一周只用 SEVD fixed-3/4。只有 H1 通过才下载真实数据。若希望最省时间先选 MEVDT；若坚持真 RGB 和高速 inter-frame dynamics，选 FRED。

## 5. 最小可行实验

### 5.1 先做结构化状态，不先改 Qwen encoder

状态定义：

```text
S_t = {object_id, class, bbox/center, zone, visible, motion_direction, confidence}
```

anchor 时刻由 RGB/标注建立状态；anchor 之间只允许使用截至查询时刻的事件流。第一版允许用 GT anchor boxes 和 IDs 作为 **oracle initialization**，目的是隔离 delta 机制；之后再替换成检测器输出。所有 oracle 项必须在结果表中明确标注。

### 5.2 输入表征

第一版使用 5 或 10 个时间 bin 的双极性 voxel grid：

```text
E[t0:tq] -> [2, B, H, W]
```

同时保留 event count、time surface 和二值非空 patch mask，方便做轻量消融。不要一开始把事件伪彩色图当普通 Qwen 图片：零样本失败只说明域不匹配，不能否定事件增量。

### 5.3 Delta 原型顺序

1. **Box-local event centroid / correlation**：在上一状态框的扩展 ROI 内估计位移，半天可实现，用于最快 kill test。
2. **E-RAFT**：官方仓库含模型、DSEC/MVSEC checkpoints 和 15-bin voxel loader，但环境较旧且需写 SEVD adapter；只有简单方法出现正信号或明显欠拟合时再接。
3. **FireNet / E2VID 类重建**：把事件变成中间灰度图再用常规光流。它会牺牲事件稀疏性，只作诊断，不作主方法。
4. **LiFR 式 feature propagation**：H1、H2 都通过后才改视觉特征；这是第二周以后的研究，不是第一版。

### 5.4 必须比较的基线

在固定 1 Hz 昂贵 RGB semantic anchor 下比较：

- Frozen anchor state；
- Causal constant-velocity extrapolation；
- RGB RAFT-small / SEA-RAFT，由相邻廉价 RGB 帧估计 delta；
- codec motion vector / residual（若数据管线可稳定取得）；
- Event centroid/correlation；
- E-RAFT（条件式加入）；
- GT optical flow oracle；
- 2/5/10 Hz RGB semantic encoding 上界。

Pixel Difference 仍可作为 gate 对照，但不是 state-delta 对照。不能只比较 Event 与 Frozen，否则结论会被“任意运动模型都能赢”解释。

### 5.5 指标

**状态指标：** center error / diagonal、bbox IoU、zone accuracy、object count F1、entry/exit recall、ID continuity；按距离 anchor 的 0.1–0.9 s 分桶。Object Entry、Exit、遮挡/解除遮挡必须单列。

**VLM 指标：** 当前数量、当前分区、左右关系、进入/离开、运动方向，以及“RGB 属性 + Event 当前位置”的组合问题。使用 exact match / macro F1，并单列静态问题，确认 Event 分支没有破坏 RGB 语义。

**效率指标：** 每个模块 warm-up 后的 P50/P95 延时、视觉 encoder 调用次数、视觉 token 数、峰值显存、端到端 query latency。FLOPs 可补充，但不能代替延时；LiFR-Seg 的数据也显示低 FLOPs 不一定有更高 FPS。

## 6. Qwen 接法与效率测量

第一版选择 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)：小于 7B、官方 Transformers 支持、可控制视频 FPS 和视觉 token 预算，适合单卡快速推理。若兼容性不稳定，再退到 Qwen2.5-VL-3B-Instruct。

按以下两条输入路径分开报告：

1. **结构化状态路径（首选诊断）**：Qwen 只在 anchor 时看 RGB；Event 分支持续更新 state table；查询时把当前 state table 作为文本上下文送入 LLM。优点是最快验证 H2 和系统调度，缺点是贡献更接近感知+状态维护。
2. **伪当前视觉路径（次选）**：用 Event flow warp anchor 的图像/patch/特征，生成当前画布和置信度 mask，再送入 Qwen。图像 warp 会再次运行视觉 encoder；只有 feature propagation 才可能真正省掉 encoder。

因此第一版不能把“少传 RGB”直接写成“少跑视觉 encoder”。结构化路径验证 state maintenance；feature propagation 才验证 encoder amortization。

效率测试先用 Transformers，分别打点 event binning、delta inference、state update、vision encode、LLM prefill、decode。算法出现正结果后再接 vLLM 或 StreamingVLM；否则推理框架优化会遮蔽机制失败。

## 7. 一周执行安排与每一步停机点

| 时间 | 本地 Codex | GPU 服务器 | 产出 | 当日停机条件 |
|---|---|---|---|---|
| Day 0–1 | 数据审计脚本、manifest、因果窗口单测 | 抽样解码/可视化 | R1 数据完整性报告 | 对齐/时间戳无法确认则暂停建模 |
| Day 2 | query 模板与指标 | Frozen、恒速、GT-flow oracle | R2 staleness 曲线 | anchor age 下没有明显退化则换数据，不接 Event 模型 |
| Day 3 | Event voxel 与 ROI delta | Event centroid/correlation 全量 | H1 初版表 | 不胜 Frozen/恒速则停止简单路线 |
| Day 4 | RGB flow / codec baseline adapter | 同预算全量对比、bootstrap | R3 机制+效率报告 | Event 不胜廉价 RGB delta 则不进 VLM |
| Day 5 | QA manifest、Qwen runner、打点 | 200–500 个查询，Qwen3-VL-2B | H2/H3 初版 | QA 无增益或成本被 2 Hz RGB 支配则停止 |
| Day 6 | 错误案例与图表 | 必要的复跑 | R4 go/no-go 汇报 | 按下述门槛做结论 |

预计投入 15–20 个专注工时。Day 2、Day 4 都是合法结束点；不需要为了“做满一周”继续。

## 8. 继续和停止的量化边界

这些门槛应在看结果前写进配置或报告：

### Gate A：数据是否能测到 Staleness

在 0.5–0.9 s anchor age 上，Frozen 相比 target GT 至少出现以下之一：zone/count/transition 指标下降 **5 个绝对百分点**，或归一化中心误差随时间显著增长。未达到时，只能说明当前片段不适合，优先换录制或 FRED/MEVDT。

### Gate B：Event delta 是否有机制价值

在 fixed-3 和 fixed-4 都满足以下之一：相对 Frozen 和恒速外推，state accuracy 提升至少 **5 个百分点**；或中心误差相对下降至少 **15%**。同时用 sequence/bootstrap 置信区间检查稳定性。

如果 Event 与廉价 RGB optical flow 差异小于 **2 个百分点**且延时不占优，停止“Event 独特增量”主张。可保留为多传感器鲁棒性工程，但不继续主论文路线。

### Gate C：VLM 下游价值

Event-sensitive QA 提升至少 **3–5 个百分点**，静态语义问题下降不超过 **1–2 个百分点**。如果结构化 state 变准但 QA 不变，说明任务或接口没有把增量价值转化为语言理解。

### Gate D：系统价值

Event 方案必须在 accuracy–P95 latency–RGB encoder calls 三维中形成非支配点。若 2 Hz RGB 在准确率和总成本上同时优于 Event+1 Hz，停止效率主张。

### Gate E：可发表性

在至少一个真实固定传感器数据集上复现方向；并明确超过 EventDrive 的范围：在线因果查询、稀疏语义 anchor、状态陈旧曲线和系统前沿。若只能在 SEVD 成立，只能作为内部机制结果或 workshop 级探索。

**项目级停止线：** SEVD 和一个真实 fixed 数据集上，Event 都不能在等成本下超过 causal RGB-only delta，则结束该方向。不要转而增加更大 VLM、更多参数或更复杂融合来回避负结果。

## 9. 主要风险

1. **Event 不含绝对外观语义。** 类别、颜色和细粒度属性必须由 RGB anchor 维持。
2. **新目标/解除遮挡不可由纯传播恢复。** 这是 LiFR v2 已明确验证的结构性问题。
3. **静止目标不产生事件。** 状态可保持，但丢失后难以重识别。
4. **Event 可能仍只是更密的运动测量。** RGB 光流、codec MV 或恒速模型可能达到相同效果。
5. **模板 QA 不一定需要 VLM。** 必须把结构化状态指标与 VLM QA 分开，不用 LLM 包装跟踪任务。
6. **零样本事件图失败不可解释。** Qwen 没见过这种表示，域差异会混淆传感器价值。
7. **SEVD 是 10 Hz 仿真标签。** 不能用它证明真实硬件的微秒响应、带宽或能耗。
8. **真实双传感器对齐。** 时间偏移、视差、外参和分辨率差异可能比算法误差更大。
9. **oracle 泄漏。** GT boxes、IDs、flow 和目标帧只能作为上界，必须与可部署方案分表。
10. **feature propagation 需要模型手术。** Qwen 视觉塔没有现成的增量 token 更新接口，投入明显高于结构化状态原型。

## 10. 阶段汇报格式

每份阶段报告固定包含：

1. 本阶段唯一假设；
2. 数据路径、序列级 split、manifest hash；
3. 查询时刻可见的输入，证明无未来信息；
4. 方法与 oracle/可部署边界；
5. 主要准确率和 P50/P95 延时表；
6. 三类代表性失败案例；
7. Gate 是否通过；
8. 只写一个下一步。

建议里程碑：

- **R0（本报告）**：论文、代码、数据与新颖性审计；
- **R1**：SEVD 对齐、时间戳和因果 manifest；
- **R2**：Frozen / 恒速 / GT-flow 的 staleness 曲线；
- **R3**：Event delta 与 RGB delta 的机制、准确率和延时；
- **R4**：Qwen QA 与最终 go/no-go。

## 11. 当前建议

**建议继续，但只批准一周的 kill-test 预算。**

本周的正确起点不是迁移 LiFR-Seg，也不是复现 EventFlash，而是先用 SEVD fixed 的 GT anchor 做一个极小的 object-state delta 实验。它能最快回答最重要的问题：在没有 ego-motion、没有检测误差、没有 VLM 域差异的情况下，Event 是否真的比 Frozen、恒速外推和 RGB optical flow 更好地维持当前状态。

只有 Gate B 通过，才接 Qwen3-VL-2B；只有 Gate C/D 通过，才投入 LiFR 式 feature propagation 或 EventFlash 式 token sparsification。

