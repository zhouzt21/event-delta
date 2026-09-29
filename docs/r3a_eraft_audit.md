# R3a E-RAFT integration audit

**Audit date:** 2026-09-29  
**Scope:** integration audit only. No checkpoint/data download, training, E-RAFT vendoring, or change to the R2 oracle implementation or results.

## Executive decision

E-RAFT is a plausible first event-flow baseline, but it is **not yet integration-ready without a server smoke test**. The official DSEC checkpoint has the right 100 ms cadence and produces forward flow in pixel units, but the model consumes **two consecutive 100 ms event packets**, not one. For SEVD transition `RGB_i -> RGB_{i+1}`, the checkpoint-compatible pair is therefore `Event_i` (the preceding interval) and `Event_{i+1}` (the target interval). This is causal at `RGB_{i+1}`, because both packets end no later than the query time.

The safest first smoke is the DSEC checkpoint, 15-bin voxels, standard/zero-init inference, and spatial resize from SEVD 1280x960 to the checkpoint's DSEC 640x480 geometry. Native-resolution inference is mathematically supported by the fully convolutional model and needs no padding, but its all-pairs correlation volume is about 16 times the DSEC-resolution size and may exceed memory.

Do not reuse `sevd_flow_to_pixel_displacement()` for E-RAFT output. E-RAFT already returns pixel displacement; the SEVD background subtraction and half-image scaling apply only to encoded SEVD GT flow.

## 1. Audited sources and provenance

### E-RAFT

- Repository: <https://github.com/uzh-rpg/E-RAFT>
- Audited commit: [`c58ce0524ea0ebfa9849991caafb547f44fe9bfd`](https://github.com/uzh-rpg/E-RAFT/tree/c58ce0524ea0ebfa9849991caafb547f44fe9bfd), dated 2022-10-08.
- Local audit clone: created outside EventDelta under `/tmp`; no upstream files were copied or modified.
- License: MIT, copyright 2021 Robotics and Perception Group ([LICENSE](https://github.com/uzh-rpg/E-RAFT/blob/c58ce0524ea0ebfa9849991caafb547f44fe9bfd/LICENSE)). This permits integration subject to retaining the license notice.
- Paper: [E-RAFT: Dense Optical Flow from Event Cameras](https://arxiv.org/abs/2108.10552).
- The repository contains evaluation/inference code and checkpoint loading, but no training entry point or optimizer/loss implementation. Reproducing training requires work beyond this audit.

### LiFR-Seg

- Paper: [LiFR-Seg: Anytime High-Frame-Rate Segmentation via Event-Guided Propagation](https://arxiv.org/abs/2603.21115).
- Linked repository: <https://github.com/Candy-Crusher/LiFR-Seg>.
- On the audit date, a fresh clone reports an empty repository: no commit/HEAD, source, environment, checkpoint, or license. The LiFR observations below are therefore paper-level facts, not source-verified implementation facts.

## 2. Environment, checkpoint, and model inventory

The upstream [`environment.yml`](https://github.com/uzh-rpg/E-RAFT/blob/c58ce0524ea0ebfa9849991caafb547f44fe9bfd/environment.yml) pins an old stack:

| Component | Upstream pin |
|---|---:|
| Python | 3.7.10 |
| PyTorch | 1.7.0 |
| torchvision | 0.8.1 |
| CUDA toolkit | 10.2.89 |
| cuDNN | 7.6.5 (through the PyTorch build) |
| NumPy | 1.19.1 |
| numba | 0.53.1 |
| h5py | 3.1.0 |
| OpenCV | 4.5.1 |
| imageio | 2.9.0 |
| SciPy | 1.5.2 |
| pandas | 1.2.3 |
| Pillow | 8.0.0 |
| matplotlib | 3.4.1 |

Other upstream dependencies include scikit-image, PyTables, GitPython, tqdm, and the usual visualization packages. Because this stack conflicts with the modern Transformers/Qwen environment, use a **separate conda environment and process boundary** for the first E-RAFT smoke. Do not downgrade the R2 environment.

Official checkpoints listed by the upstream README and consumed with `torch.load(...); checkpoint['model']` are:

| Training domain | URL | Expected filename | HTTP size observed by HEAD |
|---|---|---|---:|
| DSEC, 10 Hz / 100 ms | <https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar> | `dsec.tar` | 64,171,800 bytes |
| MVSEC, 20 Hz | <https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/mvsec_20.tar> | `mvsec_20.tar` | 64,167,512 bytes |
| MVSEC, 45 Hz | <https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/mvsec_45.tar> | `mvsec_45.tar` | 63,429,336 bytes |

The first SEVD smoke should use **`dsec.tar`**: it is the only official checkpoint trained with the same nominal 100 ms flow interval and uses 15 input bins. The repository provides no official SHA-256 digest; record the locally computed digest after download.

## 3. Exact E-RAFT input representation

### 3.1 Pairing and tensor shape

E-RAFT predicts `F_i->i+1` from two consecutive event sequences:

```text
old = events in [t_{i-1}, t_i]
new = events in [t_i, t_{i+1}]
flow = F_{t_i -> t_{i+1}}
```

This is explicit in the paper and in [`loader_dsec.py`](https://github.com/uzh-rpg/E-RAFT/blob/c58ce0524ea0ebfa9849991caafb547f44fe9bfd/loader/loader_dsec.py#L295). Each packet becomes one voxel tensor `[C,H,W]`; `DataLoader(batch_size=1)` makes the network inputs:

```text
event_volume_old: [N, C, H, W], float32
event_volume_new: [N, C, H, W], float32
N = 1
C = 15 for dsec.tar
```

The feature encoder sees both tensors with shared weights. The context encoder sees `event_volume_new` only. The stock DSEC loader hardcodes `H=480`, `W=640`, a 100 ms window, and 15 bins.

Therefore, **one isolated 100 ms packet cannot be passed to the official model/checkpoint contract**. Standard mode removes recurrent flow initialization but still requires both `old` and `new` packets. Duplicating the packet or supplying zeros would be an unvalidated distribution shift, not a legitimate independent inference mode.

### 3.2 Voxel construction

For event `j`, E-RAFT uses:

```text
t*_j = (C - 1) * (t_j - t_first) / (t_last - t_first)
value_j = 2 * p_j - 1
```

It splats `value_j` trilinearly across adjacent `x`, `y`, and time-bin locations, then standardizes only nonzero voxel cells using their per-packet mean and standard deviation. See [`VoxelGrid.convert`](https://github.com/uzh-rpg/E-RAFT/blob/c58ce0524ea0ebfa9849991caafb547f44fe9bfd/utils/dsec_utils.py#L19).

Important details:

- Timestamp units cancel during normalization. For SEVD, subtract the first `int64` ns timestamp before conversion to floating point to avoid precision loss.
- The official implementation normalizes by the **first and last event**, not the nominal packet boundaries. Empty packets are not handled, and a single-timestamp packet needs an explicit safe path. Both cases require server statistics before the adapter is finalized.
- DSEC polarity is represented as `0/1` and mapped to `-1/+1`. SEVD `pol=False/True` should map to `-1/+1`, subject to a one-sample polarity sanity check against its RGB intensity change.
- `x` is horizontal/rightward and indexes width; `y` is vertical/downward and indexes height. Voxel storage is `[bin,y,x]`. No transpose is used.
- DSEC events are rectified before voxelization. SEVD is simulated and R1.2 found a shared camera geometry; the adapter should apply no DSEC rectify map and must not introduce a flip.

### 3.3 Resolution and padding

The official DSEC loader uses 640x480. The model's [`ImagePadder`](https://github.com/uzh-rpg/E-RAFT/blob/c58ce0524ea0ebfa9849991caafb547f44fe9bfd/utils/image_utils.py#L85) zero-pads the **left and top** to multiples of 32 and removes that padding from every full-resolution prediction.

Both candidate SEVD paths need no padding:

| Path | Input `[H,W]` | Multiple of 32 | Consequence |
|---|---:|---:|---|
| DSEC-size smoke | `[480,640]` | yes | Matches checkpoint geometry and is recommended first. |
| Native SEVD | `[960,1280]` | yes | No padding, but about 16x the all-pairs correlation elements of 640x480. GPU feasibility is unverified. |

For the DSEC-size path, scale coordinates before voxelization:

```text
x_model = x_sevd * 640 / 1280
y_model = y_sevd * 480 / 960
```

Use spatial bilinear splatting, not integer truncation, so subpixel coordinates contribute to adjacent pixels. This follows the DSEC rectified-coordinate voxelizer.

## 4. Output convention

[`ERAFT.forward`](https://github.com/uzh-rpg/E-RAFT/blob/c58ce0524ea0ebfa9849991caafb547f44fe9bfd/model/eraft.py#L88) returns:

```text
flow_low_res: [N,2,H_pad/8,W_pad/8]
flow_predictions: list of `iters` tensors [N,2,H,W] (default `iters=12`)
final flow: flow_predictions[-1]
channel 0 = dx, channel 1 = dy
```

The final result is unpadded. Values are direct displacement in pixels of the model input, not normalized coordinates and not velocity. The model defines flow as `coords1 - coords0`; the paper and loaders label it `F_{t_i -> t_{i+1}}`. It is therefore **source-to-target forward flow**.

For 640x480 inference, restore a native SEVD pixel field as follows:

1. Convert `[1,2,480,640]` to `[480,640,2]`.
2. Bilinearly resize the two components to `[960,1280,2]`.
3. Multiply `dx` by `1280/640 = 2` and `dy` by `960/480 = 2`.
4. The result is native SEVD source-to-target pixel displacement `[960,1280,2]`.

No sign flip, `(-1/1024,-1/1024)` subtraction, or half-image scale is applied to E-RAFT output. Those operations are only for decoding `flow_i.npz` as GT.

## 5. SEVD-to-E-RAFT mapping

The confirmed SEVD transition is:

```text
source RGB       RGB_i
GT flow          flow_i
target events    Event_{i+1}
target RGB       RGB_{i+1}
```

The E-RAFT adapter adds the preceding event packet required by its architecture:

| E-RAFT field | SEVD source | Conversion |
|---|---|---|
| `event_volume_old` | `Event_i.dvs_events` | `(x,y,t,pol)` from interval `i-1 -> i`; resize coordinates if using 640x480; 15-bin signed voxel. |
| `event_volume_new` | `Event_{i+1}.dvs_events` | Interval `i -> i+1`; same conversion. |
| timestamp | structured `int64 t`, approximately ns | subtract packet first timestamp in integer space; normalize to `[0,14]` following upstream. |
| polarity | structured bool `pol` | `False -> -1`, `True -> +1`; verify semantics on server. |
| spatial coordinates | `uint16 x,y`, 1280x960 | identity `x=column`, `y=row`; no transpose/flip/rectification. |
| prediction | final `[1,2,H,W]` | transpose to `[H,W,2]`, then restore native resolution/vector units if resized. |
| supervision only | `flow_i.flow` | decode with the existing SEVD background and half-image-size formula to native pixel displacement. Never feed GT to the Event method. |

The first or last dataset frame must be skipped whenever either event packet or the matching GT transition is absent. For R2b's sampled windows, all event IDs from `anchor` through `anchor+9` are needed; `Event_anchor` is historical context and `Event_{anchor+k}` predicts transition `anchor+k-1 -> anchor+k`.

## 6. Warm-start causality

Warm start reuses only the preceding predicted low-resolution flow. The upstream tester forward-warps that estimate with average splatting and supplies it as `flow_init` for the next inference. It resets on sequence discontinuities. It uses no future events or GT and is therefore causal **if samples are processed chronologically and state is reset at every independent R2b window**.

Two caveats remain:

1. Warm start is a flow prior, not a recurrent hidden state; the declared `net_init` is not passed into the model.
2. The DSEC checkpoint is used by both standard and warm-start configs, but the paper reports that warm-start gains are reliable when it was included in fine-tuning. Adding warm start only at inference was not consistently beneficial.

Recommendation: validate convention with `flow_init=None` first. If that passes, compare (a) standard zero-init for every transition and (b) causal warm start reset at each anchor. Pre-register one as the formal R3a primary result before inspecting all 60-window metrics. Do not warm-start across independently sampled windows.

## 7. Contract with `flow_warp.py`

Proposed adapter result (interface draft, not implemented here):

```text
EraftFlowResult:
  flow_pixel: float32 [H_original,W_original,2]  # dx,dy; source -> target
  valid: bool [H_original,W_original]
  metadata:
    source_frame_id
    target_frame_id
    old_event_frame_id
    new_event_frame_id
    checkpoint_sha256
    eraft_commit
    bins
    model_input_hw
    original_hw
    resize_scale_xy
    padding_ltrb
    warm_start
    iterations
```

Integration into the existing propagation path must be:

```text
event_flow_native [960,1280,2] pixel displacement
  -> resize_flow_to_grid(event_flow_native, feature_hw)
  -> pixel_to_feature_displacement(..., original_hw=(960,1280), feature_hw=(30,40))
  -> forward_splat_torch(feature, displacement_grid, confidence)
```

Do **not** call `sevd_flow_to_feature_displacement()` because that function decodes the special SEVD GT encoding. Stale, Event-flow, and GT-flow features must be compared on a recorded common valid mask; Event splatting holes must not be filled with dense target features.

## 8. Domain mismatch and risks

1. **Sensor/domain:** DSEC uses a real Prophesee camera and road scenes; SEVD uses CARLA-generated events. Contrast thresholds, refractory behavior, noise, hot pixels, and event density differ.
2. **Resolution:** checkpoint inference is documented at 640x480; SEVD is 1280x960. Half-resolution inference preserves aspect ratio but merges events and halves pixel motion.
3. **Motion distribution:** DSEC contains strong ego-motion; SEVD fixed recordings have zero ego-motion and potentially sparse local object motion. Dense background flow may be poorly calibrated.
4. **Timestamp normalization:** upstream stretches first-to-last event time to the full bin range, even if activity occupies only part of a nominal 100 ms packet.
5. **Empty/sparse packets:** upstream voxel code assumes at least two timestamps and does not define the checkpoint-compatible behavior for empty packets.
6. **Polarity semantics:** the bool-to-sign mapping is conventional but has not been verified from an actual SEVD packet against intensity changes in this audit.
7. **Memory:** native 960x1280 all-pairs correlation may be impractical. Measure peak memory before any 60-window run.
8. **Dense flow in event-silent areas:** E-RAFT predicts dense flow, but fixed-camera background and newly appearing/occluded objects may be unreliable. Metrics must separate event-active, motion, static, entry, and occlusion regions where possible.
9. **Legacy runtime:** the pinned CUDA 10.2/PyTorch 1.7 stack may not run on the server's current driver or GPU architecture. A compatibility port could change numerics and must be recorded separately.

## 9. LiFR-Seg motion-branch audit

The paper's motion branch is not a drop-in copy of official E-RAFT:

- It creates two signed event voxels with **4 bins**, not E-RAFT's DSEC 15 bins.
- A RAFT-like network predicts forward motion `[2,H,W]`, initialized from zero.
- A separate ScoreNet encodes the current event voxel and predicted flow, concatenates the embeddings, and regresses a one-channel log-precision/confidence map.
- Multi-scale semantic features are propagated with confidence-weighted Softmax Splatting, followed by a two-convolution RefineNet and temporal memory attention.
- The full system is trained end-to-end with OHEM cross-entropy at future RGB segmentation timestamps. An intermediate feature is warped a second time to a supervised RGB timestamp. The paper reports AdamW, learning rate `1e-4`, weight decay `5e-3`, 10-epoch warm-up, polynomial decay, 200 epochs, batch size 4, and two RTX 4090 GPUs.
- The paper discusses DSEC-pretrained flow and an E-RAFT/E-RAFT-Lite ablation, but the empty repository prevents verification of exact module classes, weight initialization, preprocessing, losses on flow/confidence, or checkpoint availability.
- Online reuse of the previous flow as initialization is described as future work in the paper; it is not evidence of a released LiFR streaming implementation.

### Work estimate comparison

| Route | Required work | Audit recommendation |
|---|---|---|
| Direct official E-RAFT | Thin SEVD voxel adapter, legacy environment, checkpoint load, resize/unit conversion, one-sample convention validation, then R2b-compatible runner. No training for the first result. | **Proceed first**, gated by the single-sample smoke. |
| LiFR motion branch | Reconstruct unreleased 4-bin RAFT-like network and ScoreNet from paper, define/obtain weights, implement confidence Softmax Splatting, RefineNet and possibly memory, then reproduce end-to-end segmentation training before adapting it to Qwen features. | **Do not transplant now.** It is a separate research implementation/training project and violates this audit's no-training scope. |

LiFR-Seg is useful as design evidence for confidence-weighted splatting, not as the R3a integration base.

## 10. Planned R3a validation outputs (not implemented)

Keep the exact R2b recordings, 60 total windows (30 each), offsets 1..9, Qwen post-merger feature stage, motion threshold, causal recurrent feature splatting, and window bootstrap. Add, without overwriting R2 outputs:

### Flow accuracy

- Pixel EPE against decoded SEVD GT flow.
- Angular error: `acos((1 + pred·gt) / sqrt((1+||pred||^2)(1+||gt||^2)))`.
- DSEC-style `1PE`, `2PE`, and `3PE` rates.
- KITTI-style outlier rate: `EPE > 3 px` and `EPE / max(||GT||, eps) > 0.05`.
- Report all finite pixels, GT-motion pixels, event-active pixels, and their intersection separately.

### Feature propagation

- Stale anchor feature.
- Event-flow propagated feature.
- Existing GT-flow oracle propagated feature, clearly labeled oracle and left numerically unchanged in its original R2b report.
- Dense target RGB feature, evaluation only.
- Stale/Event/GT comparisons on the same common-valid tokens, plus each method's native coverage.
- Oracle retention for each distance metric when the denominator is positive:

```text
retention = (distance_stale - distance_event) /
            (distance_stale - distance_gt_oracle)
```

Report values over 1 and below 0 without clipping.

### Efficiency and aggregation

- Event NPZ decode + voxelization latency.
- E-RAFT network latency.
- Event-flow resize/unit-conversion latency.
- Feature warp latency, separately preserving the R2 efficiency definition.
- P50/P95 after warm-up, peak CUDA memory, input/output shapes, and checkpoint/environment identifiers.
- Per-offset rows, per-window macro, motion-token weighted micro, fixed-3/fixed-4, activity groups, and 2000-replicate window bootstrap with seed 20260929.

Proposed formal artifact layout:

```text
artifacts/r3a_eraft/
  config.json
  environment.json
  event_flow_per_transition.csv
  feature_per_offset_metrics.csv
  window_summary.csv
  aggregate_metrics.csv
  bootstrap.json
  efficiency.json
  summary.md
  flow_qualitative_fixed-3_high.png
  flow_qualitative_fixed-4_high.png
  feature_qualitative_fixed-3_high.png
  feature_qualitative_fixed-4_high.png
```

`environment.json` must include both the EventDelta commit and E-RAFT commit/checkpoint SHA-256. The existing R2b artifact directory remains immutable.

The 60-window run is blocked until the adapter passes direction, scale, polarity, shape, and memory checks on one sample.

## 11. Minimum server smoke and commands

### 11.1 Clone and pin outside EventDelta

```bash
export ERAFT_ROOT="$HOME/external/E-RAFT"
git clone https://github.com/uzh-rpg/E-RAFT.git "$ERAFT_ROOT"
git -C "$ERAFT_ROOT" checkout --detach c58ce0524ea0ebfa9849991caafb547f44fe9bfd
git -C "$ERAFT_ROOT" rev-parse HEAD
```

### 11.2 Create an isolated environment

```bash
conda env create -n eventdelta-eraft-r3a -f "$ERAFT_ROOT/environment.yml"
conda activate eventdelta-eraft-r3a
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If the pinned environment cannot solve or cannot support the installed GPU, stop and record the solver/driver error. Do not silently run the E-RAFT checkpoint under the Qwen environment; a modern compatibility environment must be a separately documented experiment.

### 11.3 Download exactly one checkpoint

```bash
mkdir -p "$ERAFT_ROOT/checkpoints"
curl -L --fail \
  https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar \
  -o "$ERAFT_ROOT/checkpoints/dsec.tar"
test "$(stat -c %s "$ERAFT_ROOT/checkpoints/dsec.tar")" -eq 64171800
sha256sum "$ERAFT_ROOT/checkpoints/dsec.tar" | tee "$ERAFT_ROOT/checkpoints/dsec.tar.sha256"
python -c "import torch; p=torch.load('$ERAFT_ROOT/checkpoints/dsec.tar', map_location='cpu'); print(p.keys()); print(len(p['model']))"
```

This is the only large download proposed for the next step. It was not performed during this audit.

### 11.4 Single-transition SEVD smoke contract

The minimum valid sample is **two adjacent event NPZs plus one matching GT-flow NPZ**:

```text
Event_i       historical packet i-1 -> i       (old voxel)
Event_{i+1}   target packet i -> i+1           (new voxel)
flow_i        GT only for direction/scale check
```

One event NPZ plus `flow_i` is insufficient for official E-RAFT. RGB_i/RGB_{i+1} are optional for flow-only numerical validation and required only for a qualitative overlay or feature-propagation smoke.

The intended next-step adapter CLI is frozen here as an interface draft; `tools/smoke_eraft_sevd.py` is **not implemented in this audit commit**:

```bash
conda activate eventdelta-eraft-r3a
CUDA_VISIBLE_DEVICES=0 python tools/smoke_eraft_sevd.py \
  --root "$SEVD_ROOT" \
  --recording fixed-3 \
  --source-frame-id 4592 \
  --eraft-root "$ERAFT_ROOT" \
  --checkpoint "$ERAFT_ROOT/checkpoints/dsec.tar" \
  --bins 15 \
  --input-height 480 \
  --input-width 640 \
  --iterations 12 \
  --mode standard \
  --device cuda \
  --output-dir artifacts/r3a_eraft_smoke
```

Before running, choose a source frame for which `Event_i`, `Event_{i+1}`, and `flow_i` all exist and contain finite/nonempty data. The implementation must print resolved paths and tensor shapes before inference and must fail rather than invent a packet when an input is missing.

## 12. Assumptions still requiring server verification

1. `dsec.tar` loads under the pinned environment and contains a compatible `model` state dict.
2. The server driver/GPU supports the PyTorch 1.7 + CUDA 10.2 environment; otherwise a documented compatibility port is needed.
3. `False -> -1`, `True -> +1` is the correct physical SEVD polarity interpretation.
4. Selected SEVD packets are nonempty and have at least two distinct timestamps; quantify empty/single-time incidence across the 60 windows.
5. First/last-event timestamp normalization is acceptable for SEVD, versus fixed packet-boundary normalization.
6. Half-resolution coordinate splatting exactly preserves expected motion direction and the restored output has the correct x/y scale.
7. Standard inference predicts source-to-target flow empirically: object-center displacement and GT EPE must improve for `+flow`, not `-flow`.
8. 640x480 inference is accurate enough; native 960x1280 memory, latency, and accuracy remain unknown.
9. E-RAFT dense background bias on fixed-camera SEVD is small enough not to corrupt feature propagation.
10. Standard versus within-window causal warm start must be selected before the formal 60-window result.
11. The upstream checkpoint license/distribution terms beyond the repository MIT license are not stated separately; retain provenance and do not redistribute the checkpoint.
12. LiFR-Seg has no auditable public implementation/checkpoint/license at the recorded URL; its motion branch cannot currently be reproduced from source.
