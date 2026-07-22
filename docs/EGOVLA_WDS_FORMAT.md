# EgoVLA WebDataset format

DreamZero 的原生 WDS loader 读取 frame-wise tar：一个 WebDataset sample
对应一个 30 Hz 控制 step，并包含同 key 的：

```text
<task>_ep<episode>_f<frame>.image.jpg
<task>_ep<episode>_f<frame>.chest_image.jpg
<task>_ep<episode>_f<frame>.lowdim.npy
<task>_ep<episode>_f<frame>.meta.json
```

depth 成员会被跳过。`lowdim.npy` 是 136D float32：

旧 finetune/dagger shard 把胸前相机命名为 `breast_image.jpg`；loader 会将它
映射到同一个 `video.chest`。只有两种字段都缺失时才补固定黑帧。

| Slice | 含义 |
|---|---|
| `[0:3]` | left wrist XYZ state |
| `[3:6]` | right wrist XYZ state |
| `[6:12]` | left wrist rotation6d state |
| `[12:18]` | right wrist rotation6d state |
| `[18:33]` | left five-fingertip XYZ state |
| `[33:48]` | right five-fingertip XYZ state |
| `[48:96]` | 同顺序的 next-step action |
| `[96:116]` | head extrinsic 16 + intrinsic 4 |
| `[116:136]` | chest extrinsic 16 + intrinsic 4 |

所以 `15/hand` 指五个指尖各自的 XYZ，不是 15 个关节，也不是 6 个
hand actuator。

## 24-step block

这里的 step 是 control frame。anchor 只有在 `anchor ... anchor+24` 连续时
才有效。每个 block 读取：

- anchor 时刻的一份 48D state；
- `anchor ... anchor+23` 的 24 份 48D action；
- offsets `0,3,6,9,12,15,18,21` 的八帧画面；
- `anchor+24` 的 terminal boundary 画面。

相邻 block 的 anchor 相隔 24 control steps。四个 block 的完整模型样本是
33 帧画面、4 份 state 和 96 份 action。双腕 XYZ 与双手指尖 XYZ action
相对各自 block 的 anchor state；rotation6d 保持绝对值。

## 10% 有效 step 与 shuffle

loader 先顺序读取 tar 并组成有效 anchor，再以 `keep_ratio=0.1` 对有效
anchor 做 Bernoulli 保留，最后进入有界 shuffle buffer。这样保留比例的
期望值是 10%，同时不需要随机 seek tar，也不会先解码被丢弃样本的 JPEG。
不同 rank/worker 使用不同且可复现的随机种子。

默认 buffer 是 256、初始填充 32。真实 shard 中每路 JPEG 约 103 KB，不能
使用 8192 这类会占用十几 GB 内存的 sample buffer。

## 归一化 metadata

训练前执行：

```bash
python scripts/data/compute_egovla_wds_metadata.py \
  --shards '/path/train/shard-*.tar' \
  --output /path/egovla_wds_metadata.json
```

脚本按同一个 24-step relative-action 定义计算 mean/std/min/max，并用有界
reservoir 估算 q01/q99。训练配置是
`groot/vla/configs/data/dreamzero/egovla_wds_fingertips_relative.yaml`。
