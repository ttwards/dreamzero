# DreamZero 灵巧手后训练笔记

本文记录基于 `GEAR-Dreams/DreamZero-AgiBot` checkpoint 做灵巧手后训练时，当前代码中的数据表示、相机布局和数据加载约束。

## 先决结论

发布 checkpoint 的配置是：

```text
max_state_dim = 64
max_action_dim = 32
action_horizon = 48
num_action_per_block = 48
num_state_per_block = 1
num_frame_per_block = 2
num_frames = 33
frame_seqlen = 880
max_chunk_size = 4
```

来源：[DreamZero-AgiBot config.json](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot/resolve/main/config.json)。

当前仓库的 `scripts/train/agibot_training.sh` 却传入 `action_horizon=24` 与 `num_action_per_block=24`。权重可因参数形状不变而载入，但动作的时间分块和 RoPE 位置语义已经改变；若目标是最大化继承 AgiBot checkpoint，应先恢复 48-step 设置并相应改造 sampler。

## AgiBot 原始 state / action 表示

当前 `convert_agibot.py` 写出的原始每帧向量为：

| 向量 | 维度 | 顺序 |
|---|---:|---|
| `observation.state` | 20 | 左臂关节 7、右臂关节 7、左/右 effector 1+1、头部 2、腰 pitch/lift 1+1 |
| `action` | 22 | 与 state 对齐的前 20 维位置目标，外加 `robot_velocity` 2 维 |

转换器的 metadata 将这些原始字段均标为 `absolute: true`。训练配置启用 `relative_action=true` 后，对有同名 state 的位置类 action，在每个 chunk 内计算：

```text
relative_action[t + j] = absolute_action[t + j] - state[t]
```

这不是相邻帧差分；24 或 48 个未来 action 都相对于该 chunk 起点 state。`robot_velocity` 没有同名 state，保持原表示。所有字段随后使用 q01/q99 统计归一化到 `[-1, 1]`。

输入模型时：

```text
raw state [K, 20]       -> padded state  [K, 64]
raw action [K*H, 22]    -> padded action [K*H, 32]
```

其中 `H` 是一个 action chunk 的 horizon。action mask 仅让 loss 覆盖真实的 22 维；state mask 会生成，但 action head 直接消费补零后的 64 维 state。

注意：仓库的 AgiBot 脚本和 base YAML 注释称 state 为 32 维，但转换器实际构造的是 20 维。另外 YAML 使用 `waist_position`，转换器写的是 `waist_pitch` 与 `waist_lift`。这两处不一致必须在新数据配置中修正，不能照抄。

## 3. `z` 的两种可能含义

代码将机器人本体条件命名为 `state`，不叫 `z`。如果讨论中 `z` 指本体状态，应使用上节的 `[K, 64]` state tensor。

如果 `z` 指 Wan 的视频 VAE latent，则在保留原始 AgiBot 拼图尺寸时：

```text
video input: [B, 3, 33, 352, 640]
VAE latent z: [B, 16, 9, 44, 80]
DiT tokens per latent frame: 22 * 40 = 880
```

checkpoint 的 `in_dim=36` 是 16 通道的视频 latent 加 I2V 首帧条件的 20 通道。为直接复用 checkpoint，视觉拼图应继续输出 `352x640`，从而保持 `frame_seqlen=880`。

## 4. 原始相机布局

AgiBot 原始数据有三个相机：

```text
video.top_head
video.hand_left
video.hand_right
```

也就是 head camera 加左右两个手部（可视为 wrist/hand-mounted）相机。每路先裁剪、resize 为 `176x320`，随后拼为：

```text
top-left:     head
top-right:    right hand
bottom-left:  left hand
bottom-right: black
```

拼图结果为 `352x640`。当前实现位于 `DreamTransform._prepare_video`。

我们是 `head + chest`、没有 wrist camera 时：

- 不能让输入直接退化成单个 `176x320` 画面；这样每帧只有 220 DiT token，与 checkpoint 的 880 不兼容。
- 两相机可以保留 2x2 canvas：head 放左上，chest 放一个固定格，其余两格全黑；输出仍为 `352x640`。
- 最好为灵巧手实现显式的拼图布局，而不是依赖当前通用的两视角分支（该分支默认把两个画面放在左上和左下）。
- 新 embodiment 需要增加 collator 的文字描述；否则 `DefaultDataCollator` 不认识新 tag 会报错。文字也不应继续声称存在左右 wrist view。

## 5. batch、时间 chunk 与存储 chunk

这三个概念彼此独立。

### policy action chunk

checkpoint 的结构推导如下（`K=4` 是 `max_chunk_size=4` 的满长度情况）：

```text
33 sampled video frames
  -> 9 VAE latent frames
  -> initial conditioning frame + 4 blocks x 2 latent frames

state:   [s0, s1, s2, s3]
action:  [a0(H), a1(H), a2(H), a3(H)]
```

对发布 checkpoint，`H=48`，所以满样本 action tensor 是 `[B, 192, 32]`，state tensor 是 `[B, 4, 64]`。每个 block 由 48 个 action token 和 1 个 state token 条件化。

当前 sharded sampler 将 24、8 帧采样间隔和相对动作 chunk size 写死，因此与当前脚本配合时会生成 `[B, 96, 32]`（4×24）而非 checkpoint 原始的 192 步。若切回 48，应同时改动：action offsets、chunk anchor 间隔、relative-action chunk size，以及视频采样的时间步长，使 `33 video frames -> 9 latents -> 4 blocks` 与每块 48 action 对齐。

### DataLoader batch

训练脚本设 `per_device_train_batch_size=1`。这是合理的：一个 episode 在语言片段或结尾处可能只有较少的有效 block，导致不同样本的 `K` 不同；当前 collator 直接 `np.stack`，没有 sequence padding，混合不同 `K` 的样本会失败。全局 batch 是 `NUM_GPUS * per_device_train_batch_size`。

### LeRobot 存储 chunk 与 loader shard

`data/chunk-000/episode_000000.parquet` 中的 `chunk-000` 只是磁盘上的 episode 分桶；它不是 24/48-step action chunk。

训练时的 `ShardedLeRobotMixtureDataset` 还会按约 10k steps 生成内存 shard，并按 `dataset_shard_sampling_rate` 采样；这也不同于 policy action chunk。

## 6. Dataloader 兼容的格式

当前有两个训练后端：带 GEAR metadata 的 **LeRobot v2/v3**，以及本次新增的
**EgoVLA 原生 frame-wise WebDataset**。后者直接顺序读取 tar 中的
`image.jpg`、`chest_image.jpg`、`lowdim.npy` 与 `meta.json`，不必先转成
LeRobot；归一化 metadata 由 `scripts/data/compute_egovla_wds_metadata.py`
生成。其他 ROS bag、HDF5、RLDS/TFDS 或任意 CSV 仍需先转换。下列为
LeRobot v2 结构：

```text
dataset/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.<camera>/episode_000000.mp4
  meta/info.json
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/stats.json
  meta/modality.json
  meta/embodiment.json
  meta/relative_stats_dreamzero.json
```

v3 使用共享 Parquet/MP4；episode 与文件的对应关系由 `meta/episodes/*/*.parquet` 决定，而不是文件名：

```text
dataset/
  data/chunk-000/file-000.parquet
  videos/<camera>/chunk-000/file-000.mp4
  meta/info.json                 # codebase_version: v3.0
  meta/episodes/chunk-000/file-000.parquet
  meta/tasks.parquet
  meta/stats.json
  meta/modality.json             # DreamZero/GEAR 的字段映射
```

每帧 parquet 至少要包含：

- packed float state column，例如 `observation.state`；
- packed float action column，例如 `action`；
- 对应视频列/文件，例如 `observation.images.head`、`observation.images.chest`；
- 可映射到文本任务的 annotation/task column。

`meta/modality.json` 把 packed state/action 列切成命名子字段，并且 YAML 中的 `state.*`、`action.*`、`video.*`、`annotation.*` 必须逐字匹配。标准 LeRobot v2 数据可用 `scripts/data/convert_lerobot_to_gear.py` 补齐 GEAR metadata；原始 AgiBot HDF5 需先经 `scripts/data/convert_agibot.py` 转成 LeRobot。

详细格式和转换步骤见 `docs/DATASET_TO_GEAR_AND_TRAIN.md`。

## 7. 灵巧手后训练建议

1. 新建 embodiment，不复用 `agibot` tag。
2. 当前 EgoVLA 控制向量是 48D：`max_state_dim=64` 足够，`max_action_dim` 必须扩到 48。
3. 绝对 position target 使用 relative action；原本就是 delta、velocity、torque 的通道不要再次减当前 state。
4. 加载旧 32D checkpoint 时，action encoder/decoder 的前 32 维继承权重，新增 16 维保留随机初始化。
5. 保持 2x2 视觉 canvas 与 `frame_seqlen=880`；对于 head+chest，使用固定 black slot。
6. 优先决定时间策略：严格复用 checkpoint 的 48-step，或有意识地采用当前仓库的 24-step 重新适配。
7. 原生 WDS 定义见 `egovla_wds.py` 和 `egovla_wds_fingertips_relative.yaml`。


## 8. 双臂灵巧手 embodiment（当前定义）

`dual_arm_dexterous_hand` 直接采用 EgoVLA `lowdim.npy` 前 96 维。每只手的
15D hand 不是关节角，而是五个指尖的世界系 XYZ：

```text
thumb XYZ + index XYZ + middle XYZ + ring XYZ + pinky XYZ = 15/hand
```

state 和 action 均为 48D，原始顺序固定为：

```text
left_wrist_position       [0:3]
right_wrist_position      [3:6]
left_wrist_rotation_6d    [6:12]
right_wrist_rotation_6d   [12:18]
left_fingertip_position   [18:33]
right_fingertip_position  [33:48]
```

`lowdim[48:96]` 是同顺序的 next-step action，`lowdim[96:136]` 是 head/chest
相机标定。没有 head robot DOF；head 和 chest 仅作为视觉观测。

relative action 只对双腕 XYZ 和双手指尖 XYZ 减去 block anchor state；
rotation6d 保持绝对表示。模型使用 `max_state_dim=64`、`max_action_dim=48`。
