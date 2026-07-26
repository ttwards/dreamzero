# DreamZero 14B VAE batch 与区域编译分析报告

日期：2026-07-27  
硬件：单机 8×NVIDIA A800-SXM4-80GB  
软件：PyTorch 2.8.0+cu128、DeepSpeed 0.18.4  
训练入口：`scripts/train/egosteer_lerobot_training_volcano_cloud.sh`

## 1. 结论

最终采用的方案是：

1. 把 target video 和 first-frame condition video 沿 batch 维拼成一次 VAE encode；
2. 分别编译 Wan repeated block、CLIP visual 和 VAE encode；
3. 不编译完整 action head/VLA 大图；
4. 冻结 VAE 和 CLIP 保持为每个 rank 的本地 BF16 副本，避免它们进入 ZeRO-3 参数 materialization 路径；
5. Wan block 仍由 ZeRO-3 管理，参数 hook 位于每个 compiled block 的边界外。

最终 8 卡 10-step 结果：

| 指标 | eager baseline | VAE+CLIP leaf compile | Wan blocks+VAE+CLIP | 相对 eager |
|---|---:|---:|---:|---:|
| steady step | 32.1505 s | 31.3680 s | **23.9879 s** | **-25.39%** |
| global throughput | 0.49766 sample/s | 0.51007 sample/s | **0.66700 sample/s** | **+34.03%** |
| per-GPU throughput | 0.06221 | 0.06376 | **0.08338 sample/(s·GPU)** | **+34.03%** |
| runtime MFU | 31.43% | 32.22% | **42.13%** | **+10.70 pp** |
| model forward | 9.0513 s | 8.3510 s | **6.9146 s** | **-23.61%** |
| max allocated | 57.07 GiB | 57.95 GiB | **53.85 GiB** | -3.22 GiB |
| max reserved | 65.29 GiB | 66.51 GiB | 69.25 GiB | +3.96 GiB |

相对已有的 `wan_blocks` 观测值 25.6 s、39.5% MFU、0.078 sample/(s·GPU)，最终方案 step 快 6.30%，MFU 提高 2.63 pp，单卡吞吐提高 6.89%。

最终 run 完成 10/10 step，`RUN_EXIT_CODE=0`；编译诊断为 3 unique graphs、3952 captured calls、0 graph break、0 recompile、0 guard failure。8 个 rank 均退出，GPU 已完全释放。

## 2. VAE video 与 condition 的单批计算

实现位于：

- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:641`
- `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:658`
- `groot/vla/model/dreamzero/modules/wan_video_vae.py:1208`

输入 `input_video` 为 `[B,C,T,H,W]`。condition 的构造严格为：

```python
condition_video = torch.cat(
    [
        input_video[:, :, :1],
        torch.zeros_like(input_video[:, :, 1:]),
    ],
    dim=2,
)
```

所以：

- 第 0 帧保持原值；
- 只对第 1 到第 `T-1` 帧补零；
- 不会在首帧前多 pad 一帧，也不会把首帧清零；
- target 和 condition 拼成 `[2B,C,T,H,W]` 后只调用一次 VAE；
- encode 后按 batch size split 回 target latent 和 condition latent；
- 14B 路径仍输出 4-channel mask + 16-channel VAE latent，即原来的 20-channel condition contract。

训练配置 `tiled: false`，因此 `WanVideoVAE.encode()` 会把整个 `[2B,...]` 直接传给 underlying causal VAE，而不是 Python 循环成多个单样本调用。

服务器测试覆盖：

- 首帧保留、其余帧为零；
- non-tiled 路径只发生一次真实 batch encode；
- CLIP 只在显式 compile scope 下从 ZeRO module tree 摘出。

三个测试均通过。

## 3. Compile 与 DeepSpeed ZeRO 的兼容方式

服务器上的 DeepSpeed 0.18.4 已提供 `DeepSpeedEngine.compile()`，其正常生命周期是先 `deepspeed.initialize(...)` 构造 ZeRO engine，再在 engine 上 compile，而不是先编译一个裸模型、再事后给它添加分片。

本任务没有使用 whole-engine DeepCompile，原因是：

- 用户明确不需要大图；
- 完整 VLA 会把数据预处理、冻结 encoder、Wan、checkpoint 和通信边界混在一起；
- 当前 hpZ 配置需要保留 node-local 8-rank 参数分组；
- 区域编译已经得到更小冷启动和更好的可诊断性。

实际区域：

- 40 个 `action_head.model.blocks[i].forward`；
- 1 个 `action_head.image_encoder.model.visual.forward`；
- 1 个 `action_head.vae.model.encode`。

mock 目标枚举验证得到 42 个 callable。由于 40 个 Wan block 共享相同 Python code object 和固定 shape，Dynamo 复用同一个 compiled graph，所以运行时诊断只有 3 unique graphs，而不是 42 张不同图。

冻结 VAE 一直在 ZeRO tree 外。CLIP 在 `clip`、`vae_clip` 和 `wan_blocks_vae_clip` scope 下也从 `_modules` 摘出，随后每个 rank 懒加载为本地 BF16 副本。Wan blocks 保持注册和 ZeRO 分片，DeepSpeed hook 仍在 block `__call__` 边界，compiled callable 只覆盖 block 内部 compute。

## 4. 三轮 profile

### 4.1 产物

Eager：

```text
/efs-exp/agent-workspace/xuwenxi/outputs/vae_batch_baseline_cloud_34a807d_10step/
```

VAE+CLIP leaf compile：

```text
/efs-exp/agent-workspace/xuwenxi/outputs/vae_clip_detached_compile_cloud_78c86d8_10step/
```

Wan blocks+VAE+CLIP：

```text
/efs-exp/agent-workspace/xuwenxi/outputs/wan_blocks_vae_clip_cloud_e79547c_10step/
```

每个成功 run 的 `profiling/rank_0/` 均包含 `summary.json`、Chrome trace、`top_cpu_ops.txt` 和 `top_cuda_ops.txt`。

### 4.2 Profile MFU

| 指标 | eager | VAE+CLIP | combined |
|---|---:|---:|---:|
| profiled FLOPs | 6.2292 PFLOP | 6.2226 PFLOP | 6.2216 PFLOP |
| active wall | 65.3644 s | 63.2252 s | **51.7859 s** |
| profiled TFLOP/s/GPU | 95.300 | 98.420 | **120.141** |
| profile MFU @ 312 TFLOP/s | 30.55% | 31.54% | **38.51%** |

Runtime MFU 比 profile MFU 高，是因为 profiler 的 active step 有 CUPTI/stack/event 开销。最终 hot runtime 为 23.963 s training span，而 profile active wall 为 25.893 s/step。

Runtime MFU 的估算为：

```text
F_step = 6 × 16.484390768B params × 15,940 tokens/sample × global batch 16
       = 25.2251 PFLOP/optimizer-step

Peak_node = 312 TFLOP/s × 8 = 2.496 PFLOP/s
MFU_cuda  = 25.2251 / (23.963 × 2.496) = 42.17%
```

日志中的 wall-time MFU 为 42.13%，CUDA-span MFU 为 42.17%。MFU 是 6ND 训练模型估算，不是硬件计数器给出的精确 achieved-FLOPs 比例；profile MFU 则基于 profiler 可统计的算子 FLOPs。

### 4.3 VAE

在两个 active optimizer step、每 step 两个 microbatch 的窗口内，VAE 有 4 个逻辑调用：

| 路径 | GPU annotation span | 每逻辑调用 |
|---|---:|---:|
| eager batch VAE | 3.9554 s | 988.9 ms |
| compiled VAE | 2.9289 s | 732.2 ms |

下降约 25.95%。这两个 profile 都已经使用联合 batch，所以该对比中的增量主要来自 VAE compile 和稳定 shape 的 kernel 融合；联合 batch 相对修改前双调用路径的额外收益没有单独 profile，不能混入这 25.95%。最终结构同时具备：

- target/condition 合成一次 batch encode；
- shape 稳定；
- VAE pointwise/reduction 融合；
- 避免两个独立 causal-cache encode 调用。

### 4.4 CLIP 为什么看起来快了二十多倍

旧 `summary.json` 的 `calls` 字段有口径缺陷：`key_averages()` 同时给出 CPU user annotation 和每条 CUDA stream 的 GPU annotation，原代码用同名 key 覆盖。baseline 不是执行了 8 次 CLIP；真实逻辑调用一直是 4 次：2 active optimizer steps × 2 microbatch。

Trace 证据：

| 路径 | CPU calls | GPU stream annotations | critical GPU span/call |
|---|---:|---:|---:|
| baseline ZeRO CLIP | 4 | 8（compute+comm） | 196.42 ms |
| detached+compiled CLIP | 4 | 4（compute only） | 16.76 ms |
| final combined | 4 | 4 | 16.79 ms |

因此可辩护的 CLIP 加速约为 **11.7×**，不是直接用 1.5518/0.0671 得出的约 23×。

主要原因不是 compile 单独贡献：

- baseline 每次 CLIP forward 有 38 个 NCCL all-gather kernel；
- detached 路径的 CLIP 区间内 all-gather 为 0；
- baseline compute stream 需要等待 ZeRO 参数 materialization/comm stream；
- 移除 ZeRO 通信并 compile 后，GPU kernel events 从约 740/call 降到约 327/call，并融合 layer norm、GELU、pad/cat 等 pointwise/reduction；
- 非 NCCL active kernel 时间约从 8.22 ms/call 降到 4.51 ms/call，纯 compute 部分约 1.82×。

所以 11.7× 是“冻结 CLIP 从 ZeRO 复制出来 + compile”合计，不能全部归功于 Inductor。端到端 VAE+CLIP-only run 只从 32.150 s 降到 31.368 s，即 2.49%，与两条冻结路径每 optimizer step 合计节省约 0.86 s 相符。

训练 loss 0.8039 与 0.8038 接近，同一 CLIP 对象、权重、resize/normalize/BF16/31-block 路径均保留；但没有保存固定输入的 eager/compiled 输出做逐元素误差对照，所以证据是训练级一致，不能宣称 bitwise equivalent。

## 5. Compiled block 为什么不是一个 CUDA 大算子

`fullgraph=True` 的含义是：传给 `torch.compile` 的 Python callable 必须捕获成一张无 graph break 的 FX/AOT graph。它不承诺整个 graph 生成一个 CUDA kernel。

最终 trace 中：

- `Torch-Compiled Region: 0/0`：VAE，4 次；
- `Torch-Compiled Region: 1/0`：CLIP，4 次；
- `Torch-Compiled Region: 2/0`：Wan block，320 次；
- `CompiledFunctionBackward`：160 次。

Wan 的 320 次可以由 profile 窗口精确解释：

```text
40 blocks × 4 microbatch × (正常 forward + gradient-checkpoint recompute) = 320
```

其内部仍会看到很多 CUDA kernel，原因包括：

- GEMM 继续调用 cuBLAS/cuBLASLt 或选中的 template；
- FlashAttention forward/backward 是独立库 kernel；
- 卷积继续调用 cuDNN；
- ZeRO all-gather/reduce-scatter 在 compiled block 边界外，继续是 NCCL kernel；
- Inductor 只把适合融合的 pointwise/reduction 生成 `triton_*` kernel；
- 40 个 block 是串行依赖，不可能整体合成一个单核。

所以 profiler 中“一个 compiled region 包含多个 CUDA kernel”是正确结果，不代表 graph break。判断编译是否完整应看：

- unique graph 数；
- graph breaks/recompiles/guard failures；
- `Torch-Compiled Region` 边界；
- kernel launch 数和 CPU launch gap；
- 是否出现 eager fallback。

本次 combined profile 的 launch 类事件约从 eager 的 195,232 降到 87,100，下降约 55.4%；trace 从约 1.37 GiB 降到 610 MB。compile 确实压缩了调度和大量 eager 细粒度 op，只是没有、也不应该把整个 Transformer block 变成一个 CUDA kernel。

## 6. 剩余瓶颈与取舍

### 6.1 通信

Combined active window 的 comm stream 累计：

- all-gather：5.811 s；
- reduce-scatter：5.031 s；
- `record_param_comms`：10.842 s、4712 annotations。

这些 stream 时间与 compute 有重叠，不能直接加到 51.786 s wall time。当前配置已经使用 hpZ partition size 8、overlap communication、200M reduce/all-gather bucket 和 Wan block leaf boundary。继续通过长期驻留更多 14B 参数减少 all-gather 会快速吃掉剩余显存，不是低风险优化。

### 6.2 Data loader

最终 steady step 的 `time/data_wait_s=0.000175`，占比约 0.00073%。数据管线不是稳态瓶颈，不应继续调 worker/prefetch 来追 MFU。

### 6.3 Gradient checkpointing

Trace 明确显示 40 blocks 被 recompute，关闭 checkpointing 理论上能减少计算。但当前 reserved 已到 69.25 GiB，14B video token activation 很大；仅剩余约 10.75 GiB reservation headroom。全关 checkpointing 有明显 OOM 风险，选择性 checkpoint 又会扩大代码和验证面，按“不要太别扭”的要求不做。

### 6.4 T5

训练数据使用 cached text embeddings，profile 中没有 T5 热路径。编译或复制 T5 只会增加显存，没有 steady-state 收益，不做。

### 6.5 DeepSpeed 升级

当前问题不是 DeepSpeed compile correctness bug：最终 10 step 已做到 0 graph break/recompile/error。DeepSpeed 0.18.4 + PyTorch 2.8 已能运行当前区域方案，而该方案没有调用 whole-engine DeepCompile。升级 0.19.x 会同时改变 ZeRO hook/partition 行为，缺少收益依据，因此不污染当前已验证 runtime。

### 6.6 Max autotune

额外测试了 `max-autotune-no-cudagraphs`，避免 CUDA Graph 与 ZeRO buffer 的组合风险。结果在 0/6 step 失败：

- 约 6 分钟实际 autotune；
- 328 个 candidate benchmark 事件；
- cache 峰值 3.3 GiB、88,375 个文件；
- `/tmp` 所在 20G overlay 写满，触发 `OSError: [Errno 28] No space left on device`；
- 未进入 hot step，因此没有性能收益证据。

该模式冷启动和 cache/inode 成本明显，不采用。失败 run 的专用 3.3 GiB cache 已清理，根盘从 100% 恢复到 84%、可用 3.3 GiB。成功的 default-mode cache 保留。

不尝试 `reduce-overhead` 或 `max-autotune`，因为 PyTorch 2.8 中这两个 mode 会启用 CUDA Graph；ZeRO 参数 hook 和动态 materialization buffer 使收益/正确性风险不对称。

## 7. 推荐生产配置

```bash
TORCH_COMPILE=true
TORCH_COMPILE_SCOPE=wan_blocks_vae_clip
TORCH_COMPILE_BACKEND=inductor
TORCH_COMPILE_MODE=default
TORCH_COMPILE_DYNAMIC=false
TORCH_COMPILE_FULLGRAPH=true
TORCHINDUCTOR_AUTOTUNE_POINTWISE=false
```

生产长跑建议在完成一次诊断 trace 后关闭 profiler，避免 rank 0 同步导出大 trace 时让其余 rank 等待。默认模式冷编译首 optimizer step 约 129.94 s，之后稳定约 24 s；可持久化相同版本/shape/GPU 的 Inductor cache 来降低后续冷启动，但不要把 cache 放在容量很小的 overlay `/tmp`。

## 8. Profiler summary 口径修复

`ProfCallback` 已改为同时输出：

- `calls` / `cpu_calls`：逻辑 host invocation；
- `gpu_annotation_calls`：跨 CUDA stream 投影数；
- `cpu_child_device_total_s`：host range 关联的 child device time；
- `gpu_annotation_total_s`：GPU annotation stream spans。

旧字段 `device_total_s` / `self_device_s` 保留兼容，但后续不会再把 baseline CLIP 的两个 stream 误报为 8 次 Python forward。

## 9. Git 与运行记录

主要提交：

- `34a807d`：Batch video and condition VAE encoding
- `63364ef`：Add targeted VAE and CLIP compile scope
- `78c86d8`：Detach CLIP for targeted compilation
- `e79547c`：Combine regional Wan VAE and CLIP compile

进一步的 profiler summary 口径修复与本报告在后续提交中。

## 10. 参考

- [PyTorch `torch.compile` API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch regional compilation tutorial](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [PyTorch compiler FAQ](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html)
- [DeepSpeed ZeRO documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)
- [DeepSpeed 0.18.4 engine source](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.4/deepspeed/runtime/engine.py)
