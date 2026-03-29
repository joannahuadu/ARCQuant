# Phase 3 实验记录：P3-A 引入 bf16 teacher

**目标**：将 joint/x_mask 的训练目标从 “逼近 NVFP4 dense” 改为 “逼近 bf16”，验证 teacher 目标错位是否是当前性能瓶颈。

**核心问题**：
- 当前 `cali_x_mask.py` / `cali_joint.py` 的 teacher 都是量化模型自身的 dense 输出，即 `x_mask disabled`，不是 bf16。
- 这意味着现有 `joint` 学到的是 “恢复相对 NVFP4 dense 的误差”，而不是 “恢复相对 bf16 的总误差”。
- 如果最终目标是让 **NVFP4 + 2:4 稀疏** 尽量接近 bf16，那么 teacher 应该至少在一部分实验中切到 bf16 做直接蒸馏。

---

## 背景

### 当前关键结果

| Config | arc_challenge | rte | ceval-valid | 说明 |
|--------|---------------|-----|-------------|------|
| bf16 | 0.5316 | 0.7004 | 0.5245 | 最终目标 |
| nvfp4 | 0.5145 | 0.6859 | 0.4844 | 稠密量化 baseline |
| xmask | 0.4855 | 0.7076 | 0.4398 | 稀疏后显著掉点 |
| joint_v1 | 0.5068 | 0.6643 | 0.4190 | 追回部分 sparsity loss，但仍低于 bf16/nvfp4 |

### 现象解读

- `joint_v1` 相比 `xmask` 有恢复，说明 `softmax_alpha` / `output_scale` 确实在补偿 sparsity 引入的误差。
- 但 `joint_v1` 仍低于 `nvfp4`，而训练 teacher 恰恰是 `nvfp4 dense`，说明当前目标函数最多只能逼近一个本身低于 bf16 的上界。
- `P2-A`（换 C4）和 `P2-B`（更强正则）都没有提升 arc，说明问题不太像只是超参/数据问题，更像训练目标本身错位。

---

## P3-A：Full bf16 teacher

**假设**：如果 student 直接对齐 bf16 层输出，而不是 NVFP4 dense 层输出，那么 gate、softmax alpha、output scale 会学习到同时补偿：
- activation sparsity 误差
- NVFP4 量化误差
- attention 分布偏移

从而让最终模型更接近 bf16 的任务表现，而不是只接近 NVFP4 dense。

### 训练定义

**Teacher**：
- bf16 模型对应层输出
- `x_mask` 不存在
- `softmax_alpha = 1`
- 不量化

**Student**：
- NVFP4 模型
- 开启 `x_mask`
- 开启 trainable `softmax_alpha`
- 开启 trainable `output_scale`

**Loss**：

```text
L = MSE(y_student_nvfp4_sparse, y_teacher_bf16)
  + gate_cost
  + alpha_reg * ||alpha - 1||^2
  + alpha_reg * ||output_scale - 1||^2
  + token_delta_l2 (optional)
```

### 与现有 joint 的唯一区别

不是：
- `teacher = NVFP4 dense`

而是：
- `teacher = bf16 dense`

这一步是 **目标函数修正实验**，不是超参微调实验。

---

## 为什么 P3-A 必须做

### 理论动机

当前 `joint` 的最优点，本质上更接近：

```text
argmin || sparse_nvfp4 - dense_nvfp4 ||
```

但真正想要的是：

```text
argmin || sparse_nvfp4 - bf16 ||
```

如果这两个目标不一致，那么继续在 `dataset / alpha_reg / decouple / output_scale` 上调参，收益会被 teacher 上界限制。

### 与当前目标线的冲突

- 目标线：`arc_challenge >= 0.5263`
- 当前 `nvfp4`: `0.5145`

如果 teacher 仍是 `nvfp4 dense`，那么 student 主要在追 `0.5145` 对应的内部表示；这和 `>= 0.5263` 的目标天然不一致。

---

## 实验设计

### P3-A1：joint + bf16 teacher

**目标**：最直接验证 teacher 是否是主瓶颈。

**设置**：
- dataset: `wikitext2`
- nsamples: `128`
- epochs: `30`
- trainable params:
  - x_mask gate
  - token gate mlp
  - softmax_alpha
  - output_scale
- loss target: bf16 layer outputs

**建议命名**：
- `joint_bf16_teacher`

### 推荐启动命令模板

> 备注：下面命令是记录模板；真正运行前需要先在 `model/cali_joint.py` 中补上 bf16 teacher 路径/开关。

```bash
CUDA_VISIBLE_DEVICES=0 python -u model/cali_joint.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --dataset wikitext2 \
    --nsamples 128 --cali_bsz 4 --epochs 30 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all \
    --x_mask_token_use_layer_scale \
    --x_mask_token_mlp_shared \
    --trainable_alpha --alpha_lr 1e-3 --alpha_reg 0.01 \
    --teacher_dtype bf16 \
    --exp_name joint_bf16_teacher
```

### 结果记录（待填）

| Task | bf16 | nvfp4 | joint_v1 | **P3-A1 joint_bf16_teacher** | Δ vs joint_v1 |
|------|------|-------|----------|-------------------------------|---------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | TBD | — |
| rte | 0.7004 | 0.6859 | 0.6643 | TBD | — |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | TBD | — |

---

## 判据

### 成功信号

- `arc_challenge` 明显高于 `joint_v1 = 0.5068`
- 最低要求：恢复到 `>= nvfp4 = 0.5145`
- 理想目标：达到 `>= 0.5263`

### 失败信号

- P3-A1 仍接近 `joint_v1`，提升 < 0.5%
- 或者多任务出现更严重退化，说明：
  - teacher 切换不是主因
  - 或 gate 把 quantization 误差也一并背了，训练不稳定

---

## 风险分析

### 风险 1：bf16 teacher 太强，gate 学错方向

gate 是结构约束，直接对 bf16 蒸馏可能会让 gate 试图同时补偿：
- sparsity 误差
- quantization 误差

这可能导致 gate pattern 不稳定，甚至学出不合理的保留/裁剪结构。

### 风险 2：显存/耗时翻倍

因为需要同时保留：
- bf16 teacher layer
- NVFP4 student layer

如果采用逐层双模型前向，训练成本会显著增加。

### 风险 3：最终收益只来自 alpha/scale，而不是 gate

如果真正有效的是 attention residual correction，那么 gate 本身可能仍然只适合对齐 NVFP4 dense；这时需要后续 mixed teacher 实验拆分贡献。

---

## 后续分支（P3-B 预告）

如果 P3-A1 有效，下一步继续做：

### P3-B1：mixed teacher

- gate loss 对齐 `NVFP4 dense`
- alpha/output_scale loss 对齐 `bf16`

目的：
- 让 gate 只负责 sparsity pattern 学习
- 让 alpha/output_scale 负责 bf16 residual compensation

这会比 full bf16 teacher 更稳，也更容易解释。

---

## 当前状态

| 项目 | 状态 |
|------|------|
| P3-A 方案设计 | ✅ completed |
| P3-A 实现 | 🔄 in progress |
| P3-A 训练 | ⏳ pending launch |
| P3-A 评测 | ⏳ pending |

---

## 本轮运行记录

### P3-A1：joint + bf16 teacher（wikitext2）

**计划变更**：
- 在 `model/cali_joint.py` 中新增 `--teacher_dtype {nvfp4,bf16}`
- 当 `teacher_dtype=bf16` 时，teacher 走原始 bf16 layers，student 保持 NVFP4 + x_mask + alpha + output_scale
- 为避免显存爆掉，bf16 teacher hidden state cache 放在 CPU，仅按 batch 临时搬到 GPU

**自动化**：
- 训练完成后自动启动 eval：`arc_challenge / rte / ceval-valid`
- watcher 轮询间隔：15 分钟
- watcher 日志：`results/logs/watch_p3a_eval.log`
- eval 日志：`results/logs/p3a_eval.log`
- eval 输出：`results/joint_bf16_teacher_eval.json`

### 启动信息

| 字段 | 值 |
|------|-----|
| GPU | 3 |
| Train PID | `1227735` |
| Watcher PID | `1228202` |
| 训练日志 | `results/logs/p3a_cali_bf16_teacher.log` |
| exp_dir | `outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/` |
| 输出 ckpt | `outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt` |
| 启动时间 | `2026-03-27 07:29:35 UTC` |
| 状态 | 🔄 running |

**重启说明**：
- 首次正式启动 (`exp_dir=joint_bf16_teacher_20260327_072641`) 因漏传 `CUDA_VISIBLE_DEVICES=3`，bf16 teacher 误占到物理 GPU0，与 P2-C 冲突后 OOM 退出。
- 已于 `2026-03-27 07:29:35 UTC` 按正确 GPU 重新启动。

---

## 当前结论

- `P3-A` 是必要对照实验，不是可选优化。
- 在继续细调 `P2-C / P2-D / P2-E` 之前，应该优先确认 teacher 目标是否错位。
- 如果 `P3-A` 成功，后续重点应从 “调数据/调正则” 转向 “teacher 分解设计（full bf16 vs mixed teacher）”。


## P3-A1 结果（auto-updated）

| Task | bf16 | nvfp4 | joint_v1 | **P3-A1 joint_bf16_teacher** | Δ vs joint_v1 |
|------|------|-------|----------|-------------------------------|---------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | **0.4881** | -0.0187 |
| rte | 0.7004 | 0.6859 | 0.6643 | **0.7040** | +0.0397 |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | **0.4324** | +0.0134 |

### P3-A1 诊断：bf16 teacher 的 output_scale 分布

对 `outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt` 逐层检查后，发现 `output_scale` 的异常主要集中在深层。

| Layer | min | max | mean | std | \|s-1\| mean |
|------|-----|-----|------|-----|--------------|
| 24 | 0.6048 | 1.3536 | 0.9882 | 0.0251 | 0.0181 |
| 25 | 0.5508 | 1.4710 | 0.9850 | 0.0357 | 0.0260 |
| 26 | 0.4972 | 1.6142 | 0.9815 | 0.0497 | 0.0359 |
| 27 | 0.2789 | 1.6266 | 0.9692 | 0.0597 | 0.0486 |
| 28 | 0.3872 | 1.7824 | 0.9674 | 0.0462 | 0.0424 |
| 29 | 0.5409 | 1.8121 | 0.9674 | 0.0797 | 0.0655 |
| 30 | 0.5730 | 1.6666 | 0.9557 | 0.0916 | 0.0785 |
| 31 | 0.5068 | 1.5771 | 1.0018 | 0.1447 | 0.1150 |

对比 `joint_plus`：
- `joint_plus output_scale`: min `0.8146`, max `1.2964`, mean `1.0047`
- `joint_plus softmax_alpha`: min `0.9857`, max `1.0286`, mean `1.0047`

**诊断假设**：
- `bf16 teacher` 的有效信号很强，但当前实现把大量补偿压到了深层 `output_scale`
- 这种激进补偿对 `RTE` 有利，但可能破坏 `ARC` 所需的跨层稳定性

---

## P3-A2：bf16 teacher + output_scale clip

**目的**：验证 `P3-A1` 的 ARC 退化是否主要来自 `output_scale` 在深层过度偏离 1.0。

### 变体

1. `global clip`
   - 对所有层的 `output_scale` 做后处理裁剪到 `[0.85, 1.15]`
   - ckpt:
     `outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_clip085_115_20260328/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt`

2. `deep-layer clip`
   - 仅对 `24-31` 层的 `output_scale` 做裁剪到 `[0.85, 1.15]`
   - ckpt:
     `outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_deepclip085_115_l24_31_20260328/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt`

### 当前状态

| Variant | 状态 | Eval 输出 |
|--------|------|-----------|
| global clip | 🔄 running | `results/joint_bf16_teacher_clip085_115_eval.json` |
| deep-layer clip | 🔄 running | `results/joint_bf16_teacher_deepclip085_115_l24_31_eval.json` |

### 结果（待补）

| Task | P3-A1 bf16 teacher | global clip | deep-layer clip |
|------|--------------------|-------------|-----------------|
| arc_challenge | 0.4881 | TBD | TBD |
| rte | 0.7040 | TBD | TBD |
| ceval-valid | 0.4324 | TBD | TBD |

---

## P3-B：mixed teacher

**目标**：把 teacher 信号拆开，避免 gate 和连续补偿项同时去追 bf16。

### 设计

- `gate-only phase`: teacher = `NVFP4 dense`
- `alpha/output_scale phase`: teacher = `bf16`
- 当前实现要求：`--teacher_dtype mixed` 必须与 `--decouple_training` 一起使用

### Bug 修复（2026-03-28）

`cali_joint.py` 中 `--teacher_dtype mixed` 的 `fp_outs`（nvfp4 teacher 输出）从未被计算，始终为全零。
gate-only phase 实际对着零目标训练。已修复：将 `if/else` 改为两个独立 `if`，mixed 模式同时计算 bf16 `teacher_outs` 和 nvfp4 `fp_outs`。

### P3-B v1（buggy，已废弃）

| 字段 | 值 |
|------|-----|
| GPU | 2 |
| exp_dir | `outputs/Llama-3.1-8B/NVFP4/joint_mixed_teacher_20260328_091300` |
| 启动时间 | `2026-03-28 09:12 UTC` |
| 状态 | ❌ 废弃（fp_outs=全零 bug，gate-only phase 训练目标错误） |

### P3-B v2（修复后重跑）

```bash
CUDA_VISIBLE_DEVICES=0 python -u model/cali_joint.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --dataset wikitext2 \
    --nsamples 128 --cali_bsz 4 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all \
    --x_mask_token_use_layer_scale \
    --x_mask_token_mlp_shared \
    --trainable_alpha --alpha_lr 1e-3 --alpha_reg 0.01 \
    --teacher_dtype mixed \
    --decouple_training --gate_epochs 20 --alpha_epochs 15 \
    --exp_name joint_mixed_teacher_v2
```

| 字段 | 值 |
|------|-----|
| GPU | 0 |
| Train PID | 1436270 |
| Watcher PID | 1438266 |
| 训练日志 | `results/logs/p3b_cali_mixed_teacher_v2.log` |
| exp_dir | `outputs/Llama-3.1-8B/NVFP4/joint_mixed_teacher_v2_20260328_093633/` |
| 输出 ckpt | `.../llama-3.1-8b_joint_wikitext2_max_NVFP4.pt` |
| Eval 输出 | `results/joint_mixed_teacher_v2_eval.json` |
| 启动时间 | `2026-03-28 09:36 UTC` |
| 状态 | 🔄 running |

### 结果（待补）

| Task | bf16 | nvfp4 | joint_plus | **P3-B v2 mixed teacher** |
|------|------|-------|------------|---------------------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5111 | TBD |
| rte | 0.7004 | 0.6859 | 0.6787 | TBD |
| ceval-valid | 0.5245 | 0.4844 | 0.4510 | TBD |
