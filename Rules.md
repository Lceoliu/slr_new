# 项目说明文档

**项目代号**：SignStream-RVQ

**面向代理**：请用原生 PyTorch + HuggingFace 生态（`transformers`/`accelerate`/`peft` 可安装但第一步不必使用）实现本说明文档所述代码。**禁止**使用 Lightning/DeepSpeed/庞大训练框架。

**数据**：已提供 **CSL-Daily** 数据集及其数据格式 README（以该 README 为准）。

**目标阶段**：**第一步（Stage 1）——多路 RVQ 离散器与数据/训练脚手架**。

---

## 0. 项目总目的（长线）

构建一个**实时**手语翻译系统：将 3D pose 时间序列离散为**sign tokens**，作为 **decoder-only LLM**（如 Qwen）的**前缀条件**，以 **KV cache** 支持的方式**流式**生成文本。系统核心包含三部分：

1. **多路 RVQ 离散器（本阶段实现）**：将面部/左右手/身体子流的 pose 表示离散为 token，输出语义稳定、时序平滑、码本利用率健康的离散序列。
2. **Prefix-LM 适配（后续阶段）**：将 sign tokens 作为 LLM 的前缀，<SEP> 后仅对文本计算语言建模损失。
3. **流式推理与<COMMIT>稳定机制（后续阶段）**：滑窗 KV + 摘要 token + 提交点，保证“边看边说、少回改”。

---

## 1. 第一阶段的目标（本次要交付的“能跑起来”的最小系统）

### 1.1 业务目标

* 在 CSL-Daily 上**训练一个可用的多路 RVQ 离散器**（Residual VQ，2–3 级），输入为对齐到统一坐标系/帧率的 2D pose 序列（面部/左手/右手/身体四路）。
* 产出**符号序列**，并支持将每个时间块（chunk）格式化为**可供 LLM 使用的 token 模板**（仅定义与导出，不做 LLM 训练）。
* 提供**指标与可视化**：重构损失、时序一致性、码本使用熵/困惑度、每块 token 数、运行时延统计。
* 提供**推理脚本**：给定一段样本，导出其离散 token 序列与简易 Run-Length 压缩（No-Change 标记）。
* 代码结构**清晰、可扩展**，为后续 Stage 3/4（Prefix-LM & 流式）直接接入做准备。

### 1.2 技术约束

* **框架**：PyTorch 2.1.0。
* **本机环境**: 已经配置了一个包含大多数所需库的python conda环境，请使用conda activate SLR_realtime来激活。
* **禁止** Lightning/DeepSpeed 以及将训练循环交给“黑盒 Trainer”。
* **配置**：YAML 配置文件 + 命令行覆盖；保证可复现实验。
* **日志**：TensorBoard.
* **设备**：优先单卡 3090 24G，保证后续多卡训练的可行性。

---

## 2. 数据假设与输入输出

> 以“CSL-Daily 数据格式 README”为准。CSL-Daily root dir绝对位置位于/workspace/data/public_data/CSL_Daily, README位于/workspace/data/public_data/CSL_Daily/README.md

> 在本目录下有./CSL-Daily软链接，所以可以直接访问

### 2.1 输入（训练/推理）

* **样本单位**：一段视频对应的 **2D pose 序列**，参考COCO关键点格式。
* 按照数据集README所给的格式将输入按身体部位分区为不同的子流输入：face, left_hand, right_hand, body, full_body(包含以上所有部位)

### 2.2 预处理

* **时域**：统一帧率；线性插值/低通滤波；可选时序截断到最大长度。
* **归一化**：各分区以重心/README给定的中心定义局部坐标系，所有子流坐标各自归一化；可选按骨长/下眼睑长归一。
* **块化（chunking）**：将 T 帧划分为长度 `L` 的块（建议 8–12 帧/块，或 320–400ms/块）；每块将在各子流上独立编码。
* **数据增强**：镜像（左右手交换需同步）、速度抖动（time-warp）、关节遮挡 dropout（可开关）。

### 2.3 输出（本阶段）

* **离散 token 序列**（仅 sign 部分）：为每个时间块产出若干 code id。
* **格式化模板**（供后续 LLM 使用）：

  ```
  <T_k><F:c12><LH:a37><RH:b08><B:d91><FB:e45>
  ```

  其中 `<T_k>` 为第 k 个时间块标记；`F/LH/RH/B:` 前缀区分子流来源；`a/b/c/d/e` 为各子流所属码本的 id 空间（避免与文本词表冲突）。
* **可选压缩**：相邻块若“变化极小”，输出 `<LH:NC>xK` 等 No-Change/Run-Length 记号（简化到 JSON 序列即可）。

---

## 3. 模块与目录结构

```
signstream/
  configs/
    default.yaml                  # 全局配置（数据路径/模型/训练/日志）
  data/
    datasets.py                   # CSL-Daily Dataset & DataModule
    transforms.py                 # 归一化/块化/增强
    collate.py                    # 按块组 batch，掩码/对齐
  models/
    rvq/
      encoder.py                  # 子流编码器（Face/LH/RH/Body 可共享骨干+type embed）
      quantizer.py                # 多级 RVQ（EMA/commitment/usage 正则）
      decoder.py                  # 重构解码器
      rvq_model.py                # 组合：Encoder -> RVQ -> Decoder
    metrics/
      recon.py                    # MSE/Huber/SSIM（可选）
      temporal.py                 # 时序一致性（相邻块差分）
      codebook.py                 # 使用率/困惑度/熵
  training/
    train_rvq.py                  # 训练主脚本（Accelerate）
    loop_rvq.py                   # 训练/验证循环（手写）
    optim.py                      # 优化器/调度器
    losses.py                     # 重构/commitment/usage/时序一致性/InfoNCE（留接口）
    utils_tb.py                   # 日志/可视化
    seed.py                       # 可复现
  inference/
    export_tokens.py              # 离散推理：pose->tokens(JSON)
    rle.py                        # No-Change/Run-Length
    viz.py                        # 可视化：token时间轴/码本热图
  io/
    readme_parser.py              # 读取用户提供的 README 并校验数据字段（轻量）
    dataloader_utils.py           # 多进程/缓存
  tests/
    test_quantizer.py             # 量化/反量化一致性
    test_dataset.py               # 数据切块/归一化正确性
    test_export.py                # 导出 token 与 RLE 的稳定性
  README.md
  requirements.txt
```

---

## 4. 模型设计（本阶段）

### 4.1 编码器/解码器

* **子流共享骨干**：建议一个共享 MLP/轻量 Transformer，将 `type embedding`（Face/LH/RH/B/FB）拼接/相加/Fusion融合，以减少参数且鼓励跨域组合表示。
* **时间聚合**：对块内帧做平均池化或轻量时序注意力，得到块级潜向量 `z_chunk`（每子流一份）。
* **解码器**：镜像结构，重构回块内帧的关键点（或其增量）。
* 支持可选 **`torch.compile`**。

### 4.2 RVQ 量化器

* **层数**：2–3 级残差码本；**码本大小**：每级 1024（可配置）。
* **损失**：

  * 重构：MSE 或 Huber。
  * Commitment/Codebook：标准 VQ-VAE/RVQ 公式（EMA 版）+ **usage 正则**（鼓励均匀使用，避免塌缩）。
  * **时序一致性**：相邻块 latent 的差分正则（或对齐到速度域）。
  * **可选 InfoNCE 钩子**：为后续语义对齐预留接口（本阶段可先留空/关闭）。
* **指标**：码本困惑度、码本占用率直方图、平均每块 symbol 数、压缩率。

---

## 5. 训练与评估

### 5.1 训练配置（默认建议）

* 块长度：`L = 8–12` 帧；批大小按显存自动调整。
* 优化器：AdamW；学习率与权重衰减从配置读取。
* 训练时长：以“重构收敛 + 使用率稳定”为准（示例 100–300 epoch）。
* 混合精度：bf16/fp16（`accelerate` 自动管理）。

### 5.2 评估指标（validation）

* **重构质量**：MSE/Huber（四路分别与加权求和）。
* **时序平滑**：相邻块重构差分的均值/分位数。
* **码本健康度**：使用率、困惑度、熵；空置条目比例。
* **效率**：每秒样本数、单步延迟（ms）、显存峰值。
* **导出健壮性**：随机样本导出 token 序列成功率；RLE 后长度统计。

### 5.3 验收/单元测试

* `tests/test_quantizer.py`：前向量化+反量化误差在阈内；EMA/commitment 梯度路径稳定。
* `tests/test_dataset.py`：块化后总帧覆盖率 100%，镜像与左右手交换一致。
* `tests/test_export.py`：RLE 与还原一致性；极端“全静止/全变化”样本正确处理。
* **过拟合小样本**：取 16 个样本训练，重构误差显著下降（作为 sanity check）。

---

## 6. 配置与命令行

### 6.1 `configs/default.yaml`（示意）

```yaml
data:
  root: "/path/to/CSL-Daily"
  split: {train: "train.json", val: "val.json"}
  fps: 25
  chunk_len: 10
  augment:
    mirror: true
    time_warp: true
    dropout_prob: 0.05

model:
  latent_dim: 256
  type_embed_dim: 16
  rvq:
    levels: 3
    codebook_size: 1024
    ema_decay: 0.99
    commitment_beta: 0.25
    usage_reg: 1e-3

train:
  epochs: 200
  batch_size: 16
  lr: 3e-4
  wd: 0.01
  amp: "bf16"
  log_wandb: false
  seed: 42

export:
  enable_rle: true
  rle_threshold: 0.02  # 块间变化阈值
```

### 6.2 典型命令

```bash
# 训练
accelerate launch signstream/training/train_rvq.py --config configs/default.yaml

# 验证/导出
python signstream/inference/export_tokens.py \
  --config configs/default.yaml \
  --checkpoint path/to/ckpt.pt \
  --split val \
  --num-samples 10 \
  --out ./exports/tokens.jsonl
```

---

## 7. 导出格式（供后续 LLM 阶段使用）

### 7.1 JSON Lines（每行一条样本）

```json
{
  "video_id": "xxxx",
  "fps": 25,
  "chunk_len": 10,
  "tokens": [
    {"t":0, "F":[12,7], "LH":[37], "RH":[8,8], "B":[91]},
    {"t":1, "F":[7], "LH":["NC",3], "RH":[8], "B":[13]}
  ],
  "rle": true,
  "meta": {"note":"ids按各自码本空间编码，不与文本词表混用"}
}
```

> 说明：`"LH":["NC",3]` 表示左手 3 个块无显著变化的 RLE 压缩。

### 7.2 Token 模板（示例，仅供可视化/调试）

```
<T0><F:c12><F:c7><LH:a37><RH:b8><RH:b8><B:d91><FB:e45>
<T1><F:c7><LH:NCx3><RH:b8><B:d13><FB:e87>
```

---

## 8. 非功能性要求

* **可复现**：设定随机种子；保存 `config.yaml`、`git hash`、环境依赖。
* **容错**：数据异常（缺帧/NaN）需跳过并记录。
* **性能**：DataLoader 多进程、pin memory；大样本可分块载入。
* **代码质量**：PEP8 基本规范，关键模块含文档字符串与类型注解。

---

## 9. 里程碑与交付物

* **M0（脚手架完成）**：目录/配置/数据管道/训练循环（空模型可跑通）。
* **M1（RVQ 可训练）**：重构损失下降，码本使用率>30%，困惑度>100（示例阈）。
* **M2（导出与可视化）**：能稳定导出 token 序列 + RLE 可选；提供时序条带图与码本热图。
* **M3（报告）**：在验证集产出指标：MSE 曲线、使用率直方图、平均每块符号数、导出长度统计。

**交付物**：源码仓（含 `README.md`、`requirements.txt`、`configs/default.yaml`）、1–2 个训练检查点、`exports/tokens.jsonl` 示例、日志与可视化截图。

---

## 10. 未来扩展挂钩（非本阶段实现）

* **Prefix-LM 接口**：保留 `signstream/inference/export_tokens.py` 的 `--to-llm` 选项，未来可直接拼接 `<SEP>` 后文本并构造 `labels=-100` 掩码。
* **摘要/滑窗 KV**：在导出时插入占位 `<SUMM>`（每 K 块），为后续 LLM 训练对齐。
* **<COMMIT> 稳定度**：在导出中记录块边界，后续据此生成提交点监督信号。

---

## 11. 依赖与环境

使用SLR_realtime conda 环境即可，若缺失库，直接安装并记录即可。

---

## 12. 成功标准（本阶段）

* 训练在 CSL-Daily 上可运行到收敛，**重构误差显著下降**。
* 码本**无塌缩**（使用率分布不极端），困惑度/熵稳定。
* 能对任意验证样本**稳定导出 token**；RLE 在静态片段显著压缩（>30%）。
* 所有单元测试通过；日志/可视化齐全；README 能指导他人复现。

---

如上，即为第一步的完整开发说明。请严格对照实现，并优先保证训练循环与导出流程的**简洁、稳定、可扩展**。后续阶段将直接在该脚手架上接入 LLM 前缀训练与流式解码。
