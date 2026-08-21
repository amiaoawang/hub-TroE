# Qwen3.8-Max 架构学习总结 —— 对比原始 Transformer

> 研究对象：`Qwen/Qwen3.8-2.4T-A95B`（2026-08-12 开源底座权重）
> 代码依据：`config.json` + `modeling_qwen3_5_moe.py`（transformers，model_type = `qwen3_5_moe_text`）

---

## 一、模型档案

| 项目 | 值 |
|---|---|
| 发布 | 2026-08-03 API 上线；2026-08-12 开源底座权重（BF16 / FP8 双版本） |
| 总参数 / 激活参数 | 2.4T / 约 95B（激活比 ≈ 4%） |
| 上下文 | 开源版原生 262,144；API 版最高约 1M |
| 形态 | 开源版纯文本、强制思考模式；API 版多模态 |
| 体积 | BF16 约 4.89 TB（213 个 safetensors 分片） |
| 许可 | Qwen3.8-Max 自定义许可（非 Apache-2.0） |

**架构类型**：`Qwen3_5MoeForCausalLM` —— decoder-only 因果语言模型，92 层混合堆叠（69 层线性注意力 + 23 层全注意力，3:1 交替），每层 FFN 为稀疏 MoE。

---

## 二、基线：原始 Transformer（2017）的六大瓶颈

| # | 瓶颈 | 后果 |
|---|---|---|
| ① | 自注意力 O(L²) | 序列长度翻倍，算力翻四倍 |
| ② | KV cache 线性膨胀 | 长上下文解码又贵又慢 |
| ③ | Post-LN 深层不稳 | 层数一深就发散，依赖 warmup |
| ④ | 稠密 FFN | 参数容量 = 每 token 算力，扩容必然变贵 |
| ⑤ | 绝对位置编码 | 训练长度之外外推能力差 |
| ⑥ | 词表小 | 多语言/代码 token 化效率低 |

---

## 三、单层结构对比

```
原始 Transformer 层（2017）          Qwen3.8-Max 层（×92）
─────────────────────────           ─────────────────────────
Token Embedding                     Token Embedding（词表 248K，位置不在此注入）
  ＋ 正弦绝对位置编码                RMSNorm（Pre-Norm，无 bias）
Multi-Head Attention（8 头）         Token Mixer A：Gated DeltaNet（69 层）
Add → LayerNorm（Post-LN）          Token Mixer B：门控 GQA + QK-Norm（23 层）
FFN：ReLU，4×d（稠密）              ＋ 残差
Add → LayerNorm                     RMSNorm
                                    Sparse MoE：Top-10/512 + 共享专家
                                    ＋ 残差
× 6（Encoder）+ × 6（Decoder）      × 92 层 → RMSNorm → lm_head
```

---

## 四、六处核心改动（改动 / 取舍 / 效果）

### 改动① 位置编码：正弦绝对 PE → Partial RoPE（θ=10⁷）

- **改动**：q/k 按位置旋转（内积自动编码相对距离）；仅旋转前 64/256 维（`partial_rotary_factor=0.25`），其余纯内容；基频 θ=10,000,000。
- **收益**：相对位置 → 长度泛化好；部分旋转省计算；大 θ 支撑 262K 原生上下文。
- **代价**：192/256 维无位置信号；大 θ 牺牲短距离分辨率（靠部分旋转补偿）。
- **效果**：原生 262K 上下文稳定训练；推理配 YaRN 类策略可扩至约 1M。

### 改动② 全注意力层：MHA → GQA 64:4 ＋ QK-Norm ＋ 输出门控

- **改动**：64 个 Q 头共享 4 个 KV 头（16:1）；每头 RMSNorm(q/k) 后再旋转；q_proj 输出维度 ×2，一半做 sigmoid 门控；head_dim=256。
- **收益**：KV cache 缩小 16×（每 token 每层仅 4 KB，bf16）；QK-Norm 压住 logits 漂移；输出门控增强表达力。
- **代价**：KV 共享损失头多样性；q_proj 参数翻倍（每层约 +2.7 亿）；256 维大头的 kernel 效率与数值挑战。
- **效果**：262K 序列 KV cache 约 24 GB/条（23 层）；若用 MHA 需 16×，不可行 —— GQA 是长上下文的入场券。

```python
q, gate = q_proj(x).chunk(2, dim=-1)        # 一半 query、一半门控
q, k = RoPE(q_norm(q), k_norm(k))           # 先 QK-Norm，再仅旋转前 64/256 维
attn = softmax(q @ k.T * 256**-0.5) @ v     # 4 个 KV 头 repeat 给 64 个 Q 头
out  = o_proj(attn * sigmoid(gate))         # 输出逐元素门控
```

### 改动③ 最大胆的一步：75% 层换成 Gated DeltaNet（线性注意力）

- **改动**：69/92 层不再用 softmax 注意力。通道：in_proj → 因果深度卷积（kernel=4）→ QK L2Norm；状态：每头固定矩阵 S(128×128, fp32) 共 128 头；delta 规则写入 `S←S·g + k·β(v−S·k)`（写"预测残差"而非覆盖）；可学习遗忘 g 与写入强度 β。
- **收益**：训练 O(L)（预填充按 64-token 块并行）；解码 O(1)（每层状态固定约 8.4 MB，不随长度涨）；75% 层摆脱 KV cache。
- **代价**：固定状态容量有限，精确检索弱于 softmax；必须 3:1 混入全注意力层兜底；fp32 状态与专用 kernel，工程复杂度高。
- **效果**：长序列吞吐大幅提升；与 23 个全注意力层形成"便宜的大多数 + 精确的少数"分工 —— 整体"又长又便宜"的核心来源。

```python
S = zeros(128, 128)                              # 每头固定状态（fp32）
S = S * g_t + outer(k_t, beta_t*(v_t - S @ k_t)) # 衰减旧记忆，写入"预测残差"
out_t = S @ q_t                                  # 状态直接读出，每步 O(1)
```

### 改动④ FFN：稠密 ReLU → 稀疏 MoE（512 专家，Top-10 ＋ 共享专家）

- **改动**：每层 512 个 SwiGLU 专家（中间维 2048）；Router：softmax → Top-10 → 权重归一化；＋ 1 个共享专家常开（sigmoid 门控）；aux loss 0.001 做负载均衡。
- **收益**：容量-算力解耦（2.4T 知识，每 token 仅激活 ≈4%）；共享专家承载通用知识，路由更稳。
- **代价**：推理显存 = 全量 2.4T（BF16 约 4.89 TB，只能数据中心部署）；路由训练不稳；专家并行 all-to-all 通信。
- **效果**：每 token 算力约等于 95B 稠密模型，能力却对标旗舰 —— 2.4T 容量 95B 成本；官方 FP8（block=128）再省一半。

### 改动⑤ 归一化：Post-LN（带 bias）→ Pre-RMSNorm（无 bias）

- **改动**：先 RMSNorm 再进子层（残差直通）；只做 RMS 缩放无均值中心化；eps=1e-6，权重零初始化（输出 = 1 + w）；所有 Linear 无 bias，attention_dropout=0。
- **收益**：残差梯度恒等通路 → 92 层深网稳定训练；归一化计算少约一半；省参数、访存更少。
- **代价**：Pre-Norm 有效深度打折（靠堆更多层补偿）；无均值中心化，对异常维度更敏感。
- **效果**：2.4T 参数、92 层、262K 序列长度同时训练收敛的结构前提。

### 改动⑥ 训练与工程细节

| 项目 | 内容 | 收益 | 代价 |
|---|---|---|---|
| MTP 多 token 预测 | `mtp_num_hidden_layers=1`，训练时额外预测未来 token；推理不加载 | 训练效率↑、可用于推测解码 | 训练开销↑ |
| 大词表 248,320 | 多语言/代码 token 化效率↑；嵌入不共享（untied） | 同样文本更少 token | 嵌入参数约 4.1B |
| bf16 ＋ fp32 状态 | 主干 bf16；线性注意力递归状态强制 float32 | 速度与数值稳定兼得 | 状态显存 ×2 |
| 官方 FP8 权重 | block=128 细粒度量化 | 显存/带宽约省一半 | 需 FP8 硬件 |
| 强制思考模式 | 每条回复先输出 `<think>`；effort 可调（xhigh/medium/low） | 复杂任务质量↑ | 简单任务也付推理成本 |
| 采样默认值 | temperature=1.0, top_k=20, top_p=0.95 | 长思考链采样稳定 | 需按场景手调 |

---

## 五、config.json 关键参数速查

| 参数 | 值 | 解读（对照 2017） |
|---|---|---|
| `hidden_size` | 8192 | 主干宽度（2017 为 512，×16） |
| `num_hidden_layers` | 92 | 69 线性注意力 + 23 全注意力（3:1 交替） |
| `num_attention_heads` / `num_key_value_heads` | 64 / 4 | GQA：16 个 Q 头共享 1 个 KV 头 |
| `head_dim` | 256 | 单头维度（2017 为 64，×4） |
| `partial_rotary_factor` | 0.25 | 仅旋转前 64/256 维 |
| `rope_theta` | 10,000,000 | 基频 10⁷，支持超长序列 |
| `linear_num_key/value_heads` | 16 / 128 | Gated DeltaNet 头配置 |
| `linear_key/value_head_dim` | 128 / 128 | DeltaNet 单头维度 |
| `linear_conv_kernel_dim` | 4 | 因果深度卷积核大小 |
| `num_experts` / `num_experts_per_tok` | 512 / 10 | MoE 路由 |
| `moe_intermediate_size` | 2048 | 专家中间维（共享专家同为 2048） |
| `vocab_size` | 248,320 | 大词表 |
| `max_position_embeddings` | 262,144 | 原生上下文 |
| `mtp_num_hidden_layers` | 1 | 多 token 预测头（训练期） |
| `attn_output_gate` | true | 注意力输出门控 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `tie_word_embeddings` | false | 嵌入与 lm_head 不共享 |

---

## 六、取舍总表

| 改动 | 得到（收益） | 付出（代价） |
|---|---|---|
| Partial RoPE + θ=10⁷ | 相对位置、262K 长上下文 | 3/4 维度无位置信号，短距分辨钝化 |
| GQA 64:4 | KV cache 缩小 16× | KV 头多样性下降（检索精度略降） |
| QK-Norm + 输出门控 | 训练稳定、表达力增强 | q_proj 参数 ×2，计算路径变复杂 |
| Gated DeltaNet ×69 | O(L) 训练、常数显存解码 | 精确检索弱，必须混合全注意力兜底 |
| MoE 512 专家 Top-10 | 容量≠算力：2.4T 容量 95B 成本 | 全量显存部署、路由与通信复杂 |
| Pre-RMSNorm / 无 bias | 92 层深网稳定、更快 | 有效深度微降，靠层数补偿 |
| MTP 训练头 | 训练信号密度↑ | 训练开销与实现复杂度↑ |
| 大词表 248K | 多语言/代码效率↑ | 嵌入参数约 4.1B |

**共同规律**：用"结构先验 + 工程复杂度"换"算力与显存"，再把省下的钱花在更多参数与更长上下文上。

---

## 七、效果验证

### 公开评测（极高推理档 + 工具，2026-08）

| 基准 | Qwen3.8-Max | 参照（同代旗舰） |
|---|---|---|
| PaperBench | 93.0 | Claude Fable 5：88.8 / Opus 4.8：80.3 |
| Terminal-Bench 2.1 | 86.6 | Claude Opus 4.8：84.6 |
| SWE-bench Pro | 67.7 | Claude Opus 4.8：69.2 |
| FrontierSWE | 73.5 | — |
| GPQA Diamond | 92.6 | — |
| HLE | 56.2 | 公开榜排名 13 / 181 |

### 效率账本

- 激活/总参数 = 95B / 2.4T ≈ **4%**（MoE：容量与算力解耦）
- 262K 原生上下文，75% 层解码显存为**常数**（线性注意力 + GQA）
- GB300 NVL72（FP8）：每 GPU **4K+ tok/s**（NVIDIA 参考部署）
- BF16 权重 4.89 TB → FP8 约一半

---

## 八、变与不变

**变**（每个部件都被换过一遍）：
- 正弦绝对 PE → Partial RoPE（θ=10⁷）
- MHA → GQA + QK-Norm + 输出门控
- 全注意力 → 3:1 混合线性注意力（75%）
- 稠密 FFN → 512 专家稀疏 MoE + 共享专家
- Post-LN → Pre-RMSNorm；+ MTP、大词表

**不变**（Transformer 的骨架）：
- 残差流（residual stream）仍是主干道
- "Token Mixer + Channel Mixer"交替结构未变
- 自回归 next-token prediction 目标未变
- 每处改动在 2017 论文里都有原型：注意力→delta 规则状态、FFN→门控专家、LN→RMSNorm —— **是演化，不是革命**

---

## 九、建议学习路径

1. `config.json` 逐行读（本文第五节）
2. `DecoderLayer.forward` 看清层内数据流
3. `Qwen3_5MoeAttention`（GQA / QK-Norm / 门控）
4. `GatedDeltaNet`（重点啃 chunked delta rule）
5. `SparseMoeBlock`（路由 / 共享专家）
6. 回到 2017 论文，复述每处取舍

## 十、本地资料

```
D:\program\qwen38_max_study\
├── config\
│   ├── config.json                 # 架构"身份证"（3.9 KB）
│   └── generation_config.json      # 采样参数（202 B）
├── code\
│   ├── modeling_qwen3_5_moe.py     # 模型实现（97 KB）
│   └── configuration_qwen3_5_moe.py# 配置定义（9 KB）
├── make_ppt.py                     # PPT 生成脚本
├── check_layout.py                 # 排版校验脚本
├── Qwen3.8-Max架构学习_对比原始Transformer.pptx
└── Qwen3.8-Max架构学习总结.md      # 本文件
```
