# 架构说明

本文档详细说明 PPT RAG 项目的整体架构、模块设计、数据流、边界处理与扩展点。

## 目录

- [1. 设计概览](#1-设计概览)
- [2. 整体架构](#2-整体架构)
- [3. 文件结构](#3-文件结构)
- [4. 模块详解](#4-模块详解)
- [5. 数据流](#5-数据流)
- [6. 边界处理一览](#6-边界处理一览)
- [7. 配置体系](#7-配置体系)
- [8. 错误与重试策略](#8-错误与重试策略)
- [9. 性能与资源](#9-性能与资源)
- [10. 扩展点](#10-扩展点)
- [11. 设计决策](#11-设计决策)

---

## 1. 设计概览

### 1.1 目标

构建一个 **不依赖 LangChain** 的 PPT 问答 RAG 流程，覆盖从 PPT 解析到答案生成的完整链路，并具备质量评估能力。

### 1.2 设计原则

| 原则 | 说明 |
|---|---|
| **零框架依赖** | 不使用 LangChain / LlamaIndex 等编排框架，所有逻辑用标准库 + 最小依赖实现，便于理解与改造 |
| **模块单一职责** | 每个文件负责一个环节（解析、切分、向量化…），模块间通过明确的数据结构通信 |
| **配置与代码分离** | 所有可调参数集中在 `config.py`，启动前统一校验，避免运行到一半才崩 |
| **优雅降级** | 任何可选能力（OCR / rerank / OpenAI embedding）失败都不阻断主流程，仅打 warning |
| **边界处理优先** | 对文件损坏、空输入、超长文本、网络异常、JSON 解析失败等情况都显式处理 |
| **可观测性** | 统一日志格式，关键环节打 INFO，异常打 WARNING，不静默吞错 |

### 1.3 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| PPT 解析 | `python-pptx` | 纯 Python，无系统依赖，能取文本/表格/备注/图片 |
| 切分 | 自实现递归切分 | LangChain 的 `RecursiveCharacterTextSplitter` 算法复刻，去掉依赖 |
| Embedding | `sentence-transformers` (bge-large-zh-v1.5) 或 OpenAI | 本地模型中文效果好且免费；OpenAI 兜底 |
| 向量库 | `chromadb` | 轻量、纯 Python、持久化简单，适合单机 |
| Rerank（可选） | `CrossEncoder` (bge-reranker-base) | 提升检索精度 |
| LLM | OpenAI 兼容接口 | 一套代码接 GPT / Qwen / DeepSeek / 通义千问 |
| OCR（可选） | `baidu/Unlimited-OCR` | 高精度文档解析，支持图表 |
| 评估 | LLM-as-judge | 无需标注数据也能批量评估 |

---

## 2. 整体架构

### 2.1 两阶段流水线

RAG 流程分为 **离线索引构建** 与 **在线检索生成** 两个阶段：

```
┌──────────────────────── 离线阶段（一次性） ────────────────────────┐
│                                                                  │
│   PPT 文件                                                        │
│      │                                                           │
│      ▼                                                           │
│   parser.py ──── 提取文本/表格/备注/图片(OCR)                       │
│      │                                                           │
│      ▼                                                           │
│   splitter.py ─── 递归切分 → chunks                              │
│      │                                                           │
│      ▼                                                           │
│   embedder.py ─── 向量化（bge / OpenAI）                          │
│      │                                                           │
│      ▼                                                           │
│   vectorstore.py ─ 入库（chromadb 持久化）                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────── 在线阶段（每次提问） ──────────────────────┐
│                                                                  │
│   用户问题                                                        │
│      │                                                           │
│      ▼                                                           │
│   retriever.py ─── 向量化 → Top-K 检索 → (可选 rerank)            │
│      │                                                           │
│      ▼                                                           │
│   generator.py ─── 组装 prompt → LLM 生成                         │
│      │                                                           │
│      ▼                                                           │
│   evaluator.py ─── LLM-as-judge 三维度评分（可选）                │
│      │                                                           │
│      ▼                                                           │
│   最终结果 {answer, contexts, eval}                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
main.py (CLI 入口)
   │
   ▼
pipeline.py (编排)
   ├── parser.py ──┐
   │   └── ocr.py  │ 解析（含可选 OCR）
   ├── splitter.py │ 切分
   ├── embedder.py │ 向量化
   ├── vectorstore.py │ 向量库
   ├── retriever.py │ 检索
   ├── generator.py │ 生成
   └── evaluator.py │ 评估
        │
        ▼
   config.py (配置 + 校验)
        │
        ▼
   utils.py (日志/重试/JSON/截断)
```

依赖方向单向：上层模块依赖下层，下层不感知上层；所有模块共享 `config` 与 `utils`。

---

## 3. 文件结构

```
ppt_rag/
├── config.py                  # 集中配置（dataclass + 启动校验）
├── utils.py                   # 通用工具：日志、重试、JSON 兜底、截断
├── parser.py                  # PPT 解析：文本 / 表格 / 备注 / 组合形状 / 图片(OCR)
├── ocr.py                     # Unlimited-OCR 封装（懒加载、优雅降级）
├── splitter.py                # 递归切分 + 短块合并 + 边界处理
├── embedder.py                # 向量化：本地 bge / OpenAI 双后端
├── vectorstore.py             # chromadb 封装：建库 / 载库 / 查询 / upsert
├── retriever.py               # 检索 + 可选 rerank
├── generator.py               # LLM 生成：prompt 组装 + 重试 + 上下文截断
├── evaluator.py               # LLM-as-judge 三维度评估
├── pipeline.py                # 主流程编排
├── main.py                    # CLI 入口
├── requirements.txt           # 依赖清单
├── eval_cases.example.json    # 评估用例示例
├── README.md                  # 入口文档
├── architecture.md            # 本文件
└── usage_guide.md             # 详细使用指南
```

---

## 4. 模块详解

### 4.1 config.py — 配置中心

**职责**：所有可调参数集中定义，启动前统一校验。

**结构**：用 `dataclass` 嵌套组织，分为：

- `Paths` — 文件路径（PPT / 向量库 / 评估输出 / 日志）
- `SplitConfig` — 切分参数（chunk_size / overlap / min_len / separators）
- `RetrievalConfig` — 检索参数（top_k / min_score / rerank）
- `LLMConfig` — LLM 调用参数（base_url / api_key / model / temperature / 重试）
- `EmbedConfig` — Embedding 后端选择
- `OCRConfig` — OCR 参数（model / device / image_mode / ngram）

**校验时机**：`validate(cfg, require_ppt=...)` 在 `pipeline` 启动时调用，校验失败抛 `ConfigError`，CLI 捕获后退出。

**关键校验规则**：

| 字段 | 规则 |
|---|---|
| `chunk_overlap` | `0 <= overlap < chunk_size` |
| `top_k` | `> 0` |
| `rerank_top_n` | `<= top_k`（启用 rerank 时） |
| `embed.backend` | 只能是 `local` / `openai` |
| `LLM_API_KEY` | 必填 |
| `ocr.image_mode` | 只能是 `gundam` / `base` |
| `ocr.device` | 只能是 `cuda` / `cpu` |
| PPT 文件 | 存在 + 后缀正确 + 大小非零（仅 build 模式） |

### 4.2 utils.py — 通用工具

**职责**：日志、重试装饰器、JSON 兜底解析、文本截断。

**核心组件**：

#### `get_logger(name, log_file)`
- 单例模式，重复调用返回同一 logger
- 同时输出到控制台和文件
- 文件写入失败时只打 warning，不崩

#### `with_retry(max_retries, backoff, exceptions)`
- 指数退避重试装饰器
- 只捕获 `RetryableError`，其他异常直接抛出
- 用于 LLM 调用、OpenAI Embedding 等网络操作

#### `safe_json_loads(text, default)`
- 三级兜底：直接解析 → 剥离 markdown 代码块 → 正则抽取第一个 JSON 块
- LLM 返回的 JSON 经常带额外文字，此函数统一处理

#### `truncate(text, max_len, suffix)`
- 安全截断，避免超长文本喂给 LLM

### 4.3 parser.py — PPT 解析

**职责**：用 `python-pptx` 提取每页文本 / 表格 / 备注 / 图片。

**核心函数**：

#### `parse_pptx(path, ocr_cfg)`
- 输入：PPT 路径 + 可选 OCR 配置
- 输出：`[{"page": int, "content": str, "source": str}]`
- 跳过空页，不抛异常

**解析流程**：

1. 文件级校验（存在 / 大小非零 / 能被 `python-pptx` 打开）
2. 逐 slide 遍历：
   - 调用 `_extract_shape` 递归提取每个 shape
   - shape 类型判断：文本框 / 表格 / 组合 / 图片
   - 组合形状（GROUP）递归处理子形状
   - 图片单独收集 blob（用于后续 OCR）
3. 提取备注（slide.notes_slide）
4. 如启用 OCR，对每页图片批量识别，结果以 `[图片文字]` 标记追加
5. 单页过长截断（默认 20000 字符）
6. 空页跳过

#### `_extract_shape(shape, image_collector)`
- 递归处理组合形状
- 文本框：`shape.text_frame.text`
- 表格：按行拼接单元格，用 ` | ` 分隔
- 图片：把 `shape.image.blob` 追加到 `image_collector`

#### `_save_image_blob(blob, tmp_dir, idx)`
- 通过魔数判断图片格式（jpg/png/webp/gif/bmp）
- 写入临时文件供 OCR 使用

**容错点**：

- 文件损坏 / 加密 → 抛 `ParseError`
- 单个 shape 异常 → 跳过该 shape，打 warning
- 备注提取失败 → 跳过
- 图片提取失败 → 跳过
- OCR 失败 → 返回空串，不影响其他图片
- 单页过长 → 截断

### 4.4 ocr.py — Unlimited-OCR 封装

**职责**：封装 `baidu/Unlimited-OCR`，对图片做高精度文字识别。

**设计要点**：

#### 懒加载
- `torch` / `transformers` 在首次调用时才 import
- 无 GPU 环境不会启动失败，仅当用户主动 `--use-ocr` 才加载
- 模型加载失败后标记 `_load_failed`，后续不再重试

#### 优雅降级
- torch 未装 → warning + 返回空串
- CUDA 不可用 → warning + 回退 CPU（或跳过）
- 模型下载失败 → warning + 返回空串
- 单图 OCR 失败 → warning + 返回空串

#### 双模式
- `ocr_image(path)` — 单图，支持 gundam(切图) / base 两种模式
- `ocr_images(paths)` — 多图批量，仅支持 base；失败时自动降级为逐张

#### 结果提取
- 优先用模型返回值
- 返回值为空时从输出目录读 `.txt` / `.json`
- 适配 API 返回格式差异

### 4.5 splitter.py — 递归切分

**职责**：把长文本切成可检索的 chunk。

**算法**（复刻 LangChain 的 `RecursiveCharacterTextSplitter`）：

1. 按 `separators` 优先级找第一个存在的分隔符
2. 用该分隔符切分
3. 若单个 piece 仍超长，递归用更细的分隔符
4. 用 `overlap` 保留上一段尾部，避免上下文断裂
5. 短 chunk 与相邻合并

**默认分隔符**（中文优化）：

```python
["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
```

**边界处理**：

| 情况 | 处理 |
|---|---|
| 空文本 | 返回 `[]` |
| 文本短于 chunk_size | 直接返回 `[text]` |
| 单 piece 超长 | 递归用更细的分隔符 |
| overlap >= chunk_size | config 校验阶段拦截 |
| 过短 chunk | 与相邻合并，仍过短则丢弃 |

### 4.6 embedder.py — 向量化

**职责**：实现 chromadb 的 `embedding_function` 协议。

**双后端**：

| 后端 | 类 | 适用场景 |
|---|---|---|
| `local` | `LocalEmbedder` | 本地 bge 模型，免费，中文效果好 |
| `openai` | `OpenAIEmbedder` | OpenAI / 兼容服务，按量付费 |

**协议**：`__call__(input: List[str]) -> List[List[float]]`

**容错**：

- 模型加载失败 → 抛 `EmbeddingError`
- 空输入 → 返回 `[]`
- OpenAI 限流 / 超时 → 重试（指数退避）
- OpenAI 单批最多 256 条，自动分批

### 4.7 vectorstore.py — 向量库

**职责**：封装 chromadb，提供建库 / 载库 / 查询 / upsert 接口。

**核心方法**：

| 方法 | 说明 |
|---|---|
| `build(chunks)` | 从 chunks 重新建库（先删旧 collection） |
| `load()` | 载入已有 collection |
| `exists()` | 判断 collection 是否存在 |
| `query(text, k)` | 单条查询，返回原始 chromadb 响应 |
| `upsert(chunks)` | 按 id 增量更新 |

**设计点**：

- 持久化路径自动创建
- 入库前 `_delete_collection` 避免 id 冲突
- 分批 add（每批 500），避免 sqlite 限制
- `hnsw:space=cosine`，距离越小越相似

### 4.8 retriever.py — 检索

**职责**：调用 VectorStore.query，可选 rerank。

**流程**：

1. 调 `store.query(query, n_results=top_k)`
2. 按距离过滤（`min_score`）
3. 若启用 rerank，用 CrossEncoder 重新打分排序
4. rerank 后取前 `rerank_top_n` 条

**容错**：

- 查询失败 → 返回 `[]`
- 结果空 → 返回 `[]`
- reranker 加载失败 → 降级为原始排序

### 4.9 generator.py — 生成

**职责**：组装 prompt，调 LLM 生成回答。

**Prompt 模板**：

```
你是 PPT 文档助手，严格依据下方参考资料回答问题。
要求：
1. 只使用资料中的信息，不要编造。
2. 资料不足时直接说明，不要猜测。
3. 引用页码，格式：「见第 X 页」。
【参考资料】
{context}
【问题】
{question}
【回答】
```

**边界处理**：

- 无上下文 → 让 LLM 直接拒答（不编造）
- 上下文超长（>12000 字符）→ 截断
- LLM 429 / 503 / 超时 → 重试 3 次，指数退避
- LLM 返回空 → 抛 `GeneratorError`

### 4.10 evaluator.py — 评估

**职责**：LLM-as-judge 三维度评分。

**三维度**：

| 维度 | 评估什么 | Prompt 关键点 |
|---|---|---|
| `faithfulness` | 回答是否忠于资料、有无幻觉 | 给 LLM 资料 + 问题 + 回答，问是否编造 |
| `relevance` | 回答是否切题完整 | 给 LLM 问题 + 回答，问是否完整回答 |
| `context_precision` | 检索片段是否对问题有用 | 给 LLM 问题 + 资料，问资料是否有用 |

**评分机制**：

- 强制 `response_format={"type": "json_object"}`
- 每维度返回 `{"score": 0-10, "reason": "..."}`
- `safe_json_loads` 兜底解析
- 分数越界 / 解析失败 → 标 -1，不阻塞流程

**汇总**：`summarize(results)` 计算各维度平均分（过滤 -1）。

### 4.11 pipeline.py — 编排

**职责**：把各模块串成完整流程，对外提供统一接口。

**核心方法**：

| 方法 | 说明 |
|---|---|
| `build_index()` | 解析 → 切分 → 向量化 → 入库 |
| `load_index()` | 载入已有库 |
| `answer(question, do_eval)` | 单条问答 + 可选评估 |
| `run_eval_set(cases)` | 批量评估 |
| `save_eval_results(results, path)` | 持久化评估结果 |

**设计点**：

- `load_index` 模式不强制要 PPT 文件
- 各阶段独立 try/except，错误聚合到日志
- 评估失败不影响回答返回

### 4.12 main.py — CLI

**职责**：命令行入口，子命令分发。

**子命令**：

- `build` — 从 PPT 建库
- `load` — 载入已有库

**参数覆盖默认值**：所有 `--xxx` 参数都会覆盖 `config.py` 的默认值。

---

## 5. 数据流

### 5.1 离线建库数据流

```
PPT 文件 (path)
    │
    ▼
parser.parse_pptx(path, ocr_cfg)
    │
    │ 提取每页文本/表格/备注/图片(OCR)
    ▼
List[Dict]: [{page, content, source}, ...]
    │
    ▼
splitter.build_chunks(pages, split_cfg)
    │
    │ 递归切分 + 短块合并
    ▼
List[Dict]: [{id, text, page, source}, ...]
    │
    ▼
vectorstore.build(chunks)
    │
    │ embedder 向量化 + chromadb 入库
    ▼
ChromaDB 持久化到 db_dir
```

### 5.2 在线问答数据流

```
用户问题 (str)
    │
    ▼
retriever.retrieve(query)
    │
    │ store.query → 距离过滤 → (可选 rerank)
    ▼
List[Dict]: [{id, text, page, score}, ...]
    │
    ▼
generator.generate(question, contexts)
    │
    │ 组装 prompt → 调 LLM
    ▼
answer (str)
    │
    ▼ (可选)
evaluator.evaluate(question, contexts, answer)
    │
    │ 三维度 LLM-as-judge
    ▼
eval: {faithfulness, relevance, context_precision, ...}
    │
    ▼
最终结果: {question, answer, contexts, eval}
```

### 5.3 核心数据结构

#### Page（解析后）

```python
{
    "page": int,          # 页码（从 1 开始）
    "content": str,       # 该页所有文本（含 OCR 结果）
    "source": str,        # PPT 文件路径
}
```

#### Chunk（切分后）

```python
{
    "id": str,            # 唯一 id，格式 "p{page}_{idx}"
    "text": str,          # chunk 文本
    "page": int,          # 来源页码
    "source": str,        # 来源 PPT 路径
}
```

#### Retrieved Doc（检索后）

```python
{
    "id": str,            # chunk id
    "text": str,          # chunk 文本
    "page": int,          # 来源页码
    "score": float,       # 距离（cosine，越小越相似）
    # rerank 后会多一个 "rerank_score": float
}
```

#### Answer Result（最终结果）

```python
{
    "question": str,
    "answer": str,
    "contexts": List[Retrieved Doc],
    "eval": {                       # 可选
        "faithfulness": int,        # 0-10
        "faithfulness_reason": str,
        "relevance": int,
        "relevance_reason": str,
        "context_precision": int,
        "context_precision_reason": str,
        "retrieval_distances": List[float],
        "retrieved_pages": List[int],
    }
}
```

---

## 6. 边界处理一览

| 模块 | 边界情况 | 处理方式 |
|---|---|---|
| config | overlap >= size | 启动校验抛 `ConfigError` |
| config | top_k <= 0 | 启动校验抛 `ConfigError` |
| config | api_key 缺失 | 启动校验抛 `ConfigError` |
| config | PPT 文件不存在 / 后缀错 / 大小为 0 | 启动校验抛 `ConfigError` |
| utils | 重复 add handler | logger 单例检查 |
| utils | 日志文件写入失败 | warning，继续运行 |
| utils | JSON 解析失败 | 三级兜底：直接解析 → 剥 markdown → 正则抽取 |
| parser | 文件损坏 / 加密 | 抛 `ParseError` |
| parser | 单个 shape 异常 | 跳过该 shape，warning |
| parser | 备注提取失败 | 跳过，warning |
| parser | 图片提取失败 | 跳过，warning |
| parser | 单页过长 | 截断至 20000 字符 |
| parser | 空页 | 跳过 |
| ocr | torch 未装 | warning + 返回空串 |
| ocr | CUDA 不可用 | warning + 回退 CPU |
| ocr | 模型加载失败 | warning + 标记失败不再重试 |
| ocr | 单图 OCR 失败 | warning + 返回空串 |
| ocr | 多图批量失败 | 降级为逐张处理 |
| splitter | 空文本 | 返回 `[]` |
| splitter | 单 piece 超长 | 递归用更细分隔符 |
| splitter | 过短 chunk | 与相邻合并 |
| embedder | 模型加载失败 | 抛 `EmbeddingError` |
| embedder | 空输入 | 返回 `[]` |
| embedder | OpenAI 限流 | 重试 3 次，指数退避 |
| embedder | OpenAI 单批过大 | 自动分批 256 |
| vectorstore | 目录创建失败 | 抛 `VectorStoreError` |
| vectorstore | id 冲突 | 入库前 `_delete_collection` |
| vectorstore | 单批过大 | 分批 500 |
| retriever | 查询失败 | 返回 `[]` |
| retriever | 结果空 | 返回 `[]` |
| retriever | 距离过低 | `min_score` 过滤 |
| retriever | reranker 加载失败 | 降级为原始排序 |
| generator | 无上下文 | prompt 让 LLM 拒答 |
| generator | 上下文超长 | 截断至 12000 字符 |
| generator | LLM 429/503/超时 | 重试 3 次，指数退避 |
| generator | LLM 返回空 | 抛 `GeneratorError` |
| evaluator | JSON 解析失败 | 标 -1，不阻塞 |
| evaluator | 分数越界 | 标 -1 |
| evaluator | 评估集空 | 平均分返回 0 |
| pipeline | load 模式无 PPT | 不强制要 PPT |
| pipeline | 各阶段失败 | 独立 try/except |
| main | 配置错误 | 退出码 1 |

---

## 7. 配置体系

### 7.1 配置层级

```
Config
├── paths: Paths                # 文件路径
├── split: SplitConfig          # 切分参数
├── retrieval: RetrievalConfig  # 检索参数
├── embed: EmbedConfig          # Embedding 后端
├── llm: LLMConfig              # LLM 调用
├── ocr: OCRConfig              # OCR（可选）
└── collection_name: str       # 向量库 collection 名
```

### 7.2 配置来源（优先级从高到低）

1. **CLI 参数**（`--chunk-size` 等）— 最高优先级
2. **环境变量**（`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`）
3. **config.py 默认值**（`DEFAULT` 实例）
4. **代码内硬编码常量**（如 `MAX_CONTEXT_CHARS`）

### 7.3 环境变量

| 变量 | 用途 | 默认值 |
|---|---|---|
| `LLM_API_KEY` | LLM 与 OpenAI Embedding 的 API Key | 必填 |
| `LLM_BASE_URL` | LLM 接口地址 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 生成模型名 | `gpt-4o-mini` |
| `HF_ENDPOINT` | HuggingFace 镜像（加速下载） | 官方源 |

---

## 8. 错误与重试策略

### 8.1 异常分类

| 异常类型 | 何时抛出 | 是否可重试 |
|---|---|---|
| `ConfigError` | 配置不合法 | 否，启动即终止 |
| `ParseError` | PPT 文件级损坏 | 否 |
| `EmbeddingError` | Embedding 模型加载失败 | 否 |
| `VectorStoreError` | 向量库操作失败 | 视情况 |
| `GeneratorError` | LLM 重试后仍失败 | 否 |
| `RetryableError` | 网络临时错误 | 是 |

### 8.2 重试机制

`with_retry` 装饰器：

- **触发条件**：抛出 `RetryableError`
- **退避**：指数退避，`wait = backoff ** attempt`
- **最大次数**：默认 3 次
- **最后失败**：抛出原始异常

**判定为可重试的错误关键词**：`rate` / `timeout` / `connection` / `429` / `503` / `502` / `504`

### 8.3 优雅降级

| 场景 | 降级方式 |
|---|---|
| OCR 模型加载失败 | 跳过 OCR，仅用文本 |
| Reranker 加载失败 | 用原始向量距离排序 |
| OpenAI Embedding 失败 | 抛错（无降级，需用户切换后端） |
| 多图 OCR 批量失败 | 降级为逐张 OCR |
| 评估 LLM 调用失败 | 该维度标 -1，不影响其他维度 |

---

## 9. 性能与资源

### 9.1 内存

- 切分阶段：所有 chunk 加载到内存，10MB PPT 约生成几千 chunk，可控
- Embedding 阶段：分批处理，单批 256
- 入库阶段：分批 add，单批 500

### 9.2 磁盘

- 向量库大小：约 `chunk 数 × (向量维度 × 4 字节 + 文本长度)`
- bge-large-zh：1024 维，1 万 chunk 约 40MB 向量 + 文本
- OCR 临时图：识别后自动清理

### 9.3 GPU

- Embedding（bge-large-zh）：可选 GPU，CPU 也可接受
- OCR（Unlimited-OCR）：强烈建议 GPU，CPU 极慢
- Reranker：可选 GPU

### 9.4 时间估算

| 操作 | CPU | GPU |
|---|---|---|
| 解析 100 页 PPT | < 5s | < 5s |
| 切分 1000 chunk | < 1s | < 1s |
| Embedding 1000 chunk (bge) | ~ 30s | ~ 5s |
| OCR 100 张图 (Unlimited-OCR) | 极慢（不推荐） | ~ 5-10 分钟 |
| 单次检索 | < 100ms | < 100ms |
| LLM 生成 | 1-5s | 1-5s |

---

## 10. 扩展点

### 10.1 接入其他向量库

替换 `vectorstore.py`，实现以下接口：

- `build(chunks)`
- `load()`
- `exists()`
- `query(text, k)`
- `upsert(chunks)`

例如接 Milvus / Qdrant / Pinecone。

### 10.2 接入其他 Embedding

实现 `__call__(input: List[str]) -> List[List[float]]` 协议，在 `embedder.py` 加新后端。

### 10.3 接入其他 LLM

凡是兼容 OpenAI 接口的服务（Qwen / DeepSeek / 通义千问 / Moonshot），改环境变量即可，无需改代码。

非 OpenAI 接口：替换 `generator.py` 的 `_chat` 方法。

### 10.4 接入其他 OCR

替换 `ocr.py` 的 `UnlimitedOCR` 类，实现：

- `ocr_image(path) -> str`
- `ocr_images(paths) -> str`

例如接 PaddleOCR / Tesseract。

### 10.5 增加评估维度

在 `evaluator.py` 加新 prompt 模板 + 新维度字段，例如：

- `completeness` — 回答完整性
- `conciseness` — 回答简洁性
- `citation_accuracy` — 引用准确性

### 10.6 支持多 PPT 批量入库

修改 `pipeline.build_index` 接受 PPT 列表，循环解析 + 切分，统一入库；chunk 的 `source` 字段已支持区分来源。

### 10.7 流式输出

修改 `generator.generate`，把 `stream=True` 传给 OpenAI，逐 token 返回；CLI 端逐字打印。

---

## 11. 设计决策

### 11.1 为什么不用 LangChain？

- **可控性**：每个环节都能精确控制，便于排查问题
- **依赖少**：避免 LangChain 频繁的 breaking change
- **学习成本低**：代码可读，便于二次开发
- **性能**：去除中间层，直接调底层 API

### 11.2 为什么用 chromadb？

- 纯 Python，无系统依赖
- 持久化简单（一个目录）
- 适合单机场景
- 生产环境可换 Milvus / Qdrant

### 11.3 为什么 Embedding 默认用本地 bge？

- 中文效果优于 OpenAI `text-embedding-3-small`
- 免费，无 API 成本
- 数据不出本地，隐私友好
- 1024 维，存储与检索效率平衡

### 11.4 为什么 OCR 用 Unlimited-OCR？

- 高精度文档解析（含表格、图表）
- 支持多页批量
- 中文识别效果好
- MIT 协议，可商用

### 11.5 为什么评估用 LLM-as-judge？

- 无需标注数据
- 可批量评估
- 三维度覆盖检索质量与生成质量
- 缺点：评估成本 = 一次额外 LLM 调用，可按需关闭

### 11.6 为什么 OCR 默认关闭？

- 强依赖 GPU，无 GPU 环境启用会拖慢建库数十倍
- 首次下载模型 2GB+
- 大多数 PPT 图片是装饰性图片，无文字
- 让用户按需开启，避免无谓开销
