# 使用指南

本文档详细说明 PPT RAG 的安装、配置、使用、调参与故障排查。

## 目录

- [1. 环境要求](#1-环境要求)
- [2. 安装](#2-安装)
- [3. 配置](#3-配置)
- [4. 快速开始](#4-快速开始)
- [5. CLI 完整参数](#5-cli-完整参数)
- [6. 常用场景](#6-常用场景)
- [7. Python 库调用](#7-python-库调用)
- [8. 评估功能](#8-评估功能)
- [9. OCR 功能](#9-ocr-功能)
- [10. 调参指南](#10-调参指南)
- [11. 接入其他 LLM](#11-接入其他-llm)
- [12. 故障排查](#12-故障排查)
- [13. 最佳实践](#13-最佳实践)

---

## 1. 环境要求

### 1.1 基础环境（必填）

| 项目 | 要求 |
|---|---|
| Python | 3.9+（推荐 3.10 / 3.11） |
| 操作系统 | Windows / Linux / macOS |
| 磁盘空间 | ≥ 2GB（模型 + 向量库） |
| 内存 | ≥ 4GB |

### 1.2 GPU（可选，OCR 强烈建议）

| 用途 | 是否必须 | CPU 可行性 |
|---|---|---|
| Embedding (bge) | 否 | 可，但慢 |
| Reranker | 否 | 可，但慢 |
| OCR (Unlimited-OCR) | 强烈建议 | 极慢，不推荐 |

### 1.3 网络

- 首次运行需联网下载模型（bge / Unlimited-OCR）
- 调用 OpenAI 兼容 LLM 需要可访问 `LLM_BASE_URL`

---

## 2. 安装

### 2.1 基础安装

```bash
git clone <repo_url>
cd ppt_rag

pip install -r requirements.txt
```

基础依赖：

| 包 | 用途 |
|---|---|
| `python-pptx` | PPT 解析 |
| `chromadb` | 向量库 |
| `sentence-transformers` | 本地 Embedding / Reranker |
| `openai` | OpenAI 兼容 LLM 接口 |

### 2.2 启用 OCR（可选）

```bash
pip install torch torchvision transformers Pillow einops addict easydict pymupdf
```

| 包 | 用途 |
|---|---|
| `torch` / `torchvision` | 深度学习框架 |
| `transformers` | 加载 Unlimited-OCR 模型 |
| `Pillow` | 图像处理 |
| `einops` / `addict` / `easydict` | 模型依赖 |
| `pymupdf` | PDF 转图片（Unlimited-OCR 内部用） |

### 2.3 验证安装

```bash
python -c "from config import Config; print('OK')"
python -c "import pptx, chromadb, sentence_transformers, openai; print('基础依赖 OK')"
python -c "import torch, transformers; print('OCR 依赖 OK')"
```

### 2.4 国内加速（推荐）

设置 HuggingFace 镜像，加速模型下载：

```bash
# Windows
set HF_ENDPOINT=https://hf-mirror.com

# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com
```

可写入 `~/.bashrc` / 系统环境变量永久生效。

---

## 3. 配置

### 3.1 环境变量

通过环境变量配置 LLM（必填）：

```bash
# Windows
set LLM_API_KEY=sk-xxx
set LLM_BASE_URL=https://api.openai.com/v1
set LLM_MODEL=gpt-4o-mini

# Linux / macOS
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

| 变量 | 用途 | 默认值 |
|---|---|---|
| `LLM_API_KEY` | LLM API Key（必填） | — |
| `LLM_BASE_URL` | LLM 接口地址 | `https://api.openai.com/v1` |
| `LLM_MODEL` | 生成模型名 | `gpt-4o-mini` |
| `HF_ENDPOINT` | HuggingFace 镜像 | 官方源 |

### 3.2 配置文件

所有可调参数在 [config.py](file:///./config.py) 集中定义，分 7 个 dataclass：

- `Paths` — 文件路径
- `SplitConfig` — 切分参数
- `RetrievalConfig` — 检索参数
- `LLMConfig` — LLM 调用参数
- `EmbedConfig` — Embedding 后端
- `OCRConfig` — OCR 参数
- `Config` — 顶层聚合

**优先级**：CLI 参数 > 环境变量 > config.py 默认值 > 代码内常量。

详见 [architecture.md 第 7 节](./architecture.md#7-配置体系)。

### 3.3 准备 PPT

把你的 PPT 放到任意位置，例如：

```
./data/your.pptx
```

支持 `.pptx` 和 `.ppt` 后缀。

---

## 4. 快速开始

### 4.1 最简流程（3 步）

```bash
# 1. 设置 API Key
set LLM_API_KEY=sk-xxx

# 2. 建库 + 提问
python main.py build --ppt ./data/your.pptx --question "这份 PPT 的核心结论是什么？"

# 3. 复用已有库继续提问
python main.py load --question "提到了哪些关键指标？"
```

### 4.2 输出示例

```
[2026-07-05 10:00:00] [INFO] [1/4] 初始化 embedder...
[2026-07-05 10:00:01] [INFO] 加载本地 Embedding 模型: BAAI/bge-large-zh-v1.5
[2026-07-05 10:00:10] [INFO] [2/4] 解析 PPT...
[2026-07-05 10:00:11] [INFO] 解析完成: your.pptx 共 20 页，有效页 18 页
[2026-07-05 10:00:11] [INFO] [3/4] 切分...
[2026-07-05 10:00:11] [INFO] 切分完成: 18 页 → 56 个 chunk
[2026-07-05 10:00:11] [INFO] [4/4] 入库...
[2026-07-05 10:00:12] [INFO] 入库完成: collection=ppt_docs 共 56 条
[2026-07-05 10:00:12] [INFO] 索引构建完成
[2026-07-05 10:00:12] [INFO] 问题: 这份 PPT 的核心结论是什么？
[2026-07-05 10:00:12] [INFO] 检索到 4 条上下文
[2026-07-05 10:00:13] [INFO]   [第3页] dist=0.2345 核心结论：本季度营收同比增长 20%...
[2026-07-05 10:00:13] [INFO]   [第5页] dist=0.2890 ...

=== 回答 ===
根据 PPT 内容，本季度的核心结论是：营收同比增长 20%（见第3页）...

=== 评估 ===
{
  "faithfulness": 9,
  "faithfulness_reason": "回答完全基于资料",
  "relevance": 8,
  ...
}
```

### 4.3 子命令

| 子命令 | 用途 |
|---|---|
| `build` | 从 PPT 建库（必须提供 `--ppt`） |
| `load` | 载入已有库（不需 PPT） |

---

## 5. CLI 完整参数

### 5.1 通用参数（build / load 共用）

```
python main.py <build|load> [options]
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--db-dir` | str | `./ppt_vector_db` | 向量库目录 |
| `--collection` | str | `ppt_docs` | collection 名称 |
| `--embed-backend` | `local`/`openai` | `local` | Embedding 后端 |
| `--chunk-size` | int | 500 | chunk 大小（字符） |
| `--chunk-overlap` | int | 50 | chunk 重叠（字符） |
| `--top-k` | int | 4 | 检索 Top-K |
| `--use-reranker` | flag | False | 启用 reranker |
| `--rerank-top-n` | int | 3 | rerank 后保留前 N 条 |
| `--use-ocr` | flag | False | 启用 OCR |
| `--ocr-model` | str | `baidu/Unlimited-OCR` | OCR 模型名 / 路径 |
| `--ocr-device` | `cuda`/`cpu` | `cuda` | OCR 运行设备 |
| `--ocr-image-mode` | `gundam`/`base` | `gundam` | 单图 OCR 模式 |
| `--question` | str | — | 单条问题 |
| `--eval-file` | str | — | 批量评估用例 JSON |
| `--no-eval` | flag | False | 跳过评估 |

### 5.2 build 专属参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--ppt` | 是 | PPT 文件路径 |

### 5.3 参数组合示例

```bash
# 基础建库 + 提问
python main.py build --ppt ./data/your.pptx --question "核心结论是什么？"

# 复用库 + 不评估
python main.py load --question "提到了哪些指标？" --no-eval

# 建库 + 启用 rerank + 启用 OCR
python main.py build \
  --ppt ./data/your.pptx \
  --use-reranker --rerank-top-n 3 \
  --use-ocr --ocr-device cuda \
  --question "图表里展示了什么数据？"

# 自定义切分参数
python main.py build \
  --ppt ./data/your.pptx \
  --chunk-size 800 --chunk-overlap 100 \
  --top-k 6

# 使用 OpenAI Embedding
python main.py build \
  --ppt ./data/your.pptx \
  --embed-backend openai
```

---

## 6. 常用场景

### 6.1 首次建库

```bash
python main.py build --ppt ./data/your.pptx --question "总结一下这份 PPT"
```

### 6.2 复用已有库多次提问

```bash
# 第一次会建库，后续都用 load
python main.py load --question "Q1"
python main.py load --question "Q2"
python main.py load --question "Q3"
```

### 6.3 多个 PPT 分别建库

通过 `--collection` 区分：

```bash
python main.py build --ppt ./data/ppt_a.pptx --collection ppt_a
python main.py build --ppt ./data/ppt_b.pptx --collection ppt_b

# 提问时指定对应 collection
python main.py load --collection ppt_a --question "PPT A 的核心内容"
python main.py load --collection ppt_b --question "PPT B 的核心内容"
```

### 6.4 启用 OCR 处理图片型 PPT

```bash
# 默认 gundam 模式（切图，对图表效果好）
python main.py build --ppt ./data/scanned.pptx --use-ocr --question "图表里的数据"

# 用本地模型路径（避免重复下载）
python main.py build \
  --ppt ./data/scanned.pptx \
  --use-ocr --ocr-model /path/to/Unlimited-OCR
```

启用后，每页的图片会被单独 OCR，识别文字以 `[图片文字]` 标记追加到该页 content 末尾。

### 6.5 启用 rerank 提升检索精度

```bash
python main.py load --use-reranker --rerank-top-n 3 --question "..."
```

### 6.6 批量评估调参效果

```bash
# 准备评估用例
# 见 eval_cases.example.json

# 跑评估
python main.py build --ppt ./data/your.pptx --eval-file eval_cases.json

# 查看结果
# 结果保存到 rag_eval_results.json
```

---

## 7. Python 库调用

### 7.1 基础用法

```python
from config import Config
from pipeline import RAGPipeline

cfg = Config()
cfg.paths.ppt_path = "./data/your.pptx"

pipe = RAGPipeline(cfg)
pipe.build_index()                  # 建库

result = pipe.answer("核心结论是什么？")
print(result["answer"])
print(result["eval"])
print(result["contexts"])           # 检索到的上下文
```

### 7.2 复用已有库

```python
pipe = RAGPipeline(cfg)
pipe.load_index()                   # 载入已有库
result = pipe.answer("提到了哪些指标？")
```

### 7.3 自定义配置

```python
from config import Config
from pipeline import RAGPipeline

cfg = Config()
cfg.paths.ppt_path = "./data/your.pptx"
cfg.paths.db_dir = "./my_db"

# 切分参数
cfg.split.chunk_size = 800
cfg.split.chunk_overlap = 100
cfg.split.min_chunk_len = 30

# 检索参数
cfg.retrieval.top_k = 6
cfg.retrieval.use_reranker = True
cfg.retrieval.rerank_top_n = 3

# LLM 参数
cfg.llm.model = "gpt-4o"
cfg.llm.temperature = 0.1
cfg.llm.max_retries = 5

# Embedding 后端
cfg.embed.backend = "openai"
cfg.embed.openai_model = "text-embedding-3-large"

pipe = RAGPipeline(cfg)
pipe.build_index()
```

### 7.4 启用 OCR

```python
cfg = Config()
cfg.paths.ppt_path = "./data/your.pptx"
cfg.ocr.enabled = True
cfg.ocr.device = "cuda"
cfg.ocr.image_mode = "gundam"
cfg.ocr.model_name = "baidu/Unlimited-OCR"   # 或本地路径
cfg.ocr.min_image_bytes = 2048               # 过小图片跳过

pipe = RAGPipeline(cfg)
pipe.build_index()
```

### 7.5 批量评估

```python
cases = [
    {"question": "核心结论是什么？", "expected": "..."},
    {"question": "提到了哪些指标？", "expected": "..."},
]

results = pipe.run_eval_set(cases)
pipe.save_eval_results(results, "./my_eval.json")

# 末尾会有 _summary 字段
print(results[-1]["_summary"])
```

### 7.6 直接调用底层模块

```python
from parser import parse_pptx
from splitter import build_chunks
from config import Config

cfg = Config()

# 仅解析
pages = parse_pptx("./data/your.pptx")

# 仅切分
chunks = build_chunks(pages, cfg.split)
for c in chunks:
    print(c["id"], c["page"], c["text"][:50])
```

---

## 8. 评估功能

### 8.1 三维度评分

| 维度 | 评估什么 | 分数低时如何改进 |
|---|---|---|
| `faithfulness` | 回答是否忠于资料、有无幻觉 | 缩小 chunk、加强 prompt 约束、加 rerank |
| `relevance` | 回答是否切题完整 | 调整 Top-K、改 prompt |
| `context_precision` | 检索片段是否对问题有用 | 换 embedding、改切分、加 rerank |

分数范围 0-10，由 LLM-as-judge 打分。

### 8.2 单条评估

```bash
python main.py load --question "..." 
# 默认会自动评估
```

### 8.3 批量评估

准备评估用例文件（JSON 数组）：

```json
[
  {
    "question": "核心结论是什么？",
    "expected": "本季度营收同比增长 20%"
  },
  {
    "question": "提到了哪些关键指标？",
    "expected": "营收、利润、用户数"
  },
  {
    "question": "下季度的计划是什么？",
    "expected": ""
  }
]
```

`expected` 字段可选，留空也不影响评估（评估主要看回答与资料的关系）。

执行：

```bash
python main.py build --ppt ./data/your.pptx --eval-file eval_cases.json
```

### 8.4 评估结果结构

结果保存到 `rag_eval_results.json`：

```json
[
  {
    "question": "核心结论是什么？",
    "answer": "本季度营收同比增长 20%（见第3页）",
    "expected": "本季度营收同比增长 20%",
    "contexts": [...],
    "eval": {
      "faithfulness": 9,
      "faithfulness_reason": "回答完全基于资料",
      "relevance": 8,
      "relevance_reason": "切题但略简略",
      "context_precision": 7,
      "context_precision_reason": "Top-1 相关，其他略偏",
      "retrieval_distances": [0.2345, 0.2890, 0.3120, 0.3450],
      "retrieved_pages": [3, 5, 7, 9]
    }
  },
  ...
  {
    "_summary": {
      "faithfulness": 8.5,
      "relevance": 8.0,
      "context_precision": 7.5
    }
  }
]
```

### 8.5 跳过评估

加 `--no-eval`：

```bash
python main.py load --question "..." --no-eval
```

适用于：生产部署、调试回答、节省 LLM 调用成本。

---

## 9. OCR 功能

### 9.1 何时启用

- PPT 含大量截图 / 扫描图
- 图表里有文字数据需要检索
- 流程图、架构图需要被理解

### 9.2 何时不需要

- PPT 主要是文本和表格（`python-pptx` 已能提取）
- 图片都是装饰性图标
- 无 GPU 环境（CPU 极慢）

### 9.3 启用步骤

```bash
# 1. 安装 OCR 依赖
pip install torch torchvision transformers Pillow einops addict easydict pymupdf

# 2. 设置镜像（可选，加速下载）
set HF_ENDPOINT=https://hf-mirror.com

# 3. 启用 OCR 建库
python main.py build --ppt ./data/your.pptx --use-ocr --question "..."
```

### 9.4 模式选择

| 模式 | 说明 | 适用 |
|---|---|---|
| `gundam`（默认） | 切图模式，对图表 / 复杂版面效果好 | 单图、图表、流程图 |
| `base` | 不切图，整图识别 | 多图批量、扫描页 |

```bash
# gundam（推荐用于图表）
python main.py build --ppt ./data/your.pptx --use-ocr --ocr-image-mode gundam

# base（多图默认走这个）
python main.py build --ppt ./data/your.pptx --use-ocr --ocr-image-mode base
```

### 9.5 使用本地模型路径

避免每次下载：

```bash
# 提前下载模型
git clone https://hf-mirror.com/baidu/Unlimited-OCR ./models/Unlimited-OCR

# 指定本地路径
python main.py build --ppt ./data/your.pptx --use-ocr --ocr-model ./models/Unlimited-OCR
```

### 9.6 调参

```python
cfg.ocr.min_image_bytes = 4096    # 提高阈值，跳过更多小图
cfg.ocr.max_length = 16384       # 减少最大生成长度，加速
cfg.ocr.image_size_gundam = 512  # 调整图片尺寸
```

### 9.7 OCR 限制

- 需要 NVIDIA GPU（CPU 模式不推荐）
- 首次下载模型约 2GB+
- 无 GPU / 未装 torch 时自动跳过，不阻断流程
- 识别后图片临时文件会自动清理

---

## 10. 调参指南

### 10.1 按现象调参

| 现象 | 调整方向 |
|---|---|
| 检索不到相关内容 | 增大 `--top-k`；换更好的 embedding；缩小 `--chunk-size` |
| 回答出现幻觉 | 缩小 `--chunk-size`；加强 prompt 约束；启用 `--use-reranker` |
| 回答太短不完整 | 增大 `--chunk-size`；增大 `--top-k`；调高 LLM temperature |
| 上下文超长报错 | 减小 `--top-k`；调小 `MAX_CONTEXT_CHARS` |
| 评估分数整体偏低 | 先看 `context_precision`，低则换 embedding / 加 rerank；再看 `faithfulness`，低则改 prompt |
| 建库慢 | 用 `--embed-backend openai`；或缩小 PPT；或换更小的 embedding 模型 |
| OCR 慢 | 必须用 GPU；调小 `max_length`；跳过小图 |

### 10.2 关键参数说明

#### chunk_size（默认 500）

- 太小（< 200）：上下文断裂，回答不完整
- 太大（> 1500）：检索精度下降，容易夹带无关内容
- 中文推荐 400-800

#### chunk_overlap（默认 50）

- 一般取 chunk_size 的 10%
- 太大：信息冗余，向量库膨胀
- 太小：上下文断裂

#### top_k（默认 4）

- 太小（1-2）：上下文不足，回答片面
- 太大（> 8）：噪声多，容易幻觉
- 推荐先试 4-6

#### min_score（默认 0.0）

- chromadb cosine 距离，越小越相似
- 设为 0.5 表示距离 > 0.5 的丢弃
- 用于过滤明显不相关的结果

### 10.3 调参工作流

```bash
# 1. 用默认参数建库 + 跑评估
python main.py build --ppt ./data/your.pptx --eval-file eval_cases.json
# 记录 _summary 分数

# 2. 调整 chunk_size，重新建库
python main.py build --ppt ./data/your.pptx --eval-file eval_cases.json --chunk-size 800
# 对比 _summary 分数

# 3. 启用 rerank
python main.py load --eval-file eval_cases.json --use-reranker --rerank-top-n 3
# 对比 _summary 分数

# 4. 选择分数最高的组合
```

---

## 11. 接入其他 LLM

### 11.1 OpenAI 官方

```bash
set LLM_API_KEY=sk-xxx
set LLM_BASE_URL=https://api.openai.com/v1
set LLM_MODEL=gpt-4o-mini
```

### 11.2 通义千问

```bash
set LLM_API_KEY=sk-xxx
set LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set LLM_MODEL=qwen-plus
```

### 11.3 DeepSeek

```bash
set LLM_API_KEY=sk-xxx
set LLM_BASE_URL=https://api.deepseek.com/v1
set LLM_MODEL=deepseek-chat
```

### 11.4 Moonshot (Kimi)

```bash
set LLM_API_KEY=sk-xxx
set LLM_BASE_URL=https://api.moonshot.cn/v1
set LLM_MODEL=moonshot-v1-8k
```

### 11.5 本地部署 (vLLM / Ollama)

```bash
# vLLM
set LLM_API_KEY=EMPTY
set LLM_BASE_URL=http://localhost:8000/v1
set LLM_MODEL=your-model-name

# Ollama（需启用 OpenAI 兼容接口）
set LLM_API_KEY=ollama
set LLM_BASE_URL=http://localhost:11434/v1
set LLM_MODEL=llama3
```

凡是兼容 OpenAI 接口的服务都直接可用，无需改代码。

---

## 12. 故障排查

### 12.1 PPT 里的图片文字提取不到

`python-pptx` 只能提取文本、表格、备注，无法识别图片里的文字。需启用 OCR：

```bash
pip install torch torchvision transformers Pillow einops addict easydict pymupdf
python main.py build --ppt ./data/your.pptx --use-ocr --question "图表里展示了什么？"
```

详见 [OCR 功能](#9-ocr-功能)。

### 12.2 中文检索效果不好

- 确认 Embedding 用的是中文模型（默认 `BAAI/bge-large-zh-v1.5` 已是中文优化）
- 切分时优先用中文标点分隔符（默认已配置）
- 加 reranker：`--use-reranker --rerank-top-n 3`
- 调小 `--chunk-size` 提升检索精度

### 12.3 模型首次加载慢

`sentence-transformers` 首次会从 HuggingFace 下载模型（约 1-2 GB）：

```bash
# 设置镜像
set HF_ENDPOINT=https://hf-mirror.com

# 或提前手动下载
git clone https://hf-mirror.com/BAAI/bge-large-zh-v1.5 ./models/bge
# 然后修改 config.py 的 local_model
```

### 12.4 ChromaDB 报 sqlite 版本错误

```bash
pip install pysqlite3-binary
```

或升级 Python 到 3.10+。

### 12.5 LLM 调用失败

排查清单：

1. `LLM_API_KEY` 是否设置正确
2. `LLM_BASE_URL` 是否可达（`curl $LLM_BASE_URL/models`）
3. 网络是否能访问 LLM 服务（公司代理？）
4. 是否被限流（看错误信息有没有 429）
5. `LLM_MODEL` 名称是否正确

### 12.6 OCR 模型加载失败

排查清单：

1. `pip install torch torchvision transformers` 是否装全
2. CUDA 是否可用：`python -c "import torch; print(torch.cuda.is_available())"`
3. 模型是否能下载：`git clone https://hf-mirror.com/baidu/Unlimited-OCR`
4. 显存是否足够（至少 4GB）

### 12.7 向量库载入失败

```bash
# 检查 collection 是否存在
python -c "import chromadb; c=chromadb.PersistentClient('./ppt_vector_db'); print([x.name for x in c.list_collections()])"

# 不存在则重新建库
python main.py build --ppt ./data/your.pptx
```

### 12.8 评估分数都是 -1

说明 LLM 评估调用失败：

1. 检查 LLM API 是否正常
2. 检查 LLM 是否支持 `response_format={"type": "json_object"}`
3. 不支持的话改用 `gpt-4o-mini` / `qwen-plus` 等支持的模型

### 12.9 建库时报 ConfigError

启动校验失败，检查：

- `LLM_API_KEY` 是否设置
- PPT 文件是否存在
- `chunk_overlap` 是否 < `chunk_size`
- `top_k` 是否 > 0

---

## 13. 最佳实践

### 13.1 建库

- 同一 PPT 反复提问时，第一次 build，后续都 load
- 改了切分 / embedding 参数后必须重新 build
- 不同 PPT 用不同 `--collection` 区分

### 13.2 提问

- 问题尽量具体，避免「这个 PPT 讲了什么」之类的宽泛问题
- 涉及具体数据时，明确指向（如「第 5 页的图表数据是什么」）
- 需要引用页码时，prompt 已自动要求 LLM 标注「见第 X 页」

### 13.3 评估

- 评估用例覆盖核心问题（5-10 条即可）
- 调参时跑同一份评估集对比分数
- 关注 `context_precision` —— 它低说明检索本身有问题，先改检索

### 13.4 性能

- 大 PPT（> 50 页）建议用 OpenAI Embedding（GPU 不可用时）
- OCR 必须用 GPU
- 生产环境关闭评估（`--no-eval`）

### 13.5 成本

- 本地 Embedding + 本地 Reranker：免费
- OpenAI Embedding：按 token 计费（约 $0.02 / 1M tokens）
- LLM 生成：按 token 计费
- 评估：每次问答多 3 次 LLM 调用

### 13.6 数据安全

- 本地 Embedding / OCR：数据不出本机
- OpenAI Embedding / LLM：数据会上传到服务商
- 涉密 PPT 建议全本地：`--embed-backend local` + 本地 LLM（vLLM）

---

## 附录

- [架构说明](./architecture.md)
- [入口文档](./README.md)
- [评估用例示例](./eval_cases.example.json)
