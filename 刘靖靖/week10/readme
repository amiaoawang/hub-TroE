# PPT RAG

一个不依赖 LangChain 的 PPT 问答 RAG 流程：解析 PPT → 切分 → 向量化 → 入库 → 检索 → 生成 → 评估。

支持文本 / 表格 / 备注 / **图片 OCR**（基于 [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)），并自带 **LLM-as-judge 三维度质量评估**。

## 文档导航

| 文档 | 内容 |
|---|---|
| [使用指南](./usage_guide.md) | 安装、配置、CLI 参数、Python 调用、调参、故障排查 |
| [架构说明](./architecture.md) | 设计原则、模块详解、数据流、边界处理、扩展点 |

## 文件结构

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
├── README.md                  # 本文件（入口）
├── architecture.md            # 架构说明
└── usage_guide.md             # 使用指南
```

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

详见 [使用指南 - 安装](./usage_guide.md#2-安装)。

### 2. 配置 LLM

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

接 Qwen / DeepSeek / 通义千问等兼容 OpenAI 接口的服务，改 `LLM_BASE_URL` 即可。详见 [使用指南 - 接入其他 LLM](./usage_guide.md#11-接入其他-llm)。

### 3. 建库 + 提问

```bash
python main.py build --ppt ./data/your.pptx --question "这份 PPT 的核心结论是什么？"
```

### 4. 复用已有库

```bash
python main.py load --question "提到了哪些关键指标？"
```

## 核心能力

| 能力 | 说明 |
|---|---|
| PPT 解析 | 文本 / 表格 / 备注 / 组合形状 |
| 图片 OCR | 基于 Unlimited-OCR，懒加载、优雅降级 |
| 递归切分 | 中文标点优先级切分，短块合并 |
| 双 Embedding 后端 | 本地 bge-large-zh / OpenAI |
| 可选 Rerank | CrossEncoder 提升检索精度 |
| LLM 生成 | OpenAI 兼容接口，含重试 |
| 三维度评估 | faithfulness / relevance / context_precision |

## 常用命令

```bash
# 建库 + 单条提问
python main.py build --ppt ./data/your.pptx --question "..."

# 复用库 + 不评估（生产环境）
python main.py load --question "..." --no-eval

# 启用 OCR + Rerank
python main.py build --ppt ./data/your.pptx \
  --use-ocr --ocr-device cuda \
  --use-reranker --rerank-top-n 3 \
  --question "..."

# 批量评估
python main.py build --ppt ./data/your.pptx --eval-file eval_cases.json
```

完整参数见 [使用指南 - CLI 完整参数](./usage_guide.md#5-cli-完整参数)。

## 评估维度

| 维度 | 评估什么 |
|---|---|
| `faithfulness` | 回答是否忠于资料、有无幻觉 |
| `relevance` | 回答是否切题完整 |
| `context_precision` | 检索片段是否对问题有用 |

详见 [使用指南 - 评估功能](./usage_guide.md#8-评估功能)。

## 设计原则

- **零框架依赖** — 不使用 LangChain，所有逻辑用标准库 + 最小依赖实现
- **模块单一职责** — 每个文件负责一个环节
- **配置与代码分离** — 所有参数集中在 `config.py`
- **优雅降级** — OCR / rerank / OpenAI embedding 失败都不阻断主流程
- **边界处理优先** — 对文件损坏、空输入、超长文本、网络异常都显式处理

详见 [架构说明 - 设计原则](./architecture.md#11-设计原则)。

## 常见问题

| 问题 | 解决方案 |
|---|---|
| PPT 里的图片文字提取不到 | [启用 OCR](./usage_guide.md#9-ocr-功能) |
| 中文检索效果不好 | 加 `--use-reranker --rerank-top-n 3` |
| 模型首次加载慢 | `set HF_ENDPOINT=https://hf-mirror.com` |
| ChromaDB sqlite 报错 | `pip install pysqlite3-binary` |
| 接入其他 LLM | [见使用指南](./usage_guide.md#11-接入其他-llm) |
| 评估分数都是 -1 | LLM 不支持 `response_format`，换模型 |

完整故障排查见 [使用指南 - 故障排查](./usage_guide.md#12-故障排查)。

## License

MIT
