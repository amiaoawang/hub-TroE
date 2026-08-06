---
name: learning-material-generator
description: 从演示文稿（PPTX/PDF）生成学习材料：将 PPTX 拆分为每页一个 PDF、从 PDF 提取文本，并基于文档内容制作交互式知识问答网站。当用户提出"把 PPT 拆成一页页的 PDF"、"根据 PDF/PPT 内容出题"、"做一个问答题网站"、"根据课件做测验"等需求时使用。支持四类题型（单选/判断/多选/填空），自动打散答案分布，所有题目与解析均标注原文出处。
agent_created: true
---

# Learning Material Generator

## Overview

本技能把"课件 → 学习材料"的完整流程固化为可复用工作流：

1. **PPTX → 单页 PDF**：用 PowerPoint COM（pywin32）导出完整 PDF，再用 pypdf 拆成每页一个 PDF。
2. **PDF → 文本**：提取每页文本，作为出题素材。
3. **PDF 内容 → 知识问答网站**：基于文档内容设计题库，生成自包含的单页 HTML 交互问答站。

## 工作流决策树

- 用户给的是 **PPTX**，要拆成单页 PDF → 先 `scripts/pptx_to_pdf.py` 转完整 PDF，再 `scripts/split_pdf_pages.py` 拆分。
- 用户给的是 **PDF**，要拆成单页 PDF → 直接 `scripts/split_pdf_pages.py`。
- 用户要**提取文本** / 准备出题素材 → `scripts/extract_pdf_text.py`。
- 用户要根据内容**做问答题网站** → 提取文本后设计题库 JSON，用 `scripts/build_quiz_site.js` + `assets/quiz-template.html` 一键生成站点（无需手写 HTML）。

## Step 1: PPTX → 单页 PDF

### 1.1 准备 Python 环境

Windows 下在项目目录创建虚拟环境（命名按 Python 版本，如 py313），安装依赖：

```bash
# <managed-python> -m venv <project>/py313
# <project>/py313/Scripts/python.exe -m pip install pypdf pywin32
```

注意：PowerShell 5.1 直接调用 `ExportAsFixedFormat` 会因 COM 枚举参数绑定问题报 `DISP_E_TYPEMISMATCH`，务必用 Python pywin32 而不是 PowerShell。

### 1.2 导出完整 PDF（多方案自动降级）

```bash
python scripts/pptx_to_pdf.py <input.pptx> [output.pdf]
```

脚本按以下顺序自动检测可用方案，命中即用：
1. **Microsoft PowerPoint COM**（win32com，无窗口运行，保真度最高）
2. **WPS 演示 COM**（KWPP.Application，同样无窗口）
3. **LibreOffice headless**（soffice --convert-to pdf，自动定位可执行文件）

三种方案都不可用时输出明确的安装建议。导出后用 `pres.Slides.Count` 报告页数，供后续校验。

### 1.3 拆分为单页 PDF

```bash
python scripts/split_pdf_pages.py <full.pdf> [output_dir]
```

- 输出到 `output_dir`（默认 `<输入名>_单页PDF/`），命名 `第01页.pdf`、`第02页.pdf`…（按总页数补零）。
- 拆分后用 pypdf 逐文件校验 `len(pages)==1`。

## Step 2: PDF → 文本提取（出题素材）

```bash
python scripts/extract_pdf_text.py <input.pdf> [output.txt]
```

- 按页输出 `===== PAGE N =====` 分隔，便于定位知识点所在页码。
- 读完全文后再设计题目，确保每题的解析都标注真实出处页码。

## Step 3: PDF 内容 → 知识问答网站（模板化构建）

### 3.1 设计题库 JSON

题库可以是纯数组 `[{cat,type,q,opts,ans,answers?,exp}, ...]`，或带站点配置的对象：

```json
{
  "title": "站点标题",
  "badge": "徽章文案",
  "subtitle": "副标题",
  "footer": "页脚",
  "cats": [{"key":"chapter1","name":"第一章","color":"#2563eb"}],
  "questions": [
    {"cat":"chapter1","type":"single","q":"问题？","opts":["A","B","C","D"],"ans":0,"exp":"PPT 第X页 解析"},
    {"cat":"chapter1","type":"judge","q":"判断……","opts":["正确","错误"],"ans":1,"exp":"PPT 第X页 解析"},
    {"cat":"chapter1","type":"multi","q":"……？（多选）","opts":["A","B","C","D"],"ans":[0,1,2],"exp":"PPT 第X页 解析"},
    {"cat":"chapter1","type":"fill","q":"……是____。","answers":["90","九十"],"exp":"PPT 第X页 解析"}
  ]
}
```

题型规则（与 `references/quiz-site-design.md` 一致）：
- `single`/`judge`：`ans` 为正确选项索引；`judge` 的 opts 固定 `["正确","错误"]`。
- `multi`：`ans` 为正确索引数组（≥2 项）。
- `fill`：`answers` 为可接受答案数组（含数字/中文/带单位写法），无 `opts`。
- `match`：连线匹配题，`left`/`right` 为左右两列文本（等长），`pairs` 为 `[leftIdx, rightIdx]` 正确配对数组；无 `opts`/`ans`。
- `diff`（可选）：难度 `easy` / `medium` / `hard`，缺省视为 `medium`；用于难度筛选与随机组卷。
- 每题 `exp` 必须含 "PPT 第X页" 出处；`cat` 必须对应 `cats` 中的 key（缺失时自动补默认色）。

```json
{ "cat":"evolve", "type":"match", "diff":"medium",
  "q":"将 GEPA 概念与含义连线（PPT 第17页）",
  "left":["基因","突变"], "right":["LLM 改写指令","SKILL.md 文本"],
  "pairs":[[0,1],[1,0]], "exp":"PPT 第17页 …" }
```

### 3.2 一键构建站点

```bash
node scripts/build_quiz_site.js <questions.json> [output.html] [--check-dist N]
```

脚本自动完成：
1. **题库校验**：type 合法性、ans 索引越界、fill 缺 answers、exp 缺页码等，失败即报错退出。
2. **答案分布模拟**（默认 100 次洗牌）：统计单选答案落在 A/B/C/D 的比例，任一位置占比 >40% 判失败——杜绝"答案集中"。
3. **注入模板**：把题库 JSON 注入 `assets/quiz-template.html` 的 `__QUESTIONS__`/`__CATS__` 占位符，HTML 字段自动转义防破坏结构。
4. **JS 语法校验**：`new Function()` 检查生成结果，通过后写出最终 HTML。

输出为自包含单文件 HTML（无外部依赖），含四类题型交互、章节筛选、只看错题、打乱顺序、进度统计、完成总结、历史成绩、错题本持久化、题库导入/导出。

可选参数：`--template <path>` 指定自定义模板；`--cat key:name:color` 追加章节（可多次）；`--check-dist 0` 跳过分布模拟。

### 3.3 题库与 HTML 分离 + 导入导出

- 题库以独立 JSON 文件管理（`questions.json`），HTML 仅由构建脚本生成——同一题库可生成多套站点，改题不改模板。
- 站点内置 **📤 导出题库 / 📥 导入题库**：可下载当前题库 JSON 编辑后再导入，也可直接换一套题库到同一个 HTML 文件（导入时自动校验并重新打乱选项）。
- 题库 JSON 格式见 3.1；导入支持纯数组或 `{cats, questions}` 对象两种结构。

### 3.4 答题记录持久化（localStorage）

站点自动把以下状态保存在浏览器 localStorage（key 含题量，题库变化自动隔离）：
- **作答记录**：刷新/关闭页面后恢复已答状态与结果。
- **跨会话错题本**：答错的题自动记录，「只看错题」模式基于错题本而非当前会话，可跨天复习。
- **历史成绩**：每完成一轮自动记录（时间 + 正确率，最多 20 条），折叠面板可视化展示。
- **🗑 清空记录**：一键清除错题本与历史成绩（不影响当前作答）。
- 注意：localStorage 以浏览器/站点维度隔离，换浏览器或清除浏览器数据会丢失记录。

### 3.5 一键校验脚本

```bash
node scripts/validate_quiz.js <questions.json | site.html> [--check-dist N] [--check-syntax]
```

独立校验脚本，支持两种输入：
- **JSON 题库**：结构校验（type/ans 索引/pairs 配对/页码/fill answers）+ 章节映射完整性 + 答案分布模拟 + 题干重复检测。
- **生成的 HTML**：自动抽取 QUESTIONS/CATS，额外支持 `--check-syntax` 做整段 JS 语法检查。

退出码 0=通过、1=失败；也可在 CI 或构建流程中作为门禁使用。

### 3.6 OCR 兜底（文本层缺失）

当 `extract_pdf_text.py` 提取文本为空或极少（设计类 PPT 文字多为图片）时：

```bash
python scripts/ocr_pdf_text.py <input.pdf> [output.txt] [--engine rapidocr|paddle|auto]
```

- 用 PyMuPDF 渲染每页为 PNG（2x），再由 OCR 引擎识别。
- 引擎自动降级：`rapidocr-onnxruntime`（推荐，纯 pip）→ `paddleocr`（更重）。
- 渲染图存到 `<输入名>_ocr_pages/` 便于人工核对。

### 3.7 难度分级 + 随机组卷

- 题目可选 `diff: easy|medium|hard` 字段；站点顶部有难度筛选 tab（🟢🟡🔴）。
- **📝 随机组卷**：输入题数 + 难度，从题库随机抽题生成一套试卷（Fisher-Yates 抽样），独立进度/成绩统计，可随时「退出组卷」恢复全量题库。

### 3.8 题库设计要点

- **答案分散**：模板在页面加载时用 Fisher-Yates 打乱单选/多选选项（`shuffleOpts`），正确位置天然随机，JSON 里 `ans` 写原始索引即可。
- **章节覆盖**：按文档章节定义 `cats`，题目覆盖全部章节。
- **题型搭配**：四类题型混合（如 71 单选 / 8 判断 / 10 多选 / 12 填空），判断题选正误各半。

## Resources

### scripts/

| 脚本 | 用途 |
|------|------|
| `pptx_to_pdf.py` | PPTX → PDF，自动降级链：PowerPoint COM → WPS COM → LibreOffice headless |
| `split_pdf_pages.py` | 按页拆分 PDF 为单页文件 |
| `extract_pdf_text.py` | 按页提取 PDF 文本，标注页码 |
| `ocr_pdf_text.py` | OCR 兜底：渲染 PDF 为图片 + RapidOCR/PaddleOCR 识别 |
| `build_quiz_site.js` | 题库 JSON + 模板 → 问答站 HTML（内置校验：结构/答案分布/JS 语法） |
| `validate_quiz.js` | 独立一键校验：JSON 题库或已生成 HTML（结构/章节/分布/语法/重复题干） |

### references/

| 文档 | 用途 |
|------|------|
| `quiz-site-design.md` | 问答网站题库结构、题目数据格式、交互逻辑、校验方法、导入导出与持久化 |

### assets/

| 文件 | 用途 |
|------|------|
| `quiz-template.html` | 问答站通用模板，`__QUESTIONS__`/`__CATS__`/`__TITLE__` 等占位符由 build 脚本注入 |
