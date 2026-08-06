# 知识问答网站设计与题库规范

本文档沉淀了"根据 PDF/PPT 内容制作交互式知识问答网站"的完整设计规范，来源为一次真实实践（HERMES Agent 29 页 PPT → 101 题问答站）。

## 1. 总体流程

1. `extract_pdf_text.py` 提取全文 → 通读全文。
2. 按文档章节定义 `CATS`（章节名 + 主题色），覆盖全部内容。
3. 设计题目：每题的解析都必须能指出真实出处页码（`exp` 含 "PPT 第 X 页"）。
4. 生成自包含单页 HTML（无外部依赖，CSS/JS 内嵌）。
5. 用 Node 校验：JS 语法、题数、题型分布、答案索引合法性、页码引用完整性、答案分布均匀性。

## 2. 题目数据格式

```js
{cat:"overview", type:"single", q:"问题文本",
 opts:["选项A","选项B","选项C","选项D"], ans:0,   // single: ans=正确索引
 exp:"解析（含 PPT 第X页 出处）"},

{cat:"evolve", type:"judge", q:"判断：……",
 opts:["正确","错误"], ans:1, exp:"……"},

{cat:"core", type:"multi", q:"……？（多选）",
 opts:["A","B","C","D"], ans:[0,1,2], exp:"……"},  // multi: ans=正确索引数组

{cat:"core", type:"fill", q:"……是 ____。",
 answers:["90","九十"], exp:"……"},                  // fill: answers=可接受答案数组

{cat:"evolve", type:"match", diff:"medium", q:"将……正确连线",
 left:["基因","突变"], right:["LLM 改写指令","SKILL.md 文本"],
 pairs:[[0,1],[1,0]], exp:"……"},                     // match: left/right 等长，pairs=正确配对
```

- `single` / `judge`：点击选项立即判分。
- `multi`：点击切换选中 → "提交答案"按钮判分；`ans` 为索引数组。
- `fill`：输入框 + 回车/提交；`answers` 为可接受答案（含数字、中文数字、带单位写法）。
- `match`：连线题——先点左侧一项，再点右侧对应项完成连线（显示配对编号），全部连完点"提交答案"；`pairs` 为 `[leftIdx, rightIdx]` 数组，判分要求全部配对正确。
- `diff`（可选）：`easy` / `medium` / `hard`，显示难度徽章、支持难度筛选与随机组卷。

## 3. 答案分散（关键）

**问题**：手写题库时正确答案容易全部落在 A，需自动打散。

**方案**：页面加载时对 single/multi 用 Fisher-Yates 洗牌重排选项并重映射答案：

```js
function shuffleOpts(q){
  if(q.type !== "single" && q.type !== "multi") return;
  const n = q.opts.length;
  const perm = Array.from({length:n},(_,i)=>i);
  for(let i=n-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [perm[i],perm[j]]=[perm[j],perm[i]]; }
  const oldOpts = q.opts.slice();
  q.opts = perm.map(p=>oldOpts[p]);
  if(q.type === "multi") q.ans = q.ans.map(i=>perm.indexOf(i)).sort((a,b)=>a-b);
  else q.ans = perm.indexOf(q.ans);
}
```

judge（正确/错误）与 fill 不参与打乱。

**验证**：Node 脚本克隆题库 300 次打乱，统计 single 答案位置分布，A/B/C/D 应各 ~25%（300 次 × 71 题 ≈ 各 5300 次）。

## 4. 判分逻辑

```js
function isRight(id){
  const q = QUESTIONS[id];
  if(answers[id] === undefined) return false;
  if(q.type === "multi"){
    const a = answers[id], b = q.ans;
    return a.length === b.length && a.every(v=>b.includes(v));
  }
  if(q.type === "fill"){
    const v = norm(answers[id]);
    if(!v) return false;
    return q.answers.some(k => { const nk = norm(k); return v===nk || v.includes(nk) || nk.includes(v); });
  }
  return answers[id] === q.ans;
}
function norm(s){ return String(s||"").toLowerCase().replace(/[\s,，。.、%％]/g,""); }
```

norm 归一化容忍 "11,487" 与 "11487"、"90" 与 "九十" 等写法。

## 5. 状态管理

```js
let order = QUESTIONS.map((_,i)=>i);   // 展示顺序（打乱顺序功能）
let answers = {};                      // id -> 提交内容（single:idx, multi:数组, fill:字符串, match:配对对象）
let pendingMulti = {};                 // id -> 已选集合（多选未提交前）
let pendingMatch = {};                 // id -> {leftIdx: rightIdx} 连线未提交前
let matchSel = {};                     // id -> 当前选中的左侧项
let curCat = "all";                    // 章节过滤
let curDiff = "all";                   // 难度过滤
let onlyWrongMode = false;             // 只看错题
let wrongBook = {};                    // id -> true：跨会话错题本（持久化）
let history = [];                      // [{time, right, total}]：历史成绩（持久化）
let paperMode = false;                 // 随机组卷模式
let paperOrder = [];                   // 组卷题目序列
```

## 6. 网站功能清单

- 头部：标题、副标题、数据徽章（从文档提取的关键数字）。
- 进度卡：已作答/总数、正确数、待答数、进度条。
- 章节 tab 筛选（全部 + 各章节，带题目数徽标）。
- 难度 tab 筛选（🟢 简单 / 🟡 中等 / 🔴 困难）。
- 题目卡片：题号、章节色标签、题型徽章（单选/判断/多选/填空/连线）、难度徽章、题干、选项/连线区、判分高亮（绿✓/红✗）、解析框（含出处页码）。
- 操作区：全部重做 / 只看错题 / 打乱顺序 / 导出题库 / 导入题库 / 清空记录 / 随机组卷。
- 随机组卷：输入题数 + 难度 → Fisher-Yates 抽样生成试卷，独立进度统计，可退出恢复全量。
- 完成总结：正确率 + 分档评语（100% / ≥80% / ≥60% / 其他）。
- 历史成绩：折叠面板，每轮完成自动记录（时间 + 正确率条 + 百分比，最多 20 条）。

## 6.1 题库与 HTML 分离

- 题库存独立 JSON（纯数组 或 `{title,badge,subtitle,footer,cats,questions}` 对象）。
- `build_quiz_site.js` 注入模板生成 HTML；站点内也可「📤 导出题库」下载 JSON、「📥 导入题库」换题（自动校验 + 重新打乱）。
- 导入校验：每题必须有 q/cat/type/exp；single/judge 的 ans 索引越界检查；multi ans 合法数组；fill 需 answers。

## 6.2 答题记录持久化（localStorage）

- 存储键 `lmq_state_<题量>`（题库数量变化自动隔离旧记录）。
- 持久化内容：answers（刷新恢复已答）、wrongBook（跨会话错题本）、history（历史成绩）、curCat。
- 「只看错题」基于 wrongBook 而非当前会话；答错即入错题本；完成一轮自动记一条历史（与末条相同不重复）。
- 「全部重做」只清作答，「🗑 清空记录」清除错题本+历史。
- 初始化时清理已不存在的题目 id（题库变更后的脏数据）。

## 7. Node 校验脚本要点

```js
// 语法：new Function(scriptContent) 抛异常即语法错误
// 结构：正则提取 QUESTIONS 数组 eval 后逐题检查
//   single/judge: typeof ans === 'number' 且 0<=ans<opts.length
//   multi: Array.isArray(ans) 且 2<=ans.length<=opts.length
//   fill: Array.isArray(answers) 且非空，且不应有 opts
//   exp: 必须包含 "PPT 第"（页码引用）
// 分布：克隆题库打乱 300 次，统计 single 答案落在 A/B/C/D 的次数
```

## 8. 环境经验

- Windows 下 Office COM（PowerPoint）导出 PDF 用 Python pywin32，勿用 PowerShell 5.1（DISP_E_TYPEMISMATCH）。
- cscript/wscript 被安全策略拦截时不可用作 COM 宿主。
- 虚拟环境命名按 Python 版本（py313 对应 3.13），依赖装进 venv 不污染全局。
