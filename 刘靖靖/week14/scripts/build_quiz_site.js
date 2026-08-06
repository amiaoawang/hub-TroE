#!/usr/bin/env node
/**
 * 从题库 JSON + 问答站模板生成最终 HTML 站点，并自动校验。
 *
 * 用法:
 *   node build_quiz_site.js <questions.json> [output.html] [options]
 *
 * 参数:
 *   questions.json   题库文件。支持两种结构：
 *                     A) 纯数组: [{cat,type,q,opts,ans,answers?,exp}, ...]
 *                     B) 对象: { title, badge, subtitle, footer, cats, questions }
 *   output.html      输出路径（默认 ./quiz-site.html）
 *
 * 选项:
 *   --template <path>  指定模板（默认本技能 assets/quiz-template.html）
 *   --cat <key:name:color>  追加章节映射，可多次指定
 *   --check-dist N     额外做 N 次洗牌模拟，校验答案分布（默认 100）
 *
 * 退出码: 0=成功, 1=校验失败/参数错误
 */
const fs = require('fs');
const path = require('path');

// ---------------- 参数解析 ----------------
const args = process.argv.slice(2);
if (args.length < 1) {
  console.error('用法: node build_quiz_site.js <questions.json> [output.html] [--template <path>] [--cat key:name:color] [--check-dist N]');
  process.exit(1);
}
const questionsFile = path.resolve(args[0]);
let outFile = path.resolve(args[1] || './quiz-site.html');
const opts = { template: null, cats: [], checkDist: 100 };
for (let i = 2; i < args.length; i++) {
  const a = args[i];
  if (a === '--template') { opts.template = path.resolve(args[++i]); }
  else if (a === '--cat') {
    const parts = args[++i].split(':');
    if (parts.length >= 2) opts.cats.push({ key: parts[0], name: parts[1], color: parts[2] || '#64748b' });
  }
  else if (a === '--check-dist') { opts.checkDist = parseInt(args[++i], 10); }
  else if (a.endsWith('.html')) { outFile = path.resolve(a); }
}

// ---------------- 模板定位 ----------------
function findTemplate() {
  if (opts.template && fs.existsSync(opts.template)) return opts.template;
  // 本技能目录
  const candidates = [
    path.join(__dirname, '..', 'assets', 'quiz-template.html'),
    path.join(__dirname, 'assets', 'quiz-template.html'),
    path.join(__dirname, 'quiz-template.html'),
  ];
  for (const c of candidates) if (fs.existsSync(c)) return c;
  return null;
}

// ---------------- 题库校验 ----------------
const DEFAULT_COLORS = ['#2563eb', '#7c3aed', '#0891b2', '#059669', '#d97706', '#dc2626', '#db2777', '#4f46e5'];
const VALID_TYPES = new Set(['single', 'judge', 'multi', 'fill', 'match']);
const VALID_DIFFS = new Set(['easy', 'medium', 'hard']);

function validateQuestions(questions) {
  const errors = [];
  questions.forEach((q, i) => {
    if (!q || typeof q !== 'object') { errors.push(`#${i}: 不是对象`); return; }
    if (!VALID_TYPES.has(q.type)) { errors.push(`#${i}(${q.q||''}): 非法 type=${q.type}`); }
    if (!q.q) errors.push(`#${i}: 缺少题目文本 q`);
    if (!q.cat) errors.push(`#${i}(${q.q||''}): 缺少章节 cat`);
    if (q.diff && !VALID_DIFFS.has(q.diff)) errors.push(`#${i}(${q.q||''}): 非法 diff=${q.diff}`);
    if (!q.exp || !String(q.exp).includes('PPT 第')) errors.push(`#${i}(${q.q||''}): exp 缺少 "PPT 第X页" 出处`);
    switch (q.type) {
      case 'single':
      case 'judge':
        if (!Array.isArray(q.opts) || q.opts.length < 2) errors.push(`#${i}(${q.q||''}): opts 需至少2项`);
        if (typeof q.ans !== 'number' || q.ans < 0 || q.ans >= (q.opts||[]).length) errors.push(`#${i}(${q.q||''}): ans 索引越界`);
        break;
      case 'multi':
        if (!Array.isArray(q.opts) || q.opts.length < 2) errors.push(`#${i}(${q.q||''}): opts 需至少2项`);
        if (!Array.isArray(q.ans) || q.ans.length < 2 || q.ans.some(a => typeof a !== 'number' || a < 0 || a >= (q.opts||[]).length))
          errors.push(`#${i}(${q.q||''}): multi ans 需为合法索引数组(≥2项)`);
        break;
      case 'match':
        if (!Array.isArray(q.left) || !Array.isArray(q.right) || q.left.length !== q.right.length || q.left.length < 2)
          errors.push(`#${i}(${q.q||''}): match 需 left/right 等长数组(≥2项)`);
        if (!Array.isArray(q.pairs) || q.pairs.length !== (q.left||[]).length)
          errors.push(`#${i}(${q.q||''}): match pairs 数量需等于 left 长度`);
        else {
          const rUsed = new Set();
          q.pairs.forEach(([l,r]) => {
            if (l < 0 || l >= q.left.length) errors.push(`#${i}: match pair 左索引越界 ${l}`);
            if (r < 0 || r >= q.right.length) errors.push(`#${i}: match pair 右索引越界 ${r}`);
            if (rUsed.has(r)) errors.push(`#${i}: match 右侧 ${r} 被重复配对`);
            rUsed.add(r);
          });
        }
        if (q.opts || q.ans) errors.push(`#${i}: match 不应有 opts/ans`);
        break;
      case 'fill':
        if (q.opts) errors.push(`#${i}(${q.q||''}): fill 不应有 opts`);
        if (!Array.isArray(q.answers) || q.answers.length === 0) errors.push(`#${i}(${q.q||''}): fill 缺少 answers 数组`);
        break;
    }
  });
  return errors;
}

// ---------------- 答案分布校验 ----------------
function shuffleOpts(q) {
  if (q.type !== 'single' && q.type !== 'multi') return;
  const n = q.opts.length;
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const oldOpts = q.opts.slice();
  q.opts = perm.map(p => oldOpts[p]);
  if (q.type === 'multi') q.ans = q.ans.map(i => perm.indexOf(i)).sort((a, b) => a - b);
  else q.ans = perm.indexOf(q.ans);
}

function checkDistribution(questions, n) {
  const dist = { A: 0, B: 0, C: 0, D: 0 };
  let singleCount = 0;
  for (let t = 0; t < n; t++) {
    const base = JSON.parse(JSON.stringify(questions));
    base.forEach(shuffleOpts);
    base.forEach(q => {
      if (q.type === 'single') {
        singleCount++;
        dist[String.fromCharCode(65 + q.ans)]++;
      }
    });
  }
  return { dist, singleCount };
}

// ---------------- 主流程 ----------------
function escapeHtml(s) {
  return String(s || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function main() {
  // 1. 读题库
  let raw;
  try { raw = JSON.parse(fs.readFileSync(questionsFile, 'utf8')); }
  catch (e) { console.error('题库 JSON 解析失败:', e.message); process.exit(1); }

  const payload = Array.isArray(raw) ? { questions: raw } : raw;
  const questions = payload.questions || [];
  if (!Array.isArray(questions) || questions.length === 0) {
    console.error('题库为空或缺少 questions 数组'); process.exit(1);
  }

  // 2. 校验题目
  const errs = validateQuestions(questions);
  if (errs.length) {
    console.error('❌ 题库校验失败 (' + errs.length + ' 项):');
    errs.forEach(e => console.error('  - ' + e));
    process.exit(1);
  }
  console.log(`✅ 题库校验通过: ${questions.length} 题`);

  // 3. 答案分布校验
  if (opts.checkDist > 0) {
    const { dist, singleCount } = checkDistribution(questions, opts.checkDist);
    const total = dist.A + dist.B + dist.C + dist.D;
    if (total > 0) {
      const pct = k => (dist[k] / total * 100).toFixed(1) + '%';
      console.log(`📊 答案分布(${opts.checkDist}次×${singleCount / opts.checkDist}单选): A=${pct('A')} B=${pct('B')} C=${pct('C')} D=${pct('D')}`);
      const maxPct = Math.max(dist.A, dist.B, dist.C, dist.D) / total;
      if (maxPct > 0.4) {
        console.error(`❌ 答案分布偏差过大（最高占比 ${(maxPct*100).toFixed(1)}%），检查题库 ans 是否过于集中`);
        process.exit(1);
      }
    }
  }

  // 4. 章节映射
  const cats = [...(payload.cats || [])];
  const usedKeys = [...new Set(questions.map(q => q.cat))];
  usedKeys.forEach((k, i) => {
    if (!cats.some(c => c.key === k)) {
      cats.push({ key: k, name: k, color: DEFAULT_COLORS[i % DEFAULT_COLORS.length] });
    }
  });

  // 5. 定位模板并注入
  const tplPath = findTemplate();
  if (!tplPath) { console.error('找不到 quiz-template.html 模板'); process.exit(1); }
  let tpl = fs.readFileSync(tplPath, 'utf8');

  const title = payload.title || '知识问答';
  const badge = payload.badge || '🎓 学习自测 · 题目与解析均出自原文';
  const subtitle = payload.subtitle || '单选 / 判断 / 多选 / 填空 四类题型 · 自动判分 · 章节筛选';
  const footer = payload.footer || '学习专用问答站 · 题库覆盖原文全部章节';

  const qJson = JSON.stringify(questions, null, 0).replace(/</g, '\\u003c');
  const cJson = JSON.stringify(cats);

  // HTML 注入字段转义，避免破坏页面结构
  const esc = {
    title: escapeHtml(title),
    badge: escapeHtml(badge),
    subtitle: escapeHtml(subtitle),
    footer: escapeHtml(footer),
  };
  tpl = tpl
    .replace('__QUESTIONS__', qJson)
    .replace('__CATS__', cJson)
    .replaceAll('__TITLE__', esc.title)
    .replace('__BADGE__', esc.badge)
    .replace('__SUBTITLE__', esc.subtitle)
    .replace('__FOOTER__', esc.footer);

  // 6. 语法校验
  const scriptMatch = tpl.match(/<script>([\s\S]*?)<\/script>/);
  if (!scriptMatch) { console.error('生成结果缺少 script 块'); process.exit(1); }
  try { new Function(scriptMatch[1]); }
  catch (e) { console.error('❌ 生成站点 JS 语法错误:', e.message); process.exit(1); }

  // 7. 写出
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, tpl, 'utf8');
  console.log(`✅ 站点生成: ${outFile} (${questions.length} 题, ${cats.length} 章节)`);
  process.exit(0);
}

main();
