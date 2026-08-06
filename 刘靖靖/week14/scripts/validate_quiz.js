#!/usr/bin/env node
/**
 * 一键校验问答站题库 / 已生成站点。可作为独立脚本，也可被 build 流程调用。
 *
 * 用法:
 *   node validate_quiz.js <questions.json | site.html> [--check-dist N] [--check-syntax]
 *
 * 输入:
 *   questions.json  题库文件（纯数组或 {cats, questions} 对象）
 *   site.html       已生成的问答站 HTML（自动抽取 QUESTIONS/CATS 校验）
 *
 * 选项:
 *   --check-dist N   做 N 次洗牌模拟校验答案分布（默认 100，0 关闭）
 *   --check-syntax   对 site.html 额外做 JS 语法检查（HTML 输入时默认开启）
 *
 * 退出码: 0=全部通过, 1=存在错误
 */
const fs = require('fs');
const path = require('path');

// ---------------- 参数解析 ----------------
const args = process.argv.slice(2);
if (args.length < 1) {
  console.error('用法: node validate_quiz.js <questions.json | site.html> [--check-dist N] [--check-syntax]');
  process.exit(1);
}
let target = null;
const opts = { checkDist: 100, checkSyntax: false };
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--check-dist') opts.checkDist = parseInt(args[++i], 10);
  else if (a === '--check-syntax') opts.checkSyntax = true;
  else if (target === null) target = path.resolve(a);
}
if (!target || !fs.existsSync(target)) {
  console.error('文件不存在: ' + target);
  process.exit(1);
}

// ---------------- 校验：题库结构 ----------------
const VALID_TYPES = new Set(['single', 'judge', 'multi', 'fill', 'match']);
const VALID_DIFFS = new Set(['easy', 'medium', 'hard']);

function validateQuestions(questions) {
  const errors = [];
  questions.forEach((q, i) => {
    if (!q || typeof q !== 'object') { errors.push(`#${i}: 不是对象`); return; }
    if (!VALID_TYPES.has(q.type)) errors.push(`#${i}(${q.q || ''}): 非法 type=${q.type}`);
    if (!q.q) errors.push(`#${i}: 缺少题目文本 q`);
    if (!q.cat) errors.push(`#${i}(${q.q || ''}): 缺少章节 cat`);
    if (q.diff && !VALID_DIFFS.has(q.diff)) errors.push(`#${i}(${q.q || ''}): 非法 diff=${q.diff}`);
    if (!q.exp || !String(q.exp).includes('PPT 第')) errors.push(`#${i}(${q.q || ''}): exp 缺少 "PPT 第X页" 出处`);
    switch (q.type) {
      case 'single':
      case 'judge':
        if (!Array.isArray(q.opts) || q.opts.length < 2) errors.push(`#${i}(${q.q || ''}): opts 需至少2项`);
        if (typeof q.ans !== 'number' || q.ans < 0 || q.ans >= (q.opts || []).length) errors.push(`#${i}(${q.q || ''}): ans 索引越界`);
        break;
      case 'multi':
        if (!Array.isArray(q.opts) || q.opts.length < 2) errors.push(`#${i}(${q.q || ''}): opts 需至少2项`);
        if (!Array.isArray(q.ans) || q.ans.length < 2 || q.ans.some(a => typeof a !== 'number' || a < 0 || a >= (q.opts || []).length))
          errors.push(`#${i}(${q.q || ''}): multi ans 需为合法索引数组(≥2项)`);
        break;
      case 'match':
        if (!Array.isArray(q.left) || !Array.isArray(q.right) || q.left.length !== q.right.length || q.left.length < 2)
          errors.push(`#${i}(${q.q || ''}): match 需 left/right 等长数组(≥2项)`);
        if (!Array.isArray(q.pairs) || q.pairs.length !== (q.left || []).length)
          errors.push(`#${i}(${q.q || ''}): match pairs 数量需等于 left 长度`);
        else {
          const rUsed = new Set();
          q.pairs.forEach(([l, r]) => {
            if (l < 0 || l >= q.left.length) errors.push(`#${i}: match pair 左索引越界 ${l}`);
            if (r < 0 || r >= q.right.length) errors.push(`#${i}: match pair 右索引越界 ${r}`);
            if (rUsed.has(r)) errors.push(`#${i}: match 右侧 ${r} 被重复配对`);
            rUsed.add(r);
          });
        }
        if (q.opts || q.ans) errors.push(`#${i}: match 不应有 opts/ans`);
        break;
      case 'fill':
        if (q.opts) errors.push(`#${i}(${q.q || ''}): fill 不应有 opts`);
        if (!Array.isArray(q.answers) || q.answers.length === 0) errors.push(`#${i}(${q.q || ''}): fill 缺少 answers 数组`);
        break;
    }
  });
  return errors;
}

// ---------------- 校验：答案分布 ----------------
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
  let singleTotal = 0;
  for (let t = 0; t < n; t++) {
    const base = JSON.parse(JSON.stringify(questions));
    base.forEach(shuffleOpts);
    base.forEach(q => {
      if (q.type === 'single') { singleTotal++; dist[String.fromCharCode(65 + q.ans)]++; }
    });
  }
  return { dist, singleTotal };
}

// ---------------- 主流程 ----------------
let questions = null;
let cats = null;
let ok = true;

try {
  if (target.endsWith('.html')) {
    // 从 HTML 抽取 QUESTIONS / CATS
    const html = fs.readFileSync(target, 'utf8');
    const script = html.match(/<script>([\s\S]*?)<\/script>/);
    if (!script) throw new Error('HTML 中未找到 <script> 块');
    const code = script[1];
    const qm = code.match(/const QUESTIONS = (\[[\s\S]*?\]);\n\s*const CATS = (\[[\s\S]*?\]);/);
    if (!qm) throw new Error('HTML 中未找到 QUESTIONS/CATS 定义');
    let QUESTIONS, CATS;
    eval('QUESTIONS = ' + qm[1]);
    eval('CATS = ' + qm[2]);
    questions = QUESTIONS;
    cats = CATS;
    console.log(`📄 输入: ${path.basename(target)} (HTML, ${questions.length} 题)`);
    if (opts.checkSyntax) {
      try { new Function(code); console.log('✅ JS 语法 OK'); }
      catch (e) { console.error('❌ JS 语法错误:', e.message); ok = false; }
    }
  } else {
    const raw = JSON.parse(fs.readFileSync(target, 'utf8'));
    const payload = Array.isArray(raw) ? { questions: raw } : raw;
    questions = payload.questions || [];
    cats = payload.cats || null;
    console.log(`📄 输入: ${path.basename(target)} (JSON, ${questions.length} 题)`);
  }

  if (!Array.isArray(questions) || questions.length === 0) {
    console.error('❌ 题库为空');
    process.exit(1);
  }

  // 结构校验
  const errs = validateQuestions(questions);
  if (errs.length) {
    ok = false;
    console.error(`❌ 题库结构校验失败 (${errs.length} 项):`);
    errs.forEach(e => console.error('  - ' + e));
  } else {
    console.log(`✅ 题库结构 OK (${questions.length} 题)`);
  }

  // 题型分布
  const byType = {};
  questions.forEach(q => byType[q.type] = (byType[q.type] || 0) + 1);
  console.log(`  题型分布: ${Object.entries(byType).map(([k, v]) => `${k}=${v}`).join(' ')}`);

  // 章节覆盖
  if (cats && cats.length) {
    const used = new Set(questions.map(q => q.cat));
    const missing = [...used].filter(k => !cats.some(c => c.key === k));
    if (missing.length) { ok = false; console.error(`❌ 以下章节缺少 cats 定义: ${missing.join(', ')}`); }
    else console.log(`✅ 章节映射 OK (${cats.length} 个章节)`);
  }

  // 答案分布
  if (opts.checkDist > 0) {
    const { dist, singleTotal } = checkDistribution(questions, opts.checkDist);
    const total = dist.A + dist.B + dist.C + dist.D;
    if (total > 0) {
      const pct = k => (dist[k] / total * 100).toFixed(1) + '%';
      console.log(`📊 答案分布(${opts.checkDist}次×${singleTotal / opts.checkDist}单选): A=${pct('A')} B=${pct('B')} C=${pct('C')} D=${pct('D')}`);
      const maxPct = Math.max(dist.A, dist.B, dist.C, dist.D) / total;
      if (maxPct > 0.4) { ok = false; console.error(`❌ 答案分布偏差过大（最高占比 ${(maxPct * 100).toFixed(1)}%）`); }
    }
  }

  // 题干重复检查
  const seen = new Map();
  questions.forEach((q, i) => {
    const key = q.q;
    if (seen.has(key)) console.warn(`⚠️ 第${seen.get(key) + 1}题与第${i + 1}题题干重复: ${key.slice(0, 30)}`);
    else seen.set(key, i);
  });

  console.log(ok ? '\n🎉 校验全部通过' : '\n❌ 校验未通过');
  process.exit(ok ? 0 : 1);
} catch (e) {
  console.error('校验失败: ' + e.message);
  process.exit(1);
}
