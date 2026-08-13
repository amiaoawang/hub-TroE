"""复盘与自进化（P0/P1）：指标卡 + 失败模式 + LLM 经验提炼 + playbook 沉淀注入
+ 经验执行闭环（overrides 覆盖层 + 测试门禁）。

闭环：campaign 运行 → 审计/预算数据 → 指标卡（确定性统计）
    → 失败模式（规则化诊断）→ LLM 提炼经验候选 → 人工筛选 → playbook
    → 下次运行注入 context_pack（agent 开局即携带历史经验）
    → 经验执行：ensure 型 action 自动应用（overrides.yaml）→ 测试门禁 → 标记 applied
    → 下轮验证（指标对比基线）→ 生效保留 / 恶化降级。

用法（backend/ 下）：
    python -m src.main --retro               # 基于当前 DB 跑完整复盘
    python -m src.main --playbook list       # 查看经验库
    python -m src.main --playbook accept R-001   # 采纳经验（进入注入池）
    python -m src.main --playbook reject R-001   # 拒绝经验
    python -m src.main --playbook apply          # 应用所有带 action 的已采纳经验
    python -m src.main --playbook apply R-003    # 应用单条
"""
import json
import os
import subprocess
import sys

from src import db

PLAYBOOK_FILE = None      # 模块级覆盖（测试隔离用）；None → 当前项目 playbook/
OVERRIDES_FILE = None


def playbook_dir():
    return os.path.join(db.PROJECT_DIR, "playbook")


def playbook_file():
    return PLAYBOOK_FILE or os.path.join(playbook_dir(), "playbook.json")


def overrides_file():
    return OVERRIDES_FILE or os.path.join(playbook_dir(), "overrides.yaml")

# 测试门禁默认套件（D:\gameHarness\tests\）
TEST_FILES = [f"test_{n}.py" for n in
              ("s1", "s2", "s3", "s4", "game", "temp", "bf", "retro",
               "cycle", "llm_robust", "budget", "change", "cross_review")]

_SEVERITY = {"high": "高", "medium": "中", "low": "低"}


# --------------------------------------------------------------------------
# 1) 指标卡：从 DB 确定性汇聚一次运行的指标
# --------------------------------------------------------------------------
def compute_metrics():
    """从 DB 汇聚本次运行的全部指标（不依赖 LLM，纯统计）。"""
    conn = db.connect()
    try:
        n_task = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]
        n_done = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status='done'").fetchone()["c"]
        n_doing = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status='in_progress'").fetchone()["c"]
        n_todo = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status IN ('backlog','todo')"
        ).fetchone()["c"]
        n_esc = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status='escalated'").fetchone()["c"]
        rounds = conn.execute(
            "SELECT review_rounds, max_review_rounds FROM tasks").fetchall()
        total_rounds = sum(r["review_rounds"] for r in rounds)
        rejected_tasks = sum(1 for r in rounds if r["review_rounds"] >= 1)
        near_escalate = sum(1 for r in rounds
                            if r["review_rounds"] >= (r["max_review_rounds"] or 3) - 1
                            and r["review_rounds"] > 0)

        n_ms = conn.execute("SELECT COUNT(*) c FROM milestones").fetchone()["c"]
        ms_done = conn.execute(
            "SELECT COUNT(*) c FROM milestones WHERE status='done'").fetchone()["c"]
        ms_fail = conn.execute(
            "SELECT COUNT(*) c FROM milestones WHERE status='failed'").fetchone()["c"]
        ms_review = conn.execute(
            "SELECT COUNT(*) c FROM milestones WHERE status='review'").fetchone()["c"]

        audit_actions = {r["action"]: r["c"] for r in conn.execute(
            "SELECT action, COUNT(*) c FROM audit_log GROUP BY action")}
        n_audit = sum(audit_actions.values())
        n_merge = audit_actions.get("task_merge", 0)
        n_dispatch = audit_actions.get("task_dispatch", 0)
        n_escalate_evt = audit_actions.get("task_escalate", 0)
        n_reject = audit_actions.get("task_reject", 0)
        n_arbitrate = audit_actions.get("arbitrate", 0)

        n_msg = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        n_acked = conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE ack_status='acked'").fetchone()["c"]
        n_pending = conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE ack_status='pending'").fetchone()["c"]
        n_retry = conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE retry_count>0").fetchone()["c"]
        n_null_to = conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE to_id IS NULL").fetchone()["c"]
        n_cc_archive = audit_actions.get("email_cc_archive", 0)

        n_art = conn.execute(
            "SELECT COUNT(*) c FROM artifacts WHERE status='merged'").fetchone()["c"]

        # 质量（用户反馈）：feedback 表（rating 1=通过/满意，0=拒绝/不满意）
        fb_rows = conn.execute(
            "SELECT id, milestone_id, rating, notes FROM feedback "
            "ORDER BY created_at DESC").fetchall()
        n_fb = len(fb_rows)
        n_fb_pos = sum(1 for r in fb_rows if r["rating"])
        n_fb_neg = n_fb - n_fb_pos
        approval_rate = round(n_fb_pos / n_fb, 3) if n_fb else None
        recent_notes = [r["notes"] for r in fb_rows[:3] if r["notes"]]

        budgets = [{"model": r["model"],
                    "in": r["ti"], "out": r["to_"], "cost": round(r["cu"], 4)}
                   for r in conn.execute(
                       "SELECT model, SUM(tokens_in) ti, SUM(tokens_out) to_, "
                       "SUM(cost_usd) cu FROM budget GROUP BY model ORDER BY cu DESC")]
        total_cost = round(sum(b["cost"] for b in budgets), 4)
        total_out = sum(b["out"] for b in budgets)
        total_in = sum(b["in"] for b in budgets)

        first_ts = conn.execute("SELECT MIN(ts) v FROM audit_log").fetchone()["v"]
        last_ts = conn.execute("SELECT MAX(ts) v FROM audit_log").fetchone()["v"]
        duration_s = None
        if first_ts and last_ts:
            from datetime import datetime
            try:
                duration_s = int((datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
                                  - datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")
                                  ).total_seconds())
            except ValueError:
                duration_s = None
    finally:
        conn.close()

    done_rate = round(n_done / n_task, 3) if n_task else 0.0
    reject_rate = round(rejected_tasks / n_task, 3) if n_task else 0.0
    return {
        "window": {"start_at": first_ts, "end_at": last_ts,
                   "duration_s": duration_s},
        "tasks": {"total": n_task, "done": n_done, "doing": n_doing,
                  "todo": n_todo, "escalated": n_esc,
                  "done_rate": done_rate, "reject_rate": reject_rate,
                  "rejected_tasks": rejected_tasks, "total_rounds": total_rounds,
                  "avg_rounds": round(total_rounds / n_task, 2) if n_task else 0,
                  "near_escalate": near_escalate},
        "milestones": {"total": n_ms, "done": ms_done, "failed": ms_fail,
                       "review": ms_review},
        "review": {"merge": n_merge, "dispatch": n_dispatch,
                   "reject_events": n_reject, "escalate_events": n_escalate_evt,
                   "arbitrate": n_arbitrate},
        "messages": {"total": n_msg, "acked": n_acked, "pending": n_pending,
                     "retried": n_retry, "null_to": n_null_to,
                     "cc_archived": n_cc_archive},
        "artifacts": {"merged": n_art},
        "feedback": {"total": n_fb, "positive": n_fb_pos, "negative": n_fb_neg,
                     "approval_rate": approval_rate, "recent_notes": recent_notes},
        "budget": {"models": budgets, "total_cost": total_cost,
                   "total_in": total_in, "total_out": total_out,
                   "cost_per_done": round(total_cost / n_done, 5) if n_done else 0,
                   "out_ratio": round(total_out / total_in, 2) if total_in else 0},
        "audit": {"total": n_audit, "actions": audit_actions},
    }


def render_metrics_md(m):
    """指标卡 markdown（人读）。"""
    w = m["window"]
    dur = f"{w['duration_s']}s" if w["duration_s"] is not None else "-"
    if w["duration_s"] and w["duration_s"] >= 60:
        dur = f"{w['duration_s'] // 60}分{w['duration_s'] % 60}秒"
    t, ms, rv, msg, b = m["tasks"], m["milestones"], m["review"], m["messages"], m["budget"]
    lines = [
        "# 运行指标卡",
        "",
        f"- 运行窗口：{w['start_at'] or '-'} → {w['end_at'] or '-'}（{dur}）",
        f"- 审计事件：{m['audit']['total']} 条 · 合入制品：{m['artifacts']['merged']} 件",
        "",
        "## 任务",
        f"- 完成率：{t['done']}/{t['total']}（{t['done_rate'] * 100:.1f}%）· "
        f"打回率：{t['rejected_tasks']}/{t['total']}（{t['reject_rate'] * 100:.1f}%）",
        f"- 平均打回 {t['avg_rounds']} 轮 · 接近升级仲裁：{t['near_escalate']} 个",
        f"- 进行中 {t['doing']} · 待办 {t['todo']} · 升级 {t['escalated']}",
        "",
        "## 里程碑",
        f"- done {ms['done']}/{ms['total']} · failed {ms['failed']} · review {ms['review']}",
        "",
        "## 评审",
        f"- 派发 {rv['dispatch']} · 合入 {rv['merge']} · 打回事件 {rv['reject_events']}"
        f" · 升级仲裁 {rv['escalate_events']}（仲裁 {rv['arbitrate']}）",
        "",
        "## 消息",
        f"- 总数 {msg['total']}（acked {msg['acked']} · pending {msg['pending']}"
        f" · 重投 {msg['retried']} · 空收件人 {msg['null_to']}）· CC 存档 {msg['cc_archived']}",
        "",
        "## 质量（用户反馈）",
        _render_feedback(m["feedback"]),
        "",
        "## 预算（token 效率）",
        "- " + " · ".join(
            f"**{x['model']}**: in {x['in']} / out {x['out']} · ${x['cost']}"
            for x in b["models"]) if b["models"] else "- 无",
        f"- 总成本 **${b['total_cost']}** · 每 done 任务 ${b['cost_per_done']}"
        f" · 输出/输入比 {b['out_ratio']}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 2) 失败模式：确定性诊断规则（可解释，不依赖 LLM）
# --------------------------------------------------------------------------
def find_failure_patterns(m):
    """根据指标识别失败模式，输出规则化诊断。"""
    t, ms, rv, msg = m["tasks"], m["milestones"], m["review"], m["messages"]
    patterns = []
    if t["total"] and t["done_rate"] < 1.0:
        patterns.append({
            "id": "P1", "severity": "high",
            "title": "任务未全部完成",
            "detail": f"done {t['done']}/{t['total']}，仍有 {t['doing']} 个进行中、"
                      f"{t['todo']} 个待办、{t['escalated']} 个升级",
            "rule": "阶段退出前应确认关键任务完成；未完成需触发整改而非静默推进",
        })
    if t["near_escalate"] > 0 or rv["escalate_events"] > 0:
        patterns.append({
            "id": "P2", "severity": "high",
            "title": "评审打回循环",
            "detail": f"{t['near_escalate']} 个任务接近升级仲裁上限，"
                      f"升级事件 {rv['escalate_events']} 次（仲裁 {rv['arbitrate']}）",
            "rule": "打回 ≥ 上限-1 轮时启动仲裁/换人，避免死循环；评审判定宽容化",
        })
    if msg["null_to"] > 0:
        patterns.append({
            "id": "P3", "severity": "high",
            "title": "空收件人消息",
            "detail": f"{msg['null_to']} 条消息 to_id 为 NULL（收件人缺失）",
            "rule": "发送前校验收件人；parent_id 等组织关系必须在创建时绑定",
        })
    if msg["pending"] > 0 or msg["retried"] > 0:
        patterns.append({
            "id": "P4", "severity": "medium",
            "title": "消息未及时确认",
            "detail": f"pending {msg['pending']} · 重投 {msg['retried']}",
            "rule": "收件 agent 应每轮 ack；超时重投上限应触发升级而非无限重试",
        })
    if ms["failed"] > 0 or ms["review"] > 0:
        patterns.append({
            "id": "P5", "severity": "medium",
            "title": "里程碑未通过",
            "detail": f"failed {ms['failed']} · review {ms['review']}",
            "rule": "构建失败的里程碑需自动建整改任务并回流需求池",
        })
    if m["budget"]["models"]:
        top = m["budget"]["models"][0]
        if top["cost"] and m["budget"]["total_cost"] and \
                top["cost"] / m["budget"]["total_cost"] > 0.5:
            patterns.append({
                "id": "P6", "severity": "medium",
                "title": "成本集中于单一模型",
                "detail": f"{top['model']} 占 ${top['cost']} / "
                          f"${m['budget']['total_cost']}（>{'50%'}）",
                "rule": "按任务复杂度分级路由；长产出任务限制 max_tokens 或换廉价档",
            })
    if m["budget"]["total_out"] and m["budget"]["total_in"] and \
            m["budget"]["out_ratio"] > 10:
        patterns.append({
            "id": "P7", "severity": "low",
            "title": "输出 token 远大于输入",
            "detail": f"输出/输入比 {m['budget']['out_ratio']}（"
                      f"out {m['budget']['total_out']} / in {m['budget']['total_in']}）",
            "rule": "subagent 产出设 max_tokens 上限；评审/汇报要求结构化短输出",
        })
    if t["total"] and t["reject_rate"] > 0.3:
        patterns.append({
            "id": "P8", "severity": "medium",
            "title": "打回率偏高",
            "detail": f"{t['rejected_tasks']}/{t['total']} 个任务被打回过（"
                      f"{t['reject_rate'] * 100:.0f}%）",
            "rule": "检查 DoD 是否清晰、评审标准是否与产出物匹配；考虑评审宽容化",
        })
    fb = m.get("feedback") or {}
    if fb.get("negative", 0) > 0:
        notes = "；".join(f"「{n[:40]}」" for n in fb.get("recent_notes", [])
                          if n) or "（无备注）"
        patterns.append({
            "id": "P9", "severity": "high",
            "title": "用户验收不通过",
            "detail": f"{fb['negative']}/{fb['total']} 次验收被拒绝"
                      f"（通过率 {fb['approval_rate'] * 100:.0f}%）：{notes}",
            "rule": "用户反馈是最高优先级进化信号：负反馈必须生成整改任务"
                    "并回流需求池，复盘时纳入经验提炼",
        })
    if not patterns:
        patterns.append({
            "id": "P0", "severity": "low", "title": "无显著失败模式",
            "detail": "本次运行未命中已知失败规则", "rule": "保持现状",
        })
    return patterns


# --------------------------------------------------------------------------
# 3) LLM 复盘：从指标 + 审计样本提炼可执行经验候选
# --------------------------------------------------------------------------
def generate_experiences(router, m, patterns):
    """调用强推理模型，把指标/失败模式提炼成可执行经验（返回候选列表）。"""
    audit_sample = _audit_sample(12)
    prompt = (
        "你是项目复盘专家。基于以下运行指标与审计样本，提炼 3-5 条"
        "可复用的工程经验（每条：问题 → 规则 → 适用场景）。"
        "规则必须具体可执行（如'评审判定只认明确否定词'），不要泛泛而谈。"
        "只输出 JSON 数组，不要用 markdown 代码块包裹："
        '[{"problem":"...","rule":"...","scope":"review|dispatch|scheduler|config|prompt|communication","rationale":"..."}]'
        f"\n\n【指标】\n{json.dumps(m, ensure_ascii=False)[:3000]}"
        f"\n\n【失败模式】\n{json.dumps(patterns, ensure_ascii=False)}"
        f"\n\n【用户反馈】\n{json.dumps(m.get('feedback') or {}, ensure_ascii=False)}"
        f"\n\n【审计样本】\n{audit_sample}"
    )
    try:
        text = router.complete("producer", "复盘提炼", prompt,
                               temperature=0.3, max_tokens=1500)
        if not text or not text.strip():
            return []
        data = _extract_json(text)
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict)
                and d.get("problem") and d.get("rule")]
    except Exception:  # noqa: BLE001   # LLM 输出非 JSON 时放弃本次提炼
        return []


def _extract_json(text):
    """从 LLM 输出中稳健提取 JSON（容忍 markdown 代码块/前后缀文字）。"""
    import re
    text = text.strip()
    for pat in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pat, text)
        if not m:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _audit_sample(n=12):
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT ts, actor_id, action, target FROM audit_log "
            "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return "\n".join(f"{r['ts']} {r['actor_id']} {r['action']} {r['target'] or ''}"
                         for r in rows)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 4) playbook：经验库（backend/playbook/playbook.json，进版本库）
# --------------------------------------------------------------------------
def playbook_load():
    if not os.path.exists(playbook_file()):
        return {"version": 1, "rules": []}
    with open(playbook_file(), encoding="utf-8") as f:
        return json.load(f)


def playbook_save(pb):
    os.makedirs(playbook_dir(), exist_ok=True)
    with open(playbook_file(), "w", encoding="utf-8") as f:
        json.dump(pb, f, ensure_ascii=False, indent=2)


def export_experiences(rule_ids=None, out_path=None):
    """导出当前项目经验（选择性共享）：指定 R-xxx 或全部 accepted。
    导出内容：problem/rule/scope/action + 源项目标记（provenance）。
    基线/验证历史不导出（属于源项目语境）。返回 (路径, 条数)。"""
    pb = playbook_load()
    if rule_ids:
        rules = [r for r in pb["rules"] if r["id"] in rule_ids
                 and r["status"] == "accepted"]
    else:
        rules = [r for r in pb["rules"] if r["status"] == "accepted"]
    if not rules:
        return None, 0
    payload = {
        "version": 1,
        "source_project": db.PROJECT_NAME,
        "exported_at": db.now(),
        "rules": [{k: r[k] for k in ("problem", "rule", "scope", "action")
                   if k in r} for r in rules],
    }
    out = out_path or os.path.join(
        playbook_dir(), "exports",
        f"{db.PROJECT_NAME}-{db.now().replace(':', '')[:14]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out, len(rules)


def import_experiences(path):
    """从导出文件导入经验到当前项目（选择性共享）。
    导入为 proposed 状态 + shared_from provenance；基线/验证历史不迁移
    （accept 时自动记录目标项目当前指标为新基线）。
    返回导入的新 id 列表。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != 1 or not isinstance(payload.get("rules"), list):
        raise ValueError("不是有效的经验导出文件")
    src = payload.get("source_project", "?")
    exported_at = payload.get("exported_at", "")
    imported = []
    for r in payload["rules"]:
        rid = playbook_add(r.get("problem", ""), r.get("rule", ""),
                           scope=r.get("scope", "general"),
                           source=f"shared:{src}")
        pb = playbook_load()      # 重新加载（playbook_add 已持久化）
        for rr in pb["rules"]:
            if rr["id"] == rid:
                rr["shared_from"] = {"project": src,
                                     "exported_at": exported_at}
                if r.get("action"):
                    rr["action"] = r["action"]
        playbook_save(pb)
        imported.append(rid)
    return imported


def _render_feedback(fb):
    """用户反馈章节（无数据时显示占位）。"""
    if not fb["total"]:
        return "- 暂无用户验收反馈"
    line = (f"- 验收 {fb['total']} 次 · 通过 {fb['positive']} · "
            f"拒绝 {fb['negative']} · 通过率 "
            f"{fb['approval_rate'] * 100:.0f}%")
    if fb["recent_notes"]:
        line += "\n- 最近反馈：" + "；".join(
            f"「{n[:60]}」" for n in fb["recent_notes"])
    return line


def _metrics_brief(m=None):
    """关键指标子集（经验基线/验证用，只取可对比的核心项）。"""
    m = m or compute_metrics()
    t, b, fb = m["tasks"], m["budget"], m["feedback"]
    return {
        "done_rate": t["done_rate"],
        "reject_rate": t["reject_rate"],
        "avg_rounds": t["avg_rounds"],
        "total_cost": b["total_cost"],
        "cost_per_done": b["cost_per_done"],
        "out_ratio": b["out_ratio"],
        "approval_rate": fb["approval_rate"],   # 用户验收通过率（质量真值）
    }


# 指标改善方向：True = 数值升高为改善
_IMPROVE_UP = {"done_rate", "approval_rate"}
_IMPROVE_DOWN = {"reject_rate", "avg_rounds", "total_cost",
                 "cost_per_done", "out_ratio"}


def playbook_add(problem, rule, scope="general", source="manual",
                 metrics=None, status="proposed"):
    pb = playbook_load()
    rid = f"R-{len(pb['rules']) + 1:03d}"
    pb["rules"].append({
        "id": rid, "status": status, "source": source,
        "created_at": db.now(), "problem": problem, "rule": rule,
        "scope": scope, "metrics": metrics or {},
    })
    playbook_save(pb)
    return rid


def playbook_set_status(rule_id, status):
    """accept 时自动记录基线指标（采纳时刻的运行指标），供下轮验证。"""
    pb = playbook_load()
    for r in pb["rules"]:
        if r["id"] == rule_id:
            r["status"] = status
            r["updated_at"] = db.now()
            if status == "accepted" and not r.get("baseline"):
                r["baseline"] = _metrics_brief()
            playbook_save(pb)
            return True
    return False


def playbook_verification(record=False, experiment=None):
    """验证已采纳经验的成效：当前运行指标 vs 各经验的采纳时基线。
    返回 [{"id","rule","baseline","current","deltas","verdict"}]。
    verdict: 生效 / 恶化 / 持平 / 部分生效 / 无基线（旧经验，下轮起对比）。
    record=True 时把本轮结果追加进每条经验的 verification_history（供降级判定）。
    experiment='R-xxx'：标记本轮为 A/B 实验轮（只激活该条经验，供归因分析）。"""
    cur = _metrics_brief()
    out = []
    pb = playbook_load()
    for r in pb["rules"]:
        if r["status"] != "accepted":
            continue
        bl = r.get("baseline")
        if not bl:
            out.append({"id": r["id"], "rule": r["rule"],
                        "baseline": None, "current": cur,
                        "deltas": {}, "verdict": "无基线"})
            continue
        deltas, up, down = {}, 0, 0
        for k in _IMPROVE_UP | _IMPROVE_DOWN:
            if k not in bl or k not in cur or bl[k] == cur[k]:
                continue
            diff = cur[k] - bl[k]
            good = diff > 0 if k in _IMPROVE_UP else diff < 0
            deltas[k] = {"from": round(bl[k], 4), "to": round(cur[k], 4),
                         "good": good}
            up += 1 if good else 0
            down += 0 if good else 1
        if not deltas:
            verdict = "持平"
        elif up > down:
            verdict = "生效"
        elif down > up:
            verdict = "恶化"
        else:
            verdict = "部分生效"
        out.append({"id": r["id"], "rule": r["rule"], "baseline": bl,
                    "current": cur, "deltas": deltas, "verdict": verdict})
        if record:
            hist = r.setdefault("verification_history", [])
            hist.append({"ts": db.now(), "verdict": verdict,
                         "metrics": dict(cur),
                         "experiment": experiment})
    if record:
        playbook_save(pb)
    return out


_DEGRADE_THRESHOLD = 2   # 连续 N 轮验证为「恶化」则自动降级


def _revert_overrides_for(rule):
    """撤销某条经验 action 在 overrides 中写入的值（保守：仅当当前值等于它写的期望值）。"""
    action = rule.get("action") or {}
    if action.get("type") != "config":
        return False
    path = action.get("path") or []
    expect = action.get("expect")
    ov = overrides_load()
    if _get_path(ov, path) == expect:
        import copy
        new_ov = copy.deepcopy(ov)
        _del_path(new_ov, path)
        overrides_save(new_ov)
        return True
    return False


def auto_archive_degraded(threshold=_DEGRADE_THRESHOLD):
    """护栏闭环：验证连续 N 轮「恶化」的已采纳经验自动降级（archived）：
    - 移出注入池（playbook_accepted 只认 accepted）
    - 撤销其 overrides 变更（配置回滚）
    - 审计留痕（actor=retro, action=playbook_archive）
    返回被降级的经验 id 列表。"""
    pb = playbook_load()
    archived = []
    for r in pb["rules"]:
        if r["status"] != "accepted":
            continue
        hist = r.get("verification_history") or []
        n = 0
        for h in reversed(hist):
            if h.get("verdict") == "恶化":
                n += 1
            else:
                break
        if n < threshold:
            continue
        r["status"] = "archived"
        r["archived_at"] = db.now()
        r["archive_reason"] = f"连续 {n} 轮验证恶化（阈值 {threshold}）"
        reverted = _revert_overrides_for(r)
        r["archive_reverted"] = reverted
        archived.append({"id": r["id"], "rounds": n, "reverted": reverted})
    if archived:
        playbook_save(pb)
        conn = db.connect()
        try:
            for a in archived:
                conn.execute(
                    "INSERT INTO audit_log (ts,actor_id,action,target,detail) "
                    "VALUES (?,?,?,?,?)",
                    (db.now(), "retro", "playbook_archive", a["id"],
                     json.dumps(a, ensure_ascii=False)))
            conn.commit()
        finally:
            conn.close()
    return archived


def playbook_reactivate(rule_id):
    """人工纠正：把已降级的经验重新激活（archived → accepted，清验证历史）。"""
    pb = playbook_load()
    for r in pb["rules"]:
        if r["id"] == rule_id and r["status"] == "archived":
            r["status"] = "accepted"
            r.pop("archived_at", None)
            r.pop("archive_reason", None)
            r["verification_history"] = []
            r["updated_at"] = db.now()
            playbook_save(pb)
            return True
    return False


# --------------------------------------------------------------------------
# 7) 全局退化熔断：连续多轮核心指标退化 → 暂停自动进化（人工介入）
# --------------------------------------------------------------------------
_FUSE_ROUNDS = 3      # 观察窗口：最近 N 轮
_FUSE_SCORE = 2.0     # 窗口内平均退化分阈值（≥ 则熔断）


def record_global():
    """每轮复盘记录全局指标快照（global_history），供熔断判定。"""
    pb = playbook_load()
    pb.setdefault("global_history", []).append(
        {"ts": db.now(), "metrics": _metrics_brief()})
    playbook_save(pb)


def _degrade_score(prev, cur):
    """相对上一轮的退化分：核心指标按改善方向退化计 1 分（0~7）。"""
    s = 0
    for k in _IMPROVE_UP:
        if prev.get(k) is not None and cur.get(k) is not None \
                and cur[k] < prev[k]:
            s += 1
    for k in _IMPROVE_DOWN:
        if prev.get(k) is not None and cur.get(k) is not None \
                and cur[k] > prev[k]:
            s += 1
    return s


def check_fuse(rounds=_FUSE_ROUNDS, score=_FUSE_SCORE):
    """熔断判定：最近 N 轮平均退化分 ≥ 阈值 → 熔断（暂停自动进化）。
    返回 fuse 状态 dict {"tripped", "since", "reason"}。"""
    pb = playbook_load()
    fuse = pb.get("fuse") or {"tripped": False}
    if fuse.get("tripped"):
        return fuse
    hist = pb.get("global_history") or []
    if len(hist) <= rounds:
        return fuse
    recent = hist[-rounds:]
    scores = [_degrade_score(recent[i - 1]["metrics"], recent[i]["metrics"])
              for i in range(1, len(recent))]
    avg = sum(scores) / len(scores) if scores else 0.0
    if avg >= score:
        fuse = {"tripped": True, "since": db.now(),
                "reason": f"最近 {rounds} 轮平均退化分 {avg:.1f} ≥ {score}"
                          f"（核心指标连续退化，暂停自动进化）",
                "rounds": rounds}
        pb["fuse"] = fuse
        playbook_save(pb)
        conn = db.connect()
        try:
            conn.execute(
                "INSERT INTO audit_log (ts,actor_id,action,target,detail) "
                "VALUES (?,?,?,?,?)",
                (db.now(), "retro", "fuse_trip", "global",
                 json.dumps(fuse, ensure_ascii=False)))
            conn.commit()
        finally:
            conn.close()
        return fuse
    return fuse


def fuse_status():
    pb = playbook_load()
    return pb.get("fuse") or {"tripped": False}


def fuse_reset():
    """人工解除熔断：清 fuse 状态与全局历史（重新开始观察）。"""
    pb = playbook_load()
    pb["fuse"] = {"tripped": False}
    pb["global_history"] = []
    playbook_save(pb)
    return True


# --------------------------------------------------------------------------
# 6) A/B 归因：实验轮次（单条激活）归因 + 配置类 applied 前后归因
# --------------------------------------------------------------------------
_METRIC_KEYS = ["done_rate", "reject_rate", "avg_rounds", "total_cost",
                "cost_per_done", "out_ratio", "approval_rate"]


def _verdict_of_avg(avg_before, avg_after):
    """按指标改善方向对比前后均值，得出归因 verdict。"""
    up, down = 0, 0
    for k in _METRIC_KEYS:
        if k not in avg_before or k not in avg_after:
            continue
        diff = avg_after[k] - avg_before[k]
        if abs(diff) < 1e-9:
            continue
        good = diff > 0 if k in _IMPROVE_UP else diff < 0
        up += 1 if good else 0
        down += 0 if good else 1
    if up == down == 0:
        return "持平"
    if up > down:
        return "生效"
    if down > up:
        return "恶化"
    return "部分生效"


def attribution():
    """A/B 归因：判定每条经验的独立效果。

    - 实验轮次归因：verification_history 中带 experiment 标记的轮次
      （--experiment 单条激活运行）→ verdict 分布 + 平均指标。
    - 配置类归因：action 已 applied 的经验 → applied_at 前后指标均值对比。
    - 混合轮次（experiment=None）不参与归因（诚实标注，不假装能归因）。
    """
    from collections import Counter
    pb = playbook_load()
    out = []
    for r in pb["rules"]:
        if r["status"] != "accepted":
            continue
        hist = r.get("verification_history") or []
        exp_rounds = [h for h in hist if h.get("experiment")]
        row = {"id": r["id"], "rule": r["rule"],
               "experiment_rounds": len(exp_rounds)}
        # 1) 实验轮次归因（提示词/流程类经验的主依据）
        if exp_rounds:
            cnt = Counter(h["verdict"] for h in exp_rounds)
            row["exp_distribution"] = dict(cnt)
            row["exp_verdict"] = (
                "生效" if cnt.get("生效", 0) > cnt.get("恶化", 0)
                else "恶化" if cnt.get("恶化", 0) > cnt.get("生效", 0)
                else "持平")
            avg = {}
            for k in _METRIC_KEYS:
                vals = [h["metrics"][k] for h in exp_rounds
                        if k in h.get("metrics", {})
                        and h["metrics"].get(k) is not None]
                if vals:
                    avg[k] = round(sum(vals) / len(vals), 4)
            row["exp_avg"] = avg
        # 2) 配置类归因：applied 前后指标均值对比
        action = r.get("action") or {}
        if action.get("type") == "config" and r.get("applied_at"):
            before = ([h["metrics"] for h in hist
                       if h.get("ts", "") <= r["applied_at"]]
                      + ([r["baseline"]] if r.get("baseline") else []))
            after = [h["metrics"] for h in hist
                     if h.get("ts", "") > r["applied_at"]]
            if before and after:
                avg_b = {k: round(
                    sum(x[k] for x in before if k in x and x[k] is not None)
                    / max(1, sum(1 for x in before if k in x
                                 and x[k] is not None)), 4)
                    for k in _METRIC_KEYS}
                avg_a = {k: round(
                    sum(x[k] for x in after if k in x and x[k] is not None)
                    / max(1, sum(1 for x in after if k in x
                                 and x[k] is not None)), 4)
                    for k in _METRIC_KEYS}
                row["apply_before"] = avg_b
                row["apply_after"] = avg_a
                row["apply_verdict"] = _verdict_of_avg(avg_b, avg_a)
        out.append(row)
    return out


def render_attribution_md(attrs):
    """A/B 归因报告章节。"""
    if not attrs:
        return ["## A/B 归因", "", "- 暂无已采纳经验", ""]
    lines = ["## A/B 归因", ""]
    for a in attrs:
        parts = [f"**{a['id']}**"]
        if a.get("exp_verdict"):
            parts.append(f"实验 {a['experiment_rounds']} 轮 → {a['exp_verdict']}"
                         f"（分布 {a.get('exp_distribution', {})}）")
        if a.get("apply_verdict"):
            parts.append(f"配置应用后 → {a['apply_verdict']}")
        if not a.get("exp_verdict") and not a.get("apply_verdict"):
            parts.append("未单独验证（混合轮次无法归因；用 --experiment R-xxx 跑实验轮）")
        lines.append(f"- {' · '.join(parts)}：{a['rule']}")
        lines.append("")
    return lines


def render_verification_md(verifs):
    """经验验证章节（复盘报告用）。"""
    if not verifs:
        return ["## 经验验证", "", "- 暂无已采纳经验，跳过", ""]
    lines = ["## 经验验证（对比采纳时基线）", ""]
    for v in verifs:
        tag = {"生效": "✅ 生效", "恶化": "⚠️ 恶化", "持平": "＝ 持平",
               "部分生效": "◐ 部分生效", "无基线": "⏳ 无基线"}.get(v["verdict"],
                                                          v["verdict"])
        lines.append(f"- **{v['id']}** {tag}：{v['rule']}")
        if v["deltas"]:
            for k, d in v["deltas"].items():
                arrow = "↑" if d["good"] else "↓"
                lines.append(f"  - {k}: {d['from']} → {d['to']} {arrow}")
        else:
            lines.append("  - 指标无变化")
        lines.append("")
    return lines


def playbook_accepted(only=None):
    """已被采纳的经验（注入 context_pack 用），返回简短文本列表。
    only='R-xxx'：A/B 实验模式，只返回该条经验（其余暂停注入，隔离归因）。"""
    rules = playbook_load()["rules"]
    if only:
        rules = [r for r in rules
                 if r["id"] == only and r["status"] == "accepted"]
    return [f"[{r['id']}] {r['rule']}" for r in rules
            if r["status"] == "accepted"]


def playbook_summary():
    pb = playbook_load()
    return pb, [r for r in pb["rules"] if r["status"] == "proposed"]


# --------------------------------------------------------------------------
# 5) 经验执行闭环：overrides 覆盖层 + ensure 型 action + 测试门禁
# --------------------------------------------------------------------------
def overrides_load():
    """经验层配置覆盖（运行时 merge 进 config.yaml，不污染主配置）。"""
    if not os.path.exists(overrides_file()):
        return {}
    import yaml
    with open(overrides_file(), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def overrides_save(ov):
    os.makedirs(playbook_dir(), exist_ok=True)
    import yaml
    with open(overrides_file(), "w", encoding="utf-8") as f:
        yaml.safe_dump(ov or {}, f, allow_unicode=True,
                       sort_keys=False, default_flow_style=False)


def merge_config(base, override):
    """深合并：override 的嵌套键覆盖 base（用于 config + overrides）。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def _get_path(node, path):
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _set_path(node, path, value):
    cur = node
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def _del_path(node, path):
    cur = node
    stack = []
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return
        stack.append((cur, k))
        cur = cur[k]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)
    # 递归清理空父节点（避免 overrides 残留 {'limits': {}} 之类）
    for parent, k in reversed(stack):
        if isinstance(parent.get(k), dict) and not parent[k]:
            parent.pop(k, None)
        else:
            break


def run_all_tests(test_files=None):
    """测试门禁：跑默认测试套件（backend 外的 tests/），全过返回 True。
    供经验 apply 时验证配置变更不破坏系统。
    已升级为统一 CI 门禁（gate_ci：语法 + 11 套测试），保留薄封装兼容。"""
    from src import ci
    ok, reason = ci.gate_ci()
    return ok, reason


def apply_rule_action(rule, run_tests_fn=None):
    """应用单条经验的 action（ensure 语义：确保配置达到期望值）。
    返回 {id, status, note, changed}；不直接改 playbook，由调用方持久化。
    status: applied | skipped | failed。
    """
    rid = rule["id"]
    action = rule.get("action")
    if not action:
        return {"id": rid, "status": "skipped", "note": "无 action（人工/结构类经验）",
                "changed": False}
    if action.get("type") != "config":
        return {"id": rid, "status": "skipped",
                "note": f"action 类型 {action.get('type')} 暂不支持", "changed": False}
    path = action.get("path") or []
    expect = action.get("expect")
    if not path:
        return {"id": rid, "status": "skipped", "note": "action 缺 path",
                "changed": False}

    import yaml
    with open(os.path.join(db.BASE, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cur = merge_config(cfg, overrides_load())
    cur_val = _get_path(cur, path)

    if cur_val == expect:
        return {"id": rid, "status": "applied", "changed": False,
                "note": "已满足期望值，无需变更"}

    # 需要变更：写入 overrides → 测试门禁（失败恢复 apply 前状态）
    import copy
    ov_before = overrides_load()
    ov = copy.deepcopy(ov_before)   # 深拷贝：嵌套结构修改不影响备份
    _set_path(ov, path, expect)
    overrides_save(ov)
    ok, reason = (run_tests_fn() if run_tests_fn
                  else run_all_tests())
    if ok:
        return {"id": rid, "status": "applied", "changed": True,
                "note": f"已写入 overrides.yaml（{'/'.join(path)}={expect}）"}
    overrides_save(ov_before)   # 回滚：只撤销本次写入，保留此前成功值
    return {"id": rid, "status": "failed", "changed": False,
            "note": f"测试门禁失败，已回滚: {reason}"}


def apply_playbook(rule_id=None, run_tests_fn=None):
    """应用经验库中带 action 的已采纳经验（可指定单条）。"""
    pb = playbook_load()
    if rule_id:
        targets = [r for r in pb["rules"]
                   if r["id"] == rule_id and r["status"] == "accepted"]
    else:
        targets = [r for r in pb["rules"]
                   if r["status"] == "accepted" and r.get("action")
                   and not r.get("applied")]
    if not targets:
        return [{"id": rule_id or "-", "status": "skipped",
                 "note": "无可应用经验（需 accepted + 带 action + 未 applied）",
                 "changed": False}]
    results = []
    for r in targets:
        res = apply_rule_action(r, run_tests_fn=run_tests_fn)
        results.append(res)
        # 持久化（回写 applied 状态）
        if res["status"] in ("applied", "failed"):
            for rule in pb["rules"]:
                if rule["id"] == res["id"]:
                    rule["applied"] = res["status"] == "applied"
                    rule["applied_at"] = db.now() if res["status"] == "applied" \
                        else rule.get("applied_at")
                    rule["apply_note"] = res["note"]
                    if res["status"] == "failed":
                        rule["apply_failed_at"] = db.now()
    playbook_save(pb)
    return results


# --------------------------------------------------------------------------
# 5) 复盘入口：指标卡 + 失败模式 + 经验候选（不自动合入 playbook）
# --------------------------------------------------------------------------
def run_retro(router=None, auto=False):
    """完整复盘。auto=True（campaign 结束自动跑）时不调 LLM，只出指标卡。"""
    m = compute_metrics()
    patterns = find_failure_patterns(m)
    md = [render_metrics_md(m), "", "## 失败模式诊断", ""]
    for p in patterns:
        md.append(f"- [{_SEVERITY.get(p['severity'], '?')}] **{p['title']}**："
                  f"{p['detail']}")
        md.append(f"  - 规则建议：{p['rule']}")
    # 经验验证：已采纳经验的成效对比（指标 vs 采纳时基线）+ 恶化自动降级
    verifs = playbook_verification(record=True)
    if verifs:
        md.extend(["", *render_verification_md(verifs)])
    archived = auto_archive_degraded()
    if archived:
        md.extend(["", "## 自动降级（护栏）", ""])
        for a in archived:
            md.append(f"- **{a['id']}** 连续 {a['rounds']} 轮验证恶化"
                      f" → 已 archived（移出注入池"
                      + ("，撤销 overrides" if a["reverted"] else "")
                      + "）")
    attrs = attribution()
    if any(a.get("exp_verdict") or a.get("apply_verdict") for a in attrs):
        md.extend(["", *render_attribution_md(attrs)])
    # 全局退化熔断：记录本轮指标 → 判定熔断
    record_global()
    fuse = check_fuse()
    if fuse.get("tripped"):
        md.extend(["", "## ⛔ 全局退化熔断", "",
                   f"- {fuse['reason']}", "- 自动进化已暂停：新经验不注入、"
                   "apply 建议人工评估。", "- 解除：--playbook fuse reset"])
    candidates = []
    if router and not auto:
        md.extend(["", "## 经验候选（LLM 提炼，待人工筛选）", ""])
        candidates = generate_experiences(router, m, patterns)
        for i, c in enumerate(candidates, 1):
            md.append(f"- **候选 {i}** [{c.get('scope', '')}] "
                      f"{c.get('problem', '')}")
            md.append(f"  - 规则：{c.get('rule', '')}")
        md.extend(["", "> 采纳：python -m src.main --playbook accept R-xxx"
                       "（先 add 候选入库）"])
    os.makedirs(db.LOGS_DIR, exist_ok=True)
    out = os.path.join(db.LOGS_DIR,
                       f"retro-{db.now().replace(':', '')[:14]}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return out, m, patterns, candidates


if __name__ == "__main__":
    path, m, patterns, cands = run_retro()
    print(f"复盘已生成: {path}")
