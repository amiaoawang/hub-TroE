"""变更单流程（PRD §8.3）：用户/主 agent 提「变更单」→ Producer 影响评估
（波及任务、依赖、成本）→ 批准后更新任务/宪章 → 广播（审计留痕）。

影响评估为确定性分析（不依赖 LLM，可解释可测试）：
- 波及任务：变更描述关键词命中的任务 + 它们的全部依赖链（直接/间接被依赖者）
- 预算影响：波及任务 budget_tokens 之和 + 当前已用成本
- 审批后可选 --apply：把波及任务重置 backlog（重新评估派发）

CLI（在 backend/ 下）：
    python -m src.main --change submit <标题>|<描述> [--by user]
    python -m src.main --change list [--status pending]
    python -m src.main --change assess <CR-xxx>      # 影响评估（写 impact + 审计）
    python -m src.main --change approve <CR-xxx> [--apply] [--notes ...]
    python -m src.main --change reject <CR-xxx> --notes <原因>
"""
from __future__ import annotations

import json
import re
import uuid

from src import db
from src import tickets


def _rid():
    return "CR-" + uuid.uuid4().hex[:8].upper()


def submit(title, description="", proposed_by="user"):
    """提交变更单（pending）。返回 id。"""
    rid = _rid()
    ts = db.now()
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO change_requests (id,title,description,proposed_by,"
            "status,created_at) VALUES (?,?,?,?,?,?)",
            (rid, title, description, proposed_by, "pending", ts))
        conn.commit()
    finally:
        conn.close()
    _log("change_submit", rid, {"title": title, "by": proposed_by})
    return rid


def get(rid):
    conn = db.connect()
    row = conn.execute("SELECT * FROM change_requests WHERE id=?",
                       (rid,)).fetchone()
    conn.close()
    return row


def list_requests(status=None):
    conn = db.connect()
    if status:
        rows = conn.execute(
            "SELECT * FROM change_requests WHERE status=? ORDER BY created_at",
            (status,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM change_requests ORDER BY created_at").fetchall()
    conn.close()
    return rows


# --------------------------------------------------------------------------
# 影响评估（确定性：关键词命中 + 依赖链 + 预算）
# --------------------------------------------------------------------------
def _keywords(description, title):
    """从变更描述提取候选关键词：英文 token（≥3 字母）+ 中文 4 字滑窗
    （覆盖"跳跃手感"这类跨块关键片段，避免 6 字整块切掉语义）。"""
    text = f"{title} {description}"
    words = set()
    for m in re.findall(r"[A-Za-z_][\w-]*", text):
        if len(m) >= 3:
            words.add(m.lower())
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(chunk) <= 4:
            words.add(chunk)
        else:
            for i in range(len(chunk) - 3):
                words.add(chunk[i:i + 4])
    return words or {text[:6]}


def _dependency_closure(task_ids, all_tasks):
    """波及闭包：命中任务 + 所有（直接/间接）依赖它们的下游任务。
    变更实现 → 影响验收/集成等依赖它的任务，因此沿 depends_on 反向传播。"""
    by_dep = {}                        # tid -> [依赖它的任务 id]
    for t in all_tasks:
        deps = json.loads(t["depends_on"]) if t.get("depends_on") else []
        for d in deps:
            by_dep.setdefault(d, []).append(t["id"])
    closure = set()
    stack = list(task_ids)
    while stack:
        tid = stack.pop()
        if tid in closure:
            continue
        closure.add(tid)
        for nxt in by_dep.get(tid, []):
            if nxt not in closure:
                stack.append(nxt)
    return closure


def assess(rid):
    """影响评估：波及任务（关键词命中 + 依赖闭包）+ 预算影响。
    写 impact JSON + 审计（action=change_assess）。返回 impact dict。"""
    row = get(rid)
    if not row:
        raise KeyError(f"变更单不存在: {rid}")
    conn = db.connect()
    try:
        all_tasks = [dict(r) for r in conn.execute(
            "SELECT * FROM tasks").fetchall()]
    finally:
        conn.close()
    kws = _keywords(row["description"] or "", row["title"] or "")
    hit = [t for t in all_tasks
           if any(k in f"{t['title']} {t['description'] or ''}".lower()
                  for k in kws)]
    hit_ids = [t["id"] for t in hit]
    closure = _dependency_closure(hit_ids, all_tasks)
    affected = [t for t in all_tasks if t["id"] in closure]
    # 预算影响：波及任务 budget_tokens 之和 + 已用成本（budget 表按 task_id）
    budget_total = sum(t["budget_tokens"] or 0 for t in affected)
    cost_used = 0.0
    for t in affected:
        cost_used += _task_cost(t["id"])
    impact = {
        "keywords": sorted(kws),
        "hit_tasks": hit_ids,
        "affected_tasks": [t["id"] for t in affected],
        "affected_titles": {t["id"]: t["title"] for t in affected},
        "dependency_closure_size": len(closure),
        "budget_tokens_total": budget_total,
        "cost_usd_used": round(cost_used, 4),
        "assessed_at": db.now(),
    }
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE change_requests SET impact=?, affected_tasks=? WHERE id=?",
            (db.json_dumps(impact), db.json_dumps([t["id"] for t in affected]),
             rid))
        conn.commit()
    finally:
        conn.close()
    _log("change_assess", rid, impact)
    return impact


def _task_cost(task_id):
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) v FROM budget WHERE task_id=?",
            (task_id,)).fetchone()["v"]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 决策
# --------------------------------------------------------------------------
def approve(rid, notes="", apply=False):
    """批准变更单：状态 approved + 审计；apply=True 时把波及任务重置 backlog
    （重新评估派发，让变更生效）。返回 (status, affected_ids)。"""
    row = get(rid)
    if not row:
        raise KeyError(f"变更单不存在: {rid}")
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE change_requests SET status='approved', decision_notes=?, "
            "decided_at=? WHERE id=?", (notes, db.now(), rid))
        affected = json.loads(row["affected_tasks"]) \
            if row["affected_tasks"] else []
        if apply and affected:
            for tid in affected:
                conn.execute(
                    "UPDATE tasks SET status='backlog', updated_at=? WHERE id=? "
                    "AND status IN ('backlog','todo','in_progress','in_review')",
                    (db.now(), tid))
        conn.commit()
    finally:
        conn.close()
    _log("change_approve", rid, {"notes": notes, "apply": apply,
                                 "affected": affected})
    return "approved", affected


def reject(rid, notes=""):
    """拒绝变更单：状态 rejected + 审计。"""
    row = get(rid)
    if not row:
        raise KeyError(f"变更单不存在: {rid}")
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE change_requests SET status='rejected', decision_notes=?, "
            "decided_at=? WHERE id=?", (notes, db.now(), rid))
        conn.commit()
    finally:
        conn.close()
    _log("change_reject", rid, {"notes": notes})
    return "rejected"


def _log(action, target, detail=None):
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO audit_log (ts,actor_id,action,target,detail) "
            "VALUES (?,?,?,?,?)",
            (db.now(), "producer", action, target,
             db.json_dumps(detail) if detail else None))
        conn.commit()
    finally:
        conn.close()
