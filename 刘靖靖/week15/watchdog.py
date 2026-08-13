"""看门狗（Watchdog）：实时监控 harness 是否在某个流程卡住 → 主模型诊断 → 自动修复 → 审计留痕。

与主进程（python -m src.main --demo ...）并行运行，纯 DB 判定、不依赖进程查询：

检测维度（每轮轮询，全部只读 DB）：
- 任务级：in_progress 长期无产出 / in_review 长期未评审 / backlog 依赖已满足却未派发
- Agent 级：消息重投超限被标记 blocked / in_progress 任务 owner 已退役（孤儿任务）
- 消息级：pending 超时未 ack
- 里程碑级：里程碑长期未推进
- 全局级：DB 状态快照连续 N 分钟无任何变化（主进程异常退出/死锁）

修复链路（检测到卡住后）：
1. 收集卡住证据（精简结构化文本）
2. 调主模型（config.routing 的角色，默认 main，可配 reasoning）输出结构化修复决策 JSON
3. 逐条安全执行（动作白名单 + 目标存在性校验），全部写 audit_log（actor=watchdog）
4. 防抖：同一目标冷却期内不重复修复；修复次数超上限 → 升级人工（微信/日志通知）
5. LLM 失败/输出非法 → 启发式保守回退（重派卡住任务 + 解封 blocked agent）

用法（在 backend/ 目录下）：
    py313\\Scripts\\python.exe -m src.watchdog                     # 监控 + 自动修复
    py313\\Scripts\\python.exe -m src.watchdog --no-fix            # 只检测不修复（纯观察）
    py313\\Scripts\\python.exe -m src.watchdog --once              # 跑一轮就退出（测试/调试）
    py313\\Scripts\\python.exe -m src.watchdog --keep-running      # 完成后不退出，持续监控
    py313\\Scripts\\python.exe -m src.watchdog --stale-min 3       # 覆盖任务卡住阈值（分钟）
    py313\\Scripts\\python.exe -m src.watchdog --project <name>    # 多项目监控
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # backend/ 根（src 包所在），任意 cwd 可启动

from src import db            # noqa: E402
from src import mail           # noqa: E402
from src import tickets        # noqa: E402
from src.llm import LLMRouter  # noqa: E402
from src.watch import notify_wecom  # noqa: E402  # 复用微信通知（未配置 webhook 时仅本地日志）

DEFAULT_CFG = {
    "interval_s": 30,          # 轮询间隔（秒）
    "task_stale_min": 8,       # 任务卡住阈值（分钟）：单轮干活含多次 LLM 调用，需大于最长单轮时长
    "agent_stale_min": 8,      # agent 卡住阈值（分钟）
    "msg_stale_min": 3,        # 消息 pending 卡住阈值（分钟）
    "milestone_stale_min": 15, # 里程碑无推进阈值（分钟）
    "global_stale_min": 12,    # 全局快照无变化阈值（分钟）
    "cooldown_min": 6,         # 同一目标修复冷却（分钟）
    "max_fixes": 3,            # 同一目标累计修复上限，超限升级人工
    "max_actions_per_round": 6,  # 每轮最多执行修复动作数（防 LLM 刷屏）
    "llm_role": "main",        # 修复决策用哪个路由角色（reasoning 更聪明但更贵）
    "llm_temperature": 0.2,    # 决策温度：偏低保证稳定输出
}

# 动作白名单（防止 LLM 幻觉出危险动作）
ACTION_WHITELIST = {
    "requeue_task":     "把卡住任务重置回 backlog，由主 agent 重新派发（仅限 in_progress/in_review）",
    "unblock_agent":    "解除 blocked 状态（消息重投超限导致的冻结）",
    "nudge_agent":      "由 producer 代发催促邮件，推动卡住的 agent（评审积压/未派发）",
    "rescue_orphan":    "转交孤儿任务（owner 已退役）给 supervisor 的常驻 sub",
    "resend_messages":  "重投超时未 ack 的消息（调用 mail.retry_expired）",
    "escalate_human":   "自动修复不可行，升级人工介入（微信/日志通知）",
    "noop":             "观察：当前证据不足以判定需修复（可能是正常等待依赖）",
}

TASK_TERMINAL = ("done", "escalated")
TASK_ACTIVE = ("backlog", "todo", "in_progress", "in_review")

_BJ = timezone(timedelta(hours=8))


def _parse_ts(ts):
    """'YYYY-MM-DD HH:MM:SS' → datetime（北京时间）；非法返回 None。"""
    if not ts:
        return None
    try:
        return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_BJ)
    except (ValueError, TypeError):
        return None


def _age_min(ts):
    """时间戳距今分钟数；无法解析返回 None。"""
    dt = _parse_ts(ts)
    if dt is None:
        return None
    return (datetime.now(_BJ) - dt).total_seconds() / 60.0


class Watchdog:
    """看门狗：检测 → 决策 → 修复 → 防抖。"""

    def __init__(self, cfg, router=None, webhook="", fix=True, project=None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.router = router
        self.webhook = webhook
        self.fix_enabled = fix
        self.project = project
        self.fix_state = {}      # target_key -> {"ts": float, "count": int}
        self.last_snap = None
        self.global_stale_since = None
        self.seen_data = False   # 启动保护：DB 无任务/里程碑时不判定
        self.completed = False
        self._last_context = "（无）"

    # ------------------------------------------------------------------ 审计/通知
    def log(self, action, target, detail=None):
        """watchdog 的全部动作写审计日志（actor=watchdog，可追溯）。"""
        try:
            conn = db.connect()
            conn.execute(
                "INSERT INTO audit_log (ts,actor_id,action,target,detail) VALUES (?,?,?,?,?)",
                (db.now(), "watchdog", action, target,
                 db.json_dumps(detail) if detail else None))
            conn.commit()
            conn.close()
        except Exception as e:  # noqa: BLE001
            print(f"[watchdog] 审计写入失败: {e}", flush=True)

    def notify(self, title, content):
        ok = notify_wecom(self.webhook, title, content)
        return ok

    # ------------------------------------------------------------------ 检测
    def db_snapshot(self):
        """任务+里程碑状态+最新审计时间快照（全局卡住判定）。"""
        conn = db.connect()
        try:
            tasks = [f"{r['id']}={r['status']}" for r in conn.execute(
                "SELECT id,status FROM tasks ORDER BY id")]
            mss = [f"{r['id']}={r['status']}" for r in conn.execute(
                "SELECT id,status FROM milestones ORDER BY id")]
            last_audit = conn.execute(
                "SELECT COALESCE(MAX(ts),'') v FROM audit_log").fetchone()["v"]
            return "|".join(tasks) + "||" + "|".join(mss) + "||" + last_audit
        finally:
            conn.close()

    def _stuck_tasks(self):
        """任务级卡住：
        - in_progress 超过阈值 → subagent 干活卡住
        - in_review 超过阈值 → 主 agent 评审卡住
        - backlog 依赖已满足但超过阈值未派发 → 主 agent 派发卡住
        """
        stale = self.cfg["task_stale_min"]
        conn = db.connect()
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM tasks").fetchall()]
            agents = {r["id"]: dict(r) for r in conn.execute(
                "SELECT id,role,dept,name,status,parent_id,is_temp FROM agents").fetchall()}
        finally:
            conn.close()
        out = []
        for t in rows:
            if t["status"] not in TASK_ACTIVE:
                continue
            age = _age_min(t["updated_at"])
            if age is None:
                continue
            owner = agents.get(t["owner_id"]) or {}
            sup = agents.get(t["supervisor_id"]) or {}
            item = {
                "id": t["id"], "title": t["title"], "status": t["status"],
                "dept": t["dept"], "owner": t["owner_id"],
                "owner_alive": t["owner_id"] in agents,
                "supervisor": t["supervisor_id"],
                "supervisor_alive": t["supervisor_id"] in agents,
                "age_min": round(age, 1), "review_rounds": t["review_rounds"],
                "max_review_rounds": t["max_review_rounds"],
                "depends_on": t["depends_on"] or None,
                "priority": t["priority"],
            }
            if t["status"] == "backlog":
                # 依赖未满足 = 正常等待；依赖满足仍不派发 = 派发卡住
                deps = json.loads(t["depends_on"]) if t["depends_on"] else []
                blocked_by = [d for d in deps
                              if not any(x["id"] == d and x["status"] == "done"
                                         for x in rows)]
                if blocked_by:
                    continue   # 正常依赖等待，不算卡住
                if age >= stale and t["owner_id"]:
                    item["kind"] = "dispatch_stuck"   # 就绪却未派发
                    out.append(item)
            elif t["status"] == "in_progress":
                if age >= stale:
                    item["kind"] = "work_stuck"
                    out.append(item)
            elif t["status"] == "in_review":
                if age >= stale:
                    item["kind"] = "review_stuck"
                    out.append(item)
        return out

    def _blocked_agents(self):
        conn = db.connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT id,role,dept,name,status FROM agents "
                "WHERE status='blocked'").fetchall()]
        finally:
            conn.close()
        return rows

    def _orphan_tasks(self):
        """in_progress 且 owner 已不存在/退役（孤儿任务，需要转交）。"""
        conn = db.connect()
        try:
            tasks = [dict(r) for r in conn.execute(
                "SELECT id,title,owner_id,supervisor_id,status FROM tasks "
                "WHERE status='in_progress'").fetchall()]
            alive = {r["id"] for r in conn.execute(
                "SELECT id FROM agents WHERE status!='retired'").fetchall()}
        finally:
            conn.close()
        out = []
        for t in tasks:
            if t["owner_id"] is None or t["owner_id"] not in alive:
                out.append(t)
        return out

    def _stuck_messages(self):
        stale = self.cfg["msg_stale_min"]
        conn = db.connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT id,from_id,to_id,subject,task_id,ack_status,retry_count,"
                "sent_at,last_retry_at FROM messages "
                "WHERE ack_status='pending' AND to_id IS NOT NULL").fetchall()]
        finally:
            conn.close()
        out = []
        for m in rows:
            age = _age_min(m["last_retry_at"] or m["sent_at"])
            if age is not None and age >= stale:
                m["age_min"] = round(age, 1)
                out.append(m)
        return out

    def _stuck_milestones(self):
        stale = self.cfg["milestone_stale_min"]
        conn = db.connect()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT id,name,status,planned_at,done_at FROM milestones").fetchall()]
            n_open = {r["milestone_id"]: r["c"] for r in conn.execute(
                "SELECT milestone_id, COUNT(*) c FROM tasks "
                "WHERE status NOT IN ('done','escalated') GROUP BY milestone_id").fetchall()}
        finally:
            conn.close()
        out = []
        for m in rows:
            if m["status"] in ("done", "failed"):
                continue
            age = _age_min(m["planned_at"] or m["done_at"])
            if age is not None and age >= stale and n_open.get(m["id"], 0) > 0:
                m["open_tasks"] = n_open.get(m["id"], 0)
                m["age_min"] = round(age, 1)
                out.append(m)
        return out

    def detect(self):
        """一轮检测，返回结构化卡住证据。"""
        stuck_tasks = self._stuck_tasks()
        blocked_agents = self._blocked_agents()
        orphan_tasks = self._orphan_tasks()
        stuck_msgs = self._stuck_messages()
        stuck_ms = self._stuck_milestones()

        # 全局快照无变化（主进程死锁/退出）
        snap = self.db_snapshot()
        global_stale_min = None
        if self.last_snap is not None and snap == self.last_snap:
            if self.global_stale_since is None:
                self.global_stale_since = time.time()
            elif (time.time() - self.global_stale_since) >= self.cfg["global_stale_min"] * 60:
                global_stale_min = round(
                    (time.time() - self.global_stale_since) / 60.0, 1)
        else:
            self.global_stale_since = None
            self.last_snap = snap

        # 启动保护：无任何任务/里程碑时（宪章生成阶段），跳过全局快照卡住判定
        # （agents 已注册但 tasks/milestones 为空属正常；全局"无变化"不可作为卡住证据）
        conn = db.connect()
        try:
            n_task = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]
            n_ms = conn.execute("SELECT COUNT(*) c FROM milestones").fetchone()["c"]
        finally:
            conn.close()
        booting = (n_task == 0 and n_ms == 0)
        if booting:
            global_stale_min = None
            self.global_stale_since = None

        has_stuck = bool(stuck_tasks or blocked_agents or orphan_tasks
                         or stuck_msgs or stuck_ms or global_stale_min)
        return {
            "booting": booting,
            "stuck_tasks": stuck_tasks,
            "blocked_agents": blocked_agents,
            "orphan_tasks": orphan_tasks,
            "stuck_messages": stuck_msgs,
            "stuck_milestones": stuck_ms,
            "global_stale_min": global_stale_min,
            "has_stuck": has_stuck,
        }

    # ------------------------------------------------------------------ 决策
    def _target_key(self, action):
        """防抖 key：按目标去重（escalate/noop 用全局 key）。"""
        if action["action"] in ("escalate_human", "noop"):
            return "global"
        for k in ("task_id", "agent_id", "msg_id"):
            if action.get(k):
                return f"{action['action']}:{action[k]}"
        return f"{action['action']}:global"

    def _in_cooldown(self, key):
        st = self.fix_state.get(key)
        if not st:
            return False
        return (time.time() - st["ts"]) < self.cfg["cooldown_min"] * 60

    def _over_fix_limit(self, key):
        st = self.fix_state.get(key)
        return bool(st and st["count"] >= self.cfg["max_fixes"])

    def build_context(self, ev):
        """把卡住证据压成紧凑文本（token 优化：只带卡住项，不 dump 大字段）。"""
        lines = []
        if ev.get("stuck_tasks"):
            lines.append("卡住任务（task 卡住清单）：")
            for t in ev["stuck_tasks"]:
                lines.append(
                    f"- {t['id']} [{t['kind']}] {t['title']} status={t['status']} "
                    f"owner={t['owner']} supervisor={t['supervisor']} "
                    f"age={t['age_min']}min review_rounds={t['review_rounds']}/"
                    f"{t['max_review_rounds']} deps={t['depends_on']}")
        if ev.get("blocked_agents"):
            lines.append("blocked agent（消息重投超限被冻结）：")
            for a in ev["blocked_agents"]:
                lines.append(f"- {a['id']} ({a['dept']} {a['name']})")
        if ev.get("orphan_tasks"):
            lines.append("孤儿任务（owner 已退役，in_progress 无人干）：")
            for t in ev["orphan_tasks"]:
                lines.append(f"- {t['id']} {t['title']} owner={t['owner_id']} "
                             f"supervisor={t['supervisor_id']}")
        if ev.get("stuck_messages"):
            lines.append("pending 超时消息：")
            for m in ev["stuck_messages"]:
                lines.append(f"- {m['id']} {m['from_id']}→{m['to_id']} "
                             f"task={m['task_id']} retry={m['retry_count']} "
                             f"age={m['age_min']}min")
        if ev.get("stuck_milestones"):
            lines.append("里程碑长期未推进：")
            for m in ev["stuck_milestones"]:
                lines.append(f"- {m['id']} {m['name']} status={m['status']} "
                             f"open_tasks={m['open_tasks']} age={m['age_min']}min")
        if ev.get("global_stale_min"):
            lines.append(f"全局：DB 状态快照已 {ev['global_stale_min']} 分钟无任何变化"
                         "（主进程可能异常退出/死锁）")
        return "\n".join(lines)

    def decide(self, ev):
        """主模型修复决策。返回动作列表（已按防抖过滤、含升级项）。"""
        actions = []
        escalate_reasons = []

        # 1) 主模型决策（可用则用）
        llm_actions = self._llm_decide(ev)
        if llm_actions:
            actions = llm_actions
        else:
            # 2) 启发式回退：保守可靠
            actions = self._heuristic_decide(ev)

        # 3) 防抖过滤 + 修复次数上限 → 升级人工
        final = []
        for a in actions:
            key = self._target_key(a)
            if a["action"] == "noop":
                continue
            if self._over_fix_limit(key):
                escalate_reasons.append(
                    f"{a.get('task_id') or a.get('agent_id') or '目标'} 已修复 "
                    f"{self.cfg['max_fixes']} 次仍卡住：{a.get('reason', '')}")
                continue
            if self._in_cooldown(key):
                continue
            final.append(a)
        if escalate_reasons and self.fix_enabled:
            final.append({"action": "escalate_human",
                          "reason": "；".join(escalate_reasons)})
        # 每轮动作上限
        return final[:self.cfg["max_actions_per_round"]]

    def _llm_decide(self, ev):
        if not self.router:
            return []
        context = self.build_context(ev)
        if not context:
            return []
        system = (
            "你是 Harness 多 agent 游戏开发系统的看门狗决策器。系统出现流程卡住，"
            "请根据证据诊断根因并输出修复动作。\n"
            "动作白名单（只能选这些）：" + json.dumps(ACTION_WHITELIST, ensure_ascii=False) + "\n"
            "规则：\n"
            "- 只输出一个 JSON 数组，不要任何解释文字、不要 markdown 代码块。\n"
            "- 每项形如 {\"action\": \"requeue_task\", \"task_id\": \"T-0001\", "
            "\"reason\": \"一句话根因\"}；agent 类动作用 \"agent_id\"，消息用 \"msg_id\"。\n"
            "- in_progress 卡住 → requeue_task（重新派发）；in_review 卡住 → nudge_agent "
            "（催促 supervisor 评审）；backlog 就绪未派发 → nudge_agent（催促派发）。\n"
            "- blocked agent → unblock_agent；孤儿任务 → rescue_orphan；"
            "pending 消息超时 → resend_messages。\n"
            "- 证据不足、可能只是正常等待时输出 []（空数组）。\n"
            "- 全局快照长时间无变化且无法用上述动作解决 → escalate_human。\n"
            "- 不要编造证据中不存在的 task_id/agent_id。")
        user = f"当前卡住证据：\n{context}"
        try:
            text = self.router.complete(
                self.cfg["llm_role"], system, user,
                temperature=self.cfg["llm_temperature"], max_tokens=1000)
            actions = self.parse_actions(text, ev)
            if actions is None:
                print("[watchdog] LLM 决策无法解析，回退启发式规则", flush=True)
                return []
            return actions
        except Exception as e:  # noqa: BLE001
            print(f"[watchdog] LLM 决策失败（{e}），回退启发式规则", flush=True)
            return []

    def _heuristic_decide(self, ev):
        """启发式保守回退：不调 LLM 也能自救的确定性规则。"""
        actions = []
        for t in ev.get("stuck_tasks", []):
            if t["kind"] == "work_stuck":
                actions.append({"action": "requeue_task", "task_id": t["id"],
                                "reason": f"启发式：in_progress 卡住 {t['age_min']} 分钟，重置重派"})
            elif t["kind"] in ("review_stuck", "dispatch_stuck"):
                actions.append({"action": "nudge_agent", "agent_id": t["supervisor"],
                                "task_id": t["id"],
                                "reason": f"启发式：{t['kind']} 卡住 {t['age_min']} 分钟，催促处理"})
        for a in ev.get("blocked_agents", []):
            actions.append({"action": "unblock_agent", "agent_id": a["id"],
                            "reason": "启发式：消息重投超限被冻结，解除阻塞"})
        for t in ev.get("orphan_tasks", []):
            actions.append({"action": "rescue_orphan", "task_id": t["id"],
                            "agent_id": t["supervisor_id"],
                            "reason": "启发式：owner 已退役，转交常驻 sub"})
        if ev.get("stuck_messages"):
            actions.append({"action": "resend_messages",
                            "reason": f"启发式：{len(ev['stuck_messages'])} 条消息 pending 超时"})
        if ev.get("global_stale_min"):
            actions.append({"action": "escalate_human",
                            "reason": f"全局快照 {ev['global_stale_min']} 分钟无变化，"
                                      "疑似主进程异常，需人工确认"})
        return actions

    def parse_actions(self, text, ev):
        """解析 LLM 输出为动作列表；非法则返回 None（走回退）。"""
        if not text or not text.strip():
            return None
        raw = text.strip()
        # 去掉 markdown 代码块包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:  # noqa: BLE001
            # 找第一个 [ ... ] 或 { ... }
            import re
            m = re.search(r"\[[\s\S]*\]", raw)
            if not m:
                m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    return None
            else:
                return None
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list):
            return None
        # 存在性校验（防幻觉 id）
        task_ids = {t["id"] for t in ev.get("stuck_tasks", [])} | \
                   {t["id"] for t in ev.get("orphan_tasks", [])}
        agent_ids = {a["id"] for a in ev.get("blocked_agents", [])} | \
                    {t.get("agent_id") for t in ev.get("stuck_tasks", [])} | \
                    {t.get("supervisor_id") for t in ev.get("stuck_tasks", [])} | \
                    {t.get("supervisor_id") for t in ev.get("orphan_tasks", [])}
        out = []
        for it in obj:
            if not isinstance(it, dict):
                continue
            action = it.get("action")
            if action not in ACTION_WHITELIST:
                continue
            if action == "resend_messages":
                out.append({"action": action, "reason": it.get("reason", "")})
                continue
            if action == "escalate_human":
                out.append({"action": action, "reason": it.get("reason", "")})
                continue
            if action == "nudge_agent":
                aid = it.get("agent_id")
                if aid and aid in agent_ids:
                    out.append({"action": action, "agent_id": aid,
                                "task_id": it.get("task_id"),
                                "reason": it.get("reason", "")})
                continue
            if action == "requeue_task":
                tid = it.get("task_id")
                if tid and tid in task_ids:
                    out.append({"action": action, "task_id": tid,
                                "reason": it.get("reason", "")})
                continue
            if action == "unblock_agent":
                aid = it.get("agent_id")
                if aid and aid in agent_ids:
                    out.append({"action": action, "agent_id": aid,
                                "reason": it.get("reason", "")})
                continue
            if action == "rescue_orphan":
                tid = it.get("task_id")
                if tid and tid in {t["id"] for t in ev.get("orphan_tasks", [])}:
                    out.append({"action": action, "task_id": tid,
                                "agent_id": it.get("agent_id"),
                                "reason": it.get("reason", "")})
                continue
        return out

    # ------------------------------------------------------------------ 执行
    def execute(self, action):
        """执行单个修复动作。返回 (ok, msg)。所有动作写审计 + 防抖状态。"""
        act = action["action"]
        key = self._target_key(action)
        if act in ("escalate_human", "noop"):
            pass
        else:
            st = self.fix_state.setdefault(key, {"ts": 0, "count": 0})
            st["ts"] = time.time()
            st["count"] += 1

        try:
            if act == "requeue_task":
                tid = action["task_id"]
                conn = db.connect()
                row = conn.execute(
                    "SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
                conn.close()
                if not row:
                    return False, f"任务 {tid} 不存在，跳过"
                if row["status"] not in ("in_progress", "in_review"):
                    return False, f"任务 {tid} 状态 {row['status']} 不可重派（仅 in_progress/in_review）"
                tickets.set_status(tid, "backlog")
                self.log("watchdog_requeue", tid,
                         {"reason": action.get("reason"),
                          "from": row["status"]})
                return True, f"任务 {tid} 已重置回 backlog（原 {row['status']}）"

            if act == "unblock_agent":
                aid = action["agent_id"]
                conn = db.connect()
                row = conn.execute(
                    "SELECT status FROM agents WHERE id=?", (aid,)).fetchone()
                if row and row["status"] == "blocked":
                    conn.execute("UPDATE agents SET status='idle' WHERE id=?",
                                 (aid,))
                    conn.commit()
                    self.log("watchdog_unblock", aid,
                             {"reason": action.get("reason")})
                    conn.close()
                    return True, f"agent {aid} 已解封（blocked → idle）"
                conn.close()
                return False, f"agent {aid} 不存在或未阻塞"

            if act == "nudge_agent":
                aid = action["agent_id"]
                tid = action.get("task_id")
                conn = db.connect()
                producer = conn.execute(
                    "SELECT id FROM agents WHERE role='producer' LIMIT 1").fetchone()
                conn.close()
                if not producer:
                    return False, "无 producer（agents 表为空），无法代发催促邮件"
                subject = (f"[watchdog] 催促：任务 {tid} 卡住" if tid
                           else "[watchdog] 催促：有卡住项待处理")
                body = (f"看门狗检测到流程卡住，请尽快处理：\n"
                        f"任务：{tid or '（多个）'}\n"
                        f"诊断：{action.get('reason', '')}\n"
                        f"时间：{db.now()}")
                msg_id = mail.send(producer["id"], aid, subject, body,
                                   task_id=tid, priority="urgent")
                self.log("watchdog_nudge", aid,
                         {"msg_id": msg_id, "task_id": tid,
                          "reason": action.get("reason")})
                return True, f"已由 producer 代发催促邮件给 {aid}（{msg_id}）"

            if act == "rescue_orphan":
                tid = action["task_id"]
                sup_id = action.get("agent_id")
                conn = db.connect()
                try:
                    # 找 supervisor 的常驻 sub（parent=supervisor, role=sub, 非临时）
                    sub = conn.execute(
                        "SELECT id FROM agents WHERE parent_id=? AND role='sub' "
                        "AND is_temp=0 AND status!='retired' LIMIT 1",
                        (sup_id,)).fetchone()
                    if not sub:
                        return False, f"supervisor {sup_id} 无可用常驻 sub，无法转交"
                    n = conn.execute(
                        "UPDATE tasks SET owner_id=?, updated_at=? WHERE id=? "
                        "AND status='in_progress'",
                        (sub["id"], db.now(), tid)).rowcount
                    conn.commit()
                    if n:
                        self.log("watchdog_rescue", tid,
                                 {"to": sub["id"], "from": "orphan",
                                  "reason": action.get("reason")})
                        return True, f"孤儿任务 {tid} 已转交常驻 sub {sub['id']}"
                    return False, f"任务 {tid} 不存在或不在 in_progress"
                finally:
                    conn.close()

            if act == "resend_messages":
                retried, blocked = mail.retry_expired(
                    timeout_s=max(30, self.cfg["msg_stale_min"] * 60),
                    max_retries=3)
                self.log("watchdog_resend", "messages",
                         {"retried": len(retried), "blocked": blocked,
                          "reason": action.get("reason")})
                return True, f"重投消息 {len(retried)} 条；新增 blocked {len(blocked)} 个"

            if act == "escalate_human":
                content = (f"看门狗自动修复已达上限或无法处理：\n"
                           f"{action.get('reason', '')}\n"
                           f"证据：\n{self._last_context or '（无）'}\n"
                           f"建议人工检查：DB={db.DB_PATH}")
                self.notify("🚨 Harness 需人工介入（自动修复失败）", content)
                self.log("watchdog_escalate", "human",
                         {"reason": action.get("reason")})
                return True, "已通知人工介入"

            if act == "noop":
                return False, "观察中，不动作"

            return False, f"未知动作 {act}"
        except Exception as e:  # noqa: BLE001
            self.log("watchdog_action_error", act,
                     {"error": str(e), "action": action})
            return False, f"执行 {act} 异常: {e}"

    # ------------------------------------------------------------------ 主循环
    def run_once(self, verbose=True):
        """跑一轮：检测 → 决策 → 执行。返回本轮执行的动作数。"""
        ev = self.detect()
        if ev.get("booting") and not ev.get("has_stuck"):
            if verbose:
                print("[watchdog] 尚无任务/里程碑（宪章生成中），继续等待", flush=True)
            return 0

        # 完成判定：存在里程碑且全部 done
        conn = db.connect()
        try:
            ms_rows = [dict(r) for r in conn.execute(
                "SELECT id,name,status FROM milestones ORDER BY id")]
        finally:
            conn.close()
        if ms_rows and all(m["status"] == "done" for m in ms_rows):
            if not self.completed:
                lines = "\n".join(f"  {m['id']} {m['name']}: {m['status']}" for m in ms_rows)
                self.notify("✅ Harness 运行完成（看门狗）",
                            f"里程碑：\n{lines}")
                self.log("watchdog_done", "milestones",
                         {"n": len(ms_rows)})
                self.completed = True
                if verbose:
                    print("[watchdog] 全部里程碑 done，运行完成", flush=True)
            return 0

        if not ev.get("has_stuck"):
            if verbose:
                print("[watchdog] 无卡住迹象，继续监控", flush=True)
            return 0

        context = self.build_context(ev)
        self._last_context = context
        if verbose:
            print(f"[watchdog] 检测到卡住：\n{context}", flush=True)

        if not self.fix_enabled:
            if verbose:
                print("[watchdog] --no-fix 模式：仅报告不修复", flush=True)
            self.log("watchdog_detect", "stuck", {"context": context[:500]})
            return 0

        actions = self.decide(ev)
        if not actions:
            if verbose:
                print("[watchdog] 无可执行修复动作（冷却中或证据不足）", flush=True)
            return 0
        n = 0
        for a in actions:
            ok, msg = self.execute(a)
            print(f"  [修复] {a['action']}: {'✓' if ok else '✗'} {msg}", flush=True)
            if ok:
                n += 1
        return n

    def run(self):
        interval = self.cfg["interval_s"]
        print(f"[watchdog] 启动 · 轮询 {interval}s · 任务卡住阈值 "
              f"{self.cfg['task_stale_min']}min · 修复{'启用' if self.fix_enabled else '禁用'}"
              f" · LLM 角色 {self.cfg['llm_role']}", flush=True)
        while True:
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001   # 单轮异常不退出，继续监控
                import traceback
                print(f"[watchdog] 本轮异常: {e}\n{traceback.format_exc()}", flush=True)
                self.log("watchdog_round_error", "round", {"error": str(e)})
            time.sleep(interval)


def _load_config():
    try:
        import yaml
        p = os.path.join(os.path.dirname(_HERE), "config.yaml")
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def _build_router(cfg, force_mock=False):
    llm_cfg = cfg.get("llm") or {}
    if force_mock:
        llm_cfg = json.loads(json.dumps(llm_cfg))
        for m in (llm_cfg.get("models") or {}).values():
            m["provider"] = "mock"
    return LLMRouter(llm_cfg)


def main():
    ap = argparse.ArgumentParser(description="Harness 看门狗：卡住检测 → 主模型修复 → 审计")
    ap.add_argument("--interval", type=int, default=0,
                    help="轮询间隔秒数（覆盖 config）")
    ap.add_argument("--stale-min", type=int, default=0,
                    help="任务卡住阈值分钟（覆盖 config，快捷调参）")
    ap.add_argument("--no-fix", action="store_true", help="只检测不修复（纯观察模式）")
    ap.add_argument("--once", action="store_true", help="只跑一轮就退出（测试/调试）")
    ap.add_argument("--keep-running", action="store_true",
                    help="完成后不退出，持续监控")
    ap.add_argument("--mock", action="store_true",
                    help="强制 LLM 走 mock（沙箱验证用）")
    ap.add_argument("--webhook", default=None, help="企业微信 webhook（覆盖 config）")
    ap.add_argument("--project", default=None, help="多项目：监控指定项目")
    ap.add_argument("--no-router", action="store_true",
                    help="不构建 LLM 路由（纯启发式修复，测试用）")
    args = ap.parse_args()

    if args.project:
        db.set_project(args.project)
    db.init_db()

    cfg = _load_config()
    wd_cfg = {**DEFAULT_CFG, **((cfg.get("watchdog") or {}))}
    if args.interval:
        wd_cfg["interval_s"] = args.interval
    if args.stale_min:
        wd_cfg["task_stale_min"] = args.stale_min
        wd_cfg["agent_stale_min"] = max(wd_cfg["agent_stale_min"], args.stale_min)
    webhook = args.webhook or ((cfg.get("notify") or {}).get("wecom_webhook", ""))
    router = None if args.no_router else _build_router(cfg, force_mock=args.mock)

    wd = Watchdog(wd_cfg, router=router, webhook=webhook,
                  fix=not args.no_fix, project=args.project)

    if args.once:
        n = wd.run_once()
        print(f"[watchdog] 单轮完成，执行修复 {n} 条", flush=True)
        return
    wd.run()


if __name__ == "__main__":
    main()
