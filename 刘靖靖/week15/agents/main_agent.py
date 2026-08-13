"""通用主 agent（admin）：策划/程序/美术/QA/PM 同构，由 prompt 包 + 权限区分。
职责（PRD §7.3）：派发任务单 → 收集产出 → 独立评估 → 通过则审批合入 / 打回修订。
"""
import os
import shutil

from src import db
from src import tickets
from src import validation
from src.agents.base import Agent
from src.agents.sub_agent import SubAgent


class MainAgent(Agent):
    # 部门完成通知的主题（§5.2 主题路由：CC 按主题投递给订阅者）
    TOPIC_BY_DEPT = {"策划": "数值", "程序": "构建", "美术": "美术", "QA": "质量"}

    def __init__(self, agent_id, dept, name, router=None, context_pack=None,
                 cc_targets=None, force_reject=None, topics_cfg=None,
                 parent_id="producer"):
        super().__init__(agent_id, "main", dept, name, router=router,
                         context_pack=context_pack, topics_cfg=topics_cfg,
                         parent_id=parent_id)
        self.cc_targets = cc_targets or []      # 完成时 CC 同步的主 agent 列表
        self.force_reject = force_reject        # None | "once" | "always"（演示用）
        self._rejected_once = False
        self.temp_subs = []                     # 弹性扩容的临时 subagent（本部门）
        self.resident_sub_id = None             # 常驻 sub（临时工未完成任务转交目标）
        self.budget_hold = False                # 日级预算超限（§9.2：暂停低优先级派发）

    # -- 0) 弹性扩容：临时 subagent 创建 / 回收（C4：subagent 无此权限，仅主 agent） --
    def create_temp_subagent(self, dept, name, task_pack=None, temp_max=5):
        """创建临时 subagent（内容填充期弹性扩容）。受 temp_max 上限约束。
        task_pack: 一次性上下文包（宪章章节 + 模板 + DoD 参考），临时工无跨任务记忆。"""
        conn = db.connect()
        try:
            n = conn.execute(
                "SELECT COUNT(*) c FROM agents WHERE parent_id=? AND role='sub' "
                "AND is_temp=1 AND status!='retired'", (self.id,)).fetchone()["c"]
        finally:
            conn.close()
        if n >= temp_max:
            return None
        idx = len(self.temp_subs) + 1
        temp_id = f"{self.id}-t{idx}"
        # token 优化：临时工上下文补宪章摘要（来自主 agent，不重复生成）
        ctx = dict(task_pack or {})
        if "charter_summary" not in ctx and self.context_pack.get("charter_summary"):
            ctx["charter_summary"] = self.context_pack["charter_summary"]
        sub = SubAgent(temp_id, dept, name, self.id, router=self.router,
                       context_pack=ctx, topics_cfg=self.topics_cfg,
                       is_temp=True)
        conn = db.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO agents (id,role,dept,name,parent_id,status,"
                "max_subagents,is_temp,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (temp_id, "sub", dept, name, self.id, "idle", 0, 1, db.now()))
            conn.commit()
        finally:
            conn.close()
        self.log_action("create_agent", temp_id, {"temp": True, "dept": dept})
        self.temp_subs.append(sub)
        return sub

    def retire_temp_subagent(self, sub):
        """回收临时 subagent（干完归档，审计保留，不占常驻编制）。

        回收前转交：名下还有 backlog/in_progress 的任务先转给常驻 sub
        （resident_sub_id），避免临时工退役后任务变孤儿永远没人做。
        返回转交的任务数。
        """
        reassigned = 0
        conn = db.connect()
        try:
            rows = conn.execute(
                "SELECT id FROM tasks WHERE owner_id=? "
                "AND status IN ('backlog','in_progress')", (sub.id,)).fetchall()
            if rows and self.resident_sub_id:
                conn.execute(
                    "UPDATE tasks SET owner_id=? WHERE owner_id=? "
                    "AND status IN ('backlog','in_progress')",
                    (self.resident_sub_id, sub.id))
                reassigned = len(rows)
            conn.execute("UPDATE agents SET status='retired' WHERE id=? AND is_temp=1",
                         (sub.id,))
            conn.commit()
        finally:
            conn.close()
        if reassigned:
            self.log_action("task_reassign", sub.id,
                            {"to": self.resident_sub_id, "n": reassigned})
        self.log_action("retire_agent", sub.id,
                        {"temp": True, "reassigned": reassigned})
        return reassigned

    # -- 1) 派发就绪任务（backlog → in_progress，依赖满足才派） --
    def dispatch_ready(self):
        dispatched = []
        for t in tickets.tasks_by(supervisor_id=self.id, status="backlog"):
            if not t["owner_id"]:
                continue
            # §9.2 日级预算：超限时暂停低优先级（P2/P3）派发，P0/P1 关键任务放行
            if self.budget_hold and (t["priority"] or "P2") in ("P2", "P3"):
                continue
            ok, why = tickets.can_start(t["id"])
            if not ok:
                continue
            tickets.set_status(t["id"], "in_progress")
            self.send(t["owner_id"],
                      f"[{t['id']}] {t['title']}",
                      f"任务单派发（C6：工作只由任务单驱动）。\n"
                      f"需求：{t['description']}\nDoD：{t['dod']}\n"
                      f"依赖：{t['depends_on']}\n所属阶段：{t['stage']}",
                      task_id=t["id"], priority="high")
            self.log_action("task_dispatch", t["id"])
            dispatched.append(t["id"])
        return dispatched

    # -- 2) 评估提交（in_review → done / 打回 / 升级） --
    def review_submissions(self):
        results = []
        for t in tickets.tasks_by(supervisor_id=self.id, status="in_review"):
            # S3：合入点自动校验（客观门禁）——失败直接打回，不进入主观评估
            workdir = os.path.join(db.ARTIFACTS_DIR, "workspace",
                                   t["owner_id"], t["id"])
            checks = validation.validate(t["dept"], workdir)
            if checks and not all(c.passed for c in checks):
                msgs = self._reject_validation(t, checks)
                results.append(("validation_rejected", t["id"],
                                t["review_rounds"]))
                continue
            verdict = self._evaluate(t)
            passed = not self._verdict_rejects(verdict)
            if passed:
                self._approve_and_merge(t)
                results.append(("done", t["id"], 1))
            else:
                state, rounds = tickets.register_reject(t["id"])
                if state == "escalated":
                    self.send(self.parent_id,
                              f"[{t['id']}] 升级仲裁（打回超限）",
                              f"任务 {t['id']} 已打回 {rounds} 轮（上限 "
                              f"{t['max_review_rounds']}），请求 Producer 仲裁。\n"
                              f"最近评审意见：{verdict}",
                              task_id=t["id"], priority="urgent")
                    self.log_action("task_escalate", t["id"],
                                    {"rounds": rounds})
                    results.append(("escalated", t["id"], rounds))
                else:
                    self.send(t["owner_id"],
                              f"RE: [{t['id']}] 打回（第 {rounds} 轮）",
                              verdict, task_id=t["id"])
                    results.append(("rejected", t["id"], rounds))
        return results

    def _reject_validation(self, t, checks):
        """客观校验失败 → 记录检查结果 + 打回（附具体失败项）。"""
        msgs = [f"[{c.name}] {c.message}" for c in checks if not c.passed]
        conn = db.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (id,task_id,agent_id,path,version,status,checks) "
                "VALUES (?,?,?,?,?,?,?)",
                (f"ART-{t['id']}-c", t["id"], t["owner_id"], "", 1, "rejected",
                 db.json_dumps([c.to_dict() for c in checks])))
            conn.commit()
        finally:
            conn.close()
        self.log_action("validation_reject", t["id"],
                        {"checks": [c.to_dict() for c in checks]})
        state, rounds = tickets.register_reject(t["id"])
        if state == "escalated":
            self.send(self.parent_id,
                      f"[{t['id']}] 升级仲裁（校验失败超限）",
                      "\n".join(msgs), task_id=t["id"], priority="urgent")
        else:
            self.send(t["owner_id"],
                      f"RE: [{t['id']}] 校验失败（第 {rounds} 轮）",
                      "\n".join(msgs), task_id=t["id"])
        return msgs

    @staticmethod
    def _verdict_rejects(verdict):
        """评审结论解析：只看结论行（首行/前 40 字符）是否含明确否定词。
        宽容放行：正文建议不构成打回；评审意见完整记录在邮件/审计。"""
        head = (verdict or "").strip().split("\n")[0][:40]
        for word in ("不通过", "未通过", "拒绝", "打回", "不合格"):
            if word in head:
                return True
        return False

    def _evaluate(self, t):
        if self.force_reject and not self._rejected_once and t["dept"] == self.dept:
            self._rejected_once = True
            if self.force_reject == "once":
                return "评审结论：不通过（跳跃参数需调整，重力 900 偏大，建议 780）。"
        if self.force_reject == "always" and t["dept"] == self.dept:
            return "评审结论：不通过（跳跃参数持续不达标）。"
        # 附上结构化制品材料：评审必须看到实际产出才能评估质量。
        # 注意：不附 output.md（LLM 自由文本，可能含被截断的代码/JSON 片段，
        # 评审会误当配置文件打回）；只附校验器检查的客观结构化制品。
        workdir = os.path.join(db.ARTIFACTS_DIR, "workspace",
                               t["owner_id"], t["id"])
        artifact_txt = ""
        for f in ("design.json", "report.md", "player.py", "game.html"):
            p = os.path.join(workdir, f)
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as _f:
                        artifact_txt += f"\n\n【{f}】\n{_f.read()[:600]}"
                except (OSError, UnicodeDecodeError):
                    pass
        # 格式约束：确保真实 LLM 结论可被解析（通过/不通过 开头）
        return self.llm(
            "评审",
            f"客观校验已通过。请评审任务 {t['id']} {t['title']}"
            f"（dept={t['dept']}）的产出质量。"
            f"产出材料：{artifact_txt or '（无附件材料）'}"
            f"若无明确严重问题应判定通过。请严格以『评审结论：通过』或"
            f"『评审结论：不通过』开头，随后简述理由。",
            task_id=t["id"])

    def _approve_and_merge(self, t):
        """C3：产出经主 agent 审批后整目录合入制品库主干（含结构化制品）。"""
        src_dir = os.path.join(db.ARTIFACTS_DIR, "workspace",
                               t["owner_id"], t["id"])
        dest = os.path.join(db.ARTIFACTS_DIR, "main", t["id"])
        os.makedirs(dest, exist_ok=True)
        for name in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, name), os.path.join(dest, name))
        tickets.set_status(t["id"], "done")
        tickets.update_artifact_paths(t["id"], [dest])
        workdir = os.path.join(db.ARTIFACTS_DIR, "workspace",
                               t["owner_id"], t["id"])
        checks = validation.validate(t["dept"], workdir)
        conn = db.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts (id,task_id,agent_id,path,version,status,checks,merged_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"ART-{t['id']}", t["id"], t["owner_id"], dest, 1,
                 "merged", db.json_dumps([c.to_dict() for c in checks]),
                 db.now()))
            conn.commit()
        finally:
            conn.close()
        self.log_action("task_merge", t["id"], {"artifact": dest})
        # 完成通知：To 直属（Producer），CC 其他主 agent（按部门主题路由投递）
        self.send(self.parent_id,
                  f"[{t['id']}] 已完成并合入",
                  f"任务 {t['id']} {t['title']} 验收通过，产物：{dest}",
                  cc=self.cc_targets, task_id=t["id"],
                  topic=self.TOPIC_BY_DEPT.get(self.dept))
