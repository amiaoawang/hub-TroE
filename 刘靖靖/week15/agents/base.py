"""Agent 基类：所有 agent 的共同能力（LLM 路由调用、发信）。"""
from src import mail
from src import db


class Agent:
    def __init__(self, agent_id, role, dept, name, parent_id=None, router=None,
                 context_pack=None, topics_cfg=None):
        self.id = agent_id
        self.role = role            # producer | main | sub
        self.dept = dept            # 策划/程序/美术/QA/PM/producer
        self.name = name
        self.parent_id = parent_id
        self.router = router
        self.context_pack = context_pack or {}   # 宪章章节 + 部门知识（只读）
        self.topics_cfg = topics_cfg or {}       # 主题订阅配置（§5.2）

    # -- LLM（按角色/部门路由到模型档位） --
    def llm(self, system, user, temperature=0.3, max_tokens=2000, task_id=None):
        # token 优化：摘要优先（charter_summary），无摘要才回退全文
        ctx = self.context_pack.get("charter_summary") \
            or self.context_pack.get("charter", "")
        if ctx:
            system = f"[项目宪章]\n{ctx}\n\n{system}"
        # 任务级成本归因：本次调用的 token/成本记到该任务（§9.2）
        from functools import partial
        on_budget = partial(self._on_budget, task_id=task_id or "")
        return self.router.complete(self.role, system, user,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    agent_dept=self.dept,
                                    on_budget=on_budget)

    def _on_budget(self, model_id, tokens_in, tokens_out, task_id=""):
        """按实际命中的模型单独计价，并按任务归因（§9.1/§9.2）：
        cost_usd = 输入/1e6×input_per_m + 输出/1e6×output_per_m；
        记账键 = (day, model, task_id)，task_id 空 = 全局调用（宪章/调研）。"""
        try:
            in_rate, out_rate = self.router.cost_for(model_id)
            cost = round(tokens_in / 1e6 * in_rate + tokens_out / 1e6 * out_rate, 6)
            conn = db.connect()
            try:
                day = db.today()
                row = conn.execute(
                    "SELECT * FROM budget WHERE day=? AND model=? AND task_id=?",
                    (day, model_id, task_id)).fetchone()
                if row:
                    conn.execute(
                        "UPDATE budget SET tokens_in=tokens_in+?, tokens_out=tokens_out+?,"
                        " cost_usd=cost_usd+?, updated_at=? "
                        "WHERE day=? AND model=? AND task_id=?",
                        (tokens_in, tokens_out, cost, db.now(), day, model_id,
                         task_id))
                else:
                    conn.execute(
                        "INSERT INTO budget (id,day,model,task_id,tokens_in,tokens_out,"
                        "cost_usd,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                        (f"{day}-{model_id}-{task_id}", day, model_id, task_id,
                         tokens_in, tokens_out, cost, db.now()))
                conn.commit()
            finally:
                conn.close()
        except Exception:  # 预算记账失败不阻断主流程
            pass

    # -- 消息 --
    def send(self, to_id, subject, body, cc=None, task_id=None, priority="normal",
             topic=None):
        return mail.send(self.id, to_id, subject, body, cc=cc,
                         task_id=task_id, priority=priority, topic=topic)

    def ack_all(self):
        """ack 自己收件箱里所有待确认消息（to 或 CC 可见范围内，含主题过滤）。"""
        for m in mail.inbox(self.id, topics=self.topics_cfg):
            mail.ack(m["id"], self.id)

    def log_action(self, action, target, detail=None):
        conn = db.connect()
        try:
            conn.execute(
                "INSERT INTO audit_log (ts,actor_id,action,target,detail) VALUES (?,?,?,?,?)",
                (db.now(), self.id, action, target,
                 db.json_dumps(detail) if detail else None))
            conn.commit()
        finally:
            conn.close()
