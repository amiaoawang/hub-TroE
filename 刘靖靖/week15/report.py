"""按需进度汇报（用户下达指令时触发，替代原每日晨会）：
各主 agent 向 PM 汇报进度/阻塞（已完成/进行中/待办/阻塞，CC 其他主 agent）；
PM 汇总纪要（落盘 logs/report-<时间戳>.md + 审计）并广播。
无定时任务：由用户指令触发（python -m src.main --report）。
"""
import os

from src import db


def _summary_text(main_agent):
    conn = db.connect()
    try:
        done = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE supervisor_id=? AND status='done'",
            (main_agent.id,)).fetchone()["c"]
        doing = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE supervisor_id=? AND status='in_progress'",
            (main_agent.id,)).fetchone()["c"]
        todo = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE supervisor_id=? AND status='backlog'",
            (main_agent.id,)).fetchone()["c"]
        blocked = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE supervisor_id=? AND status='escalated'",
            (main_agent.id,)).fetchone()["c"]
    finally:
        conn.close()
    return (f"已完成 {done} · 进行中 {doing} · 待办 {todo} · 阻塞/升级 {blocked}")


def run_report(pm, participants, to_id=None):
    """pm: PM 主 agent（汇总人）；participants: 各主 agent；to_id: 纪要收件人（通常 producer）。"""
    ts = db.now().replace(":", "").replace(" ", "_")
    reports = [(p, _summary_text(p)) for p in participants]
    others = [p.id for p in participants]

    # 1) 各主 agent 向 PM 汇报（CC 其他主 agent，广播主题全员可见）
    for p, text in reports:
        cc = [x for x in others if x != p.id]
        p.send(pm.id, f"进度汇报[{p.dept}] {ts}", text, cc=cc, topic="汇报")

    # 2) PM 汇总纪要
    lines = [f"# 进度汇报（用户指令触发）", f"时间：{db.now()}", ""]
    lines += [f"- {p.dept}({p.id}): {text}" for p, text in reports]
    md = "\n".join(lines)
    os.makedirs(db.LOGS_DIR, exist_ok=True)
    path = os.path.join(db.LOGS_DIR, f"report-{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    pm.log_action("report_summary", "report",
                  {"path": path, "participants": [p.id for p in participants]})

    # 3) 纪要广播（如有收件人）
    if to_id:
        pm.send(to_id, f"进度汇报 {ts}", md, cc=others, topic="汇报")
    return path
