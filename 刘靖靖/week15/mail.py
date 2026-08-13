"""消息总线（邮件模型，PRD §5）：
send 即写入 messages 表，同步写 audit_log 的 CC 副本存档（运行期不参与决策，仅事后可查）。
bcc 只进审计日志，不投递。ack 用于可靠性确认；retry_expired 实现超时重投与 blocked 升级（§5.5）。"""
import json
import uuid
from datetime import datetime, timedelta, timezone

from src import db


def send(from_id, to_id, subject, body, cc=None, bcc=None, task_id=None,
         priority="normal", msg_type="email", parent_msg_id=None, topic=None):
    msg_id = "MSG-" + uuid.uuid4().hex[:8].upper()
    thread_id = "THR-" + uuid.uuid4().hex[:8].upper()
    ts = db.now()
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO messages (id,thread_id,parent_msg_id,msg_type,from_id,to_id,cc,bcc,"
            "subject,body,task_id,priority,topic,ack_status,sent_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (msg_id, thread_id, parent_msg_id, msg_type, from_id, to_id,
             db.json_dumps(cc), db.json_dumps(bcc), subject, body, task_id,
             priority, topic, "pending", ts))
        # C5：CC/BCC 副本进审计日志（运行期不注入任何 agent 上下文）
        conn.execute(
            "INSERT INTO audit_log (ts,actor_id,action,target,detail,cc_copy) "
            "VALUES (?,?,?,?,?,?)",
            (ts, from_id, "email_cc_archive", to_id,
             json.dumps({"msg_id": msg_id, "task_id": task_id, "priority": priority,
                         "topic": topic}, ensure_ascii=False),
             json.dumps({"id": msg_id, "from": from_id, "to": to_id, "cc": cc,
                         "bcc": bcc, "subject": subject, "body": body,
                         "task_id": task_id, "topic": topic},
                        ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()
    return msg_id


def can_read(msg, agent_id, topics):
    """§5.2 主题路由：CC 收件人是否可见该消息。
    - 无主题（直发）→ 可见
    - 广播主题（宪章/里程碑/晨会）→ 全员可见
    - always_cc 里的 agent（如 PM）→ 始终可见
    - 其余 → 仅订阅了该主题的 agent 可见
    """
    t = msg["topic"]
    if t is None:
        return True
    if t in topics.get("broadcast", []):
        return True
    if agent_id in topics.get("always_cc", []):
        return True
    return agent_id in topics.get("subscriptions", {}).get(t, [])


def inbox(agent_id, msg_type="email", topics=None):
    """收件箱：to 本人 + CC 本人（主题路由过滤）。
    topics=None 时不应用主题过滤（兼容旧调用/测试）。"""
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM messages WHERE msg_type=? ORDER BY sent_at",
        (msg_type,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        if r["to_id"] == agent_id:
            out.append(r)
            continue
        cc = json.loads(r["cc"]) if r["cc"] else []
        if agent_id not in cc:
            continue
        if topics is None or can_read(r, agent_id, topics):
            out.append(r)
    return out


def unacked(agent_id=None):
    """未 ack 消息（调度器重投依据）。"""
    conn = db.connect()
    if agent_id:
        rows = conn.execute(
            "SELECT * FROM messages WHERE to_id=? AND ack_status='pending'",
            (agent_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM messages WHERE ack_status='pending'").fetchall()
    conn.close()
    return rows


def ack(msg_id, agent_id):
    conn = db.connect()
    conn.execute(
        "UPDATE messages SET ack_status='acked', acked_at=? WHERE id=? AND to_id=?",
        (db.now(), msg_id, agent_id))
    conn.commit()
    conn.close()


def retry_expired(timeout_s=30, max_retries=3):
    """§5.5 超时重投：pending 且超过 timeout 未 ack 的消息重投（retry_count+1）；
    重投达上限 → 收件 agent 标记 blocked，返回其 id（由调度器升级 Producer）。

    返回 (retried_ids, blocked_agent_ids)。
    """
    cutoff = (datetime.now(timezone(timedelta(hours=8))) -
              timedelta(seconds=timeout_s)).strftime("%Y-%m-%d %H:%M:%S")
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM messages WHERE ack_status='pending' AND to_id IS NOT NULL "
            "AND COALESCE(last_retry_at, sent_at) <= ?", (cutoff,)).fetchall()
        retried, blocked = [], []
        for r in rows:
            if r["retry_count"] >= max_retries:
                blocked.append(r["to_id"])
                conn.execute(
                    "UPDATE agents SET status='blocked' WHERE id=? AND status!='blocked'",
                    (r["to_id"],))
            else:
                conn.execute(
                    "UPDATE messages SET retry_count=retry_count+1, "
                    "delivered_at=?, last_retry_at=? WHERE id=?",
                    (db.now(), db.now(), r["id"]))
                retried.append(r["id"])
        conn.commit()
        return retried, sorted(set(x for x in blocked if x))
    finally:
        conn.close()
