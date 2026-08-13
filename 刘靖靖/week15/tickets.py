"""任务单状态机（PRD §4）：backlog → todo → in_progress → in_review → done；
打回回 in_progress，轮次超限 → escalated（Producer 仲裁）。"""
import json
import uuid

from src import db


def create_task(title, description, owner_id, supervisor_id, dept,
                milestone_id="M1", stage="prototype", priority="P2",
                dod=None, depends_on=None, budget_tokens=50000,
                max_review_rounds=3):
    tid = "T-" + str(uuid.uuid4().int % 1000000).zfill(4)
    ts = db.now()
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO tasks (id,title,description,owner_id,supervisor_id,dept,"
            "milestone_id,stage,status,priority,dod,depends_on,budget_tokens,"
            "max_review_rounds,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tid, title, description, owner_id, supervisor_id, dept, milestone_id,
             stage, "backlog", priority, db.json_dumps(dod), db.json_dumps(depends_on),
             budget_tokens, max_review_rounds, ts, ts))
        conn.commit()
    finally:
        conn.close()
    return tid


def get(tid):
    conn = db.connect()
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
    conn.close()
    return row


def set_status(tid, status):
    conn = db.connect()
    conn.execute("UPDATE tasks SET status=?, updated_at=? WHERE id=?",
                 (status, db.now(), tid))
    conn.commit()
    conn.close()


def can_start(tid):
    """依赖 DAG：所有前驱 done 才可开工（C7）。"""
    row = get(tid)
    deps = json.loads(row["depends_on"]) if row and row["depends_on"] else []
    if not deps:
        return True, ""
    for d in deps:
        r = get(d)
        if not r or r["status"] != "done":
            return False, f"依赖 {d} 未完成(status={r['status'] if r else '缺失'})"
    return True, ""


def detect_cycles():
    """C7 依赖环检测：扫描全部任务的 depends_on 构建 DAG，DFS 找环。

    返回环路径列表（如 [['T-001','T-002','T-001'], ...]），无环返回 []。
    由调度器（PM 职责）调用：发现环 → 升级 Producer 仲裁。
    """
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT id, depends_on FROM tasks").fetchall()
    finally:
        conn.close()
    graph = {}
    for r in rows:
        deps = json.loads(r["depends_on"]) if r["depends_on"] else []
        # 保留自环引用（自环也是环，DFS 的 GRAY 判断会抓到 [x, x]）
        graph[r["id"]] = deps
    if not graph:
        return []

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    stack = []            # 当前 DFS 路径（gray 栈）
    pos = {}              # 节点在 stack 中的下标
    cycles = []

    def dfs(node):
        color[node] = GRAY
        pos[node] = len(stack)
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt not in graph:          # 依赖了不存在的任务（孤儿引用，不算环）
                continue
            if color[nxt] == GRAY:        # 找到环：stack[pos[nxt]:] + nxt
                cyc = stack[pos[nxt]:] + [nxt]
                if cyc not in cycles:
                    cycles.append(cyc)
            elif color[nxt] == WHITE:
                dfs(nxt)
        stack.pop()
        del pos[node]
        color[node] = BLACK

    for n in graph:
        if color[n] == WHITE:
            dfs(n)
    return cycles


def register_reject(tid):
    """打回：轮次 +1，回 in_progress；超限 → escalated。"""
    row = get(tid)
    rounds = row["review_rounds"] + 1
    if rounds >= row["max_review_rounds"]:
        set_status(tid, "escalated")
        return "escalated", rounds
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE tasks SET review_rounds=?, status='in_progress', updated_at=? "
            "WHERE id=?", (rounds, db.now(), tid))
        conn.commit()
    finally:
        conn.close()
    return "in_progress", rounds


def tasks_by(owner_id=None, supervisor_id=None, status=None, milestone_id=None):
    conn = db.connect()
    sql = "SELECT * FROM tasks WHERE 1=1"
    args = []
    if owner_id:
        sql += " AND owner_id=?"
        args.append(owner_id)
    if supervisor_id:
        sql += " AND supervisor_id=?"
        args.append(supervisor_id)
    if status:
        sql += " AND status=?"
        args.append(status)
    if milestone_id:
        sql += " AND milestone_id=?"
        args.append(milestone_id)
    sql += " ORDER BY priority, created_at"
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return rows


def update_artifact_paths(tid, paths):
    conn = db.connect()
    conn.execute("UPDATE tasks SET artifact_paths=?, updated_at=? WHERE id=?",
                 (db.json_dumps(paths), db.now(), tid))
    conn.commit()
    conn.close()
