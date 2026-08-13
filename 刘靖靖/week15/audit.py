"""审计日志每日导出（PRD §6.3）：audit_log 落盘为 logs/audit-YYYY-MM-DD.jsonl。
按天覆盖写（审计量小，简单可靠）；后续可改为按 audit_log.id 增量追加。"""
import json
import os

from src import db


def export_logs(date=None):
    date = date or db.today()
    os.makedirs(db.LOGS_DIR, exist_ok=True)
    path = os.path.join(db.LOGS_DIR, f"audit-{date}.jsonl")
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE date(ts)=? ORDER BY id", (date,)).fetchall()
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
    finally:
        conn.close()
    return path, len(rows)
