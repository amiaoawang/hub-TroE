"""数据访问层：SQLite（S1 MVP）→ PostgreSQL 可切换。
路径约定：数据库 backend/data/harness.db，制品库 backend/artifacts/。
多项目化：--project <name> 切换项目——default 用 backend/ 根（兼容现状），
其他项目用 backend/projects/<name>/（独立 DB/制品/日志/playbook/skills）。"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # backend/

PROJECT_NAME = "default"
PROJECT_DIR = BASE                # 当前项目根（default = backend/，零迁移）
DATA_DIR = os.path.join(BASE, "data")
ARTIFACTS_DIR = os.path.join(BASE, "artifacts")
LOGS_DIR = os.path.join(BASE, "logs")

DB_PATH = os.environ.get("HARNESS_DB") or os.path.join(DATA_DIR, "harness.db")


def set_project(name=None):
    """切换项目（多项目化）。default → backend/ 根；其他 → backend/projects/<name>/。
    必须在 db 操作前调用（main --project / watch --project / 测试）。"""
    global PROJECT_NAME, PROJECT_DIR, DATA_DIR, ARTIFACTS_DIR, LOGS_DIR, DB_PATH
    PROJECT_NAME = (name or "default").strip() or "default"
    if PROJECT_NAME == "default":
        PROJECT_DIR = BASE
        DATA_DIR = os.path.join(BASE, "data")
    else:
        PROJECT_DIR = os.path.join(BASE, "projects", PROJECT_NAME)
        DATA_DIR = os.path.join(PROJECT_DIR, "data")
    ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
    LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
    if not os.environ.get("HARNESS_DB"):
        DB_PATH = os.path.join(DATA_DIR, "harness.db")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def list_projects():
    """已存在的项目列表（含 default）。"""
    projects = ["default"]
    pdir = os.path.join(BASE, "projects")
    if os.path.isdir(pdir):
        projects += sorted(d for d in os.listdir(pdir)
                           if os.path.isdir(os.path.join(pdir, d)))
    return projects


def now() -> str:
    """北京时间字符串，用于所有时间戳。"""
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def today() -> str:
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")


def json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False) if obj is not None else None


def connect() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    # 用 DELETE journal 而非 WAL：Windows 下 WAL 的 -wal/-shm 文件
    # 易被杀毒/索引服务瞬态锁定 → 偶发 "attempt to write a readonly database"。
    # 单进程场景 DELETE 模式无共享锁文件，最稳。
    conn.execute("PRAGMA journal_mode = DELETE")
    return conn


def init_db() -> None:
    from src import models
    conn = connect()
    conn.executescript(models.SCHEMA)
    conn.commit()
    conn.close()
    migrate()


def migrate() -> None:
    """轻量迁移：为已存在的库补充新列（幂等）。"""
    conn = connect()
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "retry_count" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN retry_count INTEGER DEFAULT 0")
        if "last_retry_at" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN last_retry_at TEXT")
        if "topic" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN topic TEXT")
        acols = {r["name"] for r in conn.execute("PRAGMA table_info(agents)").fetchall()}
        if "is_temp" not in acols:
            conn.execute("ALTER TABLE agents ADD COLUMN is_temp INTEGER DEFAULT 0")
        # 预算（§9）：任务级记账 + 生图计数 + 日级限额
        bcols = {r["name"] for r in conn.execute("PRAGMA table_info(budget)").fetchall()}
        if "task_id" not in bcols:
            conn.execute("ALTER TABLE budget ADD COLUMN task_id TEXT DEFAULT ''")
        if "images" not in bcols:
            conn.execute("ALTER TABLE budget ADD COLUMN images INTEGER DEFAULT 0")
        if "limit_usd" not in bcols:
            conn.execute("ALTER TABLE budget ADD COLUMN limit_usd REAL")
        # 变更单（§8.3）
        tbls = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "change_requests" not in tbls:
            conn.execute(
                "CREATE TABLE change_requests ("
                " id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT,"
                " proposed_by TEXT, status TEXT DEFAULT 'pending',"
                " affected_tasks TEXT, impact TEXT, decision_notes TEXT,"
                " created_at TEXT, decided_at TEXT)")
        conn.commit()
    finally:
        conn.close()


def reset_db() -> None:
    """清空全部表（仅 demo / 测试用）。"""
    from src import models
    conn = connect()
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for line in models.SCHEMA.strip().split(";"):
            line = line.strip()
            if line.startswith("CREATE TABLE"):
                name = line.split("(")[0].replace("CREATE TABLE IF NOT EXISTS", "").strip()
                conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.commit()
    finally:
        conn.close()
    init_db()
