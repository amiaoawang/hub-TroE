"""表定义（SQLite DDL，S1 核心集）。PostgreSQL 切换走 db.py 驱动层。"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL CHECK (role IN ('producer','main','sub')),
  dept TEXT,
  name TEXT NOT NULL,
  parent_id TEXT REFERENCES agents(id),
  status TEXT DEFAULT 'idle' CHECK (status IN ('idle','working','blocked','offline','retired')),
  max_subagents INTEGER DEFAULT 4,
  is_temp INTEGER DEFAULT 0,      -- 1=临时 subagent（弹性扩容，干完回收）
  created_at TEXT NOT NULL,
  UNIQUE(parent_id, name)
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  owner_id TEXT REFERENCES agents(id),
  supervisor_id TEXT REFERENCES agents(id),
  dept TEXT,
  milestone_id TEXT,
  stage TEXT,
  status TEXT DEFAULT 'backlog',
  priority TEXT DEFAULT 'P2',
  dod TEXT,
  depends_on TEXT,
  review_rounds INTEGER DEFAULT 0,
  max_review_rounds INTEGER DEFAULT 3,
  budget_tokens INTEGER,
  artifact_paths TEXT,
  created_at TEXT, updated_at TEXT, closed_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  thread_id TEXT,
  parent_msg_id TEXT,
  msg_type TEXT DEFAULT 'email',
  from_id TEXT NOT NULL REFERENCES agents(id),
  to_id TEXT REFERENCES agents(id),
  cc TEXT,
  bcc TEXT,
  subject TEXT,
  body TEXT,
  task_id TEXT REFERENCES tasks(id),
  priority TEXT DEFAULT 'normal',
  topic TEXT,
  ack_status TEXT DEFAULT 'pending',
  retry_count INTEGER DEFAULT 0,
  last_retry_at TEXT,
  sent_at TEXT, delivered_at TEXT, acked_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  action TEXT NOT NULL,
  target TEXT,
  detail TEXT,
  cc_copy TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  task_id TEXT REFERENCES tasks(id),
  agent_id TEXT REFERENCES agents(id),
  path TEXT NOT NULL,
  version INTEGER DEFAULT 1,
  status TEXT DEFAULT 'pending',
  checks TEXT,
  merged_at TEXT
);

CREATE TABLE IF NOT EXISTS budget (
  id TEXT PRIMARY KEY,
  day TEXT,
  model TEXT,
  task_id TEXT DEFAULT '',      -- 任务级成本归因（'' = 全局调用如宪章/调研）
  tokens_in INTEGER DEFAULT 0,
  tokens_out INTEGER DEFAULT 0,
  images INTEGER DEFAULT 0,     -- 生图计数（§9.2 画像级预算）
  cost_usd REAL DEFAULT 0,
  limit_usd REAL,               -- 日级限额（§9.2，0/空 = 不限制）
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS milestones (
  id TEXT PRIMARY KEY,
  name TEXT, stage TEXT, goal TEXT,
  status TEXT DEFAULT 'pending',   -- pending/built/review/done/failed
  build_path TEXT, planned_at TEXT, done_at TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY,
  milestone_id TEXT,
  source TEXT CHECK (source IN ('user','qa','playtest')),
  rating INTEGER, notes TEXT, affects TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS charter (
  id TEXT PRIMARY KEY, version INTEGER, body TEXT NOT NULL, updated_at TEXT
);

CREATE TABLE IF NOT EXISTS charter_amendments (
  id TEXT PRIMARY KEY, charter_id TEXT, body TEXT, proposed_by TEXT,
  status TEXT DEFAULT 'pending', broadcast_at TEXT, decided_at TEXT
);

CREATE TABLE IF NOT EXISTS llm_cache (
  key TEXT PRIMARY KEY,
  response TEXT NOT NULL,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS research (
  id TEXT PRIMARY KEY,
  title TEXT,
  genre TEXT,
  features TEXT,          -- 玩法功能清单 JSON（校验器验收依据）
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS change_requests (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  proposed_by TEXT,           -- agent_id 或 'user'（§8.3 变更单来源）
  status TEXT DEFAULT 'pending',   -- pending / approved / rejected
  affected_tasks TEXT,        -- 波及任务 JSON [task_id]
  impact TEXT,                -- 影响评估 JSON（波及任务/依赖/成本）
  decision_notes TEXT,
  created_at TEXT, decided_at TEXT
);
"""
