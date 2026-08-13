"""成本计量与预算控制（PRD §9）。

- 任务级：budget 表按 (day, model, task_id) 记账，`task_over_budget` 判定
  任务 token 预算（budget_tokens）是否超支 → 主 agent 告警（摘要模式降级）。
- 日级：当日总成本超过 `budget.daily_limit_usd` → 暂停低优先级（P2/P3）任务派发。
- 画像级：生图单独计量（images 列），每日上限 `budget.image_daily_limit`
  （生图服务接入预留：record_image 记账 + image_budget_ok 判定）。
"""
from __future__ import annotations

from src import db


# --------------------------------------------------------------------------
# 查询
# --------------------------------------------------------------------------
def task_tokens(task_id):
    """某任务累计 token（in+out）。返回 (tokens_in, tokens_out)。"""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens_in),0) ti, COALESCE(SUM(tokens_out),0) to_ "
            "FROM budget WHERE task_id=?", (task_id,)).fetchone()
        return row["ti"], row["to_"]
    finally:
        conn.close()


def task_cost(task_id):
    """某任务累计成本（美元）。"""
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) v FROM budget WHERE task_id=?",
            (task_id,)).fetchone()["v"]
    finally:
        conn.close()


def task_over_budget(task_id, budget_tokens):
    """任务级预算判定：累计 token 超过 budget_tokens → True（超支告警）。"""
    if not budget_tokens or budget_tokens <= 0:
        return False
    ti, to = task_tokens(task_id)
    return (ti + to) > budget_tokens


def daily_cost(day=None):
    """当日累计成本（美元）。day 缺省 = 今天。"""
    day = day or db.today()
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) v FROM budget WHERE day=?",
            (day,)).fetchone()["v"]
    finally:
        conn.close()


def daily_exceeded(cfg, day=None):
    """日级预算判定：当日成本 > budget.daily_limit_usd（>0 才启用，0 = 不限制）。"""
    limit = (cfg.get("budget") or {}).get("daily_limit_usd") or 0
    if limit <= 0:
        return False
    return daily_cost(day) > limit


# --------------------------------------------------------------------------
# 生图（画像级，预留：美术生图服务接入时调用）
# --------------------------------------------------------------------------
def record_image(image_count=1, cost_usd=0.0, model="image"):
    """生图记账：当日 images 计数 + 可选成本。返回当日累计生图数。"""
    day = db.today()
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM budget WHERE day=? AND model=? AND task_id=''",
            (day, model)).fetchone()
        if row:
            conn.execute(
                "UPDATE budget SET images=images+?, cost_usd=cost_usd+?, "
                "updated_at=? WHERE day=? AND model=? AND task_id=''",
                (image_count, cost_usd, db.now(), day, model))
        else:
            conn.execute(
                "INSERT INTO budget (id,day,model,task_id,tokens_in,tokens_out,"
                "images,cost_usd,updated_at) VALUES (?,?,?,?,0,0,?,?,?)",
                (f"{day}-{model}", day, model, "", image_count, cost_usd,
                 db.now()))
        conn.commit()
        return images_today()
    finally:
        conn.close()


def images_today(day=None):
    """当日累计生图数。"""
    day = day or db.today()
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT COALESCE(SUM(images),0) v FROM budget WHERE day=?",
            (day,)).fetchone()["v"]
    finally:
        conn.close()


def image_budget_ok(cfg, day=None):
    """生图预算判定：当日生图数 < budget.image_daily_limit（>0 才启用）。"""
    limit = (cfg.get("budget") or {}).get("image_daily_limit") or 0
    if limit <= 0:
        return True
    return images_today(day) < limit
