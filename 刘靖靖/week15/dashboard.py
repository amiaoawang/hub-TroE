"""进度看板（PRD §12.1 S4）：从 DB 渲染静态 HTML 快照，用户随时查看项目作战状态。
用法：python -m src.dashboard（基于当前 DB 渲染 dashboard/index.html）。
也可在 demo 结束时自动生成。无外部依赖、离线可预览。

实时进度直播页（live.html + live_state.js）：
  - export_live()  把当前 DB 状态导出为 dashboard/live_state.js（window.__LIVE__）
  - render_live()  生成 dashboard/live.html（前端每 5s 动态重载 live_state.js 刷新）
  - 独立刷新器：python -m src.main --live（循环导出，配合页面实时跟踪）
file:// 下 script 标签加载本地 js 不受同源限制，因此无需起 HTTP 服务。
"""
import html
import json


def _dash_dir():
    """看板输出目录（多项目化）：default → D:\\gameHarness\\dashboard（现状兼容）；
    其他项目 → backend/projects/<name>/dashboard（独立看板，互不覆盖）。"""
    if db.PROJECT_NAME == "default":
        return os.path.join(db.BASE, "..", "dashboard")
    return os.path.join(db.PROJECT_DIR, "dashboard")
import os
from datetime import datetime

from src import db

_STATUS_COLOR = {
    "done": "#3B6D11", "in_progress": "#185FA5", "in_review": "#854F0B",
    "backlog": "#888780", "todo": "#888780", "escalated": "#A32D2D",
    "rejected": "#A32D2D", "built": "#3B6D11", "review": "#854F0B",
    "pending": "#888780", "failed": "#A32D2D", "merged": "#3B6D11",
}


def _color(status):
    return _STATUS_COLOR.get(status or "", "#444441")


def _pill(status):
    c = _color(status)
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
            f'font-size:12px;color:#fff;background:{c};">{html.escape(str(status or "-"))}</span>')


def _render_tasks(conn):
    rows = conn.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
    if not rows:
        return "<p style='color:#888;'>暂无任务</p>"
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td>{html.escape(r['id'])}</td>"
            f"<td>{html.escape(r['dept'] or '')}</td>"
            f"<td style='max-width:300px;'>{html.escape(r['title'])}</td>"
            f"<td>{html.escape(r['milestone_id'] or '')}</td>"
            f"<td>{_pill(r['status'])}</td>"
            f"<td>{r['review_rounds']}/{r['max_review_rounds']}</td>"
            f"<td>{html.escape(r['priority'] or '')}</td>"
            "</tr>")
    return ("<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr style='text-align:left;color:#5F5E5A;border-bottom:1px solid #E3E1DA;'>"
            "<th>ID</th><th>部门</th><th>任务</th><th>里程碑</th><th>状态</th>"
            "<th>打回/上限</th><th>优先级</th></tr></thead><tbody>"
            + "".join(trs) + "</tbody></table>")


def _render_agents(conn):
    rows = conn.execute("SELECT * FROM agents ORDER BY role, dept").fetchall()
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td>{html.escape(r['id'])}</td>"
            f"<td>{html.escape(r['role'])}</td>"
            f"<td>{html.escape(r['dept'] or '')}</td>"
            f"<td>{html.escape(r['name'])}</td>"
            f"<td>{_pill(r['status'])}</td>"
            "</tr>")
    return ("<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr style='text-align:left;color:#5F5E5A;border-bottom:1px solid #E3E1DA;'>"
            "<th>ID</th><th>角色</th><th>部门</th><th>名称</th><th>状态</th></tr></thead>"
            "<tbody>" + "".join(trs) + "</tbody></table>")


def _render_milestones(conn):
    rows = conn.execute("SELECT * FROM milestones ORDER BY id").fetchall()
    if not rows:
        return "<p style='color:#888;'>暂无里程碑</p>"
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td>{html.escape(r['id'])}</td>"
            f"<td>{html.escape(r['name'])}</td>"
            f"<td>{html.escape(r['stage'] or '')}</td>"
            f"<td>{html.escape(r['goal'] or '')}</td>"
            f"<td>{_pill(r['status'])}</td>"
            f"<td style='font-size:12px;color:#5F5E5A;'>{html.escape(r['build_path'] or '')}</td>"
            "</tr>")
    return ("<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr style='text-align:left;color:#5F5E5A;border-bottom:1px solid #E3E1DA;'>"
            "<th>ID</th><th>名称</th><th>阶段</th><th>目标</th><th>状态</th><th>构建路径</th>"
            "</tr></thead><tbody>" + "".join(trs) + "</tbody></table>")


def _render_budget(conn):
    rows = conn.execute(
        "SELECT model, SUM(tokens_in) ti, SUM(tokens_out) to_, SUM(cost_usd) cu "
        "FROM budget GROUP BY model ORDER BY cu DESC").fetchall()
    if not rows:
        return "<p style='color:#888;'>暂无预算记录</p>"
    trs = []
    for r in rows:
        trs.append(
            "<tr>"
            f"<td>{html.escape(r['model'])}</td>"
            f"<td>{r['ti']}</td><td>{r['to_']}</td>"
            f"<td>${r['cu']:.4f}</td></tr>")
    return ("<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead><tr style='text-align:left;color:#5F5E5A;border-bottom:1px solid #E3E1DA;'>"
            "<th>模型</th><th>输入 token</th><th>输出 token</th><th>金额</th></tr></thead>"
            "<tbody>" + "".join(trs) + "</tbody></table>")


def _render_audit(conn):
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT 15").fetchall()
    items = []
    for r in rows:
        items.append(
            f"<li style='margin:4px 0;font-size:12px;color:#444441;'>"
            f"<span style='color:#888780;'>{html.escape(r['ts'])}</span> "
            f"<b>{html.escape(r['actor_id'])}</b> → {html.escape(r['action'])}"
            f"<span style='color:#888780;'> {html.escape(r['target'] or '')}</span></li>")
    return "<ul style='list-style:none;padding:0;margin:0;'>" + "".join(items) + "</ul>"


def _render_evolution_panel(conn):
    """自进化面板：playbook 经验统计 + 熔断状态 + 质量指标（每个项目专属）。"""
    try:
        from src import retro
    except Exception:  # noqa: BLE001
        return "<p style='color:#888;'>自进化数据不可用</p>"
    try:
        pb = retro.playbook_load()
        rules = pb.get("rules") or []
        n_acc = sum(1 for r in rules if r["status"] == "accepted")
        n_prop = sum(1 for r in rules if r["status"] == "proposed")
        n_arch = sum(1 for r in rules if r["status"] == "archived")
        n_shared = sum(1 for r in rules if r.get("shared_from"))
        fuse = retro.fuse_status()
        m = retro.compute_metrics()
    except Exception:  # noqa: BLE001
        return "<p style='color:#888;'>自进化数据不可用</p>"
    t, fb, b = m["tasks"], m["feedback"], m["budget"]
    fuse_html = ('<span style="color:#A32D2D;font-weight:500;">⛔ 熔断中</span>'
                 if fuse.get("tripped")
                 else '<span style="color:#3B6D11;">正常</span>')
    rows = [
        ("经验库", f"{n_acc} 采纳 · {n_prop} 候选 · {n_arch} 已归档"
         + (f" · {n_shared} 跨项目共享" if n_shared else "")),
        ("完成率 / 打回率",
         f"{t['done_rate'] * 100:.0f}% / {t['reject_rate'] * 100:.0f}%"),
        ("成本 / 质量",
         f"${b['total_cost']} · 验收通过率 "
         f"{fb['approval_rate'] * 100:.0f}%" if fb['approval_rate'] is not None
         else f"${b['total_cost']} · 暂无验收反馈"),
        ("全局熔断", fuse_html),
    ]
    cards = "".join(
        f"<div style='background:#F6F8FA;border:0.5px solid #E3E1DA;border-radius:10px;"
        f"padding:10px 14px;flex:1;min-width:150px;'>"
        f"<div style='font-size:12px;color:#5F5E5A;'>{k}</div>"
        f"<div style='font-size:14px;font-weight:500;margin-top:4px;'>{v}</div></div>"
        for k, v in rows)
    return (f"<div style='display:flex;gap:10px;flex-wrap:wrap;'>{cards}</div>"
            + ("<p style='font-size:12px;color:#888;margin:8px 0 0;'>"
               "跨项目共享经验："
               + "、".join(f"[{r['id']}]←{r['shared_from']['project']}"
                           for r in rules if r.get("shared_from"))
               + "</p>" if n_shared else ""))


def _render_milestone_progress(conn):
    """M1-M5 阶段进度条（done=绿 / 构建中=蓝 / pending=灰 / failed=红）。"""
    rows = conn.execute("SELECT id,name,status FROM milestones ORDER BY id").fetchall()
    if not rows:
        return "<p style='color:#888;'>暂无里程碑</p>"
    bars = []
    for r in rows:
        c = {"done": "#3B6D11", "built": "#185FA5", "review": "#BA7517",
             "pending": "#B4B2A9", "failed": "#A32D2D"}.get(
                 r["status"] or "pending", "#B4B2A9")
        bars.append(
            "<div style='flex:1;text-align:center;'>"
            f"<div style='font-size:12px;color:#5F5E5A;margin-bottom:6px;'>"
            f"{html.escape(r['id'])} {html.escape(r['name'] or '')}</div>"
            f"<div style='height:8px;background:{c};border-radius:999px;"
            f"opacity:0.85;'></div>"
            f"<div style='font-size:11px;color:#888;margin-top:4px;'>{_pill(r['status'])}</div>"
            "</div>")
    return ("<div style='display:flex;gap:10px;'>" + "".join(bars) + "</div>")


def render_dashboard(out=None):
    out = out or os.path.join(_dash_dir(), "index.html")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    conn = db.connect()
    try:
        n_task = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]
        n_done = conn.execute("SELECT COUNT(*) c FROM tasks WHERE status='done'").fetchone()["c"]
        n_doing = conn.execute("SELECT COUNT(*) c FROM tasks WHERE status='in_progress'").fetchone()["c"]
        n_todo = conn.execute("SELECT COUNT(*) c FROM tasks WHERE status IN ('backlog','todo')").fetchone()["c"]
        n_escalated = conn.execute("SELECT COUNT(*) c FROM tasks WHERE status='escalated'").fetchone()["c"]
        n_msg = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        n_art = conn.execute("SELECT COUNT(*) c FROM artifacts WHERE status='merged'").fetchone()["c"]
        n_audit = conn.execute("SELECT COUNT(*) c FROM audit_log").fetchone()["c"]
        total_cost = conn.execute("SELECT COALESCE(SUM(cost_usd),0) v FROM budget").fetchone()["v"]
        tasks_html = _render_tasks(conn)
        agents_html = _render_agents(conn)
        ms_html = _render_milestones(conn)
        budget_html = _render_budget(conn)
        audit_html = _render_audit(conn)
        ms_progress_html = _render_milestone_progress(conn)
        evol_html = _render_evolution_panel(conn)
        topics_html = ""
        for r in conn.execute(
                "SELECT topic, COUNT(*) c FROM messages GROUP BY topic ORDER BY c DESC").fetchall():
            topics_html += (f"<span style='display:inline-block;background:#E6F1FB;color:#0C447C;"
                            f"border-radius:999px;padding:2px 10px;font-size:12px;margin:2px;'>"
                            f"{html.escape(r['topic'] or '直发')} × {r['c']}</span>")
        updated = db.now()
    finally:
        conn.close()

    card = ("<div style='background:#fff;border:0.5px solid #E3E1DA;border-radius:12px;"
            "padding:1rem 1.25rem;min-width:150px;flex:1;'>"
            "<div style='font-size:13px;color:#5F5E5A;'>{}</div>"
            "<div style='font-size:24px;font-weight:500;margin-top:4px;'>{}</div></div>")
    stat_row = (
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">'
        + card.format("任务总数", n_task)
        + card.format("已完成", f'<span style="color:#3B6D11;">{n_done}</span>')
        + card.format("进行中", f'<span style="color:#185FA5;">{n_doing}</span>')
        + card.format("待办", n_todo)
        + card.format("升级/阻塞", f'<span style="color:#A32D2D;">{n_escalated}</span>')
        + card.format("消息", n_msg)
        + card.format("合入制品", n_art)
        + card.format("成本", f"${total_cost:.4f}")
        + "</div>")

    section = ('<section style="background:#fff;border:0.5px solid #E3E1DA;border-radius:12px;'
               'padding:1rem 1.25rem;margin-bottom:16px;">'
               '<h2 style="font-size:14px;font-weight:500;margin:0 0 12px;color:#1f2328;">{}</h2>'
               '{}</section>')

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Harness · {html.escape(db.PROJECT_NAME)} 项目看板</title>
</head>
<body style="margin:0;background:#F6F8FA;font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif;color:#1f2328;">
<div style="max-width:1080px;margin:0 auto;padding:24px 16px 48px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:16px;">
    <h1 style="font-size:20px;font-weight:500;margin:0;">🎮 Harness · {html.escape(db.PROJECT_NAME)} 项目看板</h1>
    <span style="font-size:12px;color:#888780;">更新于 {html.escape(updated)}</span>
  </div>
  <p style="font-size:12px;color:#888780;margin:-8px 0 16px;">
    项目：{html.escape(db.PROJECT_NAME)} · DB：{html.escape(db.DB_PATH)} · 审计 {n_audit} 条</p>
  {section.format("五阶段进度", ms_progress_html)}
  {section.format("自进化面板", evol_html)}
  {stat_row}
  {section.format("任务", tasks_html)}
  {section.format("里程碑", ms_html)}
  {section.format("Agent 组织", agents_html)}
  {section.format("消息主题分布（§5.2 主题路由）", topics_html or "<p style='color:#888;'>暂无消息</p>")}
  {section.format("成本（按模型计价）", budget_html)}
  {section.format("最近审计事件", audit_html)}
</div>
</body>
</html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return out


def _live_state(conn):
    """从 DB 汇总实时状态（直播页数据源）。"""
    ms = []
    for r in conn.execute("SELECT * FROM milestones ORDER BY id").fetchall():
        n_total = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE milestone_id=?", (r["id"],)
        ).fetchone()["c"]
        n_done = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE milestone_id=? AND status='done'",
            (r["id"],)).fetchone()["c"]
        n_art = conn.execute(
            "SELECT COUNT(*) c FROM artifacts a JOIN tasks t ON a.task_id=t.id "
            "WHERE t.milestone_id=? AND a.status='merged'", (r["id"],)).fetchone()["c"]
        ms.append({
            "id": r["id"], "name": r["name"], "stage": r["stage"] or "",
            "status": r["status"], "done_tasks": n_done, "total_tasks": n_total,
            "artifacts": n_art, "build_path": r["build_path"] or "",
        })
    tasks = [{"id": r["id"], "milestone_id": r["milestone_id"] or "",
              "title": r["title"], "dept": r["dept"] or "",
              "owner_id": r["owner_id"], "supervisor_id": r["supervisor_id"],
              "status": r["status"], "review_rounds": r["review_rounds"],
              "max_rounds": r["max_review_rounds"], "priority": r["priority"] or ""}
             for r in conn.execute(
                 "SELECT * FROM tasks ORDER BY milestone_id, created_at").fetchall()]
    # 派生忙碌状态（只读计算，不写 DB）：
    #   subagent 忙 = 有 in_progress 任务；主 agent 忙 = 有待评审(in_review)提交
    busy_owner = {t["owner_id"] for t in tasks if t["status"] == "in_progress"}
    busy_sup = {t["supervisor_id"] for t in tasks if t["status"] == "in_review"}
    agents = [{"id": r["id"], "role": r["role"], "dept": r["dept"] or "",
               "name": r["name"], "status": r["status"],
               "is_temp": bool(r["is_temp"]),
               "busy": r["id"] in busy_owner or r["id"] in busy_sup}
              for r in conn.execute(
                  "SELECT * FROM agents ORDER BY role, dept").fetchall()]
    audit = [{"ts": r["ts"], "actor": r["actor_id"], "action": r["action"],
              "target": r["target"] or ""}
             for r in conn.execute(
                 "SELECT ts,actor_id,action,target FROM audit_log "
                 "ORDER BY id DESC LIMIT 15").fetchall()]
    n_total = len(tasks)
    n_done = sum(1 for t in tasks if t["status"] == "done")
    n_doing = sum(1 for t in tasks if t["status"] == "in_progress")
    n_todo = sum(1 for t in tasks if t["status"] in ("backlog", "todo"))
    n_esc = sum(1 for t in tasks if t["status"] == "escalated")
    n_art = conn.execute(
        "SELECT COUNT(*) c FROM artifacts WHERE status='merged'").fetchone()["c"]
    cost = conn.execute(
        "SELECT COALESCE(SUM(cost_usd),0) v FROM budget").fetchone()["v"]
    first = conn.execute(
        "SELECT MIN(ts) v FROM audit_log").fetchone()["v"]
    now = db.now()
    elapsed = None
    if first:
        try:
            elapsed = int((datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
                           - datetime.strptime(first, "%Y-%m-%d %H:%M:%S")).total_seconds())
        except ValueError:
            elapsed = None
    # 当前活跃阶段：有 in_progress 任务的最新里程碑
    active_stage = None
    for t in tasks:
        if t["status"] == "in_progress":
            active_stage = t["milestone_id"]
    return {
        "generated_at": now, "elapsed_s": elapsed,
        "stats": {"total": n_total, "done": n_done, "doing": n_doing,
                  "todo": n_todo, "escalated": n_esc, "artifacts": n_art,
                  "cost_usd": round(cost, 4)},
        "milestones": ms, "tasks": tasks, "agents": agents, "audit": audit,
    }


def export_live(out=None):
    """导出 dashboard/live_state.js（window.__LIVE__ = {...}）。"""
    out = out or os.path.join(_dash_dir(), "live_state.js")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    conn = db.connect()
    try:
        state = _live_state(conn)
    finally:
        conn.close()
    js = "window.__LIVE__ = " + json.dumps(
        state, ensure_ascii=False).replace("<", "\\u003c") + ";"
    with open(out, "w", encoding="utf-8") as f:
        f.write(js)
    return out


_LIVE_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Harness 实时进度</title>
<style>
:root{--bg:#0E141B;--card:#161F2A;--card2:#1B2634;--line:#263443;
--tx:#E6EDF3;--tx2:#8B98A5;--tx3:#5B6B7A;
--green:#2EA043;--blue:#1F6FEB;--gray:#57606A;--red:#F85149;
--orange:#D29922;--cyan:#39C5CF;--purple:#A371F7;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,'Segoe UI',
'Microsoft YaHei',sans-serif;font-size:14px;line-height:1.5;}
.wrap{max-width:1120px;margin:0 auto;padding:20px 16px 60px;}
.top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:18px;}
h1{font-size:20px;font-weight:600;display:flex;align-items:center;gap:10px;}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;animation:pulse 1.6s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
.meta{font-size:12px;color:var(--tx2);text-align:right;}
.meta b{color:var(--tx);}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:16px 18px;margin-bottom:16px;}
.card h2{font-size:13px;font-weight:600;color:var(--tx2);margin-bottom:12px;
letter-spacing:.5px;text-transform:uppercase;}
/* 统计行 */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:16px;}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;}
.stat .k{font-size:12px;color:var(--tx2);}
.stat .v{font-size:22px;font-weight:600;margin-top:2px;}
/* 里程碑阶段条 */
.stages{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:16px;}
.stage{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;position:relative;}
.stage.active{border-color:var(--blue);box-shadow:0 0 0 1px var(--blue);}
.stage .nm{font-weight:600;font-size:14px;display:flex;justify-content:space-between;align-items:center;}
.stage .st{font-size:12px;color:var(--tx2);margin:4px 0 8px;}
.bar{height:6px;background:var(--card2);border-radius:4px;overflow:hidden;}
.bar>div{height:100%;background:var(--blue);border-radius:4px;transition:width .5s;}
.stage.done .bar>div{background:var(--green);}
.stage.review .bar>div{background:var(--orange);}
.stage.failed .bar>div{background:var(--red);}
.stage .sub{font-size:11px;color:var(--tx3);margin-top:6px;display:flex;justify-content:space-between;}
/* 徽章 */
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:11px;font-weight:600;}
.p-done,.p-built,.p-merged{background:rgba(46,160,67,.16);color:#3FB950;}
.p-in_progress,.p-in_review{background:rgba(31,111,235,.16);color:#58A6FF;}
.p-review{background:rgba(210,153,34,.16);color:#D29922;}
.p-escalated,.p-failed,.p-rejected{background:rgba(248,81,73,.16);color:#F85149;}
.p-backlog,.p-todo,.p-pending,.p-idle{background:rgba(87,96,106,.2);color:var(--tx2);}
.p-working{background:rgba(31,111,235,.16);color:#58A6FF;}
.p-blocked{background:rgba(248,81,73,.16);color:#F85149;}
.p-retired{background:rgba(87,96,106,.2);color:var(--tx3);}
/* 表格 */
table{width:100%;border-collapse:collapse;font-size:13px;}
th{text-align:left;color:var(--tx2);font-weight:500;font-size:12px;
border-bottom:1px solid var(--line);padding:6px 8px;}
td{padding:7px 8px;border-bottom:1px solid rgba(38,52,67,.5);}
tr:hover td{background:rgba(31,111,235,.05);}
.mid{color:var(--cyan);font-weight:600;}
.dept{font-size:12px;color:var(--tx2);}
/* 审计时间线 */
.audit{list-style:none;max-height:240px;overflow:auto;}
.audit li{padding:5px 0;border-bottom:1px dashed rgba(38,52,67,.4);
font-size:12px;color:var(--tx2);display:flex;gap:8px;}
.audit .ts{color:var(--tx3);white-space:nowrap;}
.audit .ac{color:var(--cyan);font-weight:600;white-space:nowrap;}
.audit .act{color:var(--tx);}
.active-now{background:var(--card2);border:1px solid var(--blue);border-radius:12px;
padding:10px 14px;margin-bottom:16px;font-size:13px;display:flex;gap:10px;align-items:center;}
.empty{color:var(--tx3);font-size:13px;padding:8px 0;}
.refresh-note{position:fixed;right:14px;bottom:12px;font-size:11px;color:var(--tx3);
background:var(--card);border:1px solid var(--line);border-radius:999px;padding:4px 12px;}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h1><span class="dot" id="dot"></span>Harness 实时进度</h1>
    <div class="meta" id="meta">加载中…</div>
  </div>
  <div id="activeNow"></div>
  <div class="stats" id="stats"></div>
  <div class="card"><h2>五阶段进度</h2><div class="stages" id="stages"></div></div>
  <div class="card"><h2>任务清单</h2><div id="tasks"></div></div>
  <div class="card"><h2>Agent 组织</h2><div id="agents"></div></div>
  <div class="card"><h2>最近审计事件</h2><ul class="audit" id="audit"></ul></div>
</div>
<div class="refresh-note">每 5 秒自动刷新</div>
<script>
var ST = window.__LIVE__ || null;
var COLORS = {done:'#3FB950',built:'#3FB950',merged:'#3FB950',
  in_progress:'#58A6FF',in_review:'#58A6FF',working:'#58A6FF',
  review:'#D29922',escalated:'#F85149',failed:'#F85149',rejected:'#F85149',blocked:'#F85149',
  backlog:'#8B98A5',todo:'#8B98A5',pending:'#8B98A5',idle:'#8B98A5',retired:'#5B6B7A'};
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){
  return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function pill(st){return '<span class="pill p-'+esc(st)+'">'+esc(st)+'</span>';}
function fmtDur(s){if(s==null)return '-';if(s<60)return s+'s';
  var m=Math.floor(s/60),h=Math.floor(m/60);return h>0?h+'h'+m%60+'m':m+'m'+s%60+'s';}
function render(){
  if(!ST){document.getElementById('meta').innerHTML='暂无数据（等待导出 live_state.js）';
    return;}
  var now=new Date(ST.generated_at.replace(' ','T')+'+08:00');
  var age=Math.max(0,(Date.now()-now.getTime())/1000);
  var running=ST.stats.doing>0||ST.milestones.some(function(m){return m.status=='in_progress'||m.status=='pending';});
  var dot=document.getElementById('dot');
  dot.style.background=running?(age<30?'#2EA043':'#D29922'):'#57606A';
  var meta=document.getElementById('meta');
  meta.innerHTML='数据时间 <b>'+esc(ST.generated_at)+'</b> · 已运行 <b>'+fmtDur(ST.elapsed_s)+
    '</b> · 更新于 <b>'+Math.round(age)+'s</b> 前';
  var s=ST.stats;
  var stats=document.getElementById('stats');
  stats.innerHTML=[
    ['任务总数',s.total,''],
    ['已完成',s.done,'#3FB950'],
    ['进行中',s.doing,'#58A6FF'],
    ['待办',s.todo,''],
    ['升级/阻塞',s.escalated,'#F85149'],
    ['合入制品',s.artifacts,'#39C5CF'],
    ['成本', '$'+s.cost_usd,'#A371F7']
  ].map(function(x){return '<div class="stat"><div class="k">'+x[0]+'</div>'+
    '<div class="v"'+(x[2]?' style="color:'+x[2]+'"':'')+'>'+esc(x[1])+'</div></div>';}).join('');
  // 阶段
  var stages=document.getElementById('stages');
  stages.innerHTML=ST.milestones.map(function(m){
    var pct=m.total_tasks?Math.round(m.done_tasks*100/m.total_tasks):0;
    return '<div class="stage '+esc(m.status)+(m.status=='in_progress'||m.status=='pending'?' active':'')+'">'+
      '<div class="nm"><span>'+esc(m.id)+' · '+esc(m.name)+'</span>'+pill(m.status)+'</div>'+
      '<div class="st">'+esc(m.stage)+' · '+esc(m.goal||'')+'</div>'+
      '<div class="bar"><div style="width:'+pct+'%"></div></div>'+
      '<div class="sub"><span>任务 '+m.done_tasks+'/'+m.total_tasks+'</span>'+
      '<span>制品 '+m.artifacts+'</span></div></div>';
  }).join('')||'<div class="empty">暂无里程碑（五阶段 campaign 尚未创建）</div>';
  // 活跃任务
  var doing=ST.tasks.filter(function(t){return t.status=='in_progress'||t.status=='escalated';});
  var an=document.getElementById('activeNow');
  an.innerHTML=doing.length?('<div class="active-now"><b>当前进行中：</b>'+doing.map(function(t){
    return esc(t.milestone_id)+' '+esc(t.title)+' ['+esc(t.dept)+'] '+pill(t.status);
  }).join('　')+'</div>'):'';
  // 任务表
  var tasks=document.getElementById('tasks');
  if(!ST.tasks.length){tasks.innerHTML='<div class="empty">暂无任务</div>';}
  else{
    var html='<table><thead><tr><th>ID</th><th>里程碑</th><th>任务</th><th>部门</th>'+
      '<th>状态</th><th>打回</th></tr></thead><tbody>';
    ST.tasks.forEach(function(t){
      html+='<tr><td class="mid">'+esc(t.id)+'</td><td>'+esc(t.milestone_id)+'</td>'+
        '<td>'+esc(t.title)+'</td><td class="dept">'+esc(t.dept)+'</td>'+
        '<td>'+pill(t.status)+'</td><td class="dept">'+t.review_rounds+'/'+t.max_rounds+'</td></tr>';
    });
    tasks.innerHTML=html+'</tbody></table>';
  }
  // agents
  var agents=document.getElementById('agents');
  if(!ST.agents.length){agents.innerHTML='<div class="empty">暂无 agent</div>';}
  else{
    agents.innerHTML='<table><thead><tr><th>ID</th><th>角色</th><th>部门</th><th>名称</th>'+
      '<th>运行状态</th><th>生命周期</th></tr></thead><tbody>'+ST.agents.map(function(a){
        var run=a.busy?'<span class="pill p-working">工作中</span>':'<span class="pill p-idle">空闲</span>';
        return '<tr><td class="mid">'+esc(a.id)+'</td><td>'+esc(a.role)+'</td>'+
          '<td class="dept">'+esc(a.dept)+'</td><td>'+esc(a.name)+
          (a.is_temp?' <span class="dept">(临时)</span>':'')+'</td>'+
          '<td>'+run+'</td><td>'+pill(a.status)+'</td></tr>';
      }).join('')+'</tbody></table>';
  }
  // audit
  var audit=document.getElementById('audit');
  audit.innerHTML=ST.audit.map(function(a){
    return '<li><span class="ts">'+esc(a.ts)+'</span><span class="ac">'+esc(a.actor)+'</span>'+
      '<span class="act">'+esc(a.action)+'</span><span>'+esc(a.target)+'</span></li>';
  }).join('')||'<li>暂无审计事件</li>';
}
function loadLive(){
  var s=document.createElement('script');
  s.src='live_state.js?t='+Date.now();
  s.onload=function(){ST=window.__LIVE__||ST;render();};
  document.body.appendChild(s);
}
loadLive();
setInterval(loadLive,5000);
</script>
</body>
</html>"""


def render_live(out=None):
    """生成 dashboard/live.html（前端动态渲染 + 每 5s 重载 live_state.js）。"""
    out = out or os.path.join(_dash_dir(), "live.html")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(_LIVE_HTML)
    return out


if __name__ == "__main__":
    path = render_dashboard()
    print(f"进度看板已生成: {path}")
    lpath = render_live()
    print(f"实时直播页已生成: {lpath}")
