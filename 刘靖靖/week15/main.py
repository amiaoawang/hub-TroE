"""Harness S1 入口：像素 demo「策划出需求 → 程序实现 → QA 验收」最小闭环。

用法（在 backend/ 目录下）：
    py313\\Scripts\\python.exe -m src.main --demo            # 正常闭环（一次通过）
    py313\\Scripts\\python.exe -m src.main --demo --reject   # 演示打回-修订循环
    py313\\Scripts\\python.exe -m src.main --demo --escalate # 演示升级仲裁
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import db                # noqa: E402
from src import audit             # noqa: E402
from src import build             # noqa: E402
from src import dashboard         # noqa: E402
from src import mail              # noqa: E402
from src import report            # noqa: E402
from src import retro             # noqa: E402
from src import tickets           # noqa: E402
from src.llm import LLMRouter     # noqa: E402
from src.agents.producer import Producer            # noqa: E402
from src.agents.main_agent import MainAgent         # noqa: E402
from src.agents.sub_agent import SubAgent           # noqa: E402

DEFAULT_CONFIG = {
    "llm": {
        "default": "main",
        "models": {"main": {"provider": "mock", "model": "mock-1"}},
        "routing": {"producer": "main", "main": "main", "sub": "main"},
        "fallback": [],
        # §7.6 失败矩阵：退避重试序列 + 熔断（连续失败暂停）
        "retry": {"backoff_s": [1, 5, 30],
                  "circuit_breaker": {"threshold": 5, "cooldown_s": 600}},
    },
    "limits": {"max_concurrency": 3, "subagent_max": 4,
               "review_rounds_max": 3, "max_ticks": 50,
               "ack_timeout_s": 30, "retry_max": 3},
    # §9.2 预算控制（0 = 不限制）
    "budget": {"daily_limit_usd": 0, "image_daily_limit": 20},
    "topics": {
        "broadcast": ["宪章", "里程碑", "汇报"],   # 广播主题：CC 全员可见
        "always_cc": ["main-pm"],                 # PM 始终收 CC（全局跟踪）
        "subscriptions": {                        # 普通主题：仅订阅者收 CC（§5.2）
            "数值": ["main-design", "main-program"],
            "构建": ["main-program", "main-qa"],
            "美术": ["main-art", "main-design"],
            "质量": ["main-qa", "main-program"],
        },
    },
}


def load_config():
    """读 config.yaml；文件缺失或解析失败时用内嵌默认值（mock 单模型）。
    合并经验层覆盖（playbook/overrides.yaml）——自进化执行闭环的生效层。"""
    try:
        import yaml
        path = os.path.join(db.BASE, "config.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            merged = {**DEFAULT_CONFIG, **cfg}
            merged["llm"] = {**DEFAULT_CONFIG["llm"], **(cfg.get("llm") or {})}
            llm_models = (cfg.get("llm") or {}).get("models") or {}
            merged["llm"]["models"] = {**DEFAULT_CONFIG["llm"]["models"], **llm_models}
            merged["limits"] = {**DEFAULT_CONFIG["limits"], **(cfg.get("limits") or {})}
            # 经验层覆盖：overrides.yaml 深合并（优先于 config.yaml）
            merged = retro.merge_config(merged, retro.overrides_load())
            return merged
    except Exception:
        pass
    return DEFAULT_CONFIG


def register_agent(agent, parent_id=None):
    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO agents (id,role,dept,name,parent_id,status,max_subagents,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (agent.id, agent.role, agent.dept, agent.name, parent_id, "idle",
         4, db.now()))
    conn.commit()
    conn.close()


def _load_main_agents(router, topics_cfg):
    """从数据库重建主 agent 内存对象（按需汇报用，不依赖 demo 会话）。"""
    conn = db.connect()
    rows = conn.execute("SELECT * FROM agents WHERE role='main'").fetchall()
    conn.close()
    agents = {}
    for r in rows:
        agents[r["id"]] = MainAgent(r["id"], r["dept"], r["name"],
                                    router=router, topics_cfg=topics_cfg)
    return agents


def _load_producer_id():
    conn = db.connect()
    row = conn.execute("SELECT id FROM agents WHERE role='producer' LIMIT 1").fetchone()
    conn.close()
    return row["id"] if row else None


def _spawn_temp_burst(art, design, cfg):
    """内容填充期弹性扩容：主 agent 临时创建 subagent，批量派活，返回临时 sub 列表。"""
    temp_max = cfg["limits"].get("temp_max", 5)
    task_pack = {"charter": design.context_pack.get("charter", ""),
                 "note": "内容填充批量任务：按模板与规范产出，DoD 客观可查"}
    # 美术主：3 个临时美术 sub
    art_temps = []
    for i in range(1, 4):
        t = art.create_temp_subagent("美术", f"美术外包{i}", task_pack,
                                     temp_max=temp_max)
        if t:
            art_temps.append(t)
    # 策划主：2 个临时策划 sub
    design_temps = []
    for i in range(1, 3):
        t = design.create_temp_subagent("策划", f"策划外包{i}", task_pack,
                                        temp_max=temp_max)
        if t:
            design_temps.append(t)
    # 批量任务：9 个美术素材 + 4 个数值配置，轮流分给临时 sub（受各自主 agent 派发）
    for i in range(1, 10):
        owner = art_temps[(i - 1) % len(art_temps)].id if art_temps else None
        if owner:
            tickets.create_task(
                f"批量素材 {i}", "按美术规范产出精灵素材（命名/格式合规）",
                owner, "main-art", "美术",
                dod=["命名规范", "格式合规"], budget_tokens=8000)
    for i in range(1, 5):
        owner = design_temps[(i - 1) % len(design_temps)].id if design_temps else None
        if owner:
            tickets.create_task(
                f"关卡数值配置 {i}", "产出关卡数值配置（schema 合法范围内变体）",
                owner, "main-design", "策划",
                dod=["数值表通过 schema 校验"], budget_tokens=8000)
    print(f"[弹性] 已创建临时 subagent：美术 {len(art_temps)} + 策划 {len(design_temps)}，"
          f"批量任务 9 素材 + 4 配置")
    return art_temps + design_temps


def _check_task_cycles(producer):
    """C7 依赖环检测（PM 职责）：发现环 → 环上任务标 escalated → Producer 仲裁 + 审计。
    返回发现的环数（0 = 无环）。"""
    cycles = tickets.detect_cycles()
    for cyc in cycles:
        for tid in cyc[:-1]:                       # 环上任务（去掉闭环重复尾）
            t = tickets.get(tid)
            if t and t["status"] in ("backlog", "todo", "in_progress"):
                tickets.set_status(tid, "escalated")
        producer.arbitrate(cyc[0], {"cycle": cyc, "reason": "依赖环（C7）"})
        producer.log_action("task_cycle", cyc[0], {"cycle": cyc})
        print(f"[C7] 检测到依赖环: {' → '.join(cyc)} → 升级 Producer 仲裁", flush=True)
    return len(cycles)


def _check_budget(main_agents, cfg, producer):
    """§9.2 预算控制：
    - 任务级：in_progress 任务累计 token 超 budget_tokens → 主 agent 告警
      （一次，审计留痕；告警后评审/产出自动降级为摘要模式省 token）。
    - 日级：当日成本超 daily_limit_usd → 设置各主 agent budget_hold
      （dispatch_ready 暂停 P2/P3 派发）。
    返回 (warned, daily_hold)。"""
    from src import budget as budget_mod
    warned = 0
    # in_progress（执行中）+ in_review（评审中）都在花 token，都要查
    for m in main_agents:
        for t in tickets.tasks_by(supervisor_id=m.id,
                                  status="in_progress") \
                 + tickets.tasks_by(supervisor_id=m.id, status="in_review"):
            limit = t["budget_tokens"] or 0
            if limit <= 0 or not budget_mod.task_over_budget(t["id"], limit):
                continue
            # 防重复告警：已发过则跳过（审计留痕判断）
            conn = db.connect()
            try:
                done = conn.execute(
                    "SELECT COUNT(*) c FROM audit_log WHERE action='budget_warn' "
                    "AND target=?", (t["id"],)).fetchone()["c"]
            finally:
                conn.close()
            if done:
                continue
            ti, to = budget_mod.task_tokens(t["id"])
            cost = budget_mod.task_cost(t["id"])
            m.send(m.parent_id,
                   f"[{t['id']}] 任务 token 预算超支",
                   f"任务 {t['id']} {t['title']} 已用 {ti + to}/{limit} token"
                   f"（${cost:.4f}），超出预算。已通知评估降级为摘要模式。",
                   task_id=t["id"], priority="high")
            m.log_action("budget_warn", t["id"],
                         {"used": ti + to, "limit": limit, "cost_usd": cost})
            warned += 1
            print(f"[预算] 任务 {t['id']} token 预算超支"
                  f"（{ti + to}/{limit}），已告警", flush=True)
    hold = budget_mod.daily_exceeded(cfg)
    for m in main_agents:
        m.budget_hold = hold
    if hold:
        print("[预算] 日级预算超限：低优先级任务（P2/P3）暂停派发", flush=True)
        producer.log_action("budget_daily_hold", "daily",
                            {"cost_usd": budget_mod.daily_cost()})
    return warned, hold


# 双盲评审轮转游标（§11.2：主 agent 定期抽查其他部门产出，对抗"自己人偏见"）
_CROSS_REVIEW_IDX = 0


def _cross_review_once(main_agents, producer):
    """§11.2 双盲评审：每轮轮转抽查一个「其他部门」的 in_review 任务，
    blind 评审（评审者与产出者互相不可知：prompt 不含 owner/supervisor 身份）。

    - 同一任务不重复评审（audit 查重）
    - 评审者 = 部门不同的主 agent（QA 评程序、程序评策划……）
    - 结果写审计（action=cross_review）；明确不通过 → 通知原 supervisor 参考
    - 每轮最多 1 个（成本控制）
    返回 (task_id, verdict) 或 (None, None)。
    """
    global _CROSS_REVIEW_IDX
    conn = db.connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM tasks WHERE status='in_review'").fetchall()]
        done = {r["target"] for r in conn.execute(
            "SELECT target FROM audit_log WHERE action='cross_review'")}
    finally:
        conn.close()
    candidates = [t for t in rows if t["id"] not in done]
    if not candidates:
        return None, None
    task = candidates[_CROSS_REVIEW_IDX % len(candidates)]
    _CROSS_REVIEW_IDX += 1
    # 评审者：与产出部门不同的主 agent（盲评——不泄露产出者）
    reviewers = [m for m in main_agents
                 if m.dept != (task["dept"] or "")
                 and m.id != (task["supervisor_id"] or "")]
    if not reviewers:
        return None, None
    reviewer = reviewers[0]
    # 盲评材料：只取制品内容，不含任何 agent 身份信息
    workdir = os.path.join(db.ARTIFACTS_DIR, "workspace",
                           task["owner_id"], task["id"])
    artifact_txt = ""
    for f in ("design.json", "report.md", "player.py", "game.html",
              "battlefield.html"):
        p = os.path.join(workdir, f)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as _f:
                    artifact_txt += f"\n\n【{f}】\n{_f.read()[:600]}"
            except (OSError, UnicodeDecodeError):
                pass
    verdict = reviewer.llm(
        "评审",
        f"【双盲评审】请以第三方身份评审任务 {task['id']} {task['title']}"
        f"（部门={task['dept']}）的产出质量。你不知晓产出者身份，请客观评估。"
        f"产出材料：{artifact_txt or '（无附件材料）'}"
        f"若无明确严重问题应判定通过。请严格以『评审结论：通过』或"
        f"『评审结论：不通过』开头，随后简述理由。",
        task_id=task["id"])
    rejected = reviewer._verdict_rejects(verdict)
    reviewer.log_action("cross_review", task["id"],
                        {"reviewer": reviewer.id, "dept": task["dept"],
                         "blind": True, "rejected": rejected,
                         "verdict": (verdict or "")[:120]})
    if rejected:
        reviewer.send(task["supervisor_id"],
                      f"RE: [{task['id']}] 双盲评审意见（仅供参考）",
                      f"（盲评，不泄露评审者）\n{verdict}",
                      task_id=task["id"], priority="normal")
    print(f"[盲评] {reviewer.id} 抽查 {task['id']}"
          f"（{task['dept']}）→ {'不通过' if rejected else '通过'}", flush=True)
    return task["id"], "rejected" if rejected else "passed"


def _rescue_orphan_tasks(main_agents, resident_subs):
    """救回孤儿任务：in_progress 任务的 owner 已 retired/不存在时，
    按 supervisor 转交对应主 agent 的常驻 sub。返回救回数。"""
    by_id = {m.id: m for m in main_agents}
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT id, owner_id, supervisor_id FROM tasks "
            "WHERE status='in_progress'").fetchall()
        if not rows:
            return 0
        alive = {r["id"] for r in conn.execute(
            "SELECT id FROM agents WHERE status!='retired'").fetchall()}
    finally:
        conn.close()
    rescued = 0
    for r in rows:
        if r["owner_id"] is None or r["owner_id"] in alive:
            continue   # 归属正常，不需要救
        m = by_id.get(r["supervisor_id"])
        target = m.resident_sub_id if m else None
        if not target:
            continue
        conn = db.connect()
        try:
            conn.execute(
                "UPDATE tasks SET owner_id=? WHERE id=? AND status='in_progress'",
                (target, r["id"]))
            conn.commit()
        finally:
            conn.close()
        rescued += 1
    if rescued:
        print(f"  [救援] 孤儿任务转交常驻 sub {rescued} 个（owner 已退役）", flush=True)
    return rescued


def _run_stage(main_agents, resident_subs, temp_subs, producer, cfg,
               milestone_id, max_ticks):
    """跑一个阶段的调度循环，直到该里程碑任务全 done/escalated 或连续多轮无进展。"""
    tick = 0
    stale = 0
    while tick < max_ticks:
        tick += 1
        all_agents = [producer] + main_agents + resident_subs + temp_subs
        for a in all_agents:
            a.ack_all()
        mail.retry_expired(timeout_s=cfg["limits"].get("ack_timeout_s", 30),
                           max_retries=cfg["limits"].get("retry_max", 3))
        # snap 含 review_rounds：打回循环中任务 in_progress→in_review→in_progress
        # 起终点状态相同，仅比较 (id,status) 会误判"无变化"提前 break
        snap = [(t["id"], t["status"], t["review_rounds"])
                for t in tickets.tasks_by()]
        for m in main_agents:
            m.dispatch_ready()
        for s in resident_subs + temp_subs:
            s.work_once()
        # §11.2 双盲评审：必须在 review_submissions 之前调用——
        # 趁任务还在 in_review（评审完就抽不到了），轮转抽查其他部门产出
        _cross_review_once(main_agents, producer)
        for m in main_agents:
            m.review_submissions()
        # 仲裁兜底：escalated 任务由 Producer 记录决策并重置 backlog 重新派发
        for m in main_agents:
            for t in tickets.tasks_by(supervisor_id=m.id, status="escalated"):
                producer.arbitrate(t["id"], {"rounds": t["review_rounds"],
                                             "reason": "评审超限，重置重试"})
                tickets.set_status(t["id"], "backlog")
        # C7：依赖环检测（发现环 → 升级 Producer）
        _check_task_cycles(producer)
        # §9.2 预算控制：任务级超支告警 + 日级暂停低优先级派发
        _check_budget(main_agents, cfg, producer)
        stage_tasks = tickets.tasks_by(milestone_id=milestone_id)
        if stage_tasks and all(t["status"] in ("done", "escalated")
                               for t in stage_tasks):
            break
        if snap == [(t["id"], t["status"], t["review_rounds"])
                    for t in tickets.tasks_by()]:
            stale += 1
            # 判僵局前先救孤儿任务（owner 已退役的 in_progress 任务转常驻）
            if _rescue_orphan_tasks(main_agents, resident_subs):
                stale = 0
            elif stale >= 3:      # 连续 3 轮无任何变化且无孤儿可救 → 真僵局
                break
        else:
            stale = 0
        try:      # 每轮调度后导出实时状态（直播页数据源；失败不阻断调度）
            dashboard.export_live()
        except Exception:  # noqa: BLE001
            pass
    return tick


def _theme_games_dir():
    """可玩产物目录（多项目：default → D:\\gameHarness\\game\\，
    其他项目 → backend/projects/<name>/game/）。"""
    return (os.path.join(db.BASE, "..", "game")
            if db.PROJECT_NAME == "default"
            else os.path.join(db.PROJECT_DIR, "game"))


def _copy_playable(build_dir, theme):
    """把构建快照里的可玩产物复制到可玩目录，命名为该主题的产物文件名。
    内置主题（mario/battlefield）有注册的产物名；自定义主题 = <theme>.html。
    返回 (out_path, theme_file)；无可玩产物返回 (None, None)。"""
    from src import themes
    target = themes.product_file(theme)
    for f in os.listdir(build_dir):
        if not f.endswith(".html"):
            continue
        src = os.path.join(build_dir, f)
        out_dir = _theme_games_dir()
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, target)
        shutil.copy(src, out)
        return os.path.abspath(out), target
    return None, None


def _render_game_index():
    """重建可玩入口索引页 game/index.html：动态枚举主题注册表 + 产物文件，
    生成选择页（未生成的主题显示"尚未生成"，跑对应主题 demo 后自动出现）。"""
    out_dir = _theme_games_dir()
    os.makedirs(out_dir, exist_ok=True)
    from datetime import datetime as _dt
    from src import themes
    themes_list = themes.list_themes()
    cards = []
    _PALETTE = ["#7fb069", "#c25b4e", "#4e7fb5", "#a86fb0", "#d0a33c",
                "#5aa2a0", "#8a6f5a", "#6b7a8f"]
    for i, t in enumerate(themes_list):
        key, title, fname, desc = (t["id"], t["name"], t["product"],
                                   t.get("desc", ""))
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            size_kb = max(1, round(os.path.getsize(p) / 1024))
            updated = _dt.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
            cards.append(f"""<a class="card" href="{fname}">
      <div class="thumb" style="background:{_PALETTE[i % len(_PALETTE)]}">{title}</div>
      <div class="body">
        <h3>{title} <span class="ok">✓ 可玩</span></h3>
        <p>{desc}</p>
        <div class="meta">{size_kb} KB · 更新于 {updated}</div>
      </div>
    </a>""")
        else:
            cards.append(f"""<div class="card off">
      <div class="thumb">{title}</div>
      <div class="body">
        <h3>{title} <span class="no">尚未生成</span></h3>
        <p>{desc}</p>
        <div class="meta">跑 {fname.replace('.html', '')} 主题 demo 后自动出现</div>
      </div>
    </div>""")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harness 游戏原型 · 入口</title>
<style>
  body {{ margin:0; background:#12141a; color:#e8e6e0; min-height:100vh;
         font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif; }}
  .wrap {{ max-width:860px; margin:0 auto; padding:48px 20px; }}
  h1 {{ font-size:22px; font-weight:600; margin-bottom:4px; }}
  .sub {{ color:#8b90a0; font-size:13px; margin-bottom:28px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
  .card {{ display:flex; overflow:hidden; border-radius:14px; background:#1c1f28;
          border:1px solid #2a2e3a; text-decoration:none; color:inherit;
          transition:transform .15s, border-color .15s; }}
  .card:hover {{ transform:translateY(-2px); border-color:#4a5568; }}
  .thumb {{ width:110px; display:flex; align-items:center; justify-content:center;
           font-weight:700; color:#fff; text-align:center; padding:0 8px;
           font-size:14px; letter-spacing:1px; }}
  .body {{ flex:1; padding:16px 18px; }}
  h3 {{ margin:0 0 6px; font-size:16px; }}
  .ok {{ color:#7fb069; font-size:12px; font-weight:600; }}
  .no {{ color:#c25b4e; font-size:12px; font-weight:600; }}
  p {{ margin:0 0 10px; color:#a8adbb; font-size:13px; line-height:1.6; }}
  .meta {{ color:#666c7c; font-size:12px; }}
  .card.off {{ opacity:.55; cursor:default; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>🎮 Harness 游戏原型</h1>
  <div class="sub">多智能体流水线产物 · 各主题独立入口 · 生成时间 {_dt.now().strftime('%Y-%m-%d %H:%M')}</div>
  <div class="grid">
    {chr(10).join(cards)}
  </div>
</div>
</body>
</html>"""
    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.abspath(index_path)


# ② 动态任务拆分：调研功能清单 → 系统级任务（每系统独立实现/验收）
SYSTEM_NAMES = {
    "enemy": "敌人系统", "brick": "顶砖块系统", "mushroom": "蘑菇道具系统",
    "pipe": "管道系统", "coin": "金币系统", "life": "生命系统",
    "flag": "旗杆过关系统",
}


def _dynamic_m1_tasks(features, sd, sp, sq, design, program, qa, theme="mario"):
    """M1 原型阶段任务链：按调研 features 把游戏拆成系统级任务（任意系统）。
    每个开启的系统 = 一个「实现」任务（产出 feature_<name>.js，独立验收），
    最后「集成」任务组装完整 game.html + QA「整体验收」。
    features 为空/全关时回退「原型实现 skill:<skill_id>」固定任务（任意主题）。"""
    from src import themes
    t_meta = themes.get_theme(theme) or {}
    skill_id = t_meta.get("skill", theme)
    tname = themes.theme_title(theme)
    tasks = [("玩法数值设计", "策划", sd, design,
              f"定义{tname}核心数值（受模板数值 schema 约束，产出 design.json）", None)]
    sys_titles = []
    for feat, fname in SYSTEM_NAMES.items():
        if features.get(feat):
            tasks.append(
                (f"实现{fname}[{feat}]", "程序", sp, program,
                 f"实现{fname}：数据/更新/绘制完整逻辑，产出 feature_{feat}.js",
                 ["玩法数值设计"]))
            sys_titles.append(f"实现{fname}[{feat}]")
    if sys_titles:
        tasks.append(
            ("系统集成组装完整游戏", "程序", sp, program,
             "汇总已合入的各系统 feature_*.js，组装完整可玩 game.html", sys_titles))
        tasks.append(
            ("整体功能验收", "QA", sq, qa,
             "按调研功能清单验收完整游戏玩法（" + "、".join(
                 SYSTEM_NAMES[k] for k in SYSTEM_NAMES if features.get(k)) + "）",
             ["系统集成组装完整游戏"]))
    else:
        # 回退：无功能清单（任意主题）→ 固定原型任务，走 skill 模板整体生成
        tasks.append((f"{tname}原型实现 skill:{skill_id}", "程序", sp, program,
                      f"读取策划数值，按 skill:{skill_id} 生成可玩原型（{tname}）", None))
        tasks.append(("原型验收", "QA", sq, qa, "验收原型：构建 + 结构检查", None))
    return tasks


def _bf_stages(theme, sd, sp, sq, design, program, qa, features=None):
    """五阶段任务链（任意主题）：原型→垂直切片→内容填充→打磨→发布。
    任务标题/目标注入主题名；M1 按调研 features 动态拆系统任务（② 任务动态拆分）。"""
    from src import themes
    tname = themes.theme_title(theme)
    t_meta = themes.get_theme(theme) or {}
    skill_id = t_meta.get("skill", theme)
    features = features or {}
    return [
        ("M1", "原型", "prototype", f"验证核心玩法：{tname}原型",
         _dynamic_m1_tasks(features, sd, sp, sq, design, program, qa, theme)),
        ("M2", "垂直切片", "vslice", f"{tname}一局完整流程", [
            (f"{tname}关卡设计", "策划", sd, design,
             f"垂直切片数值：{tname}核心节奏与流程", None),
            (f"{tname}切片实现 skill:{skill_id}", "程序", sp, program,
             f"实现完整一局：{tname}核心循环与胜负条件", None),
            ("切片验收", "QA", sq, qa, "验收垂直切片：全流程可用", None),
        ]),
        ("M3", "内容填充", "content", "批量关卡配置（临时 subagent 弹性扩容）", [
            ("内容抽样验收", "QA", sq, qa, "抽样评审批量配置的质量与一致性", None),
        ]),
        ("M4", "打磨", "polish", "基于 QA 反馈迭代数值手感", [
            (f"{tname}手感打磨", "策划", sd, design,
             f"根据 QA 反馈调整数值，产出打磨后 design.json", None),
            (f"{tname}打磨实现 skill:{skill_id}", "程序", sp, program,
             f"按打磨数值重新生成{tname}", None),
            ("打磨验收", "QA", sq, qa, "验收打磨后版本", None),
        ]),
        ("M5", "发布", "release", "最终回归与交付", [
            ("发布前最终回归", "QA", sq, qa, "全量回归 + 最终验收", None),
        ]),
    ]


def _campaign_bf(producer, main_agents, resident_subs, cfg, max_ticks,
                 theme="battlefield"):
    """主题化五阶段 campaign：原型→垂直切片→内容填充→打磨→发布。
    theme: 任意主题 id（内置 mario/battlefield，或已注册 skill 的 id）。"""
    by_id = {m.id: m for m in main_agents}
    design, program, art, qa, pm = (by_id["main-design"], by_id["main-program"],
                                    by_id["main-art"], by_id["main-qa"],
                                    by_id["main-pm"])
    subs = {s.id: s for s in resident_subs}
    sd, sp, sq = subs["sub-design"], subs["sub-program"], subs["sub-qa"]

    stages = _bf_stages(theme, sd, sp, sq, design, program, qa,
                        features=(by_id["main-design"].context_pack.get("research")
                                  or {}).get("features") or {})

    report = []
    tick_total = 0
    for mid, name, stage, goal, tasks in stages:
        # 续跑支持：里程碑已 done 的阶段跳过（进程中断后可从断点继续，不从头重跑）
        conn = db.connect()
        try:
            done = conn.execute(
                "SELECT COUNT(*) c FROM milestones WHERE id=? AND status='done'",
                (mid,)).fetchone()["c"]
        finally:
            conn.close()
        if done:
            print(f"===== {mid} · {name} 已完成（status=done），跳过 =====", flush=True)
            report.append({"milestone": mid, "name": name, "ok": True,
                           "files": 0, "accept": "done", "skipped": True})
            continue
        try:
            tick_total += _run_stage_phase(
                producer, main_agents, resident_subs, cfg, max_ticks,
                by_id, subs, mid, name, stage, goal, tasks, report, theme)
        except Exception as e:  # noqa: BLE001   # 阶段异常留痕，不静默中断整个 campaign
            import traceback
            err = traceback.format_exc()
            log_path = os.path.join(db.LOGS_DIR, "campaign_error.log")
            os.makedirs(db.LOGS_DIR, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== {db.now()} 阶段 {mid} 异常 ===\n{err}")
            print(f"[异常] 阶段 {mid} 失败（详见 {log_path}）：{e}", flush=True)
            report.append({"milestone": mid, "name": name, "ok": False,
                           "files": 0, "accept": "error",
                           "error": str(e)[:120]})
    return tick_total, report


def _run_stage_phase(producer, main_agents, resident_subs, cfg, max_ticks,
                     by_id, subs, mid, name, stage, goal, tasks, report,
                     theme="battlefield"):
    """单个阶段的完整流程：建里程碑 → 建任务 → 弹性扩容 → 调度 → 回收 → 构建验收。"""
    sd, sp, sq = subs["sub-design"], subs["sub-program"], subs["sub-qa"]
    design = by_id["main-design"]
    tick_total = 0
    build.create_milestone(mid, name, stage, goal)
    print(f"\n===== 阶段 {mid} · {name}（{stage}）=====")
    # 1) 创建本阶段任务（内部依赖链：设计→实现→验收）
    #    续跑去重：里程碑下已有同名任务则复用（进程中断后续跑不重复创建）
    tid_map = {}
    design_anchor = next((t[0] for t in tasks if t[1] == "策划"), None)
    conn0 = db.connect()
    try:
        existing = {r["title"]: r["id"] for r in conn0.execute(
            "SELECT id, title FROM tasks WHERE milestone_id=?", (mid,))}
    finally:
        conn0.close()
    for title, dept, sub, main_agent, desc, _dep in tasks:
        if _dep:   # 显式依赖（动态任务链）：按标题映射到 tid
            deps = [tid_map[t] for t in _dep if t in tid_map] or None
        elif title == design_anchor:
            deps = None
        elif ("实现" in title or "原型" in title or "切片" in title
              or "打磨" in title and dept == "程序"):
            deps = [tid_map.get(design_anchor)] if design_anchor else None
        elif "验收" in title or "回归" in title:
            deps = [v for k, v in tid_map.items() if k != title] or None
        else:
            deps = None
        if title in existing:
            tid = existing[title]          # 续跑：复用已有任务
            print(f"  [任务] {tid} [{dept}] {title}（续跑复用）")
        else:
            tid = tickets.create_task(title, desc, sub.id, main_agent.id, dept,
                                  milestone_id=mid, dod=["符合部门校验"],
                                  depends_on=deps)
            print(f"  [任务] {tid} [{dept}] {title}")
        tid_map[title] = tid
    # 2) M3 内容填充：临时策划 sub 批量产关卡配置
    temp_subs = []
    if stage == "content":
        temp_subs = _spawn_bf_content(design, tid_map)
    # 3) 跑本阶段调度
    ticks = _run_stage(main_agents, resident_subs, temp_subs, producer,
                       cfg, mid, max_ticks)
    tick_total += ticks
    # 4) 回收临时 sub（名下未完成任务自动转交常驻 sub）
    for m in main_agents:
        for s in list(m.temp_subs):
            m.retire_temp_subagent(s)
    # 5) 构建 + 用户验收（构建失败不标记通过）+ 复制可玩产物
    build_dir, ok, n_files = build.build_milestone(mid)
    if ok:
        accept = build.user_acceptance(mid, approved=True,
                                       notes=f"{name}验收通过")
    else:
        accept = build.user_acceptance(mid, approved=False,
                                       notes=f"{name}构建失败（任务未全部完成），反馈进需求池")
        tickets.create_task(f"{name}整改", f"{name}构建失败，需修复后重建",
                            sd.id, design.id, "策划", milestone_id=mid,
                            priority="P1",
                            dod=["数值表通过 schema 校验"])
    game_path, theme_file = _copy_playable(build_dir, theme)
    if game_path:
        print(f"  [构建] {mid} {'成功' if ok else '失败'}（{n_files} 件）· 验收 {accept}"
              f" · 可玩产物 → {game_path}（{theme_file}）")
        try:      # 刷新多主题入口选择页（失败不阻断）
            _render_game_index()
        except Exception:  # noqa: BLE001
            pass
    else:
        print(f"  [构建] {mid} {'成功' if ok else '失败'}（{n_files} 件）· 验收 {accept}"
              " · 本阶段无可玩产物")
    report.append({"milestone": mid, "name": name, "ok": ok,
                   "files": n_files, "accept": accept})
    # 5 个阶段全 done 后打印战地数值来源
    if mid == "M5":
        conn = db.connect()
        try:
            arts = conn.execute(
                "SELECT a.path FROM artifacts a JOIN tasks t ON a.task_id=t.id "
                "WHERE t.milestone_id='M4' AND a.status='merged'").fetchall()
        finally:
            conn.close()
        for a in arts:
            if a["path"] and os.path.exists(os.path.join(a["path"], "design.json")):
                pass
    return tick_total


def _spawn_bf_content(design, tid_map):
    """M3 内容填充：临时策划 sub 批量产关卡配置（按规范变体）+ 挂抽样验收依赖。
    续跑去重：M3 下已有同名配置任务则复用（不重复创建）。"""
    task_pack = {"note": "内容填充：按策划规范批量产关卡配置（数值变体）"}
    temps = [design.create_temp_subagent("策划", f"配置外包{i}", task_pack)
             for i in range(1, 3)]
    temps = [t for t in temps if t]
    conn0 = db.connect()
    try:
        existing = {r["title"]: r["id"] for r in conn0.execute(
            "SELECT id, title FROM tasks WHERE milestone_id='M3'")}
    finally:
        conn0.close()
    config_ids = []
    for i in range(1, 7):
        title = f"关卡配置 {i}"
        if title in existing:
            config_ids.append(existing[title])      # 续跑：复用
            continue
        owner = temps[(i - 1) % len(temps)].id if temps else None
        if owner:
            tid = tickets.create_task(
                title, "按规范产出关卡数值配置（schema 合法变体）",
                owner, "main-design", "策划", milestone_id="M3",
                dod=["数值表通过 schema 校验"], budget_tokens=8000)
            config_ids.append(tid)
    # 抽样验收依赖全部配置
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE tasks SET depends_on=? WHERE id=?",
            (db.json_dumps(config_ids), tid_map.get("内容抽样验收")))
        conn.commit()
    finally:
        conn.close()
    print(f"  [弹性] M3 创建临时策划 sub {len(temps)} 个，批量配置任务 {len(config_ids)} 个")
    return temps


def main():
    ap = argparse.ArgumentParser(description="Harness S1 最小闭环 demo")
    ap.add_argument("--demo", action="store_true", help="跑像素 demo 闭环")
    ap.add_argument("--fresh", action="store_true", help="清空数据库后重跑")
    ap.add_argument("--reject", action="store_true", help="演示打回-修订循环")
    ap.add_argument("--escalate", action="store_true", help="演示升级仲裁")
    ap.add_argument("--mock", action="store_true",
                    help="强制全部模型走 mock（沙箱验证用，不发真实请求）")
    ap.add_argument("--report", action="store_true",
                    help="按需汇报（用户指令触发）：各主 agent 汇报进度/阻塞，PM 汇总写日志。"
                         "可单独执行，也可 --demo 后触发")
    ap.add_argument("--milestone", action="store_true",
                    help="启用里程碑：任务归属 M1，结束后构建可玩 build + 用户验收")
    ap.add_argument("--bad-artifact", action="store_true",
                    help="演示：程序 subagent 首次产出含语法错误，被校验器打回")
    ap.add_argument("--user-reject", action="store_true",
                    help="演示：用户验收不通过 → 反馈进需求池")
    ap.add_argument("--game", action="store_true",
                    help="游戏原型模式：策划数值 → 程序生成真实 HTML5 游戏 → QA 验收 → 可玩构建")
    ap.add_argument("--temp", action="store_true",
                    help="弹性扩容演示：内容填充期临时创建 subagent 批量产任务，干完回收")
    ap.add_argument("--bf", action="store_true",
                    help="主题化五阶段 campaign：原型→垂直切片→内容填充→打磨→发布（真实 LLM 或 --mock）")
    ap.add_argument("--theme", default=None,
                    help="--bf 主题：battlefield（战地一）| mario（平台跳跃）；"
                         "缺省用启动问卷的 theme 字段，仍无则默认 battlefield")
    ap.add_argument("--export-logs", action="store_true", help="仅导出审计日志后退出")
    ap.add_argument("--cleanup", action="store_true",
                    help="清理防膨胀：build 快照留最近 N 个、report 日志留 30、平铺残留")
    ap.add_argument("--keep-builds", type=int, default=3, help="--cleanup 保留的 build 快照数")
    ap.add_argument("--deep", action="store_true",
                    help="--cleanup 深度模式：额外清理已合入任务的 workspace 暂存")
    ap.add_argument("--live", action="store_true",
                    help="实时直播：循环导出 live_state.js（配合 dashboard/live.html 每 5s 自动刷新）")
    ap.add_argument("--live-interval", type=float, default=3.0,
                    help="--live 导出间隔秒数（默认 3）")
    ap.add_argument("--retro", action="store_true",
                    help="复盘（自进化 P0）：指标卡 + 失败模式 + LLM 经验候选（不自动合入 playbook）")
    ap.add_argument("--skill", nargs="*", default=None,
                    help="skill 注册表：list（查看可用能力）")
    ap.add_argument("--ci", action="store_true",
                    help="CI 门禁：语法检查 + 全量测试 + 报告（失败 exit 1）")
    ap.add_argument("--quick", action="store_true",
                    help="--ci 快速模式：只跑核心套件（s1-s4）")
    ap.add_argument("--experiment", default=None,
                    help="A/B 实验：本轮只激活指定经验（如 --experiment R-003），"
                         "其余暂停注入，验证结果打 experiment 标记供归因")
    ap.add_argument("--playbook", nargs="*", default=None,
                    help="经验库操作：list | accept R-xxx | reject R-xxx | add <问题> | <规则> "
                         "| apply [R-xxx] | verify | attribute | reactivate R-xxx | fuse [reset]")
    ap.add_argument("--project", default=None,
                    help="多项目：指定项目名（default = backend/ 根；"
                         "其他 = backend/projects/<name>/，DB/playbook/skills/制品全隔离）")
    ap.add_argument("--list-projects", action="store_true",
                    help="列出已有项目")
    ap.add_argument("--max-ticks", type=int, default=0, help="调度轮次上限")
    # ---- 变更单（§8.3）----
    ap.add_argument("--change", nargs="*", default=None,
                    help="变更单：submit <标题>|<描述> | list | assess <CR-xxx> | "
                         "approve <CR-xxx> | reject <CR-xxx>")
    ap.add_argument("--change-by", default="user",
                    help="--change submit 的提出者（agent_id 或 user）")
    ap.add_argument("--notes", default="", help="--change approve/reject 的备注")
    ap.add_argument("--apply", action="store_true",
                    help="--change approve 时把波及任务重置 backlog（变更生效）")
    # ---- 启动问卷（§3.1/§12.1）：外部问卷文件覆盖内置问卷 ----
    ap.add_argument("--questionnaire-file", default=None,
                    help="启动问卷 JSON 文件路径（P0/P1 两级 schema，覆盖内置问卷）")
    args = ap.parse_args()

    # ---- 多项目：切换项目（必须早于一切 db 操作） ----
    if args.list_projects:
        for p in db.list_projects():
            print(p)
        return
    if args.project:
        db.set_project(args.project)
    print(f"[项目] {db.PROJECT_NAME}（DB: {db.DB_PATH}）")

    if args.fresh:
        db.reset_db()
    else:
        db.init_db()

    cfg = load_config()
    if args.mock:  # 沙箱：覆盖所有模型 provider 为 mock，忽略 API Key
        for m in (cfg["llm"].get("models") or {}).values():
            m["provider"] = "mock"
    else:
        # 真实模型：路由指向缺 Key 的模型时预回退到 default，避免每次失败-降级开销
        for role in list((cfg["llm"].get("routing") or {}).keys()):
            mid = cfg["llm"]["routing"][role]
            mcfg = (cfg["llm"].get("models") or {}).get(mid, {})
            key_env = mcfg.get("api_key_env")
            if (mcfg.get("provider") == "openai" and key_env
                    and not os.environ.get(key_env)):
                cfg["llm"]["routing"][role] = cfg["llm"]["default"]
                print(f"[LLM] 模型 {mid} 缺 Key（{key_env}），角色 {role} 回退 default")
    router = LLMRouter(cfg["llm"])
    topics_cfg = cfg.get("topics", {}) or {}
    max_ticks = args.max_ticks or cfg["limits"]["max_ticks"]

    if args.export_logs:
        path, n = audit.export_logs()
        print(f"审计日志已导出: {path}（{n} 条）")
        return

    # ---- 复盘（自进化 P0）：指标卡 + 失败模式 + LLM 经验候选 ----
    if args.retro:
        db.init_db()
        path, m, patterns, cands = retro.run_retro(router)
        print(f"[复盘] 指标卡+失败模式+经验候选 → {path}")
        print(f"[复盘] 完成率 {m['tasks']['done_rate'] * 100:.0f}% · "
              f"打回率 {m['tasks']['reject_rate'] * 100:.0f}% · "
              f"成本 ${m['budget']['total_cost']} · 失败模式 {len(patterns)} 条")
        if cands:
            print(f"[复盘] LLM 提炼经验候选 {len(cands)} 条"
                  f"（采纳: python -m src.main --playbook add <问题>|<规则>）")
        else:
            print("[复盘] 无经验候选（auto 模式或 LLM 输出不可用）")
        return

    # ---- 经验库（playbook）操作 ----
    # ---- CI 门禁（自进化护栏：进化变更必须过 CI） ----
    if args.ci:
        from src import ci
        db.init_db()
        code = ci.run_ci_cli(quick=args.quick)
        sys.exit(code)

    # ---- skill 注册表（能力热插拔，自进化改进层） ----
    if args.skill is not None:
        from src import skill as skill_reg
        if args.skill and args.skill[0] == "list":
            skills = skill_reg.list_skills()
            if not skills:
                print("[skill] 注册表为空")
            for s in skills:
                tag = "内置" if s.get("builtin") else "外部"
                print(f"  [{s['id']}] ({tag}) {s.get('name', '')} — "
                      f"{s.get('desc', '')}")
            print("[skill] 提示：任务描述带 'skill:<id>' 即自动路由；"
                  "外部模板注册用 python -m src.skill register")
        return

    if args.playbook is not None:
        db.init_db()
        op = args.playbook
        if op and op[0] == "list":
            pb, proposed = retro.playbook_summary()
            print(f"经验库共 {len(pb['rules'])} 条：")
            for r in pb["rules"]:
                flag = "★" if r["status"] == "accepted" else \
                    ("?" if r["status"] == "proposed" else "·")
                shared = f" ← 共享自 {r['shared_from']['project']}" \
                    if r.get("shared_from") else ""
                print(f"  {flag} [{r['id']}] ({r['status']}) [{r['scope']}] "
                      f"{r['rule']}{shared}")
            if proposed:
                print(f"\n待筛选 {len(proposed)} 条："
                      f"accept/reject 操作（如 --playbook accept {proposed[0]['id']}）")
        elif op and len(op) >= 2 and op[0] in ("accept", "reject"):
            ok = retro.playbook_set_status(op[1], "accepted" if op[0] == "accept"
                                           else "rejected")
            print(f"[playbook] {op[1]} → {'accepted' if op[0] == 'accept' else 'rejected'}"
                  if ok else f"[playbook] 未找到 {op[1]}")
            if ok and op[0] == "accept":
                print("[playbook] 已注入下一轮运行的 agent 上下文包")
        elif op and len(op) >= 3 and op[0] == "add":
            rid = retro.playbook_add(problem=op[1], rule=" ".join(op[2:]))
            print(f"[playbook] 已新增候选 {rid}（--playbook accept {rid} 采纳）")
        elif op and op[0] == "verify":
            # 验证 + 记录历史 + 恶化降级 + 全局熔断
            verifs = retro.playbook_verification(record=True)
            if not verifs:
                print("[playbook] 暂无已采纳经验")
            for v in verifs:
                print(f"  [{v['id']}] {v['verdict']}：{v['rule']}")
                for k, d in v["deltas"].items():
                    print(f"    {k}: {d['from']} → {d['to']}"
                          f" {'↑' if d['good'] else '↓'}")
            archived = retro.auto_archive_degraded()
            if archived:
                print("[降级] 以下经验连续恶化，已自动 archived（移出注入池）：")
                for a in archived:
                    print(f"  {a['id']}（连续 {a['rounds']} 轮，"
                          f"撤销 overrides: {a['reverted']}）")
                print("[降级] 人工纠正：--playbook reactivate <id>")
            retro.record_global()
            fuse = retro.check_fuse()
            if fuse.get("tripped"):
                print(f"[熔断] ⛔ {fuse['reason']}")
                print("[熔断] 自动进化已暂停（新经验不注入）；"
                      "人工评估后 --playbook fuse reset 解除")
        elif op and len(op) >= 2 and op[0] == "reactivate":
            ok = retro.playbook_reactivate(op[1])
            print(f"[playbook] {op[1]} 已重新激活" if ok
                  else f"[playbook] 未找到已降级的 {op[1]}")
        elif op and op[0] == "fuse":
            if len(op) > 1 and op[1] == "reset":
                retro.fuse_reset()
                print("[熔断] 已解除（人工确认），自动进化恢复；全局历史已清零")
            else:
                f = retro.fuse_status()
                if f.get("tripped"):
                    print(f"[熔断] ⛔ 已触发：{f.get('reason')}")
                    print("[熔断] 解除：--playbook fuse reset")
                else:
                    print("[熔断] 未触发（自动进化正常运行）")
        elif op and op[0] == "export":
            # 项目间经验选择性共享：导出指定 R-xxx（默认全部 accepted）
            rids = op[1:] or None
            out, n = retro.export_experiences(rule_ids=rids)
            if out:
                print(f"[导出] {n} 条经验 → {out}")
                print("[导出] 目标项目导入：--playbook import <路径>")
            else:
                print("[导出] 无可导出经验（需 accepted 或指定有效 R-xxx）")
        elif op and len(op) >= 2 and op[0] == "import":
            try:
                ids = retro.import_experiences(op[1])
                print(f"[导入] {len(ids)} 条经验已导入为候选（待筛选）："
                      f"{', '.join(ids)}")
                print(f"[导入] 采纳：--playbook accept {ids[0] if ids else ''}"
                      f"（accept 时自动记录本项目基线）")
            except Exception as e:  # noqa: BLE001
                print(f"[导入] 失败：{e}")
        elif op and op[0] == "attribute":
            # A/B 归因：基于实验轮次 + 配置应用前后对比
            attrs = retro.attribution()
            if not attrs:
                print("[归因] 暂无已采纳经验")
            for a in attrs:
                line = f"  [{a['id']}] {a['rule']}"
                if a.get("exp_verdict"):
                    line += (f" | 实验 {a['experiment_rounds']} 轮 → "
                             f"{a['exp_verdict']} {a.get('exp_distribution', {})}")
                if a.get("apply_verdict"):
                    line += f" | 配置应用后 → {a['apply_verdict']}"
                if not a.get("exp_verdict") and not a.get("apply_verdict"):
                    line += " | 未单独验证（混合轮次无法归因）"
                print(line)
            print("[归因] 提示：用 --experiment R-xxx 单独激活跑一轮，"
                  "即可获得干净归因数据")
        elif op and op[0] == "apply":
            # 经验执行闭环：apply 带 action 的已采纳经验（测试门禁 + 失败回滚）
            rid = op[1] if len(op) > 1 else None
            results = retro.apply_playbook(rule_id=rid)
            print(f"[apply] 执行 {len(results)} 条经验（测试门禁 + 失败自动回滚）:")
            for r in results:
                tag = {"applied": "✓ applied", "skipped": "· skipped",
                       "failed": "✗ failed"}.get(r["status"], r["status"])
                print(f"  [{r['id']}] {tag}：{r['note']}")
            print("[apply] 提示：overrides.yaml 已生效（load_config 自动合并）；"
                  "跑 python -m src.main --demo --fresh --mock 可验证")
        else:
            print("用法：--playbook list | accept R-xxx | reject R-xxx | "
                  "add <问题>|<规则>")
        return

    # ---- 变更单（§8.3）：用户/主 agent 提变更 → Producer 影响评估 → 决策 ----
    if args.change is not None:
        from src import changes
        db.init_db()
        op = args.change
        if op and op[0] == "submit":
            if len(op) < 2 or "|" not in op[1]:
                print("用法：--change submit <标题>|<描述> [--change-by xxx]")
                return
            title, _, desc = op[1].partition("|")
            rid = changes.submit(title.strip(), desc.strip(), args.change_by)
            print(f"[变更单] 已提交 {rid}：{title.strip()}"
                  f"（提出者 {args.change_by}）")
            print("[变更单] 影响评估：--change assess " + rid)
        elif op and op[0] == "list":
            status = None
            if len(op) > 1:
                status = op[1] if op[1] != "all" else None
            rows = changes.list_requests(status=status)
            if not rows:
                print("[变更单] 暂无记录")
            for r in rows:
                flag = {"pending": "⏳", "approved": "✓",
                        "rejected": "✗"}.get(r["status"], "·")
                impact = (json.loads(r["impact"]) if r["impact"] else {})
                n = len(impact.get("affected_tasks", []))
                print(f"  {flag} [{r['id']}] ({r['status']}) {r['title']}"
                      f" —— {r['proposed_by']} · 波及 {n} 任务"
                      + (f" · ${impact.get('cost_usd_used', 0)}" if impact else ""))
        elif op and len(op) >= 2 and op[0] == "assess":
            try:
                imp = changes.assess(op[1])
                print(f"[变更单] {op[1]} 影响评估：")
                print(f"  关键词：{', '.join(imp['keywords'])}")
                print(f"  波及任务 {len(imp['affected_tasks'])} 个："
                      + ", ".join(f"{t}({imp['affected_titles'].get(t, '')})"
                                  for t in imp["affected_tasks"][:8])
                      + ("..." if len(imp["affected_tasks"]) > 8 else ""))
                print(f"  预算：${imp['budget_tokens_total']} token 预算 · "
                      f"已用 ${imp['cost_usd_used']}")
            except KeyError as e:
                print(f"[变更单] {e}")
        elif op and len(op) >= 2 and op[0] == "approve":
            try:
                status, affected = changes.approve(
                    op[1], notes=args.notes, apply=args.apply)
                print(f"[变更单] {op[1]} → approved"
                      + (f"，{len(affected)} 个波及任务已重置 backlog（--apply）"
                         if args.apply and affected else
                         "（提示：--apply 可重置波及任务重新评估）"))
            except KeyError as e:
                print(f"[变更单] {e}")
        elif op and len(op) >= 2 and op[0] == "reject":
            try:
                changes.reject(op[1], notes=args.notes)
                print(f"[变更单] {op[1]} → rejected")
            except KeyError as e:
                print(f"[变更单] {e}")
        else:
            print("用法：--change submit <标题>|<描述> | list | "
                  "assess <CR-xxx> | approve <CR-xxx> [--apply] | "
                  "reject <CR-xxx>")
        return

    if args.cleanup:
        db.init_db()
        res = build.cleanup(keep_builds=args.keep_builds, deep=args.deep)
        print(f"[清理] build 快照删 {len(res['build_snapshots'])} · 平铺残留删 "
              f"{len(res['flat_files'])} · report 日志删 {len(res['reports'])} · "
              f"workspace 清 {len(res['workspace_dirs'])}"
              + (f" · 失败(权限/占用) {len(res['failed'])}" if res["failed"] else ""))
        return

    # ---- 实时直播：循环导出 live_state.js（Ctrl+C 退出，配合 live.html 自动刷新） ----
    if args.live:
        import time
        db.init_db()
        live_path = dashboard.render_live()          # 生成/刷新直播页壳
        state_path = dashboard.export_live()          # 首帧状态
        print(f"[直播] 页面 {live_path}")
        print(f"[直播] 状态 {state_path}（每 {args.live_interval}s 刷新，Ctrl+C 退出）")
        while True:
            time.sleep(args.live_interval)
            dashboard.export_live()
        return

    # ---- 按需汇报：用户下达指令时触发（不跑 demo，直接从现有库汇报） ----
    if args.report and not args.demo:
        agents = _load_main_agents(router, topics_cfg)
        pm = agents.get("main-pm")
        participants = [a for a in agents.values() if a.id != "main-pm"]
        to_id = _load_producer_id() or None
        if pm and participants:
            path = report.run_report(pm, participants, to_id=to_id)
            print(f"[汇报] 各主 agent 进度/阻塞已汇总（用户指令触发）→ {path}")
            # 顺带刷新看板
            dash_path = dashboard.render_dashboard()
            print(f"进度看板已刷新: {dash_path}")
        else:
            print("当前库没有完整主 agent 组织（缺 PM 或参与部门），请先跑一次 --demo")
        return

    print("=" * 64)
    print("Harness · 像素平台跳跃 demo · 策划→程序→美术→QA→PM 完整闭环")
    print(f"LLM 配置: default={cfg['llm']['default']} "
          f"models={list(cfg['llm']['models'].keys())} "
          f"routing={cfg['llm'].get('routing', {})}")
    print("=" * 64)

    # ---- 组织创建（C4：只有 Producer 建主 agent，主 agent 建 subagent） ----
    producer = Producer("producer", router=router, topics_cfg=topics_cfg)
    register_agent(producer)
    producer.log_action("create_agent", "producer")

    design = MainAgent("main-design", "策划", "策划主", router=router,
                       context_pack={}, force_reject=None,
                       topics_cfg=topics_cfg)
    program = MainAgent("main-program", "程序", "程序主", router=router,
                        context_pack={}, topics_cfg=topics_cfg,
                        force_reject="always" if args.escalate else
                        ("once" if args.reject else None))
    art = MainAgent("main-art", "美术", "美术主", router=router,
                    context_pack={}, topics_cfg=topics_cfg)
    qa = MainAgent("main-qa", "QA", "QA 主", router=router, context_pack={},
                   topics_cfg=topics_cfg)
    pm = MainAgent("main-pm", "PM", "PM 主", router=router, context_pack={},
                   topics_cfg=topics_cfg)
    main_agents = [design, program, art, qa, pm]
    for m in main_agents:
        register_agent(m, parent_id="producer")
        producer.log_action("create_agent", m.id)
        m.cc_targets = [x.id for x in main_agents if x.id != m.id]

    sub_design = SubAgent("sub-design", "策划", "策划 sub", "main-design",
                          router=router, topics_cfg=topics_cfg)
    sub_program = SubAgent("sub-program", "程序", "程序 sub", "main-program",
                           router=router, bad_once=args.bad_artifact,
                           topics_cfg=topics_cfg)
    sub_art = SubAgent("sub-art", "美术", "美术 sub", "main-art",
                       router=router, topics_cfg=topics_cfg)
    sub_qa = SubAgent("sub-qa", "QA", "QA sub", "main-qa", router=router,
                      topics_cfg=topics_cfg)
    for s in [sub_design, sub_program, sub_art, sub_qa]:
        register_agent(s, parent_id=s.parent_id)
        # C4 断言：subagent 无 create_agent 能力（其工具列表为空）
        assert not hasattr(s, "create_agent"), "C4 违反：subagent 不得拥有 create_agent"
    # 绑定常驻 sub：临时工回收时未完成任务转交到它（防孤儿任务）
    design.resident_sub_id = sub_design.id
    program.resident_sub_id = sub_program.id
    art.resident_sub_id = sub_art.id
    qa.resident_sub_id = sub_qa.id
    producer.log_action("create_agent", "sub-design")
    producer.log_action("create_agent", "sub-program")
    producer.log_action("create_agent", "sub-art")
    producer.log_action("create_agent", "sub-qa")

    # ---- 启动问卷 → 引擎选型 → 宪章 → 广播（CC 全部主 agent） ----
    # §3.1 外部问卷：--questionnaire-file 覆盖内置问卷（P0/P1 schema，
    # 缺失字段保留内置默认——Producer 兜底）
    ext_q = {}
    if args.questionnaire_file:
        try:
            with open(args.questionnaire_file, encoding="utf-8") as f:
                ext_q = json.load(f) or {}
            if not isinstance(ext_q, dict):
                raise ValueError("问卷必须是 JSON 对象")
            print(f"[问卷] 已加载外部启动问卷: {args.questionnaire_file}")
        except Exception as e:  # noqa: BLE001
            ext_q = {}
            print(f"[问卷] 问卷文件读取失败（使用内置默认）: {e}")
    # 主题：显式 --theme 优先；缺省读问卷显式 theme 字段（客户端问卷面板填写）；
    # 都没有 → 默认 battlefield。不做任何"品类→主题"隐式推导（避免误导）。
    if args.theme is None:
        args.theme = str(ext_q.get("theme") or "battlefield")
        print(f"[主题] 使用问卷指定的主题: {args.theme}")
    # 主题解析：内置 id / 已注册 skill id / 内置主题关键词描述（如"勇者救公主"）
    from src import themes as themes_mod
    t_meta = themes_mod.theme_meta(args.theme)
    if t_meta is None:
        avail = ", ".join(f"{t['id']}（{t['name']}）" for t in themes_mod.list_themes())
        print(f"[主题] ✗ 主题「{args.theme}」没有对应游戏模板。可用主题：{avail}")
        print("[主题] 新主题接入：注册一个外部 skill 模板（python -m src.skill "
              "register --id <name> --template <xxx.html>）后即可选择")
        sys.exit(2)
    print(f"[主题] 游戏模板: {t_meta['name']}（skill: {t_meta['skill']} · "
          f"产物: {t_meta['product']}）")
    is_platformer = t_meta.get("skill") == "platformer"
    questionnaire = {
        "engine": "web", "platform": "web",
        "genre": t_meta.get("genre") or ("platformer" if is_platformer else "shooter"),
        "scope": "demo", "time_budget": "2周",
        "worldview": (t_meta.get("worldview")
                      or ("治愈系像素王国：水管工英雄冒险，顶砖块、吃蘑菇、"
                          "踩敌人、收集金币、到达终点旗帜" if is_platformer
                          else "一战（WWI）欧洲堑壕战场俯视射击：步兵对射、占领旗帜、"
                               "波次进攻")),
        "art_style": "pixel",
        "audio": "asset_library", "target_player": "general", "monetization": "free",
    }
    questionnaire.update({k: v for k, v in ext_q.items()
                          if v is not None and v != ""})
    questionnaire["theme"] = args.theme   # 写回问卷，供调研/宪章消费
    # §3.2 引擎选型：Producer 按问卷推荐引擎（显式指定优先），注入宪章
    engine_id, engine_stack = producer.select_engine(questionnaire)
    questionnaire["engine_selected"] = engine_id
    questionnaire["engine_stack"] = engine_stack
    print(f"[引擎] §3.2 选型: {engine_id} → {engine_stack}")
    for m in main_agents:
        m.context_pack["engine"] = f"{engine_id}（{engine_stack}）"
        m.context_pack["theme"] = args.theme
    charter = producer.generate_charter(questionnaire)
    print(f"\n[宪章] Producer 生成项目宪章（CC 全部主 agent）:\n  {charter[:60]}...")
    for m in main_agents:
        m.context_pack["charter"] = charter
    # token 优化：生成宪章摘要，agent LLM 调用只带摘要（全文保留备用）
    summary = producer.summarize_charter(charter)
    print(f"[token] 宪章摘要 {len(summary)} 字（原文 {len(charter)} 字，"
          f"每次 LLM 调用节省 {(1 - len(summary) / max(1, len(charter))) * 100:.0f}% 上下文）")
    for m in main_agents:
        m.context_pack["charter_summary"] = summary
    for s in [sub_design, sub_program, sub_art, sub_qa]:
        s.context_pack["theme"] = args.theme
        s.context_pack["charter_summary"] = summary
    # ---- 流程第 2 环：调研（Producer 从问卷+宪章产出玩法功能清单）----
    # 功能清单是执行环节（程序 sub 生成游戏）的需求源头：游戏包含哪些系统
    research = producer.research(questionnaire, charter)
    feat_keys = list((research.get("features") or {}).keys())
    print(f"[调研] 玩法功能清单: {research.get('title', '')} · "
          f"{len(feat_keys)} 项系统: {', '.join(feat_keys)}")
    for m in main_agents:
        m.context_pack["research"] = research
    for s in [sub_design, sub_program, sub_art, sub_qa]:
        s.context_pack["research"] = research
    # 自进化：把已采纳的经验注入 agent 上下文包（开局即携带历史经验）
    # --experiment 实验模式：只注入该条经验（A/B 归因用，其余暂停）
    # 全局退化熔断：tripped 时暂停自动注入（人工评估）
    if retro.fuse_status().get("tripped"):
        print("[熔断] 全局退化熔断中，本轮跳过经验注入"
              "（解除：--playbook fuse reset）")
    else:
        playbook = retro.playbook_accepted(only=args.experiment)
        if playbook:
            for m in main_agents:
                m.context_pack["playbook"] = playbook
            tag = f"[A/B 实验 {args.experiment}] " if args.experiment else ""
            print(f"{tag}[playbook] 已注入 {len(playbook)} 条历史经验到 agent 上下文")
    producer.broadcast_charter(main_agents)

    # ---- 任务创建（普通 demo 模式；--bf 走五阶段 campaign，不建默认任务） ----
    if not args.bf:
        if args.milestone:
            build.create_milestone("M1", "原型", "prototype", "验证核心玩法可玩性")
        milestone_id = "M1"
        if args.escalate:
            max_rr = 1
        else:
            max_rr = cfg["limits"]["review_rounds_max"]
        t1 = tickets.create_task(
            "玩家移动与跳跃设计", "定义移动/跳跃数值与手感目标",
            "sub-design", "main-design", "策划", milestone_id=milestone_id,
            dod=["数值表通过 schema 校验", "引用宪章设定"], budget_tokens=30000)
        if args.game:
            t2 = tickets.create_task(
                "HTML5 游戏原型（数值驱动）",
                "读取策划 design.json 数值，生成可玩的像素平台跳跃游戏",
                "sub-program", "main-program", "程序", milestone_id=milestone_id,
                dod=["game.html 生成", "策划数值注入"], depends_on=[t1],
                budget_tokens=50000, max_review_rounds=max_rr)
            t3 = tickets.create_task(
                "主角精灵图与 UI 素材", "按宪章像素风格产出主角精灵图与基础 UI",
                "sub-art", "main-art", "美术", milestone_id=milestone_id,
                dod=["命名规范", "格式合规"], depends_on=[t1],   # 与游戏原型并行
                budget_tokens=20000)
            t4 = tickets.create_task(
                "游戏原型验收", "构建 + 冒烟 + 游戏结构检查（canvas/循环/数值注入）",
                "sub-qa", "main-qa", "QA", milestone_id=milestone_id,
                dod=["冒烟测试通过", "游戏结构检查通过"], depends_on=[t2],
                budget_tokens=30000)
        else:
            t2 = tickets.create_task(
                "移动与跳跃实现", "按设计文档实现 Player 移动/跳跃逻辑",
                "sub-program", "main-program", "程序", milestone_id=milestone_id,
                dod=["构建通过", "单测覆盖 ≥ 80%"], depends_on=[t1],
                budget_tokens=50000, max_review_rounds=max_rr)
            t3 = tickets.create_task(
                "移动与跳跃验收", "构建 + 冒烟 + 参数对照设计文档",
                "sub-qa", "main-qa", "QA", milestone_id=milestone_id,
                dod=["冒烟测试通过", "回归无新增失败"], depends_on=[t2],
                budget_tokens=30000)
            t4 = tickets.create_task(
                "主角精灵图与 UI 素材", "按宪章像素风格产出主角精灵图与基础 UI",
                "sub-art", "main-art", "美术", milestone_id=milestone_id,
                dod=["命名规范", "格式合规"], depends_on=[t1],   # 与程序任务并行（都只依赖设计）
                budget_tokens=20000)

    # ---- 主题化五阶段 campaign（替代普通 demo 流程） ----
    if args.bf:
        tick_total, stage_report = _campaign_bf(producer, main_agents,
                                                [sub_design, sub_program,
                                                 sub_art, sub_qa],
                                                cfg, max_ticks,
                                                theme=args.theme)
        for s in stage_report:
            print(f"  [{s['milestone']}] {s['name']}: 构建{'成功' if s['ok'] else '失败'} "
                  f"({s['files']} 件) · 验收 {s['accept']}")
        print(f"五阶段 campaign 完成，共 {tick_total} 调度轮")
        # 自进化 P0：跑完自动生成指标卡 + 失败模式诊断（不调 LLM，省 token）
        try:
            rpath, rm, rpatterns, _ = retro.run_retro(router=None, auto=True)
            print(f"[复盘] 指标卡已生成: {rpath}（失败模式 {len(rpatterns)} 条）")
        except Exception as e:  # noqa: BLE001
            print(f"[复盘] 指标卡生成失败: {e}")
        # 项目可视化：跑完自动渲染当前项目看板（含阶段进度 + 自进化面板）
        try:
            dash_path = dashboard.render_dashboard()
            print(f"[看板] 项目可视化已生成: {dash_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[看板] 渲染失败: {e}")
    else:
        # ---- 弹性扩容：临时 subagent 批量任务（内容填充期，--temp） ----
        temp_subs = []
        if args.temp:
            temp_subs = _spawn_temp_burst(art, design, cfg)

        # ---- 调度主循环（并行度受依赖 DAG + max_concurrency 约束） ----
        resident_subs = [sub_design, sub_program, sub_art, sub_qa]
        tick = 0
        while tick < max_ticks:
            tick += 1
            all_agents = [producer] + main_agents + resident_subs + temp_subs
            # 可靠性（§5.5）：先确认上轮消息，再重投超时未 ack 的
            for a in all_agents:
                a.ack_all()
            mail.retry_expired(timeout_s=cfg["limits"].get("ack_timeout_s", 30),
                               max_retries=cfg["limits"].get("retry_max", 3))
            snap = [(t["id"], t["status"]) for t in tickets.tasks_by()]
            done = [m.dispatch_ready() for m in main_agents]
            worked = [s.work_once() for s in resident_subs + temp_subs]
            # §11.2 双盲评审：趁任务还在 in_review（review 之后抽不到）
            _cross_review_once(main_agents, producer)
            reviewed = [m.review_submissions() for m in main_agents]
            all_tasks = tickets.tasks_by()
            if all_tasks and all(t["status"] in ("done", "escalated") for t in all_tasks):
                break
            if snap == [(t["id"], t["status"]) for t in tickets.tasks_by()]:
                break   # 无任何状态变化（阻塞/僵局），停止空转
            # C7：依赖环检测（发现环 → 升级 Producer 仲裁）
            _check_task_cycles(producer)
            # §9.2 预算控制：任务级超支告警 + 日级暂停低优先级派发
            _check_budget(main_agents, cfg, producer)

        # ---- 回收临时 subagent（干完归档，审计保留） ----
        if temp_subs:
            for m in main_agents:
                for s in list(m.temp_subs):
                    m.retire_temp_subagent(s)
            print(f"[弹性] 临时 subagent 已回收 {len(temp_subs)} 个（审计归档）")

    # ---- S3：里程碑可玩构建 + 用户验收环 ----
    if args.milestone:
        build_dir, ok, n_files = build.build_milestone("M1")
        print(f"\n[里程碑] M1 构建{'成功' if ok else '失败'}: {build_dir}"
              f"（合入制品 {n_files} 件）")
        if ok:
            # S4：里程碑构建广播（广播主题，CC 全部主 agent）
            producer.send("producer", f"【里程碑】M1 构建完成（{n_files} 件制品）",
                          f"构建目录：{build_dir}\n请各团队准备下一阶段。",
                          cc=[m.id for m in main_agents], topic="里程碑")
            if args.user_reject:
                st = build.user_acceptance("M1", approved=False,
                                           notes="跳跃手感偏硬，重力需下调")
                # 反馈进需求池：新建 backlog 任务
                tickets.create_task(
                    "按用户反馈调整跳跃手感", "重力 900 → 780，跳跃初速微调",
                    "sub-design", "main-design", "策划", milestone_id="M1",
                    priority="P1", dod=["数值表更新且校验通过"],
                    depends_on=None)
                print(f"[用户验收] M1 未通过 → 状态 {st}，反馈已进需求池")
            else:
                st = build.user_acceptance("M1", approved=True,
                                           notes="原型可玩，进入垂直切片")
                print(f"[用户验收] M1 通过 → 状态 {st}")
            # 游戏原型：普通 demo 产物是平台跳跃（game.html），固定命名为 mario.html
            # （--theme 只作用于 --bf 五阶段；普通 demo 与主题解耦，避免命名误导）
            if args.game:
                game_path, theme_file = _copy_playable(build_dir, "mario")
                if game_path:
                    print(f"[游戏] 可玩原型已就绪: {game_path}（{theme_file}）")
                try:
                    _render_game_index()
                except Exception:  # noqa: BLE001
                    pass

    # ---- 按需汇报（demo 模式：闭环后触发） ----
    if args.report:
        path = report.run_report(pm, [design, program, art, qa], to_id="producer")
        print(f"\n[汇报] 各主 agent 进度/阻塞已汇总（用户指令触发）: {path}")

    # ---- 结果汇报 ----
    print("\n" + "=" * 64)
    print("调度完成（%d 轮）" % (tick_total if args.bf else tick))
    for t in tickets.tasks_by():
        rr = f"（打回 {t['review_rounds']} 轮）" if t["review_rounds"] else ""
        print(f"  {t['id']} [{t['dept']}] {t['title']}: {t['status']}{rr}")

    conn = db.connect()
    n_msg = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
    n_cc = conn.execute("SELECT COUNT(*) c FROM audit_log WHERE action='email_cc_archive'").fetchone()["c"]
    n_audit = conn.execute("SELECT COUNT(*) c FROM audit_log").fetchone()["c"]
    n_art = conn.execute("SELECT COUNT(*) c FROM artifacts WHERE status='merged'").fetchone()["c"]
    budget_rows = conn.execute(
        "SELECT model, SUM(tokens_in) ti, SUM(tokens_out) to_, SUM(cost_usd) cu "
        "FROM budget GROUP BY model").fetchall()
    conn.close()
    print(f"\n消息: {n_msg} 封（CC 存档 {n_cc}） · 审计事件: {n_audit} · 合入制品: {n_art}")
    for b in budget_rows:
        print(f"预算[{b['model']}]: 输入 {b['ti']} + 输出 {b['to_']} token · "
              f"金额 ${b['cu']:.4f}")

    # ---- S2：审计日志每日导出 ----
    log_path, n_log = audit.export_logs()
    print(f"审计日志已导出: {log_path}（{n_log} 条）")

    # ---- S4：进度看板 ----
    dash_path = dashboard.render_dashboard()
    print(f"进度看板: {dash_path}")


if __name__ == "__main__":
    main()
