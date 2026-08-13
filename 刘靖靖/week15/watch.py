"""运行监控：检测真实运行的完成/卡住 → 微信（企业微信机器人）通知 → 可选关机。

用法（在 backend/ 目录下）：
    python -m src.watch                     # 监控 + 通知（webhook 从 config.yaml 读）
    python -m src.watch --no-shutdown       # 只通知不关机
    python -m src.watch --stale-min 15      # 卡住阈值（默认 15 分钟）
    python -m src.watch --webhook URL       # 覆盖 config 里的 webhook

检测逻辑（每 60 秒一轮，纯 DB 判定，不依赖进程查询——沙箱/权限环境不可靠）：
- 完成：存在里程碑且全部 status='done'（M1-M5 五阶段跑完）
- 卡住/中断：已有里程碑，但 DB 状态快照（任务+里程碑+审计时间）连续
  N 分钟无任何变化 → 判定卡住（或进程异常退出）
- 启动保护：数据库还没有里程碑时不判定（宪章生成阶段 DB 为空）

动作：
- 通知：POST 企业微信机器人 webhook（text 消息，含里程碑摘要）
- 关机：shutdown /s /t <delay> /c "..."（留窗口，'shutdown /a' 可取消）
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # backend/ 根（src 包所在），任意 cwd 可启动

from src import db   # noqa: E402


def _load_config():
    try:
        import yaml
        p = os.path.join(os.path.dirname(_HERE), "config.yaml")
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def db_snapshot():
    """任务+里程碑状态 + 最新审计时间快照（用于卡住检测）。"""
    conn = db.connect()
    try:
        tasks = [f"{r['id']}={r['status']}" for r in conn.execute(
            "SELECT id,status FROM tasks ORDER BY id")]
        mss = [f"{r['id']}={r['status']}" for r in conn.execute(
            "SELECT id,status FROM milestones ORDER BY id")]
        last_audit = conn.execute(
            "SELECT COALESCE(MAX(ts),'') v FROM audit_log").fetchone()["v"]
        return "|".join(tasks) + "||" + "|".join(mss) + "||" + last_audit
    finally:
        conn.close()


def milestone_summary():
    conn = db.connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT id,name,status FROM milestones ORDER BY id")]
        n_done = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status='done'").fetchone()["c"]
        n_total = conn.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]
        return rows, n_done, n_total
    finally:
        conn.close()


def notify_wecom(webhook, title, content):
    """POST 企业微信机器人 text 消息；无 webhook 时仅打印本地日志。"""
    if not webhook:
        print(f"[notify][本地日志] {title}: {content}", flush=True)
        return False
    body = json.dumps({"msgtype": "text",
                       "text": {"content": f"{title}\n{content}"}},
                      ensure_ascii=False).encode("utf-8")
    try:
        import urllib.request
        req = urllib.request.Request(webhook, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = json.loads(resp.read().decode("utf-8")).get("errcode") == 0
        print(f"[notify] 微信推送{'成功' if ok else '失败'}", flush=True)
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[notify] 微信推送异常: {e}", flush=True)
        return False


def shutdown_with_delay(delay_s, title):
    """Windows 关机，留取消窗口（shutdown /a）。"""
    msg = f"{title} - shutdown in {delay_s}s. Run 'shutdown /a' to cancel."
    try:
        subprocess.run(["shutdown", "/s", "/t", str(delay_s), "/c", msg],
                       timeout=15, check=False)
        print(f"[shutdown] 已安排 {delay_s}s 后关机（取消：shutdown /a）", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[shutdown] 关机失败: {e}", flush=True)
        return False


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Harness 运行监控：完成/卡住 → 微信通知 → 关机")
    ap.add_argument("--webhook", default=None, help="企业微信机器人 webhook（覆盖 config）")
    ap.add_argument("--stale-min", type=int, default=15, help="卡住判定阈值（分钟）")
    ap.add_argument("--no-shutdown", action="store_true", help="只通知不关机")
    ap.add_argument("--interval", type=int, default=60, help="轮询间隔（秒）")
    ap.add_argument("--project", default=None,
                    help="多项目：监控指定项目（与启动时 --project 一致）")
    args = ap.parse_args()

    if args.project:
        db.set_project(args.project)

    cfg = _load_config()
    webhook = args.webhook or (cfg.get("notify") or {}).get("wecom_webhook", "")
    delay = int((cfg.get("notify") or {}).get("shutdown_delay_s", 60))

    if not webhook:
        print("[watch] 警告：未配置 wecom_webhook，通知仅写本地日志", flush=True)
    print(f"[watch] 监控启动 · 卡住阈值 {args.stale_min} 分钟 · 关机延迟 {delay}s · "
          f"关机{'启用' if not args.no_shutdown else '禁用'}", flush=True)

    running = True
    last_snap = None
    stale_since = None
    seen_milestone = False   # 启动保护：DB 无里程碑时不判定（宪章生成阶段）

    while True:
        time.sleep(args.interval)
        snap = db_snapshot()
        rows, n_done, n_total = milestone_summary()

        if not rows:
            print("[watch] 尚无里程碑（宪章生成中），继续等待", flush=True)
            last_snap = snap
            stale_since = None
            continue

        if not seen_milestone:
            seen_milestone = True
            print(f"[watch] 检测到 {len(rows)} 个里程碑，开始监控", flush=True)

        if all(m["status"] == "done" for m in rows):
            lines = "\n".join(f"  {m['id']} {m['name']}: {m['status']}" for m in rows)
            notified = notify_wecom(webhook, "✅ Harness 五阶段运行完成",
                                    f"任务 {n_done}/{n_total} · 里程碑：\n{lines}")
            # 通知未送达（无 webhook/推送失败）则不关机——用户收不到消息不能关电脑
            if not args.no_shutdown and notified:
                shutdown_with_delay(delay, "✅ Harness 完成")
            print("[watch] 五阶段全部完成，监控结束", flush=True)
            return

        if snap == last_snap:
            if stale_since is None:
                stale_since = time.time()
            elif time.time() - stale_since >= args.stale_min * 60:
                lines = "\n".join(f"  {m['id']} {m['name']}: {m['status']}" for m in rows)
                notified = notify_wecom(webhook, "⏳ Harness 疑似卡住/中断",
                                        f"已连续 {args.stale_min} 分钟无任何状态变化（任务 "
                                        f"{n_done}/{n_total} 未全部完成）。\n里程碑：\n{lines}")
                if not args.no_shutdown and notified:
                    shutdown_with_delay(delay, "⚠️ Harness 卡住")
                print("[watch] 判定卡住/中断，监控结束", flush=True)
                return
        else:
            stale_since = None
            last_snap = snap
            print(f"[watch] 状态有变化（{len(snap)} 字符），继续监控", flush=True)


if __name__ == "__main__":
    main()
