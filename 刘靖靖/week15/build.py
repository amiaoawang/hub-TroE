"""里程碑可玩构建（PRD §8.1/8.2）：
收集 milestone 下已合入（merged）的制品 → 组装 build 快照 + MANIFEST → 冒烟。
防膨胀：快照只复制「可玩构建产物」（game.html 等），普通制品记录引用路径，
体积不随构建次数 × 制品总量线性增长。"""
import glob
import json
import os
import shutil
import uuid

from src import db

# 进入快照实体的可玩产物：任意 *.html（其余制品只在 MANIFEST 记引用）。
# 内置主题 game.html/battlefield.html 是子集；自定义主题产物（<skill>.html）同样实体复制。
PLAYABLE = (".html",)


def create_milestone(mid, name, stage, goal):
    conn = db.connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO milestones (id,name,stage,goal,status,planned_at) "
            "VALUES (?,?,?,?,?,?)",
            (mid, name, stage, goal, "pending", db.now()))
        conn.commit()
    finally:
        conn.close()


def build_milestone(mid):
    """收集 milestone 的 merged 制品 → 快照目录：可玩产物复制实体，其余记引用。"""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT a.*, t.id tid FROM artifacts a JOIN tasks t ON a.task_id=t.id "
            "WHERE t.milestone_id=? AND a.status='merged'", (mid,)).fetchall()
        stamp = db.now().replace(":", "").replace(" ", "_")
        build_dir = os.path.join(db.ARTIFACTS_DIR, "build", mid, f"build-{stamp}")
        os.makedirs(build_dir, exist_ok=True)
        manifest = {"milestone": mid, "built_at": db.now(),
                    "playable": [], "references": []}
        for r in rows:
            if not r["path"] or not os.path.exists(r["path"]):
                continue
            files = ([os.path.join(r["path"], n) for n in sorted(os.listdir(r["path"]))]
                     if os.path.isdir(r["path"]) else [r["path"]])
            for src in files:
                name = os.path.basename(src)
                if name.endswith(PLAYABLE[0]):    # 可玩产物：复制实体进快照
                    fname = f"{r['tid']}_{name}"
                    shutil.copy(src, os.path.join(build_dir, fname))
                    manifest["playable"].append({"task": r["tid"], "file": fname})
                else:                     # 普通制品：只记引用（指向 main 权威副本）
                    manifest["references"].append({"task": r["tid"],
                                                    "file": name, "path": src})
        with open(os.path.join(build_dir, "MANIFEST.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        ok = len(manifest["playable"]) + len(manifest["references"]) >= 1
        conn.execute(
            "UPDATE milestones SET status=?, build_path=?, done_at=? WHERE id=?",
            ("built" if ok else "failed", build_dir, db.now(), mid))
        conn.commit()
        return build_dir, ok, len(manifest["playable"]) + len(manifest["references"])
    finally:
        conn.close()


def cleanup(keep_builds=3, keep_reports=30, deep=False):
    """保留策略（防膨胀）：
    - build 快照只留最近 keep_builds 个；清理快照根目录平铺残留
    - report-*.md 日志只留最近 keep_reports 份
    - deep：清理已 done 任务的 workspace 暂存目录（合入 main 后不再需要）
    返回统计 dict。
    """
    removed = {"build_snapshots": [], "flat_files": [], "reports": [],
               "workspace_dirs": [], "failed": []}
    # 1) build 快照 + 根目录平铺残留
    build_root = os.path.join(db.ARTIFACTS_DIR, "build")
    if os.path.isdir(build_root):
        for mid in os.listdir(build_root):
            base = os.path.join(build_root, mid)
            if not os.path.isdir(base):
                continue
            snaps = sorted(d for d in os.listdir(base) if d.startswith("build-"))
            for old in snaps[:-keep_builds] if keep_builds > 0 else snaps:
                p = os.path.join(base, old)
                try:
                    shutil.rmtree(p)
                    removed["build_snapshots"].append(p)
                except OSError:
                    removed["failed"].append(p)
            for f in os.listdir(base):      # 非快照的平铺残留
                if not f.startswith("build-"):
                    p = os.path.join(base, f)
                    try:
                        os.remove(p)
                        removed["flat_files"].append(p)
                    except OSError:
                        removed["failed"].append(p)
    # 2) report 日志保留
    reports = sorted(glob.glob(os.path.join(db.LOGS_DIR, "report-*.md")))
    for old in reports[:-keep_reports] if keep_reports > 0 else reports:
        try:
            os.remove(old)
            removed["reports"].append(old)
        except OSError:
            removed["failed"].append(old)
    # 3) workspace 已合入任务暂存（deep）
    if deep:
        conn = db.connect()
        try:
            rows = conn.execute(
                "SELECT agent_id, task_id FROM artifacts WHERE status='merged'"
            ).fetchall()
        finally:
            conn.close()
        seen = set()
        for r in rows:
            key = (r["agent_id"], r["task_id"])
            if key in seen:
                continue
            seen.add(key)
            d = os.path.join(db.ARTIFACTS_DIR, "workspace",
                             r["agent_id"], r["task_id"])
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                    removed["workspace_dirs"].append(d)
                except OSError:
                    removed["failed"].append(d)
    return removed


def user_acceptance(mid, approved, notes=""):
    """用户验收（人在环上）：通过且任务全完成 → done；
    通过但任务未全完成 → built（构建成功未收官，续跑不跳过）；
    不通过 → 状态回 review，反馈进 feedback 表。"""
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO feedback (id,milestone_id,source,rating,notes,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"FB-{uuid.uuid4().hex[:8].upper()}", mid, "user",
             1 if approved else 0, notes, db.now()))
        if approved:
            n_total = conn.execute(
                "SELECT COUNT(*) c FROM tasks WHERE milestone_id=?",
                (mid,)).fetchone()["c"]
            n_done = conn.execute(
                "SELECT COUNT(*) c FROM tasks WHERE milestone_id=? AND status='done'",
                (mid,)).fetchone()["c"]
            if n_total and n_done == n_total:
                conn.execute(
                    "UPDATE milestones SET status='done', done_at=? WHERE id=?",
                    (db.now(), mid))
                status = "done"
            else:
                # 部分完成：构建成功但任务未完，标 built（续跑检查 done 才跳过）
                conn.execute(
                    "UPDATE milestones SET status='built', done_at=? WHERE id=?",
                    (db.now(), mid))
                status = "built"
        else:
            conn.execute("UPDATE milestones SET status='review' WHERE id=?", (mid,))
            status = "review"
        conn.commit()
        return status
    finally:
        conn.close()


def milestone_status(mid):
    conn = db.connect()
    try:
        return conn.execute("SELECT * FROM milestones WHERE id=?", (mid,)).fetchone()
    finally:
        conn.close()
