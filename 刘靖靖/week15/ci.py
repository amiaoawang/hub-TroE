"""CI 门禁（自进化护栏）：统一检查入口——语法检查 + 全量测试 + 汇总报告。

进化变更（playbook apply / 代码 / 配置）必须通过 CI 才能合入：
    python -m src.main --ci            # 全量（10 套测试 + 语法）
    python -m src.main --ci --quick    # 核心套件（s1-s4）
失败时 exit code 非 0；报告写 logs/ci-<ts>.md。
"""
import ast
import glob
import os
import subprocess
import sys
import time

from src import db

SRC_DIR = os.path.join(db.BASE, "src")
TESTS_DIR = os.path.abspath(os.path.join(db.BASE, "..", "tests"))

ALL_TESTS = ["test_s1", "test_s2", "test_s3", "test_s4",
             "test_game", "test_temp", "test_bf",
             "test_retro", "test_skill", "test_token_opt",
             "test_cycle", "test_llm_robust", "test_budget",
             "test_change", "test_cross_review", "test_theme",
             "test_deep_loop", "test_dynamic_tasks", "test_feature_gate",
             "test_research", "test_project", "test_watchdog",
             "test_ci"]
QUICK_TESTS = ["test_s1", "test_s2", "test_s3", "test_s4"]


def syntax_check():
    """语法检查 backend/src/*.py。返回 [{file, ok, detail}]。"""
    results = []
    for p in sorted(glob.glob(os.path.join(SRC_DIR, "*.py"))):
        name = os.path.basename(p)
        try:
            with open(p, encoding="utf-8") as f:
                ast.parse(f.read())
            results.append({"file": name, "ok": True, "detail": ""})
        except SyntaxError as e:
            results.append({"file": name, "ok": False,
                            "detail": f"语法错误: {e}"})
    return results


def run_tests(test_names=None, runner=None):
    """跑指定测试套件，返回 [{name, ok, seconds, detail}]。runner 可注入（测试用）。"""
    test_names = test_names or ALL_TESTS
    results = []
    for name in test_names:
        t0 = time.time()
        if runner:
            ok, detail = runner(name)
        else:
            p = os.path.join(TESTS_DIR, f"{name}.py")
            if not os.path.exists(p):
                results.append({"name": name, "ok": False,
                                "seconds": 0, "detail": "文件不存在"})
                continue
            try:
                r = subprocess.run([sys.executable, p], capture_output=True,
                                   text=True, timeout=900)
                ok = r.returncode == 0
                detail = "" if ok else (r.stdout or r.stderr)[-300:]
            except subprocess.TimeoutExpired:
                ok, detail = False, "超时"
            except Exception as e:  # noqa: BLE001
                ok, detail = False, str(e)[:200]
        results.append({"name": name, "ok": ok,
                        "seconds": round(time.time() - t0, 1), "detail": detail})
    return results


def run_ci(quick=False, test_runner=None):
    """完整 CI：语法 + 测试。返回 {ok, suites, total_seconds}。"""
    t0 = time.time()
    syntax = syntax_check()
    tests = run_tests(QUICK_TESTS if quick else ALL_TESTS,
                      runner=test_runner)
    ok = all(s["ok"] for s in syntax) and all(t["ok"] for t in tests)
    return {"ok": ok, "syntax": syntax, "tests": tests,
            "total_seconds": round(time.time() - t0, 1)}


def render_ci_md(result):
    lines = [
        "# CI 门禁报告",
        "",
        f"- 结论：**{'✅ 通过' if result['ok'] else '❌ 失败'}**"
        f"（耗时 {result['total_seconds']}s）",
        "",
        "## 语法检查",
        "",
    ]
    for s in result["syntax"]:
        mark = "✅" if s["ok"] else "❌"
        lines.append(f"- {mark} {s['file']}"
                     + (f" — {s['detail']}" if s["detail"] else ""))
    lines += ["", "## 测试套件", ""]
    for t in result["tests"]:
        mark = "✅" if t["ok"] else "❌"
        lines.append(f"- {mark} {t['name']}（{t['seconds']}s）"
                     + (f"\n  {t['detail']}" if t["detail"] else ""))
    return "\n".join(lines) + "\n"


def gate_ci():
    """门禁入口：跑全量 CI（语法 + 11 套测试），返回 (ok, 失败摘要)。
    供 playbook apply 等自动变更使用——进化变更必须过 CI 才合入。"""
    result = run_ci(quick=False)
    if result["ok"]:
        return True, ""
    fails = [s["file"] for s in result["syntax"] if not s["ok"]]
    fails += [f"{t['name']}({t['detail'][:80]})" for t in result["tests"]
              if not t["ok"]]
    return False, "; ".join(fails) or "CI 失败"


def run_ci_cli(quick=False):
    """CLI 入口：跑 CI、写报告、返回 exit code。"""
    result = run_ci(quick=quick)
    md = render_ci_md(result)
    os.makedirs(db.LOGS_DIR, exist_ok=True)
    out = os.path.join(db.LOGS_DIR, f"ci-{db.now().replace(':', '')[:14]}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"CI 报告: {out}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(run_ci_cli(quick="--quick" in sys.argv))
