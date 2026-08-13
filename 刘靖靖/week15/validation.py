"""合入点自动校验器（PRD §11.1）：客观检查先行，失败直接打回，不进入主观评估。

按部门注册：
- 策划：数值表 design.json schema + 数值范围
- 程序：player.py 语法（ast）/ 类与方法存在性（构建+冒烟的最小可执行版）
- 美术：文件命名/扩展名白名单
- QA：测试报告含冒烟通过结论

返回 [] 表示该部门无校验器（跳过客观门禁，直接走主观评估）。
"""
import ast
import json
import os
import re
import shutil
import subprocess

from src import db

# ---- 玩法功能 → game.html 中的实现标记（静态断言"系统确实实现了"）----
# 清单要求开启的功能，产物必须同时满足：注入开关开启 + 实现代码存在。
# 要求关闭的功能，注入开关必须为 false（否则说明没按清单裁剪）。
GAME_FEATURE_MARKERS = {
    "enemy": ("drawEnemy", "stomp"),          # 敌人系统：绘制 + 踩死逻辑
    "brick": ("hitBlock", "drawBlock"),       # 顶砖块：顶击 + 绘制
    "mushroom": ("mushroom", "ENABLE_MUSHROOM"),  # 蘑菇道具：实体 + 开关
    "pipe": ("pipes", "ENABLE_PIPE"),         # 管道：数据 + 开关
    "coin": ("coin", "ENABLE_COIN"),          # 金币：数据 + 开关
    "life": ("lives", "hurtPlayer"),          # 生命系统：生命值 + 受伤
    "flag": ("poleTop", "ENABLE_FLAG"),       # 旗杆过关：旗杆 + 开关
}

# feature_<name>.js（系统级任务产出）：模块格式，用数据名 + FEATURES 开关标记
FEATURE_FILE_MARKERS = {
    "enemy": ("enemies", "FEATURES.enemy"),
    "brick": ("blocks", "FEATURES.brick"),
    "mushroom": ("mushroom",),           # 数据声明即代表实现
    "pipe": ("pipes", "FEATURES.pipe"),
    "coin": ("coins", "FEATURES.coin"),
    "life": ("lives", "FEATURES.life"),
    "flag": ("flag",),                   # 数据声明即代表实现
}


def _expected_features():
    """校验依据：research 表最新玩法功能清单（Producer.research 落库）。
    无记录返回 {}（旧库/未跑调研 → 不做功能断言，向后兼容）。"""
    try:
        conn = db.connect()
        row = conn.execute(
            "SELECT features FROM research WHERE id='latest'").fetchone()
        conn.close()
        if row and row["features"]:
            data = json.loads(row["features"])
            return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    return {}


def _parse_features(html):
    """从 game.html 解析注入的 FEATURES JSON（模板 `var FEATURES = {...};`）。"""
    m = re.search(r"var FEATURES = (\{.*?\});", html)
    if m:
        try:
            data = json.loads(m.group(1))
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _game_feature_checks(html):
    """按 research 功能清单对 game.html 做逐项实现断言（④ 验收门禁）。
    返回 (checks, n_expected)。无清单时返回空（兼容旧库）；
    产物未注入标准 FEATURES（LLM 按主题创作的产物）→ 跳过清单断言（结构验收兜底）。"""
    expected = _expected_features()
    if not expected:
        return [], 0
    if "var FEATURES" not in html:   # 创作产物：无标准开关注入 → 结构验收兜底
        return [Check("game.feature.skip", True,
                      "创作产物：未注入标准 FEATURES，跳过清单断言（结构验收兜底）")], 0
    injected = _parse_features(html)
    checks = []
    n = 0
    for feat, want in expected.items():
        markers = GAME_FEATURE_MARKERS.get(feat)
        if not markers:
            continue
        n += 1
        if want:   # 要求开启：实现代码存在 且 注入开关未显式关闭
            has_code = all(m in html for m in markers)
            turned_on = injected.get(feat, True)   # 未出现在清单 = 默认开启
            ok = has_code and turned_on
            msg = (f"功能[{feat}] 已实现且开启（{markers[0]}/开关=on）" if ok
                   else f"功能[{feat}] 未实现或已关闭：代码标记 "
                        f"{'✓' if has_code else '✗'} · 开关 "
                        f"{'✓' if turned_on else '✗（应为开启）'}")
        else:      # 要求关闭：注入开关必须为 false
            ok = feat in injected and not injected[feat]
            msg = (f"功能[{feat}] 已按要求关闭" if ok
                   else f"功能[{feat}] 应关闭但仍开启（清单要求=false）")
        checks.append(Check(f"game.feature.{feat}", ok, msg))
    return checks, n


def _game_smoke_check(game_html_path):
    """headless 行为冒烟：node 无头驱动游戏逻辑，验证功能真的能玩
    （踩敌消失/吃金币+1/顶砖/蘑菇变大/旗杆胜利，按 FEATURES 条件化）。
    返回 (passed, out)。node 不可用返回 (None, 原因)（不拦截，仅提示）。"""
    smoke = os.path.normpath(os.path.join(db.BASE, "..", "tests", "smoke_mario.js"))
    if not os.path.exists(smoke):
        return None, "冒烟脚本缺失"
    node = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
    try:
        r = subprocess.run([node, smoke, game_html_path],
                           capture_output=True, text=True, timeout=90)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out[-500:]
    except FileNotFoundError:
        return None, "node 不可用"
    except subprocess.TimeoutExpired:
        return False, "冒烟超时（>90s）"


def _game_smoke_checks(workdir):
    """对 workdir 中 game.html 追加行为冒烟断言。
    产物无标准骨架（__HARNESS，LLM 创作产物）→ 跳过行为冒烟（结构验收兜底）。"""
    gg = os.path.join(workdir, "game.html")
    if not os.path.exists(gg):
        return []
    try:
        with open(gg, encoding="utf-8") as f:
            gtxt = f.read()
    except (OSError, UnicodeDecodeError):
        return []
    if "__HARNESS" not in gtxt:
        return [Check("game.smoke", True,
                      "创作产物无标准骨架（__HARNESS），跳过行为冒烟（结构验收）")]
    ok_smoke, out = _game_smoke_check(gg)
    if ok_smoke is None:
        return [Check("game.smoke", True, f"行为冒烟跳过（{out}）")]
    if ok_smoke:
        return [Check("game.smoke", True, "行为冒烟通过（node 无头驱动玩法）")]
    # 冒烟失败：解析出失败的断言行（FAIL: ...）
    fail_lines = [ln.strip() for ln in out.splitlines() if ln.startswith("FAIL")]
    detail = "; ".join(fail_lines[-3:]) if fail_lines else out[-200:]
    return [Check("game.smoke", False, f"行为冒烟失败：{detail}")]


class Check:
    def __init__(self, name, passed, message):
        self.name = name
        self.passed = bool(passed)
        self.message = message

    def to_dict(self):
        return {"name": self.name, "passed": self.passed, "message": self.message}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class BaseValidator:
    dept = None

    def validate(self, workdir):
        raise NotImplementedError


class DesignValidator(BaseValidator):
    """数值表 schema：按键集自动识别游戏类型（平台跳跃 / 战地一 / 回合制 RPG /
    自定义主题），必填键存在且数值在合理区间；自定义主题（generic）只要求合法 JSON。"""
    dept = "策划"
    PLATFORMER = {"move_speed": (50, 500), "jump_vel": (200, 1000),
                  "gravity": (300, 3000)}
    BATTLEFIELD = {"player_speed": (100, 400), "fire_interval_s": (0.5, 3.0),
                   "reload_s": (1.0, 5.0), "max_ammo": (3, 10),
                   "player_hp": (3, 20), "enemy_hp": (1, 3),
                   "enemy_speed": (30, 150), "wave_size": (1, 10),
                   "flag_capture_s": (1.0, 8.0), "kill_target": (5, 50)}
    RPG = {"player_hp": (30, 300), "player_mp": (5, 200),
           "player_atk": (3, 80), "player_def": (0, 40),
           "enemy_hp": (5, 200), "enemy_atk": (1, 80),
           "boss_hp": (30, 500), "boss_atk": (3, 150),
           "exp_to_level": (10, 300), "heal_cost": (1, 50),
           "fire_cost": (1, 80), "potion_count": (1, 20)}

    # 各类型「特征键」（不含重叠键，用于判别类型）：
    # BATTLEFIELD 与 RPG 都可能有 player_hp/enemy_hp，须用特征键区分。
    _BF_KEYS = ("player_speed", "fire_interval_s", "reload_s", "max_ammo",
                "enemy_speed", "wave_size", "flag_capture_s", "kill_target")
    _RPG_KEYS = ("player_mp", "player_atk", "player_def", "enemy_atk",
                 "boss_hp", "boss_atk", "exp_to_level", "heal_cost",
                 "fire_cost", "potion_count")
    _PLAT_KEYS = ("move_speed", "jump_vel", "gravity")

    def validate(self, workdir):
        p = os.path.join(workdir, "design.json")
        if not os.path.exists(p):
            return [Check("design.schema", False, "缺少 design.json 数值表")]
        try:
            data = _read_json(p)
        except Exception as e:  # noqa: BLE001
            return [Check("design.schema", False, f"design.json 解析失败: {e}")]
        if not isinstance(data, dict):
            return [Check("design.schema", False, "design.json 必须是对象")]
        # 按特征键判别类型（避免 player_hp/enemy_hp 在 BATTLEFIELD 与 RPG 间歧义）
        if any(k in data for k in self._BF_KEYS):
            schema = self.BATTLEFIELD
        elif any(k in data for k in self._RPG_KEYS):
            schema = self.RPG
        elif any(k in data for k in self._PLAT_KEYS):
            schema = self.PLATFORMER
        else:
            return [Check("design.generic", True, "自定义主题数值表（宽松校验）")]
        checks = []
        for key, (lo, hi) in schema.items():
            v = data.get(key)
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi
            msg = f"{key}={v} 应在 [{lo}, {hi}]" if not ok else f"{key}={v} 合规"
            checks.append(Check(f"design.{key}", ok, msg))
        return checks


class ProgramValidator(BaseValidator):
    """构建+冒烟最小版：语法检查 + 类与方法契约。"""
    dept = "程序"

    def validate(self, workdir):
        p = os.path.join(workdir, "player.py")
        if not os.path.exists(p):
            return [Check("program.build", False, "缺少 player.py（构建产物）")]
        try:
            with open(p, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except SyntaxError as e:
            return [Check("program.syntax", False, f"语法错误: {e}")]
        checks = [Check("program.syntax", True, "语法检查通过")]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        methods = {m.name for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef)
                   for m in n.body if isinstance(m, ast.FunctionDef)}
        checks.append(Check("program.class", "Player" in classes, "含 Player 类"))
        checks.append(Check("program.methods", {"move", "jump"} <= methods,
                            "含 move/jump 方法"))
        # 战地原型：battlefield.html 结构 + 数值注入 + 功能标志
        gp = os.path.join(workdir, "battlefield.html")
        if os.path.exists(gp):
            with open(gp, encoding="utf-8") as f:
                g = f.read()
            checks += [
                Check("bf.canvas", "<canvas" in g, "含 <canvas>"),
                Check("bf.loop", "requestAnimationFrame" in g, "含游戏循环"),
                Check("bf.design", "PLAYER_SPEED" in g and "FLAG_CAPTURE_S" in g
                      and "KILL_TARGET" in g, "策划数值已注入"),
                Check("bf.features", "reload" in g and "ammo" in g
                      and "flags_won" in g, "上弹/弹药/旗帜占领功能存在"),
            ]
        # ④ 平台跳跃原型：game.html 结构 + 数值注入 + 按调研清单逐项验证玩法功能
        gg = os.path.join(workdir, "game.html")
        if os.path.exists(gg):
            with open(gg, encoding="utf-8") as f:
                g2 = f.read()
            checks += [
                Check("game.canvas", "<canvas" in g2, "含 <canvas>"),
                Check("game.loop", "requestAnimationFrame" in g2, "含游戏循环"),
                Check("game.design", "MOVE_SPEED" in g2 and "JUMP_VEL" in g2
                      and "GRAVITY" in g2, "策划数值已注入"),
                Check("game.features",
                      ("FEATURES" in g2 and "ENABLE_ENEMY" in g2)
                      or "FEATURES" not in g2,   # 创作产物无标准开关 → 放行（结构验收兜底）
                      "功能清单开关存在（FEATURES/ENABLE_*）" if "FEATURES" in g2
                      else "创作产物：无 FEATURES 开关（跳过清单断言）"),
            ]
            feat_checks, n_exp = _game_feature_checks(g2)
            if n_exp:
                checks += feat_checks
                if not all(c.passed for c in feat_checks):
                    checks.append(Check(
                        "game.feature.total",
                        False,
                        f"玩法功能验收未过：{sum(1 for c in feat_checks if not c.passed)}/"
                        f"{len(feat_checks)} 项未按调研清单实现"))
            # 行为冒烟：node 无头驱动验证功能真的能玩（拦"代码在但逻辑坏"）
            checks += _game_smoke_checks(workdir)
        # 回合制 RPG 原型：rpg.html 结构（DOM 非 canvas）+ 数值注入 + 回合制战斗
        rp = os.path.join(workdir, "rpg.html")
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                g4 = f.read()
            checks += [
                Check("rpg.structure", 'id="game"' in g4, "含 RPG 容器"),
                Check("rpg.values", "PLAYER_HP" in g4 and "EXP_TO_LEVEL" in g4
                      and "PLAYER_MP" in g4, "RPG 数值已注入"),
                Check("rpg.battle", "startBattle" in g4 and "levelUp" in g4
                      and ("回合" in g4 or "turn" in g4.lower()),
                      "回合制战斗/升级存在"),
            ]
        # 自定义主题产物（<skill_id>.html，非内置名字）：结构校验
        for fn in sorted(os.listdir(workdir)):
            if fn.endswith(".html") and fn not in ("game.html", "battlefield.html",
                                                   "rpg.html"):
                try:
                    with open(os.path.join(workdir, fn), encoding="utf-8") as f:
                        g3 = f.read()
                except (OSError, UnicodeDecodeError):
                    continue
                checks += [
                    Check(f"{fn}.canvas", "<canvas" in g3, "含 <canvas>"),
                    Check(f"{fn}.loop", "requestAnimationFrame" in g3,
                          "含游戏循环"),
                ]
        # ② 系统级任务：feature_<name>.js（动态任务拆分的单系统实现）
        for fn in sorted(os.listdir(workdir)):
            if fn.startswith("feature_") and fn.endswith(".js"):
                fname = fn[len("feature_"):-3]
                with open(os.path.join(workdir, fn), encoding="utf-8") as f:
                    seg = f.read()
                marker = FEATURE_FILE_MARKERS.get(fname)
                ok = bool(seg.strip())
                msg = (f"系统模块 {fname} 已实现（{len(seg)} 字符）" if ok
                       else f"系统模块 {fname} 为空")
                if ok and marker and not all(m in seg for m in marker):
                    ok = False
                    msg = f"系统模块 {fname} 缺少实现标记 {marker[0]}/{marker[1]}"
                checks.append(Check(f"feature.{fname}", ok, msg))
        return checks


class ArtValidator(BaseValidator):
    """命名/格式/风格：小写、无空格、扩展名白名单。"""
    dept = "美术"
    WHITELIST = {".png", ".jpg", ".json", ".md", ".svg", ".mp3"}

    def validate(self, workdir):
        bad = []
        for f in sorted(os.listdir(workdir)):
            name, ext = os.path.splitext(f)
            if ext.lower() not in self.WHITELIST:
                bad.append(f"{f}(扩展名非法)")
            if " " in f or f != f.lower():
                bad.append(f"{f}(命名不规范)")
        return [Check("art.naming", not bad, "；".join(bad) if bad else "命名/格式合规")]


class QAValidator(BaseValidator):
    """测试报告：冒烟 3/3 且结论通过；游戏原型另查 game.html 结构。"""
    dept = "QA"

    def validate(self, workdir):
        p = os.path.join(workdir, "report.md")
        if not os.path.exists(p):
            return [Check("qa.report", False, "缺少 report.md 测试报告")]
        with open(p, encoding="utf-8") as f:
            text = f.read()
        ok = ("3/3" in text) and ("通过" in text or "PASS" in text.upper())
        checks = [Check("qa.smoke", ok,
                        "冒烟 3/3 且结论通过" if ok else "冒烟未全过或结论未通过")]
        gp = os.path.join(workdir, "game.html")
        if os.path.exists(gp):
            with open(gp, encoding="utf-8") as f:
                g = f.read()
            checks += [
                Check("game.canvas", "<canvas" in g, "含 <canvas>"),
                Check("game.loop", "requestAnimationFrame" in g, "含游戏循环"),
                Check("game.design", "MOVE_SPEED" in g and "GRAVITY" in g,
                      "策划数值已注入"),
            ]
        return checks


REGISTRY = {v.dept: v() for v in (DesignValidator, ProgramValidator,
                                  ArtValidator, QAValidator)}


def validate(dept, workdir):
    """对某部门产物目录跑客观校验；无该部门校验器或目录不存在时返回 []。"""
    v = REGISTRY.get(dept)
    if v is None or not workdir or not os.path.isdir(workdir):
        return []
    return v.validate(workdir)
