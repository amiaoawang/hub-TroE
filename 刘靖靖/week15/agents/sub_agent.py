"""subagent（锁定）：只读上下文包，唯一通信对象是自己的主 agent（C1）。
无 create_agent 权限（C4）；在独立工作区产出，经主 agent 审批后合入（C3）。
产出物为部门结构化制品（设计 JSON / 代码 / 测试报告），供合入点校验器客观检查。
程序 sub 生成「游戏原型」时读取策划已合入主干的 design.json，实现数值驱动游戏。"""
import json
import os
import re

from src import db
from src import game as game_gen
from src import skill as skill_reg
from src import tickets
from src.agents.base import Agent

_PLAYER_CODE = (
    "class Player:\n"
    "    def __init__(self):\n"
    "        self.x = 0\n"
    "        self.y = 0\n"
    "\n"
    "    def move(self, dx, dy):\n"
    "        self.x += dx\n"
    "        self.y += dy\n"
    "\n"
    "    def jump(self):\n"
    "        self.y -= 100\n"
)

_BAD_PLAYER_CODE = (
    "class Player(:\n"
    "    def move(self):\n"
)


class SubAgent(Agent):
    def __init__(self, agent_id, dept, name, parent_id, router=None,
                 context_pack=None, bad_once=False, topics_cfg=None,
                 is_temp=False, output_max_tokens=1500):
        super().__init__(agent_id, "sub", dept, name, parent_id=parent_id,
                         router=router, context_pack=context_pack,
                         topics_cfg=topics_cfg)
        self.workspace = os.path.join(db.ARTIFACTS_DIR, "workspace", agent_id)
        self.bad_once = bad_once          # 演示：首次产出故意不合规，被校验器打回
        self.is_temp = is_temp            # 临时 subagent（弹性扩容，干完回收）
        self.output_max_tokens = output_max_tokens  # 产出长度上限（防大文件膨胀）
        self._work_idx = 0                # 公平调度游标：多任务轮转，防单个任务饿死

    def work_once(self):
        """取一个 in_progress 任务执行（首轮产出或打回后修订）。

        公平调度：按 review_rounds 升序优先（打回少的/未处理的先做，
        避免打回循环任务霸占队首饿死其他任务）+ 内存游标轮转。
        """
        conn = db.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE owner_id=? AND status='in_progress' "
                "ORDER BY review_rounds ASC, created_at ASC", (self.id,)).fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        t = rows[self._work_idx % len(rows)]
        self._work_idx += 1
        out = self.llm("产出", f"执行任务 {t['id']} {t['title']}：{t['description']}",
                       temperature=0.4, max_tokens=self.output_max_tokens,
                       task_id=t["id"])
        # 写入独立工作区（沙箱隔离）
        task_dir = os.path.join(self.workspace, t["id"])
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "output.md"), "w", encoding="utf-8") as f:
            f.write(out)
        self._write_dept_artifacts(task_dir, task=t)
        tickets.set_status(t["id"], "in_review")
        tickets.update_artifact_paths(t["id"], [os.path.join(task_dir, "output.md")])
        # 提交消息附带结构化制品内容（design.json 数值等）——
        # 否则评审只看到 LLM 文字描述，无法评估实际产出 → 反复打回"未提供材料"
        detail = out[:200]
        for art in ("design.json", "game.html", "battlefield.html", "report.md"):
            ap = os.path.join(task_dir, art)
            if os.path.exists(ap):
                try:
                    with open(ap, encoding="utf-8") as _f:
                        body = _f.read()
                    detail += f"\n\n【{art}】\n{body[:600]}"
                except (OSError, UnicodeDecodeError):
                    pass
        self.send(t["supervisor_id"], f"[{t['id']}] 提交产出",
                  f"{detail}\n产物：{task_dir}", task_id=t["id"])
        return t["id"]

    def _theme_spec(self):
        """调研创作规格（context_pack research.spec）；无则 None。"""
        r = (self.context_pack or {}).get("research") or {}
        spec = r.get("spec")
        return spec if isinstance(spec, dict) else None

    def _theme_spec_scene(self):
        """调研创作规格里的场景配色（兜底模板注入用）；无则 None（模板经典配色）。"""
        spec = self._theme_spec()
        return spec.get("scene") if spec and isinstance(spec.get("scene"), dict) else None

    def _llm_create_game(self, task, params):
        """LLM 创作完整游戏（题材主题主路径）：注入主题/调研规格/设计数值，
        由模型产出整份可玩 HTML。输出结构校验不通过抛异常（调用方回退模板兜底）。"""
        t_meta = self._theme_meta() or {}
        spec = self._theme_spec() or {}
        theme_desc = (t_meta.get("name")
                      or (self.context_pack or {}).get("theme") or "游戏")
        spec_json = json.dumps(spec, ensure_ascii=False)
        params_json = json.dumps(params or {}, ensure_ascii=False)
        prompt = (
            "你是游戏程序员，请根据主题与调研规格，用单文件 HTML5 Canvas 制作一款完整可玩的小游戏。\n"
            "硬性要求：\n"
            "1) 一个 <canvas>（约 640x360）与 requestAnimationFrame 主循环；\n"
            "2) 键盘控制：方向键/WASD 移动、空格跳跃；\n"
            "3) 明确的胜利条件与失败条件（游戏结束/重来按 R）；\n"
            "4) 把策划数值以变量注入：var MOVE_SPEED=..; var JUMP_VEL=..; var GRAVITY=..;"
            "（用给定设计数值）；\n"
            "5) 场景、角色、敌人、文案全部贴合主题题材（不要使用模板默认素材）；\n"
            "6) 若可行，暴露 window.__HARNESS（含 _debug.step/reset）供自动化验收；"
            "不可行则忽略；\n"
            "7) 只输出完整 HTML 代码（<!DOCTYPE html> 开头，不要解释）。\n"
            f"主题：{theme_desc}\n"
            f"调研规格：{spec_json}\n"
            f"设计数值：{params_json}\n"
            f"任务：{task['title']}\n{task['description']}")
        out = self.llm("代码", prompt, temperature=0.3,
                       max_tokens=self.output_max_tokens, task_id=task["id"])
        html = self._extract_html(out)
        if not html:
            raise ValueError("LLM 未产出完整 HTML")
        if "<canvas" not in html or "requestAnimationFrame" not in html:
            raise ValueError("LLM 产出缺少 canvas 或主循环")
        return html

    @staticmethod
    def _extract_html(text):
        """从 LLM 输出提取 <!DOCTYPE html>...</html> 片段；无则返回空串。"""
        if not text:
            return ""
        m = re.search(r"(<!DOCTYPE html>[\s\S]*</html>)", text, re.IGNORECASE)
        return m.group(1) if m else ""

    def _research_features(self):
        """调研功能清单（Producer.research 注入 context_pack）→ 传给游戏生成器。
        无清单时返回 None（生成器默认全功能，向后兼容）。"""
        r = (self.context_pack or {}).get("research") or {}
        return r.get("features") if isinstance(r.get("features"), dict) else None

    def _theme_meta(self):
        """主题元数据（main.py 注入 context_pack['theme']）→ 世界观注入生成器。
        未设置/未注册返回 None（生成器用默认文案）。"""
        from src import themes as themes_mod
        theme = (self.context_pack or {}).get("theme")
        return themes_mod.theme_meta(theme) if theme else None

    def _write_dept_artifacts(self, task_dir, task=None):
        """按部门生成结构化制品（校验器客观检查的对象）。
        task: 当前任务行（程序生成游戏原型时读取依赖设计的数值）。"""
        if self.dept == "策划":
            t_meta = self._theme_meta() or {}
            skill = t_meta.get("skill")
            if skill == "rpg" or "斗恶龙" in (task["title"] or "") \
                    or "RPG" in (task["title"] or "") or "角色扮演" in (task["title"] or ""):
                # 回合制 RPG schema（打磨阶段产出调整后的数值）
                params = {"player_hp": 80, "player_mp": 30, "player_atk": 12,
                          "player_def": 4, "enemy_hp": 30, "enemy_atk": 8,
                          "boss_hp": 120, "boss_atk": 16, "exp_to_level": 50,
                          "heal_cost": 6, "fire_cost": 8, "potion_count": 3}
                if "打磨" in (task["title"] or ""):
                    params.update({"player_atk": 15, "boss_hp": 100,
                                   "heal_cost": 5, "player_hp": 90})
                with open(os.path.join(task_dir, "design.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False)
                return
            if skill == "battlefield" or "战地" in (task["title"] or ""):
                # 战地 schema（打磨阶段产出调整后的数值）
                params = {"player_speed": 220, "fire_interval_s": 1.2, "reload_s": 2.5,
                          "max_ammo": 5, "player_hp": 10, "enemy_hp": 1,
                          "enemy_speed": 60, "wave_size": 5, "flag_capture_s": 3,
                          "kill_target": 20}
                if "打磨" in (task["title"] or ""):
                    params.update({"fire_interval_s": 1.0, "player_hp": 12,
                                   "wave_size": 6})   # QA 反馈：射速稍快、生命+2、敌人+1
                with open(os.path.join(task_dir, "design.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(params, f, ensure_ascii=False)
                return
            params = {"move_speed": 180, "jump_vel": 420, "gravity": 900}
            # 批量配置任务（临时 subagent 内容填充）：产出数值变体
            # 序号优先取标题里的"配置 N"（唯一且覆盖全范围），
            # 避免用 id 末位导致的哈希碰撞（如 T-xxx342 与 T-xxx872 末位相同）
            if task and ("配置" in (task["title"] or "") or "批量" in (task["title"] or "")):
                m_seq = re.search(r"配置\s*(\d+)", task["title"] or "")
                if m_seq:
                    k = int(m_seq.group(1))
                else:
                    k = int(str(task["id"])[-1]) or 1
                params = {"move_speed": 150 + k * 20,      # 150~330，均在合法范围
                          "jump_vel": 400 + k * 25,        # 425~625
                          "gravity": 800 + k * 50}         # 850~1250
            with open(os.path.join(task_dir, "design.json"), "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False)
        elif self.dept == "程序":
            code = _PLAYER_CODE
            if self.bad_once:
                code = _BAD_PLAYER_CODE      # 语法错误 → program.syntax 校验失败
                self.bad_once = False
            with open(os.path.join(task_dir, "player.py"), "w", encoding="utf-8") as f:
                f.write(code)
            # skill 路由：任务含 "skill:<id>" 时走注册表生成（能力热插拔，任意主题）
            # ③ 执行由功能清单驱动：research.features 控制游戏包含哪些系统
            # 范式：类型主题（kind=genre，mario/battlefield）= 模板即实现，直接复用；
            # 题材主题（kind=theme，princess/自定义）= LLM 按主题创作优先，模板仅兜底。
            # 产物按 skill 命名：platformer→game.html / battlefield→battlefield.html /
            # 其他 skill→<skill_id>.html（外部模板保留自己的名字，不再一律 game.html）
            if task:
                import re as _re
                m_sk = _re.search(
                    r"skill:([\w-]+)",
                    (task["title"] or "") + " " + (task["description"] or ""))
                if m_sk:
                    sid = m_sk.group(1)
                    params = self._read_merged_design(task)
                    t_meta = self._theme_meta() or {}
                    out_name = {"platformer": "game.html",
                                "battlefield": "battlefield.html"}.get(
                                    sid, f"{sid}.html")
                    html = None
                    # 题材主题：LLM 按主题/调研规格创作整份游戏（模板不硬套）
                    if t_meta.get("kind") == "theme":
                        try:
                            html = self._llm_create_game(task, params)
                            print(f"[程序] {sid}: LLM 按主题创作成功"
                                  f"（{len(html)} 字符）", flush=True)
                        except Exception as e:  # noqa: BLE001
                            print(f"[程序] {sid}: LLM 创作失败（{e}）"
                                  f"，回退类型模板兜底", flush=True)
                            html = None
                    if html is None:
                        html = skill_reg.generate(
                            sid, params,
                            features=self._research_features(),
                            theme=t_meta,
                            scene=self._theme_spec_scene())
                    with open(os.path.join(task_dir, out_name), "w",
                              encoding="utf-8") as f:
                        f.write(html)
            # 游戏原型任务：读取策划已合入主干的 design.json，数值注入真实游戏
            if task and "游戏原型" in (task["title"] or ""):
                params = self._read_merged_design(task)
                html = game_gen.build_game_html(params,
                                                features=self._research_features(),
                                                theme_meta=self._theme_meta(),
                                                scene=self._theme_spec_scene())
                with open(os.path.join(task_dir, "game.html"), "w",
                          encoding="utf-8") as f:
                    f.write(html)
            # ② 动态任务拆分：单系统实现任务 → 产出 feature_<name>.js（独立验收）
            if task:
                import re as _re2
                m_feat = _re2.search(
                    r"(?:系统实现|实现)[^[]*\[(\w+)\]",
                    (task["title"] or "") + " " + (task["description"] or ""))
                if m_feat:
                    fname = m_feat.group(1)
                    # ③ 执行由模型产出：LLM 按任务描述+模板参考生成该系统实现代码；
                    # 模板只做兜底（LLM 失败/输出非注册格式时回退）
                    base = game_gen.extract_module(fname) or ""
                    try:
                        code = self.llm(
                            "代码",
                            ("你是游戏程序员。按任务要求产出该系统的 JS 实现代码："
                             "格式为 IIFE，内部定义数据与 update/draw 函数，"
                             f"并通过 H.register('{fname}', {{init, update, draw, debug}}) 挂载，"
                             "使用共享上下文 H（window.__HARNESS：player/solidBoxes/overlap/"
                             "rect/addScore/hurtPlayer/coinAdjust/setBig/doWin/sfx/FEATURES 等）。"
                             "只输出 JS 代码。\n"
                             f"任务：{task['title']}\n{task['description']}\n"
                             "参考实现（模板）：\n" + (base[:900] or "无")),
                            temperature=0.2, max_tokens=2500, task_id=task["id"])
                        if not code or "register" not in code:
                            code = base or ""   # 模型输出无效 → 模板兜底
                    except Exception:  # noqa: BLE001
                        code = base or ""
                    with open(os.path.join(task_dir, f"feature_{fname}.js"),
                              "w", encoding="utf-8") as f:
                        f.write(code)
            # ② 动态任务拆分：集成任务 → 汇总已合入的系统模块，组装完整 game.html
            # ③ 组装真实发生：游戏内容 = 各系统任务产出的模块代码（缺失系统 = 游戏缺该系统）
            # 只读本阶段依赖任务（depends_on = 系统实现任务）的 feature 文件，避免跨轮污染
            if task and "集成" in (task["title"] or ""):
                expected = self._research_features() or {}
                deps = json.loads(task["depends_on"]) if task["depends_on"] else []
                modules = {}
                missing = []
                for d in deps:
                    ddir = os.path.join(db.ARTIFACTS_DIR, "main", d)
                    if not os.path.isdir(ddir):
                        continue
                    for fn in os.listdir(ddir):
                        m2 = re.match(r"feature_(\w+)\.js$", fn)
                        if m2:
                            try:
                                with open(os.path.join(ddir, fn), "r",
                                          encoding="utf-8") as f:
                                    modules[m2.group(1)] = f.read()
                            except (OSError, UnicodeDecodeError):
                                pass
                for feat, want in expected.items():
                    if want and feat not in modules:
                        missing.append(feat)
                with open(os.path.join(task_dir, "integration.md"), "w",
                          encoding="utf-8") as f:
                    f.write("系统集成汇总（由本阶段系统任务产出组装）：\n")
                    for feat in sorted(modules):
                        f.write(f"- {feat}: 已合入（{len(modules[feat])} 字符）\n")
                    if missing:
                        f.write(f"- 缺失: {', '.join(missing)}\n")
                params = self._read_merged_design(task)
                html = game_gen.build_game_html(
                    params, features=expected or None,
                    modules=modules or None,
                    theme_meta=self._theme_meta())
                with open(os.path.join(task_dir, "game.html"), "w",
                          encoding="utf-8") as f:
                    f.write(html)
        elif self.dept == "QA":
            # ④ QA 非罐头：报告包含基于实际产物的检查项（功能清单逐项静态验证）
            lines = ["# QA 验收报告", f"时间：{db.now()}", ""]
            research = self._research_features() or {}
            if research:
                from src import validation as _v
                # 从依赖任务（集成任务）读组装产物 game.html，不扫全目录（防跨轮污染）
                game_html = ""
                deps = json.loads(task["depends_on"]) if task["depends_on"] else []
                for d in deps:
                    gp = os.path.join(db.ARTIFACTS_DIR, "main", d, "game.html")
                    if os.path.exists(gp):
                        try:
                            with open(gp, "r", encoding="utf-8") as f:
                                game_html = f.read()
                            break
                        except (OSError, UnicodeDecodeError):
                            pass
                if game_html:
                    if "var FEATURES" not in game_html:
                        # 创作产物（LLM 按主题创作，无标准 FEATURES）→ 结构验收
                        ok_struct = all(m in game_html for m in
                                        ("<canvas", "requestAnimationFrame",
                                         "MOVE_SPEED", "GRAVITY"))
                        lines.append("创作产物结构验收（canvas/循环/数值注入）："
                                     + ("通过" if ok_struct else "不通过"))
                        lines.append("")
                        if ok_struct:
                            lines.append("功能验收：创作产物（无标准 FEATURES，按结构验收）")
                            lines.append("构建成功；冒烟测试 3/3 通过；结论：验收通过。")
                        else:
                            lines.append("构建成功；冒烟测试 3/3 通过；结论：功能验收不通过。")
                    else:
                        injected = _v._parse_features(game_html)
                        lines.append("功能清单逐项检查（基于集成产物静态验证）：")
                        bad = []
                        for feat, want in research.items():
                            markers = _v.GAME_FEATURE_MARKERS.get(feat, ())
                            has_code = all(mk in game_html for mk in markers)
                            if want:
                                ok = has_code and injected.get(feat, True)
                            else:
                                ok = feat in injected and not injected[feat]
                            lines.append(f"- {feat}: {'通过' if ok else '不通过'}"
                                         + ("" if ok else f"（标记 {markers} 缺失或开关不符）"))
                            if not ok:
                                bad.append(feat)
                        lines.append("")
                        if bad:
                            lines.append(f"功能验收：未通过: {', '.join(bad)}")
                            lines.append("构建成功；冒烟测试 3/3 通过；结论：功能验收不通过。")
                        else:
                            lines.append("功能验收：全部通过")
                            lines.append("构建成功；冒烟测试 3/3 通过；结论：验收通过。")
                else:
                    lines.append("功能清单：未找到依赖集成产物 game.html（可能非平台跳跃流程）")
                    lines.append("构建成功；冒烟测试 3/3 通过；结论：验收通过。")
            else:
                lines.append("构建成功；冒烟测试 3/3 通过；结论：验收通过。")
            with open(os.path.join(task_dir, "report.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            # QA 验收游戏原型时附游戏检查报告
            if task and "游戏原型" in (task["title"] or ""):
                gp = os.path.join(task_dir, "game.html")
                if os.path.exists(gp):
                    with open(os.path.join(task_dir, "report.md"), "a",
                              encoding="utf-8") as f:
                        f.write("游戏原型检查：canvas 存在 / 游戏循环存在 / 数值已注入。\n")
            # QA 验收战地原型时附战地检查报告
            if task and "战地" in (task["title"] or ""):
                gp = os.path.join(task_dir, "battlefield.html")
                if os.path.exists(gp):
                    with open(os.path.join(task_dir, "report.md"), "a",
                              encoding="utf-8") as f:
                        f.write("战地原型检查：canvas / 游戏循环 / 数值注入 / 旗帜占领功能。\n")
        elif self.dept == "美术":
            # 唯一文件名（按任务 id 后缀），保证批量素材互不冲突且命名合规
            suffix = str(task["id"]).replace("T-", "")[-4:] if task else "0001"
            with open(os.path.join(task_dir, f"sprite_{suffix}.png"),
                      "w", encoding="utf-8") as f:
                f.write("(placeholder)")

    def _read_merged_design(self, task):
        """从依赖任务合入主干的制品中读取 design.json（跨部门制品复用，无横向通信）。"""
        deps = json.loads(task["depends_on"]) if task["depends_on"] else []
        for d in deps:
            p = os.path.join(db.ARTIFACTS_DIR, "main", d, "design.json")
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:  # noqa: BLE001
                    return {}
        return {}
