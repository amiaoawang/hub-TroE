"""LLM 层：多模型配置与路由（PRD §10.1）。

- 默认单模型（default + main 即可用）。
- 支持多模型档位，按角色 routing 分发；失败走 fallback 降级链。
- S1 沙箱默认 provider=mock（离线确定性输出，便于验证闭环）；
  接真实模型改为 provider=openai + 环境变量 API Key。
- 成本计量：每次调用通过 on_budget 回调记账（按实际命中模型单价）。
- token 优化：llm_cache 响应缓存（相同 system+user+模型 命中直接返回，
  不调 API 不计费；打回重评/复盘重跑等高复用场景显著降本）。
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

from src import db

# §7.6 失败矩阵默认值：退避重试序列（1s/5s/30s）+ 熔断（连续 5 次失败暂停 10min）
DEFAULT_BACKOFF_S = [1, 5, 30]
DEFAULT_BREAKER = {"threshold": 5, "cooldown_s": 600}


# ---- 题材主题创作规格（mock 调研确定性产出；真实 LLM 由模型按主题自由生成）----
# 女巫城堡场景：夜空/石砖/金甲勇者/魔物/水晶（模板仅提供类型机制，视觉归属创作层）
_CASTLE_SCENE = {
    "sky": "#2a1a4a", "cloud": "#6b5b8a", "hill": "#3a2a5a", "hill2": "#241640",
    "ground": "#5a5a66", "ground_top": "#7a7a8a", "ground_dot": "#4a4a55",
    "player_hat": "#d0a33c", "player_skin": "#f5c4b3",
    "player_pants": "#4e7fb5", "player_boot": "#8a5a2b",
    "enemy_shell": "#c25b4e", "enemy_body": "#7a2a2a", "enemy_face": "#e8b4a0",
    "block": "#8a7a5a", "block_hi": "#d0a33c",
    "block_used": "#6a5a4a", "block_used_in": "#8a7a5a",
    "pipe": "#5a5a66", "pipe_light": "#7a7a8a",
    "pipe_hl": "#9a9aaa", "pipe_in": "#c8c8d4",
    "coin": "#7fd4c1", "coin_hl": "#e8fff8",
    "mush_cap": "#b06ad0", "mush_stem": "#f0e8ff",
    "mush_dot": "#ffffff", "mush_foot": "#5a3a6a",
    "flag_pole": "#8a7a5a", "flag": "#d0a33c",
}


def _spec_for_theme(tid, tmeta, raw_theme):
    """题材主题的创作规格（mock 确定性产出）：主题化场景/文案，供 LLM 创作游戏。
    真实 LLM 调研时由模型按主题自由生成，本函数仅为 mock 提供确定性结果。"""
    raw = str(raw_theme or "")
    if tid == "princess" or "公主" in raw or "勇者" in raw:
        return {
            "title": "勇者救公主 · 城堡冒险",
            "desc": "勇者穿越女巫城堡：跳跃平台、顶宝箱、击败魔物、收集力量水晶，抵达城堡救出公主",
            "controls": "方向键/WASD 移动 · 空格跳跃 · 顶宝箱(上方向)",
            "scene": dict(_CASTLE_SCENE),
            "win_text": "你救出了公主！",
            "hud_tip": "方向键 / WASD 移动 · 空格跳跃 · 顶宝箱 · 击败魔物 · 抵达城堡救出公主",
            "state_goal": "向城堡进发！击败魔物 / 收集力量水晶",
        }
    return {
        "title": (tmeta or {}).get("name") or "自定义冒险",
        "desc": (tmeta or {}).get("desc", "") or "自定义题材冒险",
        "controls": "方向键/WASD 移动 · 空格跳跃",
        "win_text": "胜利！",
        "hud_tip": "方向键 / WASD 移动 · 空格跳跃",
        "state_goal": "向前进发！",
    }


def _mock_created_game(user_text):
    """mock 的 LLM 创作：从创作请求 prompt 提取 调研规格 + 设计数值，
    按规格生成主题化完整游戏（模板类型机制 + 主题场景/文案）。
    真实 LLM 模式下由模型自由创作整份 HTML，本函数仅为 mock 确定性产物。"""
    import re as _re
    from src import game as _game_gen
    spec, params = {}, {}
    mq = _re.search(r"调研规格[^\n]*?(\{[\s\S]*?\})\n", user_text)
    if mq:
        try:
            spec = json.loads(mq.group(1)) or {}
        except Exception:  # noqa: BLE001
            spec = {}
    mp = _re.search(r"设计数值[^\n]*?(\{[\s\S]*?\})\n", user_text)
    if mp:
        try:
            params = json.loads(mp.group(1)) or {}
        except Exception:  # noqa: BLE001
            params = {}
    theme_meta = {
        "title": spec.get("title") or "冒险游戏",
        "hud_tip": spec.get("hud_tip") or "方向键/WASD 移动 · 空格跳跃",
        "state_goal": spec.get("state_goal") or "出发！",
        "win_text": spec.get("win_text") or "胜利！",
    }
    scene = spec.get("scene") if isinstance(spec.get("scene"), dict) else None
    return _game_gen.build_game_html(params, features=None,
                                     theme_meta=theme_meta, scene=scene)


class LLMError(Exception):
    pass


class CircuitBreaker:
    """模型级熔断器（§7.6）：连续 threshold 次失败 → open（暂停 cooldown_s）；
    冷却结束 half-open 放行一个试探请求，成功即 close（失败则立即重新熔断）。"""

    def __init__(self, threshold=5, cooldown_s=600, clock=time.monotonic):
        self.threshold = max(1, threshold)
        self.cooldown_s = max(0, cooldown_s)
        self._clock = clock
        self._fails = {}          # key -> 连续失败计数
        self._open_until = {}     # key -> 熔断结束时间戳

    def allow(self, key):
        """当前是否允许调用（False = 熔断中）。冷却到期 → half-open 放行试探。"""
        until = self._open_until.get(key)
        if until is None:
            return True
        if self._clock() >= until:
            self._open_until.pop(key, None)   # half-open：放行一个请求
            return True
        return False

    def record_success(self, key):
        self._fails.pop(key, None)
        self._open_until.pop(key, None)       # 成功后彻底关闭

    def record_failure(self, key):
        n = self._fails.get(key, 0) + 1
        self._fails[key] = n
        if n >= self.threshold:
            self._open_until[key] = self._clock() + self.cooldown_s

    def state(self, key):
        """调试/测试：'open' | 'half-open' | 'closed' | 'closed(fails=N)'。"""
        until = self._open_until.get(key)
        if until is not None:
            return "open" if self._clock() < until else "half-open"
        n = self._fails.get(key, 0)
        return "closed" if n == 0 else f"closed(fails={n})"


class Provider:
    name = "base"

    def complete(self, system, user, temperature=0.3, max_tokens=2000):
        raise NotImplementedError


class MockProvider(Provider):
    """确定性离线模型：按角色/部门返回罐头输出，用于沙箱验证。"""

    name = "mock"

    def __init__(self, cfg):
        self.model = cfg.get("model", "mock-1")
        self.cost = cfg.get("cost", {}) or {}
        self.calls = 0                 # 调用计数（缓存测试断言用）

    def complete(self, system, user, temperature=0.3, max_tokens=2000,
                 backoff=None):
        self.calls += 1
        u = user.lower()
        s = (system or "").lower()
        if "看门狗" in s or ("修复决策" in u and "卡住" in u):
            # watchdog 修复决策：返回可解析的 JSON 动作（测试/沙箱用）
            import re
            tid = re.search(r"\bT-\d{4}\b", u)
            aid = re.search(r"sub-\w+", u)
            actions = []
            if "in_progress" in u and tid:
                actions.append({"action": "requeue_task", "task_id": tid.group(0),
                                "reason": "mock 决策：in_progress 卡住，重置重派"})
            if "blocked" in u and aid:
                actions.append({"action": "unblock_agent", "agent_id": aid.group(0),
                                "reason": "mock 决策：解封 blocked agent"})
            if "孤儿" in u and tid:
                sup = re.search(r"supervisor=(\S+)", u)
                actions.append({"action": "rescue_orphan", "task_id": tid.group(0),
                                "agent_id": sup.group(1) if sup else None,
                                "reason": "mock 决策：转交孤儿任务"})
            return json.dumps(actions, ensure_ascii=False) or "[]"
        if "代码" in s or "JS 实现" in u:
            # 代码生成：由模型产出
            # - 原型实现/游戏原型/切片实现/打磨实现（题材主题）→ 整份主题化游戏 HTML
            #   （mock 按调研规格确定性创作；真实 LLM 由模型按主题自由写整份游戏）
            # - 系统实现任务（类型模板拆分）→ 单系统模块实现（模板参考，类型复用）
            import re as _re2
            from src import game as _game_gen
            if ("原型实现" in u or "游戏原型" in u
                    or "切片实现" in u or "打磨实现" in u):
                return _mock_created_game(u)
            m_feat = _re2.search(r"\[(\w+)\]", u)
            seg = _game_gen.extract_module(m_feat.group(1)) if m_feat else None
            if not seg:
                return "// 无法生成实现（任务未指定系统）"
            return seg
        if "调研" in s or "功能清单" in u:
            # 调研：真正消费用户输入——解析问卷，按品类决定功能清单 + 创作规格。
            # 范式：类型主题（mario/battlefield）= 模板即实现，给 7 系统清单（类型复用）；
            # 题材主题（princess/自定义）= 由 LLM 创作，features 为空 + spec 创作规格。
            import re as _re3
            feats = {"enemy": True, "brick": True, "mushroom": True,
                     "pipe": True, "coin": True, "life": True, "flag": True}
            q = {}
            mq = _re3.search(r"启动问卷：(\{[\s\S]*?\})", u)
            if mq:
                try:
                    q = json.loads(mq.group(1)) or {}
                except Exception:  # noqa: BLE001
                    q = {}
            genre = str(q.get("genre", "") or "")
            worldview = str(q.get("worldview", "") or "")
            title_hint = str(q.get("title", "") or "")
            theme = str(q.get("theme") or "")
            # 主题解析（main.py 会写回 questionnaire["theme"]）：
            # 内置 id / 已注册 skill id / 关键词描述（如"勇者救公主"→princess）
            from src import themes as _themes_mod
            _tmeta = _themes_mod.theme_meta(theme) if theme else None
            # 战地（类型主题）：空清单 + 堑壕创作规格（走 bf 模板 = 类型实现）
            if theme == "battlefield":
                return json.dumps({"title": "战地一 · 堑壕突击", "genre": "俯视射击",
                                   "features": {},
                                   "spec": {"desc": "一战堑壕俯视射击：步兵对射、占领旗帜、波次进攻",
                                            "controls": "WASD 移动 · 鼠标瞄准射击",
                                            "win_text": "胜利！占领全部旗帜",
                                            "hud_tip": "向旗帜推进 · 弹药耗尽自动上弹",
                                            "state_goal": "夺取旗帜 / 达成击杀目标"}},
                                  ensure_ascii=False)
            # 题材主题（princess / 自定义 / 描述解析）：空清单 + 主题化创作规格 →
            # M1 回退「原型实现」任务，由 LLM 按 spec 创作游戏（模板仅同类型兜底）
            if _tmeta and _tmeta.get("kind") == "theme":
                spec = _spec_for_theme(_tmeta.get("id") or theme, _tmeta, theme)
                return json.dumps({"title": _tmeta.get("name") or theme,
                                   "genre": genre or "奇幻冒险",
                                   "features": {}, "spec": spec},
                                  ensure_ascii=False)
            # 回合制 RPG（类型主题，skill=rpg，如 dragonquest/勇者斗恶龙）：
            # 空清单 + RPG 创作规格 → M1 回退「原型实现 skill:rpg」任务
            if _tmeta and _tmeta.get("skill") == "rpg":
                return json.dumps(
                    {"title": _tmeta.get("name") or "勇者斗恶龙 · 回合制 RPG",
                     "genre": "rpg", "features": {},
                     "spec": {"desc": "回合制 RPG：地图探索/遇敌/回合制战斗/升级/讨伐魔王",
                              "controls": "方向键/WASD 移动 · 空格攻击",
                              "win_text": "击败魔王！"}},
                    ensure_ascii=False)
            # 无主题字段（旧调用/测试直接调 research）：按品类判断
            if "战地" in worldview or "射击" in genre or "shooter" in genre:
                return json.dumps({"title": "战地一 · 堑壕突击", "genre": "俯视射击",
                                   "features": {},
                                   "spec": {"desc": "堑壕俯视射击", "controls": "WASD+鼠标",
                                            "win_text": "胜利！"}}, ensure_ascii=False)
            is_platformer = (
                "platformer" in genre or "平台跳跃" in genre
                or "马里奥" in worldview or "水管工" in worldview
                or (_tmeta and _tmeta.get("kind") == "genre"
                    and _tmeta.get("skill") == "platformer"))
            if not is_platformer:
                gname = genre or title_hint or "自定义游戏"
                return json.dumps({"title": gname, "genre": genre,
                                   "features": {},
                                   "spec": {"desc": "自定义玩法", "controls": "",
                                            "win_text": "胜利！"}},
                                  ensure_ascii=False)
            # 类型主题 / 品类=平台跳跃：7 系统（类型模板复用）+ 世界观裁剪
            if "没有敌人" in worldview or "无敌人" in worldview or "不战斗" in worldview:
                feats["enemy"] = False
            if "没有砖块" in worldview or "无砖块" in worldview:
                feats["brick"] = False
            if "没有蘑菇" in worldview or "无蘑菇" in worldview:
                feats["mushroom"] = False
            if "没有管道" in worldview or "无管道" in worldview:
                feats["pipe"] = False
            if "没有金币" in worldview or "无金币" in worldview or "不收集" in worldview:
                feats["coin"] = False
            if "没有生命" in worldview or "无生命" in worldview or "无敌" in worldview:
                feats["life"] = False
            if "没有旗杆" in worldview or "无旗杆" in worldview or "无终点" in worldview:
                feats["flag"] = False
            if "马里奥" in worldview or "水管工" in worldview:
                title, genre_name = "超级马里奥 · 像素冒险", "平台跳跃"
            else:
                title, genre_name = "像素平台跳跃", "平台跳跃"
            return json.dumps({"title": title, "genre": genre_name,
                               "features": feats,
                               "spec": {"desc": "平台跳跃：移动/跳跃/收集/到达终点",
                                        "controls": "方向键/WASD 移动 · 空格跳跃",
                                        "win_text": "过关！"}},
                              ensure_ascii=False)
        if "摘要" in u:
            return ("宪章摘要：Web 游戏 demo；玩法/数值由策划 design.json 驱动；"
                    "命名 snake_case。")
        if "宪章" in u or "charter" in u or "问卷" in u:
            # 宪章消费问卷：解析问卷 JSON 提取品类/世界观/主题，生成对应宪章文本
            import re as _re5
            mq2 = _re5.search(r"启动问卷：(\{[\s\S]*?\})", u)
            q2 = {}
            if mq2:
                try:
                    q2 = json.loads(mq2.group(1)) or {}
                except Exception:  # noqa: BLE001
                    q2 = {}
            g2 = str(q2.get("genre", "") or "web 游戏")
            wv = str(q2.get("worldview", "") or "通用设定")[:40]
            th = str(q2.get("theme", "") or "")
            th_txt = f"；主题 = {th}" if th else ""
            eng = str(q2.get("engine_selected", "") or "web")
            return (f"项目宪章 v1：{g2} 游戏；技术栈 = {eng}；"
                    f"世界观：{wv}{th_txt}；规范：命名 snake_case，数值表 JSON。")
        if "复盘" in u or "提炼" in u:
            return ('[{"problem":"真实 LLM 评审措辞多变，含建议类文字即误判打回",'
                    '"rule":"评审判定只认明确否定词（不通过/未通过/拒绝/打回/不合格）",'
                    '"scope":"review","rationale":"mock 复盘"}]')
        if "评审" in s or "评估" in s or "review" in u:
            return "评审结论：通过（DoD 全部满足）。"
        if "设计" in s or "策划" in s or "design" in u:
            return ("设计文档：玩家支持左右移动(方向键/WASD)与跳跃(空格)；移动速度 180px/s；"
                    "跳跃初速 420px/s；重力 900px/s²；数值见 docs/design.json。")
        if "程序" in s or "实现" in s or "code" in u:
            return ("代码实现完成：src/player.py 实现 Player 类（move/jump/update），"
                    "含单元测试 test_player.py，冒烟测试通过。")
        if "qa" in s or "qa" in u or "验收" in u or "测试" in s:
            return ("QA 报告：构建成功；冒烟 3/3 通过；移动/跳跃参数符合设计文档；"
                    "结论：验收通过。")
        return f"[mock:{self.model}] 已处理：{u[:80]}"


class OpenAICompatProvider(Provider):
    """OpenAI 兼容 /chat/completions（DeepSeek / OpenAI 等）。"""

    name = "openai"

    def __init__(self, cfg):
        self.base_url = cfg.get("base_url", "https://api.deepseek.com/v1").rstrip("/")
        self.model = cfg.get("model", "deepseek-chat")
        self.api_key_env = cfg.get("api_key_env", "LLM_API_KEY")
        self.api_key = os.environ.get(self.api_key_env, "")
        self.cost = cfg.get("cost", {})

    def complete(self, system, user, temperature=0.3, max_tokens=2000,
                 backoff=None):
        if not self.api_key:
            raise LLMError(f"缺少 API Key（环境变量 {self.api_key_env}）")
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        # §7.6 退避重试：网络/超时类错误按 backoff 序列重试（1s/5s/30s）；
        # HTTP 错误（认证/限流/4xx/5xx）立即抛出，不重试
        delays = list(backoff or DEFAULT_BACKOFF_S) + [None]
        last_err = None
        for delay in delays:
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")
            except Exception as e:    # noqa: BLE001
                last_err = e
                if delay is not None:
                    time.sleep(delay)
        raise LLMError(str(last_err))


class LLMRouter:
    def __init__(self, config):
        self.config = config
        self.providers = {}
        for mid, mcfg in (config.get("models") or {}).items():
            ptype = mcfg.get("provider", "openai")
            cls = MockProvider if ptype == "mock" else OpenAICompatProvider
            self.providers[mid] = cls(mcfg)
        self.default = config.get("default", "main")
        self.routing = config.get("routing", {})
        self.fallback = config.get("fallback", [])
        # §7.6 退避重试 + 熔断（llm.retry 可配置，默认 1s/5s/30s + 5 次/10min）
        retry = config.get("retry") or {}
        self.backoff = list(retry.get("backoff_s") or DEFAULT_BACKOFF_S)
        br = retry.get("circuit_breaker") or {}
        self.breaker = CircuitBreaker(
            br.get("threshold", DEFAULT_BREAKER["threshold"]),
            br.get("cooldown_s", DEFAULT_BREAKER["cooldown_s"]))

    def resolve(self, agent_role, agent_dept=None):
        """路由（PRD §10.1）：优先 dept 键 → 其次 role 键 → 最后 default。
        路由值必须指向 models 中已定义的模型；否则回退，避免 KeyError。"""
        for key in (agent_dept, agent_role):
            if not key:
                continue
            mid = self.routing.get(key)
            if mid in self.providers:
                return mid
        if self.default in self.providers:
            return self.default
        return next(iter(self.providers), None)

    def cost_for(self, model_id):
        """该模型每百万 token 单价 (input_per_m, output_per_m)，用于预算计价。"""
        p = self.providers.get(model_id)
        if p is None:
            return 0, 0
        c = getattr(p, "cost", {}) or {}
        return c.get("input_per_m", 0), c.get("output_per_m", 0)

    @staticmethod
    def _cache_key(model_id, system, user, temperature, max_tokens):
        raw = f"{model_id}|{temperature}|{max_tokens}|{system}|{user}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key):
        try:
            conn = db.connect()
            row = conn.execute(
                "SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
            conn.close()
            return row["response"] if row else None
        except Exception:  # noqa: BLE001   # 缓存不可用不影响主流程
            return None

    def _cache_put(self, key, response):
        try:
            conn = db.connect()
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key,response,created_at) "
                "VALUES (?,?,?)", (key, response, db.now()))
            conn.commit()
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    def complete(self, agent_role, system, user, temperature=0.3,
                 max_tokens=2000, agent_dept=None, on_budget=None):
        if not self.providers:
            raise LLMError("未配置任何模型（llm.models 为空）")
        model_id = self.resolve(agent_role, agent_dept)
        if model_id is None:
            raise LLMError("模型路由失败：default 与 routing 均未指向有效模型")
        cache = bool(self.config.get("cache", True))
        if cache:
            key = self._cache_key(model_id, system, user, temperature, max_tokens)
            hit = self._cache_get(key)
            if hit is not None:
                return hit   # 命中缓存：不调 API、不计费
        tried = set()
        while True:
            if model_id in tried:
                raise LLMError(f"模型 {model_id} 及其降级链均失败")
            tried.add(model_id)
            # 熔断检查：熔断中不发起调用（省请求），直接尝试 fallback 替代模型
            if not self.breaker.allow(model_id):
                nxt = self._fallback_of(model_id)
                if nxt is None or nxt == model_id:
                    raise LLMError(
                        f"模型 {model_id} 熔断中（连续失败暂停 "
                        f"{self.breaker.cooldown_s}s，且无可用降级）")
                model_id = nxt
                continue
            try:
                text = self.providers[model_id].complete(
                    system, user, temperature, max_tokens, backoff=self.backoff)
                self.breaker.record_success(model_id)
                if on_budget:
                    on_budget(model_id, len(system) + len(user), len(text))
                if cache:
                    self._cache_put(key, text)
                return text
            except LLMError:
                self.breaker.record_failure(model_id)
                nxt = self._fallback_of(model_id)
                if nxt is None or nxt == model_id:
                    raise
                model_id = nxt

    def _fallback_of(self, model_id):
        """fallback 链中 model_id 的降级目标（无则 None）。"""
        for rule in self.fallback:
            if rule.get("from") == model_id:
                return rule.get("to")
        return None
