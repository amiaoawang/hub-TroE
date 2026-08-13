"""Producer（superadmin）：生成宪章、调研玩法功能清单、广播宪章（CC 全部主 agent）、仲裁（S1 占位）。"""
import json
import re

from src import db
from src.agents.base import Agent

# 调研回退清单：LLM 输出不可用时，保证完整超级马里奥功能不丢
DEFAULT_RESEARCH = {
    "title": "超级马里奥 · 像素冒险",
    "genre": "平台跳跃",
    "features": {
        "enemy": True,      # 敌人系统：蘑菇怪巡逻/可踩死
        "brick": True,      # 顶砖块：? 砖顶出金币/蘑菇
        "mushroom": True,   # 蘑菇道具：吃后变大（双形态）
        "pipe": True,       # 管道：装饰 + 碰撞
        "coin": True,       # 金币收集
        "life": True,       # 生命系统：3 条命/受伤/掉落
        "flag": True,       # 旗杆过关：到达终点胜利
    },
    # 创作规格（LLM 按主题生成；题材主题据此由 LLM 创作游戏，模板仅同类型兜底）
    "spec": {
        "desc": "平台跳跃：移动/跳跃/顶砖块/吃蘑菇/收集金币/到达旗帜",
        "controls": "方向键/WASD 移动 · 空格跳跃",
        "scene": {"sky": "#5c94fc", "hill": "#4a8a3a",
                  "ground": "#854f0b", "player_hat": "#e63e2e",
                  "enemy_shell": "#c47f17", "coin": "#ef9f27"},
        "win_text": "过关！",
    },
}

# §3.2 引擎选型参考表：engine_id → 技术栈描述（宪章注入用）
ENGINE_STACK = {
    "web": "Web（Phaser/Three.js/PlayCanvas，JS）· 即点即玩，几日出可玩版",
    "godot": "Godot（GDScript/C#）· 2D 极快，开源零授权费",
    "unity": "Unity（C#）· 2D/3D 中型，成熟管线快",
    "unreal": "Unreal UE5（C++/蓝图）· 3A 画质，管线重、吃硬件",
}


class Producer(Agent):
    def __init__(self, agent_id, router=None, topics_cfg=None):
        super().__init__(agent_id, "producer", "producer", "Producer 总控",
                         router=router, topics_cfg=topics_cfg)
        self.charter = ""
        self.research_data = {}

    def generate_charter(self, questionnaire):
        text = self.llm("宪章", f"根据启动问卷生成项目宪章："
                        f"{json.dumps(questionnaire, ensure_ascii=False)}",
                        max_tokens=600)
        self.charter = text
        self.context_pack["charter"] = text
        self.log_action("charter_generate", "charter", {"version": "v1"})
        return text

    @staticmethod
    def select_engine(questionnaire):
        """§3.2 引擎选型：问卷 engine 显式指定 → 用之；
        producer_recommend/缺省 → 按品类+规模推荐（对应 §3.2 参考表）。

        返回 (engine_id, stack_desc)。
        """
        engine = str(questionnaire.get("engine", "") or "").strip()
        if engine and engine != "producer_recommend":
            return engine, ENGINE_STACK.get(engine, "")
        genre = str(questionnaire.get("genre", "") or "").lower()
        scope = str(questionnaire.get("scope", "") or "").lower()
        # 规则（§3.2）：3A/动作/FPS → Unreal；中型/RPG/手游 → Unity；
        # 2D 平台/休闲 → Web 或 Godot；demo 一律 Web（最快验证）
        if scope == "demo" or genre in ("casual", "puzzle", "platformer",
                                        "休闲", "平台跳跃", "解谜"):
            return "web", ENGINE_STACK["web"]
        if genre in ("shooter", "action", "fps", "射击", "动作"):
            return "unreal", ENGINE_STACK["unreal"]
        if genre in ("rpg", "simulation", "模拟"):
            return "unity", ENGINE_STACK["unity"]
        return "web", ENGINE_STACK["web"]

    def research(self, questionnaire, charter=""):
        """调研（流程第 2 环）：从问卷+宪章产出玩法功能清单（结构化 JSON）。
        功能清单是后续任务拆分（②）与程序执行（③）的需求源头；
        LLM 输出不可解析时回退 DEFAULT_RESEARCH（保证完整玩法不丢）。"""
        prompt = (f"启动问卷：{json.dumps(questionnaire, ensure_ascii=False)}\n"
                  f"宪章：{(charter or '')[:500]}")
        try:
            text = self.llm(
                "调研",
                "你是游戏策划调研员。根据启动问卷与宪章，产出该游戏的玩法功能清单与"
                "创作规格。模板只用于同类型游戏复用，游戏内容应体现主题题材。"
                "只输出一个 JSON 对象（不要解释、不要 markdown 代码块）："
                '{"title":"游戏标题","genre":"游戏类型",'
                '"features":{"enemy":true(敌人系统),"brick":true(顶砖/宝箱系统),'
                '"mushroom":true(变大道具),"pipe":true(障碍碰撞),'
                '"coin":true(收集物),"life":true(生命系统),"flag":true(终点过关)},'
                '"spec":{"desc":"一句话玩法描述","controls":"操作说明",'
                '"scene":{"sky":"天空色","hill":"远景色","ground":"平台色",'
                '"player_hat":"角色主色","enemy_shell":"敌人色","coin":"收集物色"},'
                '"win_text":"胜利文案","hud_tip":"HUD 操作引导","state_goal":"进行中目标文案"}}。'
                "features 的值用 true/false 表示是否包含该系统；"
                "spec 按主题世界观填写（题材、场景、角色都要贴合主题）。"
                f"\n\n{prompt}",
                max_tokens=700)
            data = self._parse_research(text)
        except Exception as e:  # noqa: BLE001
            print(f"[调研] LLM 失败（{e}），回退完整功能清单", flush=True)
            data = dict(DEFAULT_RESEARCH)
        self.research_data = data
        self.context_pack["research"] = data
        # 落库：功能清单是校验器（④）验收"功能是否实现"的依据
        try:
            conn = db.connect()
            conn.execute(
                "INSERT OR REPLACE INTO research (id,title,genre,features,created_at) "
                "VALUES (?,?,?,?,?)",
                ("latest", data.get("title"), data.get("genre"),
                 json.dumps(data.get("features") or {}, ensure_ascii=False),
                 db.now()))
            conn.commit()
            conn.close()
        except Exception as e:  # noqa: BLE001
            print(f"[调研] 清单落库失败（不影响运行）: {e}", flush=True)
        self.log_action("research_features", "research",
                        {"title": data.get("title"),
                         "features": list((data.get("features") or {}).keys())})
        return data

    @staticmethod
    def _parse_research(text):
        """解析调研 JSON；非法则回退默认完整清单。"""
        if text:
            raw = text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("```", 1)[0].strip()
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    data = json.loads(m.group(0))
                    if isinstance(data, dict) and isinstance(data.get("features"), dict):
                        return data
                except Exception:  # noqa: BLE001
                    pass
        return dict(DEFAULT_RESEARCH)

    def summarize_charter(self, charter, max_len=1200):
        """宪章摘要（token 优化）：LLM 压缩成要点，注入 agent 上下文替代全文。
        摘要失败时回退确定性截断（保头部要点区）。"""
        try:
            text = self.llm(
                "宪章摘要",
                "把项目宪章压缩成要点摘要（≤400字），必须保留："
                "项目定位/核心玩法/数值规范/范围外(Out-of-Scope)/风格规范。"
                f"\n\n宪章原文：\n{charter[:4000]}",
                max_tokens=500)
            summary = text.strip()
        except Exception:  # noqa: BLE001
            summary = charter[:max_len]
        if len(summary) > max_len:
            summary = summary[:max_len]
        self.log_action("charter_summarize", "charter",
                        {"full_len": len(charter), "summary_len": len(summary)})
        return summary

    def broadcast_charter(self, main_agents):
        """宪章注入：To 每个主 agent，CC 其他全部主 agent（广播主题，全员可见）。"""
        others = [a.id for a in main_agents]
        for m in main_agents:
            cc = [x for x in others if x != m.id]
            self.send(m.id, "【宪章】项目宪章 v1", self.charter, cc=cc,
                      topic="宪章")
        self.log_action("charter_broadcast", "charter",
                        {"to": [a.id for a in main_agents]})

    def arbitrate(self, task_id, detail):
        """仲裁：S1 占位——记录决策并标记任务 back 到 backlog 由主 agent 重新处理。"""
        self.log_action("arbitrate", task_id, detail)
        return {"decision": "重新派发", "task_id": task_id}
