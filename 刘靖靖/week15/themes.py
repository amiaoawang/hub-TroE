"""主题注册表：游戏主题 → 模板/skill/产物名 的单一事实源。

主题分两类（kind 字段）：
- genre（类型模板主题）：如 mario（平台跳跃）/ battlefield（俯视射击）——
  主题即玩法类型，模板就是该类型的完整实现，直接复用（同类型游戏复用）。
- theme（题材主题）：如 princess（勇者救公主）——主题是题材/世界观，
  不绑定玩法类型；由 LLM 按主题与调研规格创作游戏，skill 仅作兜底模板
  （LLM 失败时回退到同类型玩法，保证流水线不断）。

自定义主题（skill.register_skill 注册）视为题材主题（LLM 创作）。
主题解析（get_theme/resolve_theme）：
1. 精确匹配内置主题 id（mario/battlefield/princess）
2. 精确匹配已注册外部 skill id
3. 内置主题关键词匹配——支持自然语言描述（如「勇者救出被困在女巫城堡里的公主」
   → princess），显式关键词表，不做隐式推导
"""
from __future__ import annotations

from src import skill as skill_reg

# 内置主题注册（skill 值必须对应 skill.py 的 _BUILTIN 或已注册外部 skill）
# keywords：自然语言描述 → 主题的关键词表（显式映射，无歧义）
# worldview：主题默认世界观（问卷未指定时注入宪章）
# genre：主题默认品类（问卷未指定时决定调研功能清单）
# kind：genre=类型模板主题（模板即实现，直接复用）| theme=题材主题（LLM 创作，模板兜底）
_BUILTIN_THEMES = {
    "mario": {
        "name": "马里奥 · 平台跳跃",
        "kind": "genre",                # 主题即类型：模板即完整实现，直接复用
        "skill": "platformer",          # skill 注册表 id
        "product": "mario.html",        # 可玩产物文件名（game/ 目录）
        "desc": "平台跳跃：移动/跳跃/顶砖块/吃蘑菇/收集金币/到达旗帜",
        "design_schema": "platformer",  # DesignValidator 数值 schema 键
        "genre": "platformer",
        "worldview": "治愈系像素王国：水管工英雄冒险，顶砖块、吃蘑菇、踩敌人、收集金币、到达终点旗帜",
        "keywords": ["马里奥", "水管工", "顶砖", "蘑菇王国"],
    },
    "battlefield": {
        "name": "战地一 · 堑壕射击",
        "kind": "genre",                # 类型模板主题：俯视射击模板即实现
        "skill": "battlefield",
        "product": "battlefield.html",
        "desc": "一战堑壕俯视射击：步兵对射/占领旗帜/波次进攻",
        "design_schema": "battlefield",
        "genre": "shooter",
        "worldview": "一战（WWI）欧洲堑壕战场俯视射击：步兵对射、占领旗帜、波次进攻",
        "keywords": ["战地", "堑壕", "一战", "索姆河", "战争"],
    },
    "princess": {
        "name": "勇者救公主 · 城堡冒险",
        "kind": "theme",                # 题材主题：LLM 按主题创作，platformer 仅兜底
        "skill": "platformer",          # 兜底模板（LLM 创作失败时回退）
        "product": "princess.html",
        "desc": "奇幻城堡冒险：勇者穿越女巫城堡，击败魔物、收集力量水晶，救出被囚的公主",
        "design_schema": "platformer",
        "genre": "platformer",
        "worldview": "奇幻王国：女巫掳走公主囚于城堡，勇者踏上营救之路——跳跃平台、顶宝箱、收集水晶、击败魔物，抵达城堡救出公主",
        "keywords": ["公主", "女巫", "城堡", "救公主", "营救", "骑士"],
        "game_text": {                  # 兜底模板的世界观文案（LLM 创作时由 LLM 自定）
            "title": "勇者救公主 · 城堡冒险",
            "hud_tip": "方向键 / WASD 移动 · 空格跳跃 · 顶宝箱 · 击败魔物 · 吃力量水晶变大 · 抵达城堡救出公主",
            "state_goal": "向城堡进发！击败魔物 / 顶宝箱 / 收集水晶",
            "win_text": "你救出了公主！",
        },
    },
    "dragonquest": {
        "name": "勇者斗恶龙 · 回合制 RPG",
        "kind": "genre",                # 类型主题：回合制 RPG 模板即实现
        "skill": "rpg",
        "product": "dragonquest.html",
        "desc": "勇者斗恶龙式回合制 RPG：地图探索/遇敌/回合制战斗/升级/讨伐魔王",
        "design_schema": "rpg",
        "genre": "rpg",
        "worldview": "奇幻王国：黑暗魔王复活，勇者踏上讨伐之旅——探索地图、击败魔物、升级、讨伐魔王",
        "keywords": ["勇者斗恶龙", "斗恶龙", "角色扮演", "回合制", "RPG", "rpg"],
        "game_text": {
            "title": "勇者斗恶龙 · 回合制 RPG",
            "enemy_name": "史莱姆",
            "boss_name": "魔王",
            "spell_name": "火球术",
            "heal_name": "治愈术",
            "story_intro": "黑暗笼罩大地，魔王复活了。你作为被选中的勇者，踏上讨伐魔王的旅程。\n在地图中探索，击败魔物、收集金币、提升等级，最终在出口挑战魔王！",
            "win_text": "你击败了魔王，世界恢复了和平！",
            "hud_tip": "方向键 / WASD 移动 · 踩到敌人进入战斗 · 空格攻击 · 走到出口挑战魔王",
        },
    },
}


def list_themes():
    """全部可用主题：内置 + 已注册外部 skill 派生（每个 skill id 一个主题）。"""
    out = []
    for tid, meta in _BUILTIN_THEMES.items():
        out.append({"id": tid, "name": meta["name"], "builtin": True,
                    "desc": meta["desc"], "skill": meta["skill"],
                    "product": meta["product"]})
    for s in skill_reg.list_skills():
        if s["id"] in ("platformer", "battlefield", "rpg"):   # 内置 skill 已被内置主题覆盖
            continue
        out.append({"id": s["id"], "name": s.get("name", s["id"]),
                    "builtin": False, "desc": s.get("desc", ""),
                    "skill": s["id"], "product": f"{s['id']}.html"})
    return out


def resolve_theme(theme):
    """主题解析核心：自然语言/ID → 主题 id（确定性映射，无隐式推导）。
    顺序：内置 id 精确 → 内置主题关键词匹配（最长命中优先）→ 外部 skill id。
    关键词匹配先于 skill id：让 "rpg" 命中 dragonquest 主题而非 rpg 外部 skill。
    均不命中返回 None。"""
    theme = (theme or "").strip()
    if not theme:
        return None
    if theme in _BUILTIN_THEMES:
        return theme
    best = None
    best_len = 0
    for tid, meta in _BUILTIN_THEMES.items():
        for kw in meta.get("keywords", []):
            if kw and kw in theme and len(kw) > best_len:
                best = tid
                best_len = len(kw)
    if best:
        return best
    if skill_reg.get_skill(theme):          # 外部已注册 skill：id 精确匹配
        return theme
    return None


def get_theme(theme):
    """解析主题：内置 → 元数据；未内置但存在对应 skill → 派生主题；
    自然语言描述命中内置主题关键词 → 对应内置主题；
    都无 → None（未注册模板，不可用）。"""
    tid = resolve_theme(theme)
    if tid is None:
        return None
    if tid in _BUILTIN_THEMES:
        return dict(_BUILTIN_THEMES[tid], id=tid)
    sk = skill_reg.get_skill(tid)
    if sk:
        return {"id": tid, "name": sk.get("name", tid), "skill": tid,
                "product": f"{tid}.html",
                "desc": sk.get("desc", "自定义主题（外部 skill 模板）"),
                "design_schema": "generic", "genre": "other",
                "kind": "theme", "worldview": "", "keywords": []}
    return None


def product_file(theme):
    """该主题的可玩产物文件名（game/ 目录内）。"""
    meta = get_theme(theme)
    return meta["product"] if meta else f"{theme}.html"


def design_schema(theme):
    """该主题的数值表校验 schema 键（platformer/battlefield/generic）。"""
    meta = get_theme(theme)
    return meta.get("design_schema", "generic") if meta else "generic"


def theme_title(theme):
    """主题显示名。"""
    meta = get_theme(theme)
    return meta["name"] if meta else theme


def theme_meta(theme):
    """主题元数据 dict（含 skill/product/design_schema/genre/worldview）；
    未注册返回 None。生成器世界观注入等场景使用。"""
    return get_theme(theme)
