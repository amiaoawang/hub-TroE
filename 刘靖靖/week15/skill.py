"""skill 注册表：能力热插拔（自进化改进层——新增能力不改 agent 代码）。

- 内置 skill：映射 game.py 生成器（battlefield / platformer），数值表驱动。
- 外部 skill：模板文件（@@TOKEN@@ 占位符约定）+ 元数据注册，generate 时按参数填充。
- 对自进化的意义：新游戏类型/新产出模板 = 注册一个 skill，subagent 任务描述
  带 skill:<id> 即自动路由，实现"能力可扩展、可复用、可沉淀"。

用法：
    from src import skill
    skill.list_skills()
    skill.generate("battlefield", {...数值...})
    skill.register_skill("neon-racer", "霓虹赛车", "俯视赛车", "/path/tpl.html")
"""
import json
import os
import re
import shutil

from src import db
from src import game as game_gen

SKILLS_DIR = None      # 模块级覆盖（测试隔离用）；None → 当前项目 skills/


def skills_dir():
    return SKILLS_DIR or os.path.join(db.PROJECT_DIR, "skills")


def registry_file():
    return os.path.join(skills_dir(), "registry.json")


def templates_dir():
    return os.path.join(skills_dir(), "templates")

# 内置 skill：模板逻辑在 game.py，此处只声明元数据（fn = game.py 生成函数名）
_BUILTIN = {
    "battlefield": {
        "name": "一战堑壕俯视射击",
        "fn": "build_battlefield_html",
        "desc": "数值注入的单文件 HTML5 俯视射击（波次敌人/旗帜占领/胜利失败）",
        "tokens": ["PLAYER_SPEED", "FIRE_INTERVAL_S", "RELOAD_S", "MAX_AMMO",
                   "PLAYER_HP", "ENEMY_HP", "ENEMY_SPEED", "WAVE_SIZE",
                   "FLAG_CAPTURE_S", "KILL_TARGET"],
    },
    "platformer": {
        "name": "像素平台跳跃",
        "fn": "build_game_html",
        "desc": "数值注入的单文件 HTML5 平台跳跃（移动/跳跃/收集/旗帜）",
        "tokens": ["MOVE_SPEED", "JUMP_VEL", "GRAVITY"],
    },
    "rpg": {
        "name": "回合制角色扮演",
        "fn": "build_rpg_html",
        "desc": "数值注入的单文件 HTML5 回合制 RPG（地图探索/遇敌/回合制战斗/升级/剧情）",
        "tokens": ["PLAYER_HP", "PLAYER_MP", "PLAYER_ATK", "PLAYER_DEF",
                   "ENEMY_HP", "ENEMY_ATK", "BOSS_HP", "BOSS_ATK",
                   "EXP_TO_LEVEL", "HEAL_COST", "FIRE_COST", "POTION_COUNT"],
    },
}


def _registry():
    if os.path.exists(registry_file()):
        with open(registry_file(), encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "skills": []}


def _save(reg):
    os.makedirs(skills_dir(), exist_ok=True)
    with open(registry_file(), "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def list_skills():
    """全部可用 skill（内置 + 外部注册）。"""
    skills = [{"id": sid, "name": m["name"], "builtin": True,
               "desc": m["desc"], "tokens": m["tokens"]}
              for sid, m in _BUILTIN.items()]
    skills += [dict(s) for s in _registry()["skills"]]
    return skills


def get_skill(skill_id):
    for s in list_skills():
        if s["id"] == skill_id:
            return s
    return None


def register_skill(skill_id, name, desc, template_path):
    """注册外部 skill：模板文件用 @@KEY@@ 占位符，generate 时按参数填充。
    模板会复制进 skills/templates/（进版本库，可复用可回滚）。"""
    if skill_id in _BUILTIN:
        raise ValueError(f"{skill_id} 是内置 skill，不可覆盖")
    if not os.path.exists(template_path):
        raise FileNotFoundError(template_path)
    os.makedirs(templates_dir(), exist_ok=True)
    dst = os.path.join(templates_dir(), f"{skill_id}.html")
    shutil.copy(template_path, dst)
    reg = _registry()
    reg["skills"] = [s for s in reg["skills"] if s["id"] != skill_id]
    reg["skills"].append({"id": skill_id, "name": name, "desc": desc,
                          "builtin": False,
                          "template": f"templates/{skill_id}.html"})
    _save(reg)
    return skill_id


def generate(skill_id, params, features=None, theme=None, scene=None):
    """按 skill 生成可玩 HTML。params: 数值字典（非法/缺省由生成器兜底）。
    features: 调研功能清单（仅 platformer 类消费，battlefield/rpg 忽略）。
    theme: 主题元数据 dict（themes.theme_meta()）——platformer/rpg 世界观文案，None=默认。
    scene: 场景配色 dict（创作层按主题提供）——platformer 场景注入，None=经典配色。"""
    meta = _BUILTIN.get(skill_id)
    if meta:
        fn = getattr(game_gen, meta["fn"])
        if meta["fn"] == "build_game_html":
            return fn(params or {}, features=features, theme_meta=theme,
                      scene=scene)
        if meta["fn"] == "build_rpg_html":
            return fn(params or {}, theme_meta=theme)
        return fn(params or {})
    reg = _registry()
    for s in reg["skills"]:
        if s["id"] == skill_id:
            tpl = os.path.join(skills_dir(), s["template"])
            if not os.path.exists(tpl):
                raise FileNotFoundError(tpl)
            with open(tpl, encoding="utf-8") as f:
                text = f.read()
            return re.sub(r"@@(\w+)@@",
                          lambda m: str(params.get(m.group(1), "")), text)
    raise KeyError(f"未知 skill: {skill_id}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="skill 注册表操作")
    ap.add_argument("op", choices=["list", "register"], help="操作")
    ap.add_argument("--id", help="skill id（register）")
    ap.add_argument("--name", help="skill 名称（register）")
    ap.add_argument("--desc", default="", help="skill 描述（register）")
    ap.add_argument("--template", help="模板文件路径（register，@@KEY@@ 占位符）")
    args = ap.parse_args()
    if args.op == "list":
        for s in list_skills():
            tag = "内置" if s["builtin"] else "外部"
            print(f"  [{s['id']}] ({tag}) {s['name']} — {s['desc']}")
    elif args.op == "register":
        sid = register_skill(args.id, args.name, args.desc, args.template)
        print(f"已注册 skill: {sid}")
