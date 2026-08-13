"""真实小游戏原型生成器：单文件 HTML5 Canvas 像素风平台跳跃（零依赖、即开即玩）。
数值（移速/跳跃初速/重力）由策划的 design.json 注入——体现「数值表驱动游戏」链路。
模板内数值用 @@TOKEN@@ 占位，避免与 CSS/JS 花括号冲突。
"""
import json
import os
import re


# 模板默认场景配色（马里奥经典）：主题化场景由创作层按 spec 提供，
# 缺省保持经典视觉——模板 = 类型机制 + 可配置场景，视觉归属创作层。
DEFAULT_SCENE = {
    "sky": "#5c94fc", "cloud": "#ffffff", "hill": "#4a8a3a", "hill2": "#3f7a32",
    "ground": "#854f0b", "ground_top": "#ba7517", "ground_dot": "#a06a10",
    "player_hat": "#e63e2e", "player_skin": "#f5c4b3",
    "player_pants": "#2c5fd8", "player_boot": "#7a4a2b",
    "enemy_shell": "#c47f17", "enemy_body": "#8a5a2b", "enemy_face": "#f5c4b3",
    "block": "#c47f17", "block_hi": "#f2b23a",
    "block_used": "#8a6a3a", "block_used_in": "#a5814e",
    "pipe": "#2f9e44", "pipe_light": "#40c057",
    "pipe_hl": "#51cf66", "pipe_in": "#8ce99a",
    "coin": "#ef9f27", "coin_hl": "#fff3cd",
    "mush_cap": "#e63e2e", "mush_stem": "#f5c4b3",
    "mush_dot": "#ffffff", "mush_foot": "#7a4a2b",
    "flag_pole": "#444441", "flag": "#e63e2e",
}


def build_game_html(params, features=None, modules=None, theme_meta=None,
                    scene=None):
    """params: dict(move_speed, jump_vel, gravity) → 完整可玩 HTML。非法/缺省用安全默认值。
    features: 调研功能清单 dict（如 {"enemy": true, "brick": false}）——
    控制游戏包含哪些系统（敌人/顶砖/蘑菇/管道/金币/生命/旗杆），None = 全功能。
    modules: dict(name -> 系统模块 JS 代码)——系统任务产出，按需组装进骨架；
    None = 用内置全部模块（默认全功能，向后兼容）。
    theme_meta: 主题元数据 dict（themes.theme_meta()）——世界观文案注入：
    title/hud_tip/state_goal/win_text 覆盖模板文本（None = 马里奥默认文案，向后兼容）。
    scene: 场景配色 dict（创作层按主题提供，键见 DEFAULT_SCENE）；None = 经典配色。"""
    from src import game_templates as gt
    p = {"move_speed": 180, "jump_vel": 420, "gravity": 900}
    for k in ("move_speed", "jump_vel", "gravity"):
        v = params.get(k) if isinstance(params.get(k), (int, float)) else None
        if v is not None and not isinstance(v, bool) and v > 0:
            p[k] = int(v)
    feat = {}
    if isinstance(features, dict):
        for k in ("enemy", "brick", "mushroom", "pipe", "coin", "life", "flag"):
            if k in features:
                feat[k] = bool(features[k])
    if modules is None:
        sys_code = "\n".join(gt._SYS_MODULES[k] for k in gt._SYS_ORDER)
    else:
        # 系统任务产出组装：缺失系统 = 游戏缺该系统（验收可发现）
        sys_code = "\n".join(modules[k] for k in gt._SYS_ORDER
                             if modules.get(k) and modules[k].strip())
    # ---- 世界观注入（主题化文本；缺省 = 马里奥默认文案） ----
    tm = theme_meta if isinstance(theme_meta, dict) else {}
    # 主题专属文案：优先取 game_text 块（themes 内置主题），其次顶层 title 字段
    gt_override = tm.get("game_text") if isinstance(tm.get("game_text"), dict) else {}
    theme_text = {
        "title": "超级马里奥 · 像素冒险",
        "hud_tip": "方向键 / WASD 移动 · 空格跳跃 · 顶 ? 砖块 · 踩敌人 · 吃蘑菇变大 · 到达旗杆过关",
        "state_goal": "向旗杆进发！踩敌人 / 顶砖块 / 吃蘑菇",
        "win_text": "过关！",
    }
    for k in theme_text:
        v = gt_override.get(k) or tm.get(k)
        if isinstance(v, str) and v.strip():
            theme_text[k] = v.strip()
    # ---- 场景配色注入（创作层按主题提供；缺省经典配色，缺失键回退默认） ----
    scene_cfg = dict(DEFAULT_SCENE)
    if isinstance(scene, dict):
        for k in scene_cfg:
            v = scene.get(k)
            if isinstance(v, str) and v.strip():
                scene_cfg[k] = v.strip()
    html = (gt._GAME_SKELETON
            .replace("@@THEME_TITLE@@", theme_text["title"])
            .replace("@@HUD_TIP@@", theme_text["hud_tip"])
            .replace("@@STATE_GOAL@@", theme_text["state_goal"])
            .replace("@@WIN_TEXT@@", theme_text["win_text"]))
    return (html
            .replace("@@MOVE_SPEED@@", str(p["move_speed"]))
            .replace("@@JUMP_VEL@@", str(p["jump_vel"]))
            .replace("@@GRAVITY@@", str(p["gravity"]))
            .replace("@@FEATURES@@", json.dumps(feat, ensure_ascii=False))
            .replace("@@SCENE@@", json.dumps(scene_cfg, ensure_ascii=False))
            .replace("/*__SYS_SCRIPTS__*/", sys_code))


def extract_module(name):
    """提取单个玩法系统的实现模块代码（系统级任务的产出载体）。
    返回带标记的模块代码文本；无该模块返回 None。"""
    from src import game_templates as gt
    return gt._SYS_MODULES.get(name)


def module_names():
    """可提取的系统模块列表（模板内置，按组装顺序）。"""
    from src import game_templates as gt
    return list(gt._SYS_ORDER)


_BF_DEFAULTS = {
    "player_speed": 200, "fire_interval_s": 1.2, "reload_s": 2.5,
    "max_ammo": 5, "player_hp": 10, "enemy_hp": 1, "enemy_speed": 60,
    "wave_size": 5, "flag_capture_s": 3, "kill_target": 20,
}

_BF_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>战地一 · 堑壕突击（网页原型）</title>
<style>
  body { margin:0; background:#1b1e23; display:flex; flex-direction:column;
         align-items:center; justify-content:center; min-height:100vh;
         font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif; color:#e8e6e0; }
  canvas { border:2px solid #6b5d45; border-radius:10px; cursor:crosshair;
           box-shadow:0 6px 24px rgba(0,0,0,.5); }
  #hud { margin-top:10px; font-size:14px; color:#c9c4b8; text-align:center; line-height:1.8;
         background:#262a31; border-radius:10px; padding:8px 20px; min-width:900px; }
  #hud b { color:#e8d9a8; }
  .bar { display:inline-block; width:140px; height:10px; background:#3a3f47;
         border-radius:5px; vertical-align:middle; margin:0 6px; overflow:hidden; }
  .bar i { display:block; height:100%; background:#8a9a4a; border-radius:5px; }
  .bar i.hp { background:#c25b4e; }
  .badge { display:inline-block; background:#3a3f47; border-radius:6px; padding:2px 10px;
           margin-left:8px; font-size:12px; color:#c9c4b8; }
</style>
</head>
<body>
<canvas id="bf" width="960" height="540"></canvas>
<div id="hud">
  <span>生命 <b id="hp">10/10</b></span>
  <span class="bar"><i class="hp" id="hpbar"></i></span>
  <span>弹药 <b id="ammo">5</b> / 5</span>
  <span>击杀 <b id="kills">0</b></span>
  <span>占领 <b id="flags">0</b>/3</span>
  <span class="badge" id="state">1916 · 索姆河 · 向旗帜推进</span>
</div>
<script>
(function () {
  "use strict";
  // ===== 策划数值注入（战地 schema）=====
  var PLAYER_SPEED = @@PLAYER_SPEED@@;
  var FIRE_INTERVAL_S = @@FIRE_INTERVAL_S@@;
  var RELOAD_S = @@RELOAD_S@@;
  var MAX_AMMO = @@MAX_AMMO@@;
  var PLAYER_HP = @@PLAYER_HP@@;
  var ENEMY_HP = @@ENEMY_HP@@;
  var ENEMY_SPEED = @@ENEMY_SPEED@@;
  var WAVE_SIZE = @@WAVE_SIZE@@;
  var FLAG_CAPTURE_S = @@FLAG_CAPTURE_S@@;
  var KILL_TARGET = @@KILL_TARGET@@;

  var cv = document.getElementById("bf");
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height;
  var el = { hp: document.getElementById("hp"), hpbar: document.getElementById("hpbar"),
             ammo: document.getElementById("ammo"), kills: document.getElementById("kills"),
             flags: document.getElementById("flags"), state: document.getElementById("state") };

  // 战场：左侧英军堑壕，右侧德军堑壕，中央旗帜
  var TRENCH_W = 90;
  var FLAG = { x: W / 2, y: H / 2, r: 34, by_player: false };

  var keys = {};
  var mouse = { x: W / 2, y: H / 2, down: false };
  var player = { x: TRENCH_W - 60, y: H / 2, r: 12, hp: PLAYER_HP, ammo: MAX_AMMO,
                 reloading: 0, cooldown: 0, recoil: 0 };
  var enemies = [], pbullets = [], ebullets = [], puffs = [], flags_won = 0;
  var kills = 0, capture_progress = 0, state = "playing";  // playing | won | lost
  var flash = 0, spawn_timer = 1.0;

  document.addEventListener("keydown", function (e) {
    keys[e.code] = true;
    if (e.code === "Space") e.preventDefault();
    if (state !== "playing" && e.code === "KeyR") location.reload();
  });
  document.addEventListener("keyup", function (e) { keys[e.code] = false; });
  cv.addEventListener("mousemove", function (e) {
    var r = cv.getBoundingClientRect();
    mouse.x = (e.clientX - r.left) * (W / r.width);
    mouse.y = (e.clientY - r.top) * (H / r.height);
  });
  cv.addEventListener("mousedown", function () { mouse.down = true; });
  cv.addEventListener("mouseup", function () { mouse.down = false; });

  function startReload() { player.reloading = RELOAD_S; el.state.textContent = "上弹中…"; }

  function fire() {
    if (player.reloading > 0 || player.cooldown > 0) return;
    if (player.ammo <= 0) { startReload(); return; }
    player.ammo -= 1;
    player.cooldown = FIRE_INTERVAL_S;
    player.recoil = 6;
    flash = 0.06;
    var ang = Math.atan2(mouse.y - player.y, mouse.x - player.x);
    var spread = (Math.random() - 0.5) * 0.06;
    pbullets.push({ x: player.x + Math.cos(ang) * 16, y: player.y + Math.sin(ang) * 16,
                    vx: Math.cos(ang + spread) * 620, vy: Math.sin(ang + spread) * 620, life: 2 });
    sfxShoot();
    if (player.ammo === 0) startReload();
  }

  // 简易音效（WebAudio 振荡器）
  var actx = null;
  function sfx(freq, dur, type) {
    try {
      if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
      var o = actx.createOscillator(), g = actx.createGain();
      o.type = type || "square"; o.frequency.value = freq;
      g.gain.setValueAtTime(0.12, actx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + dur);
      o.connect(g); g.connect(actx.destination);
      o.start(); o.stop(actx.currentTime + dur);
    } catch (e) {}
  }
  function sfxShoot() { sfx(180, 0.18, "sawtooth"); }
  function sfxHit() { sfx(90, 0.22, "square"); }
  function sfxFlag() { sfx(520, 0.3, "sine"); setTimeout(function () { sfx(780, 0.3, "sine"); }, 120); }

  function spawnWave() {
    for (var i = 0; i < WAVE_SIZE; i++) {
      enemies.push({
        x: W - TRENCH_W + 20 + Math.random() * 40, y: 50 + Math.random() * (H - 100),
        r: 11, hp: ENEMY_HP, shoot: 1 + Math.random()
      });
    }
  }

  function update(dt) {
    if (state !== "playing") return;
    // 移动
    var dx = 0, dy = 0;
    if (keys["KeyA"] || keys["ArrowLeft"]) dx -= 1;
    if (keys["KeyD"] || keys["ArrowRight"]) dx += 1;
    if (keys["KeyW"] || keys["ArrowUp"]) dy -= 1;
    if (keys["KeyS"] || keys["ArrowDown"]) dy += 1;
    if (dx || dy) { var m = Math.hypot(dx, dy); player.x += dx / m * PLAYER_SPEED * dt; player.y += dy / m * PLAYER_SPEED * dt; }
    // 边界（玩家被限制在英军侧与无人区）
    player.x = Math.max(20, Math.min(W / 2 + 60, player.x));
    player.y = Math.max(20, Math.min(H - 20, player.y));
    // 冷却/上弹/后坐力
    if (player.cooldown > 0) player.cooldown -= dt;
    if (player.reloading > 0) {
      player.reloading -= dt;
      if (player.reloading <= 0) { player.ammo = MAX_AMMO; el.state.textContent = "已上弹"; }
    }
    if (player.recoil > 0) player.recoil = Math.max(0, player.recoil - dt * 20);
    // 射击
    if (mouse.down) fire();
    // 敌人 AI
    for (var i = enemies.length - 1; i >= 0; i--) {
      var e = enemies[i];
      var dist = Math.hypot(player.x - e.x, player.y - e.y);
      if (dist > 260) {
        var ang2 = Math.atan2(player.y - e.y, player.x - e.x);
        e.x += Math.cos(ang2) * ENEMY_SPEED * dt;
        e.y += Math.sin(ang2) * ENEMY_SPEED * dt;
      } else {
        e.shoot -= dt;
        if (e.shoot <= 0) {
          e.shoot = 1.6 + Math.random() * 1.2;
          var a3 = Math.atan2(player.y - e.y, player.x - e.x);
          ebullets.push({ x: e.x, y: e.y, vx: Math.cos(a3) * 260, vy: Math.sin(a3) * 260, life: 3 });
        }
      }
    }
    // 玩家子弹
    for (var j = pbullets.length - 1; j >= 0; j--) {
      var b = pbullets[j];
      b.x += b.vx * dt; b.y += b.vy * dt; b.life -= dt;
      if (b.life <= 0 || b.x < 0 || b.x > W || b.y < 0 || b.y > H) { pbullets.splice(j, 1); continue; }
      var hit = false;
      for (var k = enemies.length - 1; k >= 0; k--) {
        var e2 = enemies[k];
        if (Math.hypot(b.x - e2.x, b.y - e2.y) < e2.r + 3) {
          e2.hp -= 1;
          puffs.push({ x: e2.x, y: e2.y, life: 0.25 });
          sfxHit();
          if (e2.hp <= 0) {
            enemies.splice(k, 1); kills += 1;
            puffs.push({ x: e2.x, y: e2.y, life: 0.5 });
          }
          pbullets.splice(j, 1); hit = true; break;
        }
      }
      if (hit) continue;
      if (b.x > W - TRENCH_W) { pbullets.splice(j, 1); }  // 子弹落入德军堑壕
    }
    // 敌人子弹命中玩家
    for (var m2 = ebullets.length - 1; m2 >= 0; m2--) {
      var eb = ebullets[m2];
      eb.x += eb.vx * dt; eb.y += eb.vy * dt; eb.life -= dt;
      if (eb.life <= 0 || eb.x < 0 || eb.x > W || eb.y < 0 || eb.y > H) { ebullets.splice(m2, 1); continue; }
      if (Math.hypot(eb.x - player.x, eb.y - player.y) < player.r + 3) {
        player.hp -= 1;
        puffs.push({ x: player.x, y: player.y, life: 0.3 });
        sfx(120, 0.2, "sawtooth");
        ebullets.splice(m2, 1);
        if (player.hp <= 0) { state = "lost"; el.state.textContent = "阵亡…按 R 重来"; }
      }
    }
    // 旗帜占领
    var near = Math.hypot(player.x - FLAG.x, player.y - FLAG.y) < FLAG.r + 8;
    if (near && !FLAG.by_player) {
      capture_progress += dt;
      if (capture_progress >= FLAG_CAPTURE_S) {
        FLAG.by_player = true; flags_won += 1; capture_progress = 0;
        sfxFlag();
        if (flags_won >= 3) { state = "won"; el.state.textContent = "胜利！占领全部旗帜"; }
      }
    } else if (!near) { capture_progress = Math.max(0, capture_progress - dt); }
    // 波次补充
    if (enemies.length < WAVE_SIZE) {
      spawn_timer -= dt;
      if (spawn_timer <= 0) { spawn_timer = 2.0; spawnWave(); }
    }
    // 胜利条件：击杀目标
    if (kills >= KILL_TARGET) { state = "won"; el.state.textContent = "胜利！达成击杀目标"; }
    // HUD
    el.hp.textContent = Math.max(0, player.hp) + "/" + PLAYER_HP;
    el.hpbar.style.width = Math.max(0, player.hp / PLAYER_HP * 100) + "%";
    el.ammo.textContent = player.ammo;
    el.kills.textContent = kills;
    el.flags.textContent = flags_won;
  }

  function trench(x, w, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, 20, w, H - 40);
    ctx.fillStyle = "rgba(0,0,0,0.25)";
    ctx.fillRect(x + 14, 20, 6, H - 40);
    ctx.fillStyle = "#8a7a55";
    ctx.fillRect(x + 8, 20, 4, H - 40);
    ctx.fillRect(x + w - 12, 20, 4, H - 40);
  }

  function draw() {
    // 地面
    ctx.fillStyle = "#5c6b3a"; ctx.fillRect(0, 0, W, H);
    // 无人区弹坑
    ctx.fillStyle = "rgba(0,0,0,0.18)";
    ctx.beginPath(); ctx.arc(340, 180, 22, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(560, 340, 28, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.arc(700, 140, 18, 0, 7); ctx.fill();
    // 铁丝网装饰
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    for (var wg = 0; wg < 3; wg++) {
      ctx.beginPath();
      ctx.moveTo(200 + wg * 90, 300 + wg * 20);
      ctx.lineTo(230 + wg * 90, 280 + wg * 20);
      ctx.lineTo(260 + wg * 90, 320 + wg * 20);
      ctx.stroke();
    }
    // 堑壕
    trench(0, TRENCH_W, "#6f5f3f");          // 英军
    trench(W - TRENCH_W, TRENCH_W, "#5d5140"); // 德军
    // 旗帜
    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.beginPath(); ctx.arc(FLAG.x, FLAG.y, FLAG.r, 0, 7); ctx.fill();
    ctx.fillStyle = FLAG.by_player ? "#8a9a4a" : "#c25b4e";
    ctx.fillRect(FLAG.x - 2, FLAG.y - 34, 4, 68);
    ctx.beginPath();
    ctx.moveTo(FLAG.x + 2, FLAG.y - 34);
    ctx.lineTo(FLAG.x + 30, FLAG.y - 26);
    ctx.lineTo(FLAG.x + 2, FLAG.y - 18);
    ctx.fill();
    // 占领进度环
    if (capture_progress > 0) {
      ctx.strokeStyle = "#e8d9a8"; ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(FLAG.x, FLAG.y, FLAG.r + 8, -Math.PI / 2,
              -Math.PI / 2 + capture_progress / FLAG_CAPTURE_S * Math.PI * 2);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
    // 玩家（英军绿）
    ctx.fillStyle = "#4a5d40";
    ctx.beginPath(); ctx.arc(player.x, player.y, player.r, 0, 7); ctx.fill();
    ctx.fillStyle = "#6b7a52";
    ctx.beginPath(); ctx.arc(player.x - 2, player.y - 2, player.r - 4, 0, 7); ctx.fill();
    ctx.strokeStyle = "#2c2c2a";
    ctx.lineWidth = 2;
    var ang = Math.atan2(mouse.y - player.y, mouse.x - player.x) - player.recoil * 0.02;
    ctx.beginPath();
    ctx.moveTo(player.x, player.y);
    ctx.lineTo(player.x + Math.cos(ang) * 26, player.y + Math.sin(ang) * 26);
    ctx.stroke();
    ctx.lineWidth = 1;
    // 敌人（德灰）
    for (var i = 0; i < enemies.length; i++) {
      var e = enemies[i];
      ctx.fillStyle = "#6b6b6b";
      ctx.beginPath(); ctx.arc(e.x, e.y, e.r, 0, 7); ctx.fill();
      ctx.fillStyle = "#8a8a8a";
      ctx.beginPath(); ctx.arc(e.x - 2, e.y - 2, e.r - 4, 0, 7); ctx.fill();
    }
    // 子弹
    ctx.fillStyle = "#ffd977";
    for (var j = 0; j < pbullets.length; j++) {
      ctx.fillRect(pbullets[j].x - 2, pbullets[j].y - 1, 5, 2);
    }
    ctx.fillStyle = "#ff7b5e";
    for (var m2 = 0; m2 < ebullets.length; m2++) {
      ctx.fillRect(ebullets[m2].x - 2, ebullets[m2].y - 1, 5, 2);
    }
    // 命中特效
    for (var q = puffs.length - 1; q >= 0; q--) {
      var pf = puffs[q];
      pf.life -= 1 / 60;
      ctx.fillStyle = "rgba(255,220,120," + Math.max(0, pf.life * 3) + ")";
      ctx.beginPath(); ctx.arc(pf.x, pf.y, 10, 0, 7); ctx.fill();
      if (pf.life <= 0) puffs.splice(q, 1);
    }
    // 枪口闪光
    if (flash > 0) {
      flash -= 1 / 60;
      ctx.fillStyle = "rgba(255,220,120,0.8)";
      ctx.beginPath(); ctx.arc(player.x + Math.cos(ang) * 28, player.y + Math.sin(ang) * 28, 6, 0, 7); ctx.fill();
    }
    // 准星
    ctx.strokeStyle = "#e8d9a8"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(mouse.x - 8, mouse.y); ctx.lineTo(mouse.x - 3, mouse.y);
    ctx.moveTo(mouse.x + 3, mouse.y); ctx.lineTo(mouse.x + 8, mouse.y);
    ctx.moveTo(mouse.x, mouse.y - 8); ctx.lineTo(mouse.x, mouse.y - 3);
    ctx.moveTo(mouse.x, mouse.y + 3); ctx.lineTo(mouse.x, mouse.y + 8);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  var last = performance.now();
  function loop(now) {
    var dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    update(dt); draw();
    requestAnimationFrame(loop);
  }
  spawnWave();
  requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""

def build_battlefield_html(params):
    """战地一原型：params 战地 schema 数值 → 完整可玩 HTML。非法/缺省用安全默认。"""
    p = dict(_BF_DEFAULTS)
    for k in p:
        v = params.get(k) if isinstance(params.get(k), (int, float)) else None
        if v is not None and not isinstance(v, bool) and v > 0:
            p[k] = float(v) if k.endswith("_s") else int(v)
    html = _BF_TEMPLATE
    for k, v in p.items():
        html = html.replace("@@%s@@" % k.upper(), str(v))
    return html


def load_design_params(path):
    """从策划的 design.json 读数值（缺失键用默认值）。"""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# 回合制 RPG 原型生成器（勇者斗恶龙式：地图探索 + 遇敌 + 回合制战斗 + 升级 + 剧情）
# 数值（HP/MP/攻击/防御/经验曲线/法术消耗）由策划 design.json 注入。
# 世界观文案（标题/敌人名/法术名/剧情/胜利文本）由主题 theme_meta 注入。
# ======================================================================
_RPG_DEFAULTS = {
    "player_hp": 80, "player_mp": 30, "player_atk": 12, "player_def": 4,
    "enemy_hp": 30, "enemy_atk": 8, "boss_hp": 120, "boss_atk": 16,
    "exp_to_level": 50, "heal_cost": 6, "fire_cost": 8, "potion_count": 3,
}

_RPG_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
  body { margin:0; background:#141821; display:flex; justify-content:center;
         align-items:center; min-height:100vh;
         font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif; color:#e8e6e0; }
  #game { width:720px; background:#1b2029; border:1px solid #2a3040;
          border-radius:14px; padding:16px; box-shadow:0 8px 30px rgba(0,0,0,.5); }
  #status { display:flex; gap:14px; flex-wrap:wrap; font-size:13px; color:#c9c4b8;
            background:#222836; border-radius:10px; padding:10px 16px; margin-bottom:12px; }
  #status b { color:#e8d9a8; }
  .bar { width:120px; height:10px; background:#3a3f47; border-radius:5px;
         overflow:hidden; display:inline-block; vertical-align:middle; margin:0 4px; }
  .bar i { display:block; height:100%; background:#8a9a4a; border-radius:5px; }
  .bar i.hp { background:#c25b4e; }
  .bar i.mp { background:#4e7fb5; }
  #screen { display:flex; gap:12px; min-height:300px; }
  #mapwrap { flex:1; }
  #map { display:grid; gap:2px; }
  .cell { aspect-ratio:1; display:flex; align-items:center; justify-content:center;
          font-size:16px; background:#2a3040; border-radius:4px; }
  .cell.wall { background:#3a3f47; }
  .cell.player { background:#4a5d40; }
  .cell.enemy { background:#5d3a3a; }
  .cell.boss { background:#6a3a5d; }
  .cell.chest { background:#6b5d20; }
  .cell.exit { background:#4a5d40; }
  #battle { flex:1; background:#222836; border-radius:10px; padding:12px;
           display:flex; flex-direction:column; }
  #enemy { text-align:center; font-size:48px; margin:8px 0; }
  #log { flex:1; background:#141821; border-radius:8px; padding:10px; overflow:auto;
         font-size:13px; line-height:1.7; min-height:120px; }
  #menu { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
  #menu button { padding:9px; font-size:14px; background:#2a3040; color:#e8e6e0;
                 border:1px solid #3a4455; border-radius:8px; cursor:pointer; }
  #menu button:hover { background:#35405a; }
  #menu button:disabled { opacity:.4; cursor:default; }
  #intro { text-align:center; padding:40px 20px; }
  #intro h2 { font-size:22px; color:#e8d9a8; margin:0 0 16px; }
  #intro p { font-size:14px; line-height:1.8; color:#c9c4b8; }
  #intro button { margin-top:20px; padding:10px 26px; font-size:15px; background:#2f6feb;
                  color:#fff; border:none; border-radius:8px; cursor:pointer; }
  .hint { text-align:center; font-size:12px; color:#8b90a0; margin-top:10px; }
</style>
</head>
<body>
<div id="game">
  <div id="status">
    <span>@@TITLE@@</span>
    <span>Lv <b id="lv">1</b></span>
    <span>HP <b id="hp">0/0</b><span class="bar"><i class="hp" id="hpbar"></i></span></span>
    <span>MP <b id="mp">0/0</b><span class="bar"><i class="mp" id="mpbar"></i></span></span>
    <span>EXP <b id="exp">0/0</b></span>
    <span>金币 <b id="gold">0</b></span>
  </div>
  <div id="screen">
    <div id="intro">
      <h2>@@TITLE@@</h2>
      <p>@@STORY_INTRO@@</p>
      <button onclick="startGame()">开始冒险</button>
    </div>
    <div id="mapwrap" style="display:none;"><div id="map"></div></div>
    <div id="battle" style="display:none;">
      <div id="enemy">👾</div>
      <div id="log"></div>
      <div id="menu"></div>
    </div>
  </div>
  <div class="hint">@@HUD_TIP@@</div>
</div>
<script>
(function () {
  "use strict";
  // ===== 策划数值注入（命名变量，便于校验器静态断言已注入） =====
  var PLAYER_HP = @@PLAYER_HP@@, PLAYER_MP = @@PLAYER_MP@@,
      PLAYER_ATK = @@PLAYER_ATK@@, PLAYER_DEF = @@PLAYER_DEF@@,
      ENEMY_HP = @@ENEMY_HP@@, ENEMY_ATK = @@ENEMY_ATK@@,
      BOSS_HP = @@BOSS_HP@@, BOSS_ATK = @@BOSS_ATK@@,
      EXP_TO_LEVEL = @@EXP_TO_LEVEL@@, HEAL_COST = @@HEAL_COST@@,
      FIRE_COST = @@FIRE_COST@@, POTION_COUNT = @@POTION_COUNT@@;
  var CFG = {
    player_hp: PLAYER_HP, player_mp: PLAYER_MP,
    player_atk: PLAYER_ATK, player_def: PLAYER_DEF,
    enemy_hp: ENEMY_HP, enemy_atk: ENEMY_ATK,
    boss_hp: BOSS_HP, boss_atk: BOSS_ATK,
    exp_to_level: EXP_TO_LEVEL, heal_cost: HEAL_COST,
    fire_cost: FIRE_COST, potion_count: POTION_COUNT,
  };
  var TXT = { enemy: "@@ENEMY_NAME@@", boss: "@@BOSS_NAME@@",
              fire: "@@SPELL_NAME@@", heal: "@@HEAL_NAME@@", win: "@@WIN_TEXT@@" };

  var MAP = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,2,0,0,0,3,0,1],
    [1,0,1,0,0,1,0,0,0,1],
    [1,0,1,0,0,1,0,2,0,1],
    [1,0,0,0,1,0,0,0,1,1],
    [1,0,1,0,0,0,1,0,0,1],
    [1,0,1,0,2,0,0,0,3,1],
    [1,0,0,0,0,1,0,0,0,1],
    [1,4,0,0,0,0,0,0,5,1],
  ];
  var W = MAP[0].length, H = MAP.length;
  var player = { x:1, y:8, hp:CFG.player_hp, maxhp:CFG.player_hp,
                 mp:CFG.player_mp, maxmp:CFG.player_mp, lv:1, exp:0,
                 atk:CFG.player_atk, def:CFG.player_def, gold:0,
                 potions:CFG.potion_count, dead:false };
  var enemies = {};   // key "x,y" -> {name,hp,maxhp,atk,boss,dead}
  var inBattle = false, currentEnemy = null, won = false;

  function addLog(t) {
    var el = document.getElementById('log');
    el.innerHTML += '<div>' + t + '</div>';
    el.scrollTop = el.scrollHeight;
  }
  function updateStatus() {
    document.getElementById('lv').textContent = player.lv;
    document.getElementById('hp').textContent = Math.max(0,player.hp) + '/' + player.maxhp;
    document.getElementById('hpbar').style.width = Math.max(0, player.hp/player.maxhp*100) + '%';
    document.getElementById('mp').textContent = player.mp + '/' + player.maxmp;
    document.getElementById('mpbar').style.width = Math.max(0, player.mp/player.maxmp*100) + '%';
    document.getElementById('exp').textContent = player.exp + '/' + CFG.exp_to_level;
    document.getElementById('gold').textContent = player.gold;
  }
  function renderMap() {
    var m = document.getElementById('map');
    m.style.gridTemplateColumns = 'repeat(' + W + ', 1fr)';
    m.innerHTML = '';
    for (var y=0; y<H; y++) for (var x=0; x<W; x++) {
      var c = document.createElement('div'); c.className = 'cell';
      var t = MAP[y][x];
      if (t===1) c.className += ' wall';
      if (x===player.x && y===player.y) { c.className += ' player'; c.textContent = '@'; }
      else if (t===2 && !enemies[x+','+y]) c.textContent = '👾';
      else if (t===3 && !enemies[x+','+y]) c.textContent = '💎';
      else if (t===4) { c.className += ' exit'; c.textContent = '🚪'; }
      else if (t===5) { c.className += ' boss'; c.textContent = '🐉'; }
      m.appendChild(c);
    }
  }
  function move(dx,dy) {
    if (inBattle || won) return;
    var nx = player.x+dx, ny = player.y+dy;
    if (nx<0||ny<0||nx>=W||ny>=H) return;
    var t = MAP[ny][nx];
    if (t===1) return;
    player.x=nx; player.y=ny;
    if (t===2 && !enemies[nx+','+ny]) startBattle(TXT.enemy, false);
    else if (t===3 && !enemies[nx+','+ny]) { player.gold += 30; enemies[nx+','+ny]=true;
      addLog('发现宝箱，获得 30 金币！'); }
    else if (t===5 && !enemies[nx+','+ny]) startBattle(TXT.boss, true);
    else if (t===4) { enemies[nx+','+ny]=true; victory(); }
    renderMap(); updateStatus();
  }
  function showBattle(show) {
    document.getElementById('battle').style.display = show ? 'flex' : 'none';
    document.getElementById('mapwrap').style.display = show ? 'none' : 'block';
  }
  function startBattle(name, boss) {
    inBattle = true; currentEnemy = { name:name, boss:boss,
      hp: boss?CFG.boss_hp:CFG.enemy_hp, maxhp: boss?CFG.boss_hp:CFG.enemy_hp,
      atk: boss?CFG.boss_atk:CFG.enemy_atk };
    document.getElementById('enemy').textContent = boss ? '🐉' : '👾';
    document.getElementById('log').innerHTML = '';
    addLog('遭遇了 ' + name + '！');
    showBattle(true); renderMenu(); updateStatus();
  }
  function renderMenu() {
    var m = document.getElementById('menu');
    m.innerHTML = '';
    [['攻击','a'],['法术','s'],['道具','i'],['逃跑','f']].forEach(function(k){
      var b=document.createElement('button'); b.textContent=k[0];
      b.onclick=function(){ act(k[1]); }; m.appendChild(b);
    });
  }
  function act(key) {
    if (key==='a') { var d = Math.max(1, player.atk - (currentEnemy.boss?2:0) + (Math.random()*4|0));
      hit(d, '你攻击 ' + currentEnemy.name + '，造成 ' + d + ' 点伤害'); }
    else if (key==='s') spellMenu();
    else if (key==='i') { if (player.potions>0) { player.potions--; player.hp=Math.min(player.maxhp, player.hp+40);
      addLog('使用药水，恢复 40 HP（剩 ' + player.potions + ' 个）'); enemyTurn(); }
      else { addLog('没有药水了'); return; } }
    else if (key==='f') { if (Math.random()<0.5) { addLog('成功逃脱！'); inBattle=false;
      showBattle(false); renderMap(); return; } else { addLog('逃跑失败！'); enemyTurn(); } }
    updateStatus();
  }
  function hit(d, msg) { currentEnemy.hp -= d; addLog(msg);
    if (currentEnemy.hp<=0) { winBattle(); return; } enemyTurn(); }
  function spellMenu() {
    var m = document.getElementById('menu'); m.innerHTML='';
    [['火球术('+CFG.fire_cost+'MP)','fire'],['治愈术('+CFG.heal_cost+'MP)','heal'],['返回','back']].forEach(function(k){
      var b=document.createElement('button'); b.textContent=k[0];
      b.onclick=function(){ spell(k[1]); }; m.appendChild(b);
    });
  }
  function spell(k) {
    if (k==='back') { renderMenu(); return; }
    if (k==='fire') { if (player.mp<CFG.fire_cost) { addLog('MP 不足'); return; }
      player.mp-=CFG.fire_cost; var d = Math.round(player.atk*1.6 + (Math.random()*4));
      hit(d, TXT.fire + ' 造成 ' + d + ' 点伤害'); }
    else if (k==='heal') { if (player.mp<CFG.heal_cost) { addLog('MP 不足'); return; }
      player.mp-=CFG.heal_cost; player.hp=Math.min(player.maxhp, player.hp+25);
      addLog(TXT.heal + ' 恢复 25 HP'); enemyTurn(); }
    updateStatus();
  }
  function enemyTurn() {
    var d = Math.max(1, currentEnemy.atk - player.def + (Math.random()*3|0));
    player.hp -= d; addLog(currentEnemy.name + ' 攻击，造成 ' + d + ' 点伤害');
    if (player.hp<=0) { player.hp=0; addLog('你被击败了……'); defeat(); }
    updateStatus();
  }
  function winBattle() {
    var exp = currentEnemy.boss ? CFG.exp_to_level : Math.round(CFG.exp_to_level*0.4);
    var gold = currentEnemy.boss ? 100 : 20;
    player.exp += exp; player.gold += gold;
    addLog('击败 ' + currentEnemy.name + '！获得 ' + exp + ' EXP、' + gold + ' 金币');
    if (currentEnemy.boss) { enemies[player.x+','+player.y]=true; }
    else enemies[player.x+','+player.y]=true;
    inBattle=false; showBattle(false);
    while (player.exp>=CFG.exp_to_level) { levelUp(); }
    renderMap(); updateStatus();
  }
  function levelUp() {
    player.exp -= CFG.exp_to_level; player.lv++;
    player.maxhp += 12; player.maxmp += 6; player.atk += 3; player.def += 1;
    player.hp = player.maxhp; player.mp = player.maxmp;
    addLog('升级！现在是 Lv ' + player.lv + '，能力提升了');
  }
  function defeat() {
    inBattle=false; document.getElementById('menu').innerHTML='';
    var l=document.getElementById('log');
    l.innerHTML = '<div>你被击败了…… 按「重新开始」再试一次</div>';
    var b=document.createElement('button'); b.textContent='重新开始';
    b.onclick=function(){ location.reload(); };
    document.getElementById('menu').appendChild(b);
  }
  function victory() {
    won=true; showBattle(false);
    document.getElementById('intro').style.display='block';
    document.getElementById('intro').innerHTML =
      '<h2>胜利！</h2><p>@@WIN_TEXT@@</p>' +
      '<p>Lv ' + player.lv + ' · 金币 ' + player.gold + '</p>';
    document.getElementById('mapwrap').style.display='none';
  }
  window.startGame = function () {
    document.getElementById('intro').style.display='none';
    document.getElementById('mapwrap').style.display='block';
    renderMap(); updateStatus();
  };
  document.addEventListener('keydown', function (e) {
    if (e.code==='ArrowUp'||e.code==='KeyW') move(0,-1);
    else if (e.code==='ArrowDown'||e.code==='KeyS') move(0,1);
    else if (e.code==='ArrowLeft'||e.code==='KeyA') move(-1,0);
    else if (e.code==='ArrowRight'||e.code==='KeyD') move(1,0);
    else if (e.code==='Space'||e.code==='KeyJ') { if (inBattle) act('a'); e.preventDefault(); }
  });
  updateStatus();
})();
</script>
</body>
</html>
"""

# RPG 主题化文案默认值（theme_meta 未提供时使用）
_RPG_THEME_DEFAULTS = {
    "title": "勇者斗恶龙 · 回合制 RPG",
    "enemy_name": "史莱姆", "boss_name": "魔王",
    "spell_name": "火球术", "heal_name": "治愈术",
    "story_intro": "黑暗笼罩大地，魔王复活了。你作为被选中的勇者，踏上讨伐魔王的旅程。\n在地图中探索，击败魔物、收集金币、提升等级，最终在出口挑战魔王！",
    "win_text": "你击败了魔王，世界恢复了和平！",
    "hud_tip": "方向键 / WASD 移动 · 踩到敌人进入战斗 · 空格攻击 · 走到出口挑战魔王",
}


def build_rpg_html(params, theme_meta=None):
    """回合制 RPG 原型：params 为 RPG schema 数值，theme_meta 提供世界观文案。
    非法/缺省数值用安全默认值；未提供主题文案用 _RPG_THEME_DEFAULTS。"""
    p = dict(_RPG_DEFAULTS)
    for k in p:
        v = params.get(k) if isinstance(params.get(k), (int, float)) else None
        if v is not None and not isinstance(v, bool) and v > 0:
            p[k] = int(v)
    # 主题文案：优先 theme_meta.game_text 块，其次顶层字段，最后默认
    tm = theme_meta if isinstance(theme_meta, dict) else {}
    gt = tm.get("game_text") if isinstance(tm.get("game_text"), dict) else {}
    txt = dict(_RPG_THEME_DEFAULTS)
    for k in txt:
        v = gt.get(k) or tm.get(k)
        if isinstance(v, str) and v.strip():
            txt[k] = v.strip()
    html = _RPG_TEMPLATE
    for k, v in p.items():
        html = html.replace("@@%s@@" % k.upper(), str(v))
    for k, v in txt.items():
        html = html.replace("@@%s@@" % k.upper(), v)
    return html
