# -*- coding: utf-8 -*-
"""平台跳跃游戏模板：骨架 + 系统模块（模块化架构）。
- _GAME_SKELETON：玩家/物理/输入/循环/绘制框架 + __HARNESS 共享上下文 + 模块挂载点
- _SYS_MODULES：7 个玩法系统模块（注册式，通过 __HARNESS 访问骨架能力）
- 系统任务产出 = 单个模块代码（LLM 生成，模板兜底）；集成任务 = 骨架 + 各模块组装
"""

_GAME_SKELETON = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@THEME_TITLE@@</title>
<style>
  body { margin:0; background:#f6f8fa; display:flex; flex-direction:column;
         align-items:center; justify-content:center; min-height:100vh;
         font-family:-apple-system,'Segoe UI','Microsoft YaHei',sans-serif; color:#1f2328; }
  canvas { border:2px solid #d0d7de; border-radius:12px; background:#5c94fc;
           box-shadow:0 4px 16px rgba(0,0,0,.08); image-rendering:pixelated; }
  #hud { margin-top:10px; font-size:14px; color:#5f5e5a; text-align:center; line-height:1.7; }
  #hud b { color:#0c447c; }
  .win { color:#3b6d11; font-weight:500; }
  .lose { color:#c0392b; font-weight:500; }
</style>
</head>
<body>
<canvas id="game" width="640" height="360"></canvas>
<div id="hud">
  @@HUD_TIP@@<br>
  <span id="score">分数 0</span> · <span id="lives">生命 3</span> · <span id="state"></span>
</div>
<script>
(function () {
  "use strict";
  // ===== 策划数值注入（design.json）=====
  var MOVE_SPEED = @@MOVE_SPEED@@;   // 移速 px/s
  var JUMP_VEL = @@JUMP_VEL@@;       // 跳跃初速
  var GRAVITY = @@GRAVITY@@;         // 重力加速度

  // ===== 调研功能清单注入（research.features）：各系统独立开关 =====
  var FEATURES = @@FEATURES@@;
  var ENABLE_ENEMY = FEATURES.enemy !== false;
  var ENABLE_BRICK = FEATURES.brick !== false;
  var ENABLE_MUSHROOM = FEATURES.mushroom !== false;
  var ENABLE_PIPE = FEATURES.pipe !== false;
  var ENABLE_COIN = FEATURES.coin !== false;
  var ENABLE_LIFE = FEATURES.life !== false;
  var ENABLE_FLAG = FEATURES.flag !== false;

  // ===== 场景配色注入（创作层按主题提供；缺省 = 马里奥经典配色） =====
  var SCENE = @@SCENE@@;

  var cv = document.getElementById("game");
  var ctx = cv.getContext("2d");
  var scoreEl = document.getElementById("score");
  var livesEl = document.getElementById("lives");
  var stateEl = document.getElementById("state");

  var W = cv.width, H = cv.height, GROUND = H - 24;
  var keys = {};
  var state = "playing";            // playing | won | gameover
  var score = 0, lives = ENABLE_LIFE ? 3 : 0, coinCount = 0;
  var big = false, invuln = 0, dying = 0;   // invuln=受伤无敌帧; dying=角色消失动画
  var particles = [];

  // ---- 玩家（小 14x16 / 大 18x28）----
  var player = { x: 48, y: GROUND - 16, w: 14, h: 16, vx: 0, vy: 0, face: 1 };

  // ---- 平台：地面 + 浮动平台（骨架基础碰撞体）----
  var platforms = [
    { x: 0, y: GROUND, w: W, h: 24 },
    { x: 120, y: 292, w: 100, h: 12 },
    { x: 300, y: 250, w: 110, h: 12 },
    { x: 500, y: 208, w: 100, h: 12 }
  ];

  // ---- 管道（基础碰撞体；pipe 模块负责绘制）----
  var pipes = [];
  if (ENABLE_PIPE) {
    pipes = [
      { x: 60, y: GROUND - 32, w: 32, h: 32 },
      { x: 420, y: GROUND - 48, w: 32, h: 48 }
    ];
  }

  // ---- 终点旗杆（flag 模块消费）----
  var flag = { x: 602, poleTop: GROUND - 88, poleH: 88 };

  // ---- 简单音效（WebAudio 振荡器）----
  var actx = null;
  function sfx(freq, dur, type, slide) {
    try {
      if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
      var o = actx.createOscillator(), g = actx.createGain();
      o.type = type || "square"; o.frequency.value = freq;
      g.gain.setValueAtTime(0.08, actx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + dur);
      o.connect(g); g.connect(actx.destination);
      if (slide) o.frequency.exponentialRampToValueAtTime(slide, actx.currentTime + dur);
      o.start(); o.stop(actx.currentTime + dur);
    } catch (e) {}
  }
  var sfxJump = function () { sfx(320, 0.12, "square", 520); };
  var sfxCoin = function () { sfx(988, 0.09, "square", 1319); };
  var sfxStomp = function () { sfx(240, 0.14, "sawtooth", 90); };
  var sfxBlock = function () { sfx(140, 0.1, "square"); };
  var sfxMushroom = function () { sfx(392, 0.2, "square", 784); };
  var sfxHurt = function () { sfx(400, 0.25, "sawtooth", 120); };
  var sfxDie = function () { sfx(500, 0.4, "sawtooth", 80); };
  var sfxWin = function () {
    sfx(523, 0.15, "square"); setTimeout(function () { sfx(659, 0.15, "square"); }, 150);
    setTimeout(function () { sfx(784, 0.3, "square"); }, 300);
  };

  document.addEventListener("keydown", function (e) {
    keys[e.code] = true;
    if (e.code === "Space") e.preventDefault();
    if ((state === "gameover" || state === "won") && e.code === "KeyR") location.reload();
  });
  document.addEventListener("keyup", function (e) { keys[e.code] = false; });

  // ---- 基础工具 ----
  function rect(x, y, w, h, color) {
    ctx.fillStyle = color; ctx.fillRect(Math.round(x), Math.round(y), w, h);
  }
  function overlap(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  }
  function solidBoxes() {
    return platforms.concat(pipes);
  }
  function onGround() {
    for (var i = 0; i < platforms.length; i++) {
      var p = platforms[i];
      if (player.x + player.w > p.x && player.x < p.x + p.w &&
          Math.abs(player.y + player.h - p.y) < 4) return true;
    }
    return false;
  }
  function resetPlayer() {
    player.x = 48; player.y = GROUND - player.h;
    player.vx = 0; player.vy = 0; invuln = 1.2; dying = 0;
  }
  function addScore(n, sx, sy) {
    score += n;
    particles.push({ x: sx || player.x, y: sy || player.y - 10, vy: -60, life: 0.8, text: "+" + n });
  }
  function loseLife() {
    lives -= 1;
    livesEl.textContent = "生命 " + Math.max(0, lives);
    sfxDie();
    if (lives <= 0) {
      state = "gameover";
      stateEl.innerHTML = '<span class="lose">游戏结束！得分 ' + score + ' · 按 R 重来</span>';
    } else {
      dying = 0.9;
      setTimeout(resetPlayer, 600);
    }
  }
  function hurtPlayer() {
    if (!ENABLE_LIFE || invuln > 0 || dying > 0 || state !== "playing") return;
    if (big) {
      big = false;
      player.h = 16; player.w = 14;
      player.y += 12;
      invuln = 1.5;
      sfxHurt();
    } else {
      loseLife();
    }
  }

  // ---- 主循环 ----
  function update(dt) {
    if (state !== "playing") return;
    if (dying > 0) { dying -= dt; return; }
    if (invuln > 0) invuln -= dt;

    // 输入
    var left = keys["ArrowLeft"] || keys["KeyA"];
    var right = keys["ArrowRight"] || keys["KeyD"];
    player.vx = 0;
    if (left) { player.vx = -MOVE_SPEED; player.face = -1; }
    if (right) { player.vx = MOVE_SPEED; player.face = 1; }
    if ((keys["Space"] || keys["ArrowUp"] || keys["KeyW"]) && onGround()) {
      player.vy = -JUMP_VEL;
      sfxJump();
    }

    // 物理（x / y 分离碰撞 vs 平台+管道）
    player.vy += GRAVITY * dt;
    player.x += player.vx * dt;
    var solids = solidBoxes();
    for (var i = 0; i < solids.length; i++) {
      var s = solids[i];
      if (overlap(player, s)) {
        if (player.vx > 0) player.x = s.x - player.w;
        else if (player.vx < 0) player.x = s.x + s.w;
      }
    }
    player.y += player.vy * dt;
    for (var j = 0; j < solids.length; j++) {
      var s2 = solids[j];
      if (overlap(player, s2)) {
        if (player.vy > 0) {
          player.y = s2.y - player.h;
          player.vy = 0;
        } else if (player.vy < 0) {
          player.y = s2.y + s2.h;
          player.vy = 0;
        }
      }
    }

    // 边界
    if (player.x < 0) player.x = 0;
    if (player.x > W - player.w) player.x = W - player.w;
    if (player.y > H + 20) {
      if (ENABLE_LIFE) { loseLife(); } else { resetPlayer(); }
      return;
    }
    scoreEl.textContent = "分数 " + score;

    // 系统模块 update（敌人/顶砖/蘑菇/金币/旗杆等，由各自任务实现）
    for (var k in __SYS) { if (__SYS[k].update) __SYS[k].update(dt); }

    // 粒子
    for (var p = particles.length - 1; p >= 0; p--) {
      var pt = particles[p];
      pt.y += pt.vy * dt; pt.vy *= 0.9; pt.life -= dt;
      if (pt.life <= 0) particles.splice(p, 1);
    }
  }

  function draw() {
    // 天空与远景（场景配色由创作层注入）
    rect(0, 0, W, H, SCENE.sky);
    rect(70, 46, 46, 14, SCENE.cloud); rect(130, 70, 34, 12, SCENE.cloud);
    rect(560, 60, 40, 12, SCENE.cloud);
    rect(0, GROUND - 40, 200, 40, SCENE.hill);
    rect(150, GROUND - 60, 140, 60, SCENE.hill2);
    rect(400, GROUND - 46, 180, 46, SCENE.hill);
    // 平台
    for (var i = 0; i < platforms.length; i++) {
      var p = platforms[i];
      rect(p.x, p.y, p.w, p.h, SCENE.ground);
      rect(p.x, p.y, p.w, 4, SCENE.ground_top);
      for (var tx = p.x; tx < p.x + p.w; tx += 12) {
        ctx.fillStyle = SCENE.ground_dot; ctx.fillRect(tx + 6, p.y + 10, 2, 2);
      }
    }
    // 系统模块 draw
    for (var k in __SYS) { if (__SYS[k].draw) __SYS[k].draw(); }
    // 玩家
    drawPlayer();
    // 粒子
    for (var p2 = particles.length - 1; p2 >= 0; p2--) {
      var pt2 = particles[p2];
      ctx.fillStyle = "rgba(60,60,60," + Math.max(0, pt2.life) + ")";
      ctx.font = "bold 10px monospace";
      ctx.fillText(pt2.text, pt2.x, pt2.y);
    }
    if (state === "playing") {
      stateEl.textContent = "@@STATE_GOAL@@";
    }
  }

  function drawPlayer() {
    if (dying > 0 && Math.floor(dying * 10) % 2 === 0) return;
    if (invuln > 0 && Math.floor(invuln * 12) % 2 === 0) return;
    var px = player.x, py = player.y;
    if (big) {
      rect(px + 2, py + 1, 14, 6, SCENE.player_hat);
      rect(px + 1, py + 7, 16, 8, SCENE.player_skin);
      rect(px + 4, py + 9, 3, 3, "#2c2c2a");
      rect(px + 11, py + 9, 3, 3, "#2c2c2a");
      rect(px + 3, py + 14, 5, 4, SCENE.player_boot);
      rect(px + 10, py + 14, 5, 4, SCENE.player_boot);
      rect(px + 1, py + 15, 16, 9, SCENE.player_pants);
      rect(px + 4, py + 15, 4, 4, SCENE.player_hat);
      rect(px + 10, py + 15, 4, 4, SCENE.player_hat);
      rect(px + 2, py + 24, 5, 4, SCENE.player_boot);
      rect(px + 11, py + 24, 5, 4, SCENE.player_boot);
    } else {
      rect(px + 1, py + 1, 12, 5, SCENE.player_hat);
      rect(px + 2, py + 6, 12, 6, SCENE.player_skin);
      rect(px + 3, py + 8, 2, 2, "#2c2c2a");
      rect(px + 8, py + 8, 2, 2, "#2c2c2a");
      rect(px + 2, py + 12, 5, 3, SCENE.player_pants);
      rect(px + 7, py + 12, 5, 3, SCENE.player_pants);
      rect(px + 1, py + 15, 4, 1, SCENE.player_boot);
      rect(px + 9, py + 15, 4, 1, SCENE.player_boot);
    }
  }

  // ---- 模块注册表（各系统模块通过 __HARNESS.register 挂载）----
  var __SYS = {};
  // 跨系统状态操作（骨架闭包持有真实状态，模块通过接口调用）
  function coinAdjust(n) {
    coinCount += n;
    scoreEl.textContent = "分数 " + score;
  }
  function setBig() {
    if (!big) {
      big = true;
      player.w = 18; player.h = 28;
      player.y -= 12;
    }
  }
  function doWin() {
    if (state !== "playing") return;
    state = "won";
    scoreEl.textContent = "分数 " + score;
    stateEl.innerHTML = '<span class="win">@@WIN_TEXT@@得分 ' + score +
      ' · 金币 ' + coinCount + ' · 按 R 再来一局</span>';
    sfxWin();
  }
  window.__HARNESS = {
    FEATURES: FEATURES,
    SCENE: SCENE,
    ctx: ctx,
    W: W, H: H, GROUND: GROUND, GRAVITY: GRAVITY, JUMP_VEL: JUMP_VEL, MOVE_SPEED: MOVE_SPEED,
    player: player, platforms: platforms, pipes: pipes, flag: flag,
    solidBoxes: solidBoxes, onGround: onGround, overlap: overlap, rect: rect,
    addScore: addScore, hurtPlayer: hurtPlayer, loseLife: loseLife,
    coinAdjust: coinAdjust, setBig: setBig, doWin: doWin,
    sfx: { jump: sfxJump, coin: sfxCoin, stomp: sfxStomp, block: sfxBlock,
           mushroom: sfxMushroom, hurt: sfxHurt, die: sfxDie, win: sfxWin },
    register: function (n, m) { __SYS[n] = m; },
    _debug: {
      keys: keys,
      state: function () {
        return { score: score, coinCount: coinCount, lives: lives,
                 state: state, big: big, player: player, flag: flag };
      },
      sys: function (n) { return __SYS[n]; },
      step: function (dt) { update(dt); draw(); },
      reset: function () {
        // 冒烟隔离：重置玩家/分数/状态（模块数据保留，便于逐步断言）
        state = "playing";
        lives = ENABLE_LIFE ? 3 : 0;
        score = 0; coinCount = 0;
        big = false; invuln = 0; dying = 0;
        resetPlayer();
      }
    }
  };

  /*__SYS_SCRIPTS__*/

  // 初始化系统模块（各模块的 init 填充数据）
  for (var k2 in __SYS) { if (__SYS[k2].init) __SYS[k2].init(); }

  function loop() {
    update(1 / 60);
    draw();
    requestAnimationFrame(loop);
  }
  loop();
})();
</script>
</body>
</html>

"""

_SYS_ORDER = ['enemy', 'brick', 'mushroom', 'pipe', 'coin', 'life', 'flag']

_SYS_MODULES = {
    'enemy': """/*==MODULE:enemy==*/
(function () {
  if (!window.__HARNESS) return;
  var H = window.__HARNESS;
  var ENABLE = H.FEATURES.enemy !== false;
  var enemies = [];
  function drawEnemy(en) {
    if (!en.alive) return;
    H.rect(en.x + 1, en.y, 12, 6, H.SCENE.enemy_body);
    H.rect(en.x, en.y + 5, 14, 4, H.SCENE.enemy_shell);
    H.rect(en.x + 2, en.y + 9, 10, 4, H.SCENE.enemy_face);
    H.rect(en.x + 3, en.y + 13, 8, 1, "#2c2c2a");
    H.rect(en.x + 3, en.y + 7, 2, 2, "#2c2c2a");
    H.rect(en.x + 9, en.y + 7, 2, 2, "#2c2c2a");
  }
  H.register("enemy", {
    init: function () {
      if (!ENABLE) return;
      enemies = [
        { x: 200, y: H.GROUND - 14, w: 14, h: 14, vx: 40, alive: true },
        { x: 340, y: H.GROUND - 14, w: 14, h: 14, vx: -45, alive: true },
        { x: 480, y: H.GROUND - 14, w: 14, h: 14, vx: 50, alive: true }
      ];
    },
    update: function (dt) {
      if (!ENABLE) return;
      var solids = H.solidBoxes();
      for (var e = enemies.length - 1; e >= 0; e--) {
        var en = enemies[e];
        if (!en.alive) continue;
        en.x += en.vx * dt;
        for (var se = 0; se < solids.length; se++) {
          if (H.overlap({ x: en.x, y: en.y, w: en.w, h: en.h }, solids[se])) {
            if (en.vx > 0) en.x = solids[se].x - en.w; else en.x = solids[se].x + solids[se].w;
            en.vx = -en.vx;
          }
        }
        if (en.x < 0) { en.x = 0; en.vx = Math.abs(en.vx); }
        if (en.x > H.W - en.w) { en.x = H.W - en.w; en.vx = -Math.abs(en.vx); }
        en.vy = (en.vy || 0) + H.GRAVITY * dt;
        en.y += en.vy * dt;
        for (var eg = 0; eg < solids.length; eg++) {
          var s3 = solids[eg];
          if (H.overlap({ x: en.x, y: en.y, w: en.w, h: en.h }, s3) && en.vy >= 0) {
            en.y = s3.y - en.h; en.vy = 0;
          }
        }
        if (H.overlap({ x: en.x, y: en.y, w: en.w, h: en.h },
                      { x: H.player.x, y: H.player.y, w: H.player.w, h: H.player.h })) {
          var stomp = H.player.vy > 0 && (H.player.y + H.player.h - en.y) < 10;
          if (stomp) {
            en.alive = false;
            H.player.vy = -H.JUMP_VEL * 0.6;
            H.addScore(100, en.x, en.y);
            H.sfx.stomp();
          } else {
            H.hurtPlayer();
          }
        }
      }
    },
    draw: function () {
      if (!ENABLE) return;
      for (var e = 0; e < enemies.length; e++) drawEnemy(enemies[e]);
    },
    debug: function () {
      return { alive: enemies.filter(function (e) { return e.alive; }).length,
               total: enemies.length,
               first: enemies.length ? { x: enemies[0].x, y: enemies[0].y } : null };
    }
  });
})();
/*==END:enemy==*/""",
    'brick': """/*==MODULE:brick==*/
(function () {
  if (!window.__HARNESS) return;
  var H = window.__HARNESS;
  var ENABLE = H.FEATURES.brick !== false;
  var blocks = [];
  function hitBlock(blk) {
    if (blk.used) { H.sfx.block(); return; }
    blk.used = true;
    H.sfx.block();
    if (blk.kind === "coin") {
      H.coinAdjust(1);
      H.addScore(100, blk.x + 12, blk.y - 4);
      H.sfx.coin();
    } else if (blk.kind === "mushroom" && H.FEATURES.mushroom !== false) {
      if (H.spawnMushroom) H.spawnMushroom(blk.x, blk.y);
      H.sfx.mushroom();
    }
  }
  function drawBlock(blk) {
    var ctx = H.ctx;
    if (blk.used) {
      H.rect(blk.x, blk.y, blk.w, blk.h, H.SCENE.block_used);
      H.rect(blk.x + 4, blk.y + 4, blk.w - 8, blk.h - 8, H.SCENE.block_used_in);
      return;
    }
    H.rect(blk.x, blk.y, blk.w, blk.h, H.SCENE.block);
    H.rect(blk.x + 3, blk.y + 3, blk.w - 6, blk.h - 6, H.SCENE.block_hi);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 13px monospace";
    ctx.fillText("?", blk.x + 8, blk.y + 18);
  }
  H.register("brick", {
    init: function () {
      if (!ENABLE) return;
      blocks = [
        { x: 150, y: 250, w: 24, h: 24, used: false, kind: "coin" },
        { x: 178, y: 250, w: 24, h: 24, used: false, kind: "coin" },
        { x: 262, y: 210, w: 24, h: 24, used: false, kind: "mushroom" },
        { x: 330, y: 210, w: 24, h: 24, used: false, kind: "coin" },
        { x: 470, y: 168, w: 24, h: 24, used: false, kind: "coin" }
      ];
    },
    update: function (dt) {
      if (!ENABLE) return;
      if (H.player.vy < 0) {
        var head = { x: H.player.x + 3, y: H.player.y, w: H.player.w - 6, h: 4 };
        for (var hb = 0; hb < blocks.length; hb++) {
          var hblk = blocks[hb];
          if (H.overlap(head, hblk)) {
            H.player.y = hblk.y + hblk.h;
            H.player.vy = 0;
            hitBlock(hblk);
          }
        }
      }
    },
    draw: function () {
      if (!ENABLE) return;
      for (var b = 0; b < blocks.length; b++) drawBlock(blocks[b]);
    },
    debug: function () {
      return blocks.map(function (b) { return b.kind + (b.used ? ":used" : ""); });
    }
  });
})();
/*==END:brick==*/""",
    'mushroom': """/*==MODULE:mushroom==*/
(function () {
  if (!window.__HARNESS) return;
  var H = window.__HARNESS;
  var ENABLE = H.FEATURES.mushroom !== false;
  var mushroom = null;
  function spawn(x, y) {
    mushroom = { x: x, y: y - 14, vx: 60, vy: -220, active: true };
  }
  function drawMushroom() {
    if (!mushroom || !mushroom.active) return;
    H.rect(mushroom.x + 1, mushroom.y, 12, 5, H.SCENE.mush_cap);
    H.rect(mushroom.x, mushroom.y + 3, 14, 4, H.SCENE.mush_stem);
    H.rect(mushroom.x + 4, mushroom.y + 1, 2, 2, H.SCENE.mush_dot);
    H.rect(mushroom.x + 9, mushroom.y + 1, 2, 2, H.SCENE.mush_dot);
    H.rect(mushroom.x + 2, mushroom.y + 7, 10, 5, H.SCENE.mush_stem);
    H.rect(mushroom.x + 3, mushroom.y + 12, 8, 2, H.SCENE.mush_foot);
  }
  H.register("mushroom", {
    init: function () {
      mushroom = null;
      H.spawnMushroom = spawn;   // 供 brick 模块顶砖时调用
    },
    update: function (dt) {
      if (!ENABLE || !mushroom || !mushroom.active) return;
      mushroom.vy += H.GRAVITY * dt;
      mushroom.x += mushroom.vx * dt;
      mushroom.y += mushroom.vy * dt;
      var mbox = { x: mushroom.x, y: mushroom.y, w: 14, h: 14 };
      var solids = H.solidBoxes();
      for (var sm = 0; sm < solids.length; sm++) {
        var sms = solids[sm];
        if (H.overlap(mbox, sms)) {
          if (mushroom.vy > 0) { mushroom.y = sms.y - 14; mushroom.vy = 0; }
          else if (mushroom.vx !== 0) { mushroom.vx = -mushroom.vx; }
        }
      }
      if (mushroom.y > H.H + 20) { mushroom.active = false; mushroom = null; }
      if (mushroom && H.overlap(mbox,
          { x: H.player.x, y: H.player.y, w: H.player.w, h: H.player.h })) {
        H.setBig();
        H.addScore(100, mushroom.x, mushroom.y);
        mushroom.active = false;
        mushroom = null;
      }
    },
    draw: function () { if (ENABLE) drawMushroom(); },
    debug: function () {
      return mushroom ? { active: mushroom.active, x: mushroom.x, y: mushroom.y } : null;
    }
  });
})();
/*==END:mushroom==*/""",
    'pipe': """/*==MODULE:pipe==*/
(function () {
  if (!window.__HARNESS) return;
  var H = window.__HARNESS;
  var ENABLE = H.FEATURES.pipe !== false;
  H.register("pipe", {
    init: function () {},
    update: function () {},
    draw: function () {
      if (!ENABLE) return;
      for (var pi = 0; pi < H.pipes.length; pi++) {
        var pipe = H.pipes[pi];
        H.rect(pipe.x, pipe.y, pipe.w, pipe.h, H.SCENE.pipe);
        H.rect(pipe.x + 4, pipe.y, pipe.w - 8, pipe.h, H.SCENE.pipe_light);
        H.rect(pipe.x, pipe.y, pipe.w, 6, H.SCENE.pipe_hl);
        H.rect(pipe.x + 4, pipe.y + 2, pipe.w - 8, 4, H.SCENE.pipe_in);
      }
    },
    debug: function () { return H.pipes.length; }
  });
})();
/*==END:pipe==*/""",
    'coin': """/*==MODULE:coin==*/
(function () {
  if (!window.__HARNESS) return;
  var H = window.__HARNESS;
  var ENABLE = H.FEATURES.coin !== false;
  var coins = [];
  H.register("coin", {
    init: function () {
      if (!ENABLE) return;
      coins = [
        { x: 140, y: 322, r: 6 }, { x: 345, y: 224, r: 6 },
        { x: 512, y: 182, r: 6 }, { x: 240, y: 268, r: 6 }
      ];
    },
    update: function (dt) {
      if (!ENABLE) return;
      for (var c = coins.length - 1; c >= 0; c--) {
        var coin = coins[c];
        var dx = H.player.x + H.player.w / 2 - coin.x;
        var dy = H.player.y + H.player.h / 2 - coin.y;
        if (dx * dx + dy * dy < 324) {
          coins.splice(c, 1);
          H.coinAdjust(1);
          H.addScore(50, coin.x, coin.y);
          H.sfx.coin();
        }
      }
    },
    draw: function () {
      if (!ENABLE) return;
      var ctx = H.ctx;
      for (var c = 0; c < coins.length; c++) {
        var coin = coins[c];
        ctx.fillStyle = H.SCENE.coin;
        ctx.beginPath(); ctx.arc(coin.x, coin.y, coin.r, 0, 7); ctx.fill();
        ctx.fillStyle = H.SCENE.coin_hl;
        ctx.beginPath(); ctx.arc(coin.x - 2, coin.y - 2, coin.r - 3, 0, 7); ctx.fill();
      }
    },
    debug: function () { return { left: coins.length }; }
  });
})();
/*==END:coin==*/""",
    'life': """/*==MODULE:life==*/
(function () {
  if (!window.__HARNESS) return;
  var H = window.__HARNESS;
  var ENABLE = H.FEATURES.life !== false;
  H.register("life", {
    init: function () {},
    update: function () {},
    draw: function () {},
    debug: function () { return ENABLE ? H._debug.state().lives : 0; }
  });
})();
/*==END:life==*/""",
    'flag': """/*==MODULE:flag==*/
(function () {
  if (!window.__HARNESS) return;
  var H = window.__HARNESS;
  var ENABLE = H.FEATURES.flag !== false;
  H.register("flag", {
    init: function () {},
    update: function () {
      if (!ENABLE) return;
      if (H._debug.state().state !== "playing") return;
      var flag = H.flag;
      if (H.player.x + H.player.w > flag.x && H.player.x < flag.x + 4 &&
          H.player.y + H.player.h > flag.poleTop) {
        H.doWin();
      }
    },
    draw: function () {
      if (!ENABLE) return;
      var flag = H.flag;
      H.rect(flag.x, flag.poleTop, 4, flag.poleH, H.SCENE.flag_pole);
      H.rect(flag.x + 4, flag.poleTop, 20, 12, H.SCENE.flag);
      var ctx = H.ctx;
      ctx.beginPath();
      ctx.fillStyle = "#fff";
      ctx.arc(flag.x + 2, flag.poleTop, 3, 0, 7); ctx.fill();
    },
    debug: function () { return ENABLE ? "ready" : "off"; }
  });
})();
/*==END:flag==*/""",
}
