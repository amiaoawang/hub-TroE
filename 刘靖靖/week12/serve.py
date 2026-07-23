"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  POST /reset         - 清空指定会话的多轮历史
  GET  /health        - 健康检查

多轮对话：
  请求体可携带 session_id。同一 session_id 下的历史 Q&A 会被注入到
  ReAct 的 system 与本轮 user 之间，形成跨轮次上下文记忆。
  ReAct 中间的 Thought/Action/Observation 不进入历史，避免上下文膨胀。

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 多轮会话存储 ──────────────────────────────────────────────────────────────
# session_id -> [{"role": "user"|"assistant", "content": str}, ...]
# 仅保存每轮的 用户问题 + Final Answer，不保存中间 ReAct 步骤
SESSIONS: dict[str, list] = {}
MAX_HISTORY_MESSAGES = 10  # 最多保留最近 10 条消息（约 5 轮），防止 token 膨胀


def _get_history(session_id: str) -> list:
    """读取指定会话的历史（返回拷贝，避免被 worker 修改）"""
    return list(SESSIONS.get(session_id, []))


def _append_history(session_id: str, question: str, answer: str):
    """将本轮 Q&A 追加到会话历史，并裁剪到上限"""
    history = SESSIONS.get(session_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        # 成对裁剪：从开头删掉偶数条，保证仍以 user 开头
        drop = len(history) - MAX_HISTORY_MESSAGES
        if drop % 2 == 1:
            drop += 1  # 凑成偶数，保证消息成对
        history = history[drop:]
    SESSIONS[session_id] = history


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str
    max_steps: int = 10
    session_id: str | None = None  # 不传则新建会话


class ResetRequest(BaseModel):
    session_id: str


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str, session_id: str):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    多轮：启动前读取 session_id 对应的历史并注入；结束后把本轮
    Q&A（user 问题 + Final Answer）写回历史，供下一轮使用。
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    final_answer_holder = {"answer": ""}  # worker 线程回传最终答案

    history = _get_history(session_id)

    def _worker():
        try:
            for step_data in react_run(question, max_steps=max_steps, history=history):
                queue.put_nowait(step_data)
                if step_data.get("type") == "final":
                    final_answer_holder["answer"] = step_data.get("answer", "")
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({
        "type": "start",
        "question": question,
        "mode": mode,
        "session_id": session_id,
        "history_turns": len(history) // 2,  # 历史轮次数，便于 UI 展示
    })

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    # 持久化本轮 Q&A 到会话历史（即便没拿到 Final Answer 也写入空串，保持成对）
    _append_history(session_id, question, final_answer_holder["answer"])

    yield _sse({"type": "done", "session_id": session_id})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
async def reset_session(req: ResetRequest):
    """清空指定会话的多轮历史，开始新对话"""
    SESSIONS.pop(req.session_id, None)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.getenv("AGENT_MODEL", "qwen-max"),
        "active_sessions": len(SESSIONS),
    }


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
