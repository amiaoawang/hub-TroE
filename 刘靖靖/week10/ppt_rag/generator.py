"""生成：调用 OpenAI 兼容接口，组装 prompt，处理重试与超时。

边界处理：
- 无上下文 → 直接让 LLM 拒答
- 上下文过长 → 截断
- API 失败 → 重试 + 指数退避
- 响应空 → 兜底返回
"""
from __future__ import annotations

from typing import Dict, List

from config import LLMConfig
from utils import RetryableError, get_logger, truncate, with_retry

logger = get_logger()

# 喂给 LLM 的上下文总长度上限（字符），防止超出 token 限制
MAX_CONTEXT_CHARS = 12000


class GeneratorError(RuntimeError):
    """生成失败。"""


def build_prompt(question: str, contexts: List[Dict]) -> str:
    """组装 RAG prompt。"""
    if not question.strip():
        raise GeneratorError("问题不能为空")

    if not contexts:
        return f"""你是 PPT 文档助手。本次未检索到任何相关资料。

【问题】
{question}

请说明：根据当前 PPT 资料无法回答该问题。
"""

    # 截断超长上下文
    parts: List[str] = []
    used = 0
    for c in contexts:
        chunk_text = c["text"]
        if used + len(chunk_text) > MAX_CONTEXT_CHARS:
            chunk_text = truncate(chunk_text, MAX_CONTEXT_CHARS - used)
            parts.append(f"[第{c['page']}页] {chunk_text}")
            break
        parts.append(f"[第{c['page']}页] {chunk_text}")
        used += len(chunk_text)

    context_str = "\n\n".join(parts)
    return f"""你是 PPT 文档助手，严格依据下方参考资料回答问题。

要求：
1. 只使用资料中的信息，不要编造。
2. 资料不足时直接说明，不要猜测。
3. 引用页码，格式：「见第 X 页」。

【参考资料】
{context_str}

【问题】
{question}

【回答】
"""


class Generator:
    """LLM 生成器。"""

    def __init__(self, cfg: LLMConfig):
        if not cfg.api_key:
            raise GeneratorError("LLM api_key 未配置")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise GeneratorError("未安装 openai，请 pip install openai") from e
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)

    @with_retry(max_retries=3, backoff=1.5, exceptions=(RetryableError,))
    def _chat(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperature,
            )
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("rate", "timeout", "connection", "429", "503", "502", "504")):
                raise RetryableError(f"LLM 暂时失败: {e}") from e
            raise GeneratorError(f"LLM 调用失败: {e}") from e

        if not resp.choices:
            raise GeneratorError("LLM 返回空 choices")
        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise GeneratorError("LLM 返回空内容")
        return content.strip()

    def generate(self, question: str, contexts: List[Dict]) -> str:
        prompt = build_prompt(question, contexts)
        try:
            return self._chat(prompt)
        except RetryableError as e:
            raise GeneratorError(f"LLM 重试后仍失败: {e}") from e
