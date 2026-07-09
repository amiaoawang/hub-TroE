"""通用工具：日志、重试、JSON 兜底解析、安全文本拼接。"""
from __future__ import annotations

import json
import logging
import re
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


# ------------------------------------------------------------
# 日志：同时输出到控制台和文件
# ------------------------------------------------------------
def get_logger(name: str = "ppt_rag", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:           # 避免重复 add handler
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError as e:
            logger.warning(f"无法写入日志文件 {log_file}: {e}")
    return logger


# ------------------------------------------------------------
# 重试装饰器：捕获可重试异常，指数退避
# ------------------------------------------------------------
class RetryableError(Exception):
    """可重试的临时错误（网络、限流等）。"""


def with_retry(max_retries: int = 3, backoff: float = 1.5,
               exceptions: tuple = (RetryableError,)):
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_err: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    if attempt == max_retries:
                        break
                    wait = backoff ** attempt
                    logging.getLogger("ppt_rag").warning(
                        f"{fn.__name__} 第 {attempt}/{max_retries} 次失败: {e}，{wait:.1f}s 后重试")
                    time.sleep(wait)
            assert last_err is not None
            raise last_err
        return wrapper
    return decorator


# ------------------------------------------------------------
# JSON 兜底解析：LLM 有时返回带额外文字的 JSON
# ------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def safe_json_loads(text: str, default: Any = None) -> Any:
    """尝试解析 JSON；失败则正则抽取第一个 JSON 块再试；仍失败返回 default。"""
    if not text:
        return default
    text = text.strip()
    # 去掉常见 markdown 代码块包裹
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return default


# ------------------------------------------------------------
# 文本安全截断：避免超长喂给 LLM
# ------------------------------------------------------------
def truncate(text: str, max_len: int, suffix: str = "...") -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix
