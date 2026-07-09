"""向量化：本地 sentence-transformers 与 OpenAI 双后端，统一接口。

实现 chromadb 的 embedding_function 协议：__call__(input: List[str]) -> List[List[float]]。
"""
from __future__ import annotations

from typing import List, Optional

from config import EmbedConfig, LLMConfig
from utils import RetryableError, get_logger, with_retry

logger = get_logger()


class EmbeddingError(RuntimeError):
    """向量化失败。"""


class LocalEmbedder:
    """基于 sentence-transformers 的本地 Embedding。"""

    def __init__(self, model_name: str, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "未安装 sentence-transformers，请 pip install sentence-transformers") from e
        try:
            logger.info(f"加载本地 Embedding 模型: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise EmbeddingError(f"加载本地 Embedding 模型失败: {model_name} - {e}") from e

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        # 过滤空串，避免某些模型报错
        cleaned = [t if t.strip() else " " for t in input]
        try:
            vecs = self.model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()
        except Exception as e:
            raise EmbeddingError(f"本地 Embedding 编码失败: {e}") from e


class OpenAIEmbedder:
    """OpenAI 兼容的 Embedding 接口。"""

    def __init__(self, llm_cfg: LLMConfig, model: str):
        if not llm_cfg.api_key:
            raise EmbeddingError("OpenAI Embedding 需要 api_key")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbeddingError("未安装 openai，请 pip install openai") from e
        self.client = OpenAI(base_url=llm_cfg.base_url, api_key=llm_cfg.api_key,
                             timeout=llm_cfg.timeout)
        self.model = model
        self.llm_cfg = llm_cfg

    @with_retry(max_retries=3, backoff=1.5, exceptions=(RetryableError,))
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            # 限流 / 超时 / 网络错误可重试
            msg = str(e).lower()
            if any(k in msg for k in ("rate", "timeout", "connection", "429", "503", "502")):
                raise RetryableError(f"OpenAI Embedding 暂时失败: {e}") from e
            raise EmbeddingError(f"OpenAI Embedding 调用失败: {e}") from e

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        # OpenAI 单批最多 2048 条，这里按 256 分批
        out: List[List[float]] = []
        batch_size = 256
        for i in range(0, len(input), batch_size):
            out.extend(self._embed_batch(input[i:i + batch_size]))
        return out


def build_embedder(embed_cfg: EmbedConfig, llm_cfg: LLMConfig) -> object:
    """根据配置返回对应的 embedder 实例。"""
    if embed_cfg.backend == "local":
        return LocalEmbedder(embed_cfg.local_model)
    elif embed_cfg.backend == "openai":
        return OpenAIEmbedder(llm_cfg, embed_cfg.openai_model)
    raise EmbeddingError(f"未知 embed.backend: {embed_cfg.backend}")
