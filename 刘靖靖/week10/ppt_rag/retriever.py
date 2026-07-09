"""检索：调用 VectorStore.query，可选 rerank。

边界处理：
- 检索结果为空 → 返回空列表
- 距离过滤 → 丢弃相似度过低的项
- reranker 加载失败 → 退化为原始排序
"""
from __future__ import annotations

from typing import Dict, List, Optional

from config import RetrievalConfig
from utils import get_logger
from vectorstore import VectorStore

logger = get_logger()


class Retriever:
    """向量检索 + 可选 rerank。"""

    def __init__(self, store: VectorStore, cfg: RetrievalConfig):
        self.store = store
        self.cfg = cfg
        self._reranker = None
        if cfg.use_reranker and cfg.rerank_top_n > 0:
            self._reranker = self._load_reranker(cfg.reranker_model)

    @staticmethod
    def _load_reranker(model_name: str):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载 reranker: {model_name}")
            return CrossEncoder(model_name)
        except Exception as e:
            logger.warning(f"reranker 加载失败，退化为原始排序: {e}")
            return None

    def retrieve(self, query: str) -> List[Dict]:
        """返回 [{id, text, page, score}]，score 是距离（越小越相似）。"""
        try:
            res = self.store.query(query, n_results=self.cfg.top_k)
        except Exception as e:
            logger.warning(f"检索失败: {e}")
            return []

        if not res or not res.get("ids") or not res["ids"][0]:
            logger.info("检索结果为空")
            return []

        docs: List[Dict] = []
        for i in range(len(res["ids"][0])):
            dist = float(res["distances"][0][i])
            # 距离上限过滤（cosine 距离越小越相似）
            if dist < self.cfg.min_score:
                continue
            docs.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "page": res["metadatas"][0][i].get("page"),
                "score": dist,
            })

        if not docs:
            logger.info(f"检索后无有效结果 (min_score={self.cfg.min_score})")
            return []

        # rerank
        if self._reranker is not None and self.cfg.rerank_top_n > 0:
            docs = self._rerank(query, docs)

        return docs

    def _rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        try:
            pairs = [(query, d["text"]) for d in docs]
            scores = self._reranker.predict(pairs)
            for d, s in zip(docs, scores):
                d["rerank_score"] = float(s)
            docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            n = self.cfg.rerank_top_n
            return docs[:n]
        except Exception as e:
            logger.warning(f"rerank 失败，使用原始排序: {e}")
            return docs
