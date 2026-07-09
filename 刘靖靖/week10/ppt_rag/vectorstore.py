"""向量库管理：基于 chromadb，支持建库 / 载库 / 删库 / 查询 / 增量入库。

边界处理：
- 入库前清空旧 collection，避免 id 冲突
- 分批 add，避免单批过大
- 路径不存在时自动创建
- 已有库可重复 load
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from utils import get_logger

logger = get_logger()

# 单批 add 上限，chromadb 默认 sqlite 限制较多
BATCH_SIZE = 500


class VectorStoreError(RuntimeError):
    """向量库操作失败。"""


class VectorStore:
    """chromadb 持久化向量库的封装。"""

    def __init__(self, db_dir, collection_name: str, embedder):
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.embedder = embedder
        try:
            self.db_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise VectorStoreError(f"无法创建向量库目录 {self.db_dir}: {e}") from e

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e:
            raise VectorStoreError(f"连接 chromadb 失败: {e}") from e

    # ---------- collection 管理 ----------
    def _delete_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"已删除旧 collection: {self.collection_name}")
        except Exception:
            # 不存在就算了
            pass

    def build(self, chunks: List[Dict]) -> object:
        """从 chunks 重新建库（会先清空旧库）。"""
        if not chunks:
            raise VectorStoreError("无 chunk 可入库")

        self._delete_collection()
        try:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedder,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"创建 collection 失败: {e}") from e

        # 分批 add
        total = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            try:
                collection.add(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[{"page": c["page"], "source": c["source"]} for c in batch],
                )
                total += len(batch)
            except Exception as e:
                logger.warning(f"第 {i}-{i+len(batch)} 批入库失败: {e}")
        logger.info(f"入库完成: collection={self.collection_name} 共 {collection.count()} 条")
        return collection

    def load(self) -> object:
        """载入已有 collection。"""
        try:
            collection = self.client.get_collection(
                name=self.collection_name, embedding_function=self.embedder)
            logger.info(f"载入 collection: {self.collection_name} ({collection.count()} 条)")
            return collection
        except Exception as e:
            raise VectorStoreError(
                f"载入 collection 失败: {self.collection_name}，可能未建库 - {e}") from e

    def exists(self) -> bool:
        """collection 是否已存在。"""
        try:
            collections = [c.name for c in self.client.list_collections()]
            return self.collection_name in collections
        except Exception:
            return False

    # ---------- 检索 ----------
    def query(self, query_text: str, n_results: int = 4) -> Dict:
        """单条查询，返回原始 chromadb 响应。"""
        if not query_text or not query_text.strip():
            raise VectorStoreError("查询文本不能为空")
        try:
            return self.load().query(
                query_texts=[query_text],
                n_results=n_results,
            )
        except Exception as e:
            raise VectorStoreError(f"查询失败: {e}") from e

    # ---------- 增量入库 ----------
    def upsert(self, chunks: List[Dict]) -> None:
        """按 id 增量更新（已存在则覆盖）。"""
        if not chunks:
            return
        try:
            collection = self.load()
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                collection.upsert(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[{"page": c["page"], "source": c["source"]} for c in batch],
                )
            logger.info(f"upsert {len(chunks)} 条，当前共 {collection.count()} 条")
        except Exception as e:
            raise VectorStoreError(f"upsert 失败: {e}") from e
