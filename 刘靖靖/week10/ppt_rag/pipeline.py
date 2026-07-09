"""主流程编排：把各模块串起来，提供 build_index / load_index / answer / run_eval_set 接口。"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from config import Config, validate
from embedder import build_embedder
from evaluator import Evaluator, summarize
from generator import Generator, GeneratorError
from parser import ParseError, parse_pptx
from retriever import Retriever
from splitter import build_chunks
from utils import get_logger
from vectorstore import VectorStore, VectorStoreError

logger = get_logger()


class RAGPipeline:
    """PPT RAG 流程封装。"""

    def __init__(self, cfg: Config):
        # load 模式不需要 ppt 文件，所以分两步校验
        validate(cfg, require_ppt=False)
        self.cfg = cfg
        self.embedder = None
        self.store: Optional[VectorStore] = None
        self.retriever: Optional[Retriever] = None
        self.generator: Optional[Generator] = None
        self.evaluator: Optional[Evaluator] = None

    # ---------- 索引构建 ----------
    def build_index(self) -> None:
        """从 PPT 解析 → 切分 → 向量化 → 入库。"""
        validate(self.cfg, require_ppt=True)

        logger.info("[1/4] 初始化 embedder...")
        self.embedder = build_embedder(self.cfg.embed, self.cfg.llm)

        logger.info("[2/4] 解析 PPT...")
        try:
            pages = parse_pptx(self.cfg.paths.ppt_path, ocr_cfg=self.cfg.ocr)
        except ParseError as e:
            logger.error(f"PPT 解析失败: {e}")
            raise
        if not pages:
            logger.error("解析后无有效内容，终止")
            return

        logger.info("[3/4] 切分...")
        chunks = build_chunks(pages, self.cfg.split)
        if not chunks:
            logger.error("切分后无 chunk，终止")
            return

        logger.info("[4/4] 入库...")
        self.store = VectorStore(self.cfg.paths.db_dir, self.cfg.collection_name, self.embedder)
        self.store.build(chunks)

        self._init_runtime()
        logger.info("索引构建完成")

    # ---------- 载入已有库 ----------
    def load_index(self) -> None:
        """复用已建好的向量库。"""
        logger.info("载入已有向量库...")
        self.embedder = build_embedder(self.cfg.embed, self.cfg.llm)
        self.store = VectorStore(self.cfg.paths.db_dir, self.cfg.collection_name, self.embedder)
        if not self.store.exists():
            raise VectorStoreError(
                f"collection 不存在: {self.cfg.collection_name}，请先 build_index")
        self.store.load()
        self._init_runtime()
        logger.info("载入完成")

    def _init_runtime(self) -> None:
        self.retriever = Retriever(self.store, self.cfg.retrieval)
        self.generator = Generator(self.cfg.llm)
        self.evaluator = Evaluator(self.cfg.llm)

    # ---------- 单条问答 ----------
    def answer(self, question: str, do_eval: bool = True) -> Dict:
        if not question or not question.strip():
            raise ValueError("问题不能为空")
        if self.retriever is None or self.generator is None:
            raise RuntimeError("未初始化 retriever/generator，请先 build_index 或 load_index")

        logger.info(f"问题: {question}")
        contexts = self.retriever.retrieve(question)
        logger.info(f"检索到 {len(contexts)} 条上下文")
        for c in contexts:
            preview = c["text"][:60].replace("\n", " ")
            logger.info(f"  [第{c['page']}页] dist={c['score']:.4f} {preview}...")

        try:
            answer_text = self.generator.generate(question, contexts)
        except GeneratorError as e:
            logger.error(f"生成失败: {e}")
            answer_text = f"(生成失败: {e})"

        result: Dict = {"question": question, "answer": answer_text, "contexts": contexts}
        if do_eval and self.evaluator is not None:
            try:
                ev = self.evaluator.evaluate(question, contexts, answer_text)
            except Exception as e:
                logger.warning(f"评估失败: {e}")
                ev = {"error": str(e)}
            result["eval"] = ev
        return result

    # ---------- 批量评估 ----------
    def run_eval_set(self, cases: List[Dict]) -> List[Dict]:
        """批量跑评估集，输出平均分。"""
        results: List[Dict] = []
        for i, case in enumerate(cases, 1):
            q = case.get("question", "").strip()
            if not q:
                logger.warning(f"第 {i} 条用例问题为空，跳过")
                continue
            logger.info(f"--- 评估用例 {i}/{len(cases)} ---")
            r = self.answer(q, do_eval=True)
            r["expected"] = case.get("expected", "")
            results.append(r)

        summary = summarize(results)
        logger.info("=== 评估汇总 ===")
        for k, v in summary.items():
            logger.info(f"{k:20s} avg = {v}")
        results.append({"_summary": summary})
        return results

    # ---------- 持久化评估结果 ----------
    def save_eval_results(self, results: List[Dict], path=None) -> None:
        import os
        out = path or self.cfg.paths.eval_output
        try:
            os.makedirs(os.path.dirname(str(out)) or ".", exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"评估结果已保存: {out}")
        except OSError as e:
            logger.error(f"保存评估结果失败: {e}")
