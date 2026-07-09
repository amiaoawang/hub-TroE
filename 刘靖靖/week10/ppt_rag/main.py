"""PPT RAG CLI 入口。

用法：
  # 建库 + 单条提问
  python main.py build --ppt ./data/your.pptx --question "核心结论是什么？"

  # 复用已有库提问
  python main.py load --question "提到了哪些指标？"

  # 批量评估
  python main.py build --ppt ./data/your.pptx --eval-file ./eval_cases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import Config, ConfigError, DEFAULT, validate
from pipeline import RAGPipeline
from utils import get_logger

logger = get_logger(log_file=str(DEFAULT.paths.log_file))


def load_eval_cases(path: str):
    p = Path(path)
    if not p.exists():
        logger.error(f"评估用例文件不存在: {p}")
        sys.exit(1)
    try:
        with open(p, "r", encoding="utf-8") as f:
            cases = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"评估用例文件 JSON 解析失败: {e}")
        sys.exit(1)
    if not isinstance(cases, list):
        logger.error("评估用例文件应为 JSON 数组")
        sys.exit(1)
    return cases


def make_cfg(args) -> Config:
    cfg = Config()
    if args.ppt:
        cfg.paths.ppt_path = Path(args.ppt)
    if args.db_dir:
        cfg.paths.db_dir = Path(args.db_dir)
    if args.collection:
        cfg.collection_name = args.collection
    if args.embed_backend:
        cfg.embed.backend = args.embed_backend
    if args.chunk_size:
        cfg.split.chunk_size = args.chunk_size
    if args.chunk_overlap:
        cfg.split.chunk_overlap = args.chunk_overlap
    if args.top_k:
        cfg.retrieval.top_k = args.top_k
    if args.use_reranker:
        cfg.retrieval.use_reranker = True
        cfg.retrieval.rerank_top_n = args.rerank_top_n or min(cfg.retrieval.top_k, 3)
    # OCR 参数
    if args.use_ocr:
        cfg.ocr.enabled = True
    if args.ocr_model:
        cfg.ocr.model_name = args.ocr_model
    if args.ocr_device:
        cfg.ocr.device = args.ocr_device
    if args.ocr_image_mode:
        cfg.ocr.image_mode = args.ocr_image_mode
    return cfg


def cmd_build(args):
    cfg = make_cfg(args)
    try:
        validate(cfg, require_ppt=True)
    except ConfigError as e:
        logger.error(f"配置错误: {e}")
        sys.exit(1)

    pipe = RAGPipeline(cfg)
    try:
        pipe.build_index()
    except Exception as e:
        logger.error(f"建库失败: {e}")
        sys.exit(1)

    _after_build(args, pipe)


def cmd_load(args):
    cfg = make_cfg(args)
    try:
        validate(cfg, require_ppt=False)
    except ConfigError as e:
        logger.error(f"配置错误: {e}")
        sys.exit(1)

    pipe = RAGPipeline(cfg)
    try:
        pipe.load_index()
    except Exception as e:
        logger.error(f"载入失败: {e}")
        sys.exit(1)

    _after_build(args, pipe)


def _after_build(args, pipe):
    if args.eval_file:
        cases = load_eval_cases(args.eval_file)
        results = pipe.run_eval_set(cases)
        pipe.save_eval_results(results)
    elif args.question:
        r = pipe.answer(args.question, do_eval=not args.no_eval)
        print("\n=== 回答 ===")
        print(r["answer"])
        if "eval" in r:
            print("\n=== 评估 ===")
            print(json.dumps(r["eval"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="PPT RAG (no LangChain)")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db-dir", help="向量库目录")
    common.add_argument("--collection", help="collection 名称")
    common.add_argument("--embed-backend", choices=["local", "openai"], help="embedding 后端")
    common.add_argument("--chunk-size", type=int, help="chunk 大小")
    common.add_argument("--chunk-overlap", type=int, help="chunk 重叠")
    common.add_argument("--top-k", type=int, help="检索 Top-K")
    common.add_argument("--use-reranker", action="store_true", help="启用 reranker")
    common.add_argument("--rerank-top-n", type=int, help="rerank 后保留前 N 条")
    # OCR 参数
    common.add_argument("--use-ocr", action="store_true",
                        help="启用 OCR（需安装 torch + transformers，建议有 GPU）")
    common.add_argument("--ocr-model", help="Unlimited-OCR 模型名或本地路径")
    common.add_argument("--ocr-device", choices=["cuda", "cpu"], help="OCR 运行设备")
    common.add_argument("--ocr-image-mode", choices=["gundam", "base"],
                        help="单图 OCR 模式：gundam(切图, 推荐) 或 base")
    common.add_argument("--question", help="单条问题")
    common.add_argument("--eval-file", help="批量评估用例 JSON 文件")
    common.add_argument("--no-eval", action="store_true", help="不进行评估")

    p_build = sub.add_parser("build", parents=[common], help="从 PPT 建库")
    p_build.add_argument("--ppt", required=True, help="PPT 文件路径")
    p_build.set_defaults(func=cmd_build)

    p_load = sub.add_parser("load", parents=[common], help="载入已有库")
    p_load.set_defaults(func=cmd_load)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
