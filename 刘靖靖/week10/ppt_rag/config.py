"""集中配置：所有可调参数在此定义，启动时校验环境变量与路径。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


class ConfigError(RuntimeError):
    """配置不合法时抛出。"""


@dataclass
class Paths:
    ppt_path: Path = Path("./data/your.pptx")
    db_dir: Path = Path("./ppt_vector_db")
    eval_output: Path = Path("./rag_eval_results.json")
    log_file: Path = Path("./ppt_rag.log")


@dataclass
class SplitConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_len: int = 20          # 过短的 chunk 直接丢弃
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )


@dataclass
class RetrievalConfig:
    top_k: int = 4
    min_score: float = 0.0           # 距离上限，超过则丢弃（cosine 距离越小越相似）
    rerank_top_n: int = 0             # 0 表示不启用 rerank；>0 表示 rerank 后取前 N
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"


@dataclass
class LLMConfig:
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_retries: int = 3
    retry_backoff: float = 1.5        # 指数退避基数（秒）
    timeout: float = 60.0


@dataclass
class EmbedConfig:
    backend: str = "local"            # "local" 或 "openai"
    local_model: str = "BAAI/bge-large-zh-v1.5"
    openai_model: str = "text-embedding-3-small"
    # openai 后端复用 LLMConfig 的 base_url/api_key


@dataclass
class OCRConfig:
    enabled: bool = False             # 是否启用 OCR（默认关闭，避免无 GPU 环境出错）
    model_name: str = "baidu/Unlimited-OCR"
    device: str = "cuda"              # "cuda" / "cpu"（cpu 极慢，不推荐）
    image_mode: str = "gundam"        # 单图模式: "gundam"(切图) 或 "base"
    # 单图 gundam 推荐参数
    base_size: int = 1024
    image_size_gundam: int = 640
    image_size_base: int = 1024
    crop_mode: bool = True
    max_length: int = 32768
    no_repeat_ngram_size: int = 35
    ngram_window_single: int = 128
    ngram_window_multi: int = 1024
    # 小于此字节数的图片跳过 OCR（避免图标/装饰）
    min_image_bytes: int = 2048
    # 单张图片 OCR 超时（秒）
    timeout: float = 300.0
    # 临时图目录
    tmp_dir: Path = Path("./_ocr_tmp")


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    collection_name: str = "ppt_docs"


def validate(cfg: Config, require_ppt: bool = True) -> None:
    """启动前校验，避免运行到一半才崩。"""
    if cfg.split.chunk_size <= 0:
        raise ConfigError("chunk_size 必须 > 0")
    if not (0 <= cfg.split.chunk_overlap < cfg.split.chunk_size):
        raise ConfigError("chunk_overlap 必须满足 0 <= overlap < chunk_size")
    if cfg.split.min_chunk_len < 0:
        raise ConfigError("min_chunk_len 不能为负")
    if cfg.retrieval.top_k <= 0:
        raise ConfigError("top_k 必须 > 0")
    if cfg.retrieval.rerank_top_n > cfg.retrieval.top_k:
        raise ConfigError("rerank_top_n 不应大于 top_k")
    if cfg.llm.timeout <= 0 or cfg.llm.max_retries < 0:
        raise ConfigError("timeout / max_retries 不合法")
    if cfg.embed.backend not in ("local", "openai"):
        raise ConfigError("embed.backend 只能是 local 或 openai")
    if cfg.embed.backend == "openai" and not cfg.llm.api_key:
        raise ConfigError("使用 openai embedding 时必须设置 LLM_API_KEY")
    if not cfg.llm.api_key:
        raise ConfigError("未设置 LLM_API_KEY，无法调用生成模型")
    # OCR 校验
    if cfg.ocr.enabled:
        if cfg.ocr.image_mode not in ("gundam", "base"):
            raise ConfigError(f"ocr.image_mode 只能是 gundam 或 base，当前: {cfg.ocr.image_mode}")
        if cfg.ocr.device not in ("cuda", "cpu"):
            raise ConfigError(f"ocr.device 只能是 cuda 或 cpu，当前: {cfg.ocr.device}")
        if cfg.ocr.max_length <= 0:
            raise ConfigError("ocr.max_length 必须 > 0")
        if cfg.ocr.min_image_bytes < 0:
            raise ConfigError("ocr.min_image_bytes 不能为负")
    if require_ppt:
        p = cfg.paths.ppt_path
        if not p.exists():
            raise ConfigError(f"PPT 文件不存在: {p}")
        if p.suffix.lower() not in (".pptx", ".ppt"):
            raise ConfigError(f"文件后缀应为 .pptx 或 .ppt，当前: {p.suffix}")
        if p.stat().st_size == 0:
            raise ConfigError(f"PPT 文件大小为 0: {p}")


# 全局默认实例
DEFAULT = Config()
