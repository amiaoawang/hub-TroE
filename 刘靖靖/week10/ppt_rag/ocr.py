"""Unlimited-OCR 封装：基于 baidu/Unlimited-OCR，对 PPT 中的图片做文字识别。

设计要点：
- 懒加载：仅当启用 OCR 时才 import torch/transformers，避免无 GPU 环境启动失败
- 优雅降级：torch 未装 / CUDA 不可用 / 模型加载失败 → 返回空串并打 warning
- 单图模式 gundam(切图) 与 base 二选一
- 多图走 infer_multi（仅支持 base）
- 临时图自动清理
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from config import OCRConfig
from utils import get_logger, truncate

logger = get_logger()


class OCRError(RuntimeError):
    """OCR 失败。"""


class UnlimitedOCR:
    """baidu/Unlimited-OCR 模型封装。"""

    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_failed = False
        self._load_reason = ""

    # ---------- 懒加载 ----------
    def _ensure_loaded(self) -> bool:
        """首次调用时加载模型。失败后不再重试。"""
        if self._loaded:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            self._load_failed = True
            self._load_reason = f"未安装 torch/transformers: {e}"
            logger.warning(f"OCR 不可用：{self._load_reason}（pip install torch transformers）")
            return False

        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，OCR 将回退到 CPU（极慢）或直接跳过")
            # 不强制失败，让用户自己决定

        try:
            logger.info(f"加载 Unlimited-OCR 模型: {self.cfg.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, trust_remote_code=True)
            dtype = torch.bfloat16 if self.cfg.device == "cuda" else torch.float32
            self._model = AutoModel.from_pretrained(
                self.cfg.model_name,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=dtype,
            )
            if self.cfg.device == "cuda":
                self._model = self._model.eval().cuda()
            else:
                self._model = self._model.eval()
            self._loaded = True
            logger.info("Unlimited-OCR 模型加载完成")
            return True
        except Exception as e:
            self._load_failed = True
            self._load_reason = str(e)
            logger.warning(f"OCR 模型加载失败，后续图片将跳过 OCR: {e}")
            return False

    # ---------- 单图 OCR ----------
    def ocr_image(self, image_path) -> str:
        """对单张图片做 OCR，返回识别文本。失败返回空串。"""
        p = Path(image_path)
        if not p.exists() or p.stat().st_size == 0:
            logger.warning(f"图片不存在或为空: {p}")
            return ""

        if not self._ensure_loaded():
            return ""

        cfg = self.cfg
        is_gundam = cfg.image_mode == "gundam"
        image_size = cfg.image_size_gundam if is_gundam else cfg.image_size_base
        ngram_window = cfg.ngram_window_single

        # 准备输出目录（模型 save_results=True 时写入）
        out_dir = tempfile.mkdtemp(prefix="ocr_out_")

        try:
            result = self._model.infer(
                self._tokenizer,
                prompt="<image>document parsing.",
                image_file=str(p),
                output_path=out_dir,
                base_size=cfg.base_size,
                image_size=image_size,
                crop_mode=cfg.crop_mode if is_gundam else False,
                max_length=cfg.max_length,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                ngram_window=ngram_window,
                save_results=True,
            )
            return self._extract_text(result, out_dir, p.stem)
        except Exception as e:
            logger.warning(f"OCR 失败: {p.name} - {e}")
            return ""

    # ---------- 多图 OCR ----------
    def ocr_images(self, image_paths: List[str]) -> str:
        """对多张图片做批量 OCR（仅支持 base 模式）。"""
        if not image_paths:
            return ""
        if len(image_paths) == 1:
            return self.ocr_image(image_paths[0])

        if not self._ensure_loaded():
            return ""

        cfg = self.cfg
        out_dir = tempfile.mkdtemp(prefix="ocr_multi_")

        try:
            result = self._model.infer_multi(
                self._tokenizer,
                prompt="<image>Multi page parsing.",
                image_files=[str(p) for p in image_paths],
                output_path=out_dir,
                image_size=cfg.image_size_base,    # multi 仅支持 base
                max_length=cfg.max_length,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                ngram_window=cfg.ngram_window_multi,
                save_results=True,
            )
            return self._extract_text(result, out_dir, "multi")
        except Exception as e:
            logger.warning(f"多图 OCR 失败: {e}")
            # 失败时降级为逐张处理
            logger.info("降级为逐张 OCR")
            texts = []
            for p in image_paths:
                t = self.ocr_image(p)
                if t:
                    texts.append(t)
            return "\n\n".join(texts)

    # ---------- 结果提取 ----------
    @staticmethod
    def _extract_text(result, out_dir: str, stem: str) -> str:
        """从模型返回值 / 输出文件中提取文本。"""
        # 1. 优先用返回值
        if isinstance(result, str) and result.strip():
            return result.strip()
        if isinstance(result, dict):
            for k in ("text", "result", "content", "output"):
                v = result.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        # 2. 从输出目录里读 .txt / .json
        try:
            for f in sorted(Path(out_dir).iterdir()):
                if f.suffix.lower() == ".txt":
                    return f.read_text(encoding="utf-8").strip()
                if f.suffix.lower() == ".json":
                    import json
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        if isinstance(data, dict):
                            for k in ("text", "result", "content", "output"):
                                v = data.get(k)
                                if isinstance(v, str) and v.strip():
                                    return v.strip()
                        if isinstance(data, list):
                            return "\n".join(str(x) for x in data).strip()
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"读取 OCR 输出文件失败: {e}")
        return ""


# ---------- 工厂 ----------
_GLOBAL_OCR: Optional[UnlimitedOCR] = None


def get_ocr(cfg: OCRConfig) -> Optional[UnlimitedOCR]:
    """获取全局 OCR 实例（懒加载 + 单例）。未启用时返回 None。"""
    global _GLOBAL_OCR
    if not cfg.enabled:
        return None
    if _GLOBAL_OCR is None:
        _GLOBAL_OCR = UnlimitedOCR(cfg)
    return _GLOBAL_OCR


def reset_ocr() -> None:
    """重置全局 OCR 实例（用于切换配置 / 测试）。"""
    global _GLOBAL_OCR
    _GLOBAL_OCR = None
