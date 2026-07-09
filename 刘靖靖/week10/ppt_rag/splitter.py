"""递归切分：按分隔符优先级拆分，超过 chunk_size 时再递归。

边界处理：
- 文本本身短于 chunk_size → 直接返回
- 切分后某些块仍超长 → 用更细的分隔符或硬切
- 过短 chunk → 丢弃
- overlap 不能 >= chunk_size
"""
from __future__ import annotations

from typing import Dict, List

from config import SplitConfig
from utils import get_logger

logger = get_logger()


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int,
                     separators: List[str]) -> List[str]:
    """核心递归切分逻辑。"""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # 找当前文本中存在的最高优先级分隔符
    sep = separators[-1]
    for s in separators[:-1]:    # 最后一个通常是空串（硬切），先用前面的
        if s and s in text:
            sep = s
            break

    pieces = text.split(sep)
    chunks: List[str] = []
    current = ""

    for piece in pieces:
        piece = piece + (sep if sep else "")
        # 单个 piece 自身就超长 → 递归用更细的分隔符
        if len(piece) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            sub_seps = [s for s in separators if s not in ("", sep)]
            sub_seps.append("")
            sub = _recursive_split(piece.rstrip(sep), chunk_size, chunk_overlap, sub_seps)
            chunks.extend(sub)
            continue

        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            # overlap：取上一段尾部
            if chunk_overlap > 0 and chunks:
                tail = chunks[-1][-chunk_overlap:]
                current = tail + piece
            else:
                current = piece

    if current:
        chunks.append(current)
    return chunks


def _merge_short_chunks(chunks: List[str], min_len: int, max_len: int) -> List[str]:
    """过短 chunk 与相邻 chunk 合并，避免检索时上下文不足。"""
    if not chunks:
        return []
    merged: List[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        if merged and len(merged[-1]) < min_len and len(merged[-1]) + len(ch) + 1 <= max_len:
            merged[-1] = merged[-1] + "\n" + ch
        else:
            merged.append(ch)
    # 末尾如果合并后仍过短，保留但不丢弃（至少有一段内容）
    return merged


def split_text(text: str, cfg: SplitConfig) -> List[str]:
    """对外接口：把单段文本切成多个 chunk。"""
    chunks = _recursive_split(
        text=text,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=list(cfg.separators),
    )
    chunks = _merge_short_chunks(chunks, cfg.min_chunk_len, cfg.chunk_size)
    return [c for c in chunks if len(c.strip()) >= cfg.min_chunk_len]


def build_chunks(pages: List[Dict], cfg: SplitConfig) -> List[Dict]:
    """对每个 page 调用 split_text，输出带元数据的 chunk 列表。"""
    all_chunks: List[Dict] = []
    total_chunks = 0
    for p in pages:
        try:
            parts = split_text(p["content"], cfg)
        except Exception as e:
            logger.warning(f"第 {p['page']} 页切分失败，整页作为单 chunk: {e}")
            parts = [p["content"]] if p["content"] else []

        if not parts:
            logger.info(f"第 {p['page']} 页切分后无 chunk，跳过")
            continue

        for c in parts:
            all_chunks.append({
                "id": f"p{p['page']}_{total_chunks}",
                "text": c.strip(),
                "page": p["page"],
                "source": p["source"],
            })
            total_chunks += 1

    logger.info(f"切分完成: {len(pages)} 页 → {len(all_chunks)} 个 chunk")
    if not all_chunks:
        logger.warning("切分后无任何 chunk，请检查 PPT 内容或切分参数")
    return all_chunks
