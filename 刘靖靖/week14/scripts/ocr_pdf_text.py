# -*- coding: utf-8 -*-
"""PDF 文本 OCR 兜底：当 PDF 无文本层（文字是图片）时，渲染每页为图片并 OCR 识别。

用法:
    python ocr_pdf_text.py <input.pdf> [output.txt]

依赖（按需安装，OCR 引擎二选一）:
    pip install pymupdf            # 渲染 PDF 为图片（必需）
    pip install rapidocr-onnxruntime  # 推荐：纯 pip 安装，中文效果好，CPU 可跑
    或
    pip install paddleocr paddlepaddle  # 备选：PaddleOCR，效果更好但更重

输出:
    每页以 `===== PAGE N =====` 分隔；不指定 output.txt 时打印到 stdout。
    渲染图片默认存到 <输入名>_ocr_pages/ 便于人工核对。
"""
import os
import sys


def render_pages(pdf_path: str, page_dir: str, scale: float = 2.0) -> list:
    """渲染 PDF 每页为 PNG，返回图片路径列表（按页码顺序）。"""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("未安装 pymupdf，请先 pip install pymupdf")
    os.makedirs(page_dir, exist_ok=True)
    paths = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        out = os.path.join(page_dir, f"page_{i:03d}.png")
        pix.save(out)
        paths.append(out)
    doc.close()
    return paths


def ocr_images(image_paths: list, engine: str = "auto") -> list:
    """对图片列表逐张 OCR，返回每页文本列表。engine: auto/rapidocr/paddle。"""
    engine = engine.lower()
    if engine in ("auto", "rapidocr"):
        try:
            return _ocr_rapidocr(image_paths)
        except ImportError:
            if engine == "rapidocr":
                raise RuntimeError("未安装 rapidocr-onnxruntime，请先 pip install rapidocr-onnxruntime")
    if engine in ("auto", "paddle"):
        try:
            return _ocr_paddle(image_paths)
        except ImportError:
            if engine == "paddle":
                raise RuntimeError("未安装 paddleocr，请先 pip install paddleocr paddlepaddle")
    raise RuntimeError(
        "未检测到可用 OCR 引擎。请安装其一：\n"
        "  pip install rapidocr-onnxruntime   （推荐，轻量）\n"
        "  pip install paddleocr paddlepaddle （备选，更重但效果更好）"
    )


def _ocr_rapidocr(image_paths):
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    results = []
    for p in image_paths:
        res, _ = ocr(p)
        text = "\n".join(line[1] for line in res) if res else ""
        results.append(text)
    return results


def _ocr_paddle(image_paths):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    results = []
    for p in image_paths:
        res = ocr.ocr(p, cls=True)
        lines = []
        if res and res[0]:
            for item in res[0]:
                lines.append(item[1][0])
        results.append("\n".join(lines))
    return results


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python ocr_pdf_text.py <input.pdf> [output.txt] [--engine rapidocr|paddle|auto]", file=sys.stderr)
        return 1
    src = os.path.abspath(sys.argv[1])
    if not os.path.isfile(src):
        print(f"文件不存在: {src}", file=sys.stderr)
        return 1
    engine = "auto"
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--engine" and i + 1 < len(args):
            engine = args[i + 1]

    base = os.path.splitext(src)[0]
    page_dir = base + "_ocr_pages"

    print(f"[1/2] 渲染 PDF 为图片 -> {page_dir}", flush=True)
    images = render_pages(src, page_dir)
    print(f"      共 {len(images)} 页", flush=True)

    print(f"[2/2] OCR 识别（引擎: {engine}）", flush=True)
    texts = ocr_images(images, engine)

    parts = []
    for i, t in enumerate(texts, start=1):
        parts.append(f"===== PAGE {i} =====")
        parts.append(t)
    out_text = "\n".join(parts)

    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        out = os.path.abspath(sys.argv[2])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"written: {out} ({len(out_text)} chars)", flush=True)
    else:
        print(out_text, flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
