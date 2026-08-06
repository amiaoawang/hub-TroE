# -*- coding: utf-8 -*-
"""按页提取 PDF 文本，标注页码，作为出题素材。

用法:
    python extract_pdf_text.py <input.pdf> [output.txt]

依赖:
    pip install pypdf

输出:
    每页以 `===== PAGE N =====` 分隔；不指定 output.txt 时打印到 stdout。
"""
import os
import sys
from pypdf import PdfReader


def extract_pdf_text(src: str) -> str:
    reader = PdfReader(src)
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"===== PAGE {i} =====")
        parts.append(text)
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python extract_pdf_text.py <input.pdf> [output.txt]", file=sys.stderr)
        return 1
    src = os.path.abspath(sys.argv[1])
    if not os.path.isfile(src):
        print(f"文件不存在: {src}", file=sys.stderr)
        return 1

    text = extract_pdf_text(src)
    total_chars = len(text)

    if len(sys.argv) > 2:
        out = os.path.abspath(sys.argv[2])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"written: {out} ({total_chars} chars)", flush=True)
    else:
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
