# -*- coding: utf-8 -*-
"""将完整 PDF 按页拆分为单独 PDF 文件。

用法:
    python split_pdf_pages.py <input.pdf> [output_dir]

依赖:
    pip install pypdf

输出:
    <output_dir>/第01页.pdf、第02页.pdf ...（按总页数补零）
    默认 output_dir 为 <输入文件名不带扩展名>_单页PDF/
"""
import os
import sys
from pypdf import PdfReader, PdfWriter


def split_pdf(src: str, out_dir: str) -> int:
    reader = PdfReader(src)
    total = len(reader.pages)
    print(f"total pages: {total}", flush=True)

    width = len(str(total))
    for i, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        name = f"第{i:0{width}d}页.pdf"
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            writer.write(f)
        print(f"  {name}  OK", flush=True)
    return total


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python split_pdf_pages.py <input.pdf> [output_dir]", file=sys.stderr)
        return 1
    src = os.path.abspath(sys.argv[1])
    if not os.path.isfile(src):
        print(f"文件不存在: {src}", file=sys.stderr)
        return 1
    out_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_单页PDF"
    os.makedirs(out_dir, exist_ok=True)

    total = split_pdf(src, out_dir)

    # 校验：每个输出文件都必须是单页
    bad = []
    for name in sorted(os.listdir(out_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(out_dir, name)
        if len(PdfReader(path).pages) != 1:
            bad.append(name)
    if bad:
        print(f"警告: 以下文件非单页: {bad}", file=sys.stderr)
        return 1
    print(f"ALL DONE, {total} single-page files in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
