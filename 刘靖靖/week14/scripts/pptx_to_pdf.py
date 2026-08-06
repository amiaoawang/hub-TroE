# -*- coding: utf-8 -*-
"""将 PPTX 导出为完整 PDF（多方案自动降级）。

用法:
    python pptx_to_pdf.py <input.pptx> [output.pdf]

依赖（按检测顺序，命中任一即可）:
    1. Microsoft PowerPoint COM（win32com，需安装 PowerPoint + pywin32）
    2. WPS 演示 COM（kwpp，需安装 WPS Office + pywin32）
    3. LibreOffice headless（soffice 命令行，需安装 LibreOffice）

注意:
    - 不要用 PowerShell 5.1 调 ExportAsFixedFormat（COM 枚举参数绑定会报
      DISP_E_TYPEMISMATCH），务必用 pywin32 或 LibreOffice。
    - 三个方案都不可用时，脚本会给出明确的安装建议。
"""
import os
import shutil
import subprocess
import sys


# ---------------- 方案 1: Microsoft PowerPoint COM ----------------
def _try_powerpoint_com(src: str, out: str):
    try:
        import win32com.client
    except ImportError:
        raise RuntimeError("未安装 pywin32，请先 pip install pywin32")
    try:
        ppt = win32com.client.Dispatch("PowerPoint.Application")
    except Exception as e:
        raise RuntimeError(f"未检测到 PowerPoint COM: {e}")
    slides = 0
    try:
        pres = ppt.Presentations.Open(src, ReadOnly=True, Untitled=False, WithWindow=False)
        slides = pres.Slides.Count
        print(f"[PowerPoint COM] opened, slides={slides}", flush=True)
        # ExportAsFixedFormat(Path, FixedFormatType, Intent, FrameSlides,
        #                     HandoutOrder, OutputType, PrintHiddenSlides,
        #                     PrintRange, RangeType)
        # 2=ppFixedFormatTypePDF, 1=ppFixedFormatIntentScreen, 0=msoFalse,
        # 1=ppPrintHandoutVerticalFirst, 1=ppPrintOutputSlides, 1=ppPrintAll
        pres.ExportAsFixedFormat(
            out, 2, 1, 0, 1, 1, 0, None, 1
        )
        print("[PowerPoint COM] exported OK", flush=True)
        pres.Close()
    finally:
        ppt.Quit()
    return slides


# ---------------- 方案 2: WPS 演示 COM ----------------
def _try_wps_com(src: str, out: str):
    try:
        import win32com.client
    except ImportError:
        raise RuntimeError("未安装 pywin32，请先 pip install pywin32")
    try:
        app = win32com.client.Dispatch("KWPP.Application")
    except Exception as e:
        raise RuntimeError(f"未检测到 WPS 演示 COM: {e}")
    slides = 0
    try:
        pres = app.Presentations.Open(src, ReadOnly=True, Untitled=False, WithWindow=False)
        slides = pres.Slides.Count
        print(f"[WPS COM] opened, slides={slides}", flush=True)
        # WPS 兼容 Office 的 ExportAsFixedFormat 枚举值
        pres.ExportAsFixedFormat(out, 2, 1, 0, 1, 1, 0, None, 1)
        print("[WPS COM] exported OK", flush=True)
        pres.Close()
    finally:
        app.Quit()
    return slides


# ---------------- 方案 3: LibreOffice headless ----------------
def _find_soffice():
    names = ["soffice", "soffice.com", "libreoffice"]
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "/usr/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _try_libreoffice(src: str, out: str):
    soffice = _find_soffice()
    if not soffice:
        raise RuntimeError("未找到 LibreOffice soffice 可执行文件")
    out_dir = os.path.dirname(out) or "."
    # soffice 输出文件名 = 输入名去掉扩展名 + .pdf
    exp = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + ".pdf")
    # 用独立 user profile 避免多实例锁冲突
    profile = os.path.join(out_dir, f".lo_profile_{os.getpid()}")
    cmd = [
        soffice, "--headless",
        f"-env:UserInstallation=file:///{profile.replace(os.sep, '/')}",
        "--convert-to", "pdf", "--outdir", out_dir, src,
    ]
    print(f"[LibreOffice] running: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if os.path.isfile(exp):
        if os.path.abspath(exp) != os.path.abspath(out):
            os.replace(exp, out)
        # 清理临时 profile
        shutil.rmtree(profile, ignore_errors=True)
        print("[LibreOffice] exported OK", flush=True)
        from pypdf import PdfReader
        return len(PdfReader(out).pages)
    raise RuntimeError(f"LibreOffice 转换失败: {r.stderr or r.stdout}")


def export_pptx_to_pdf(src: str, out: str) -> int:
    """按优先级尝试三个方案，返回页数。全部失败抛 RuntimeError。"""
    errors = []
    # 1) PowerPoint COM
    try:
        return _try_powerpoint_com(src, out)
    except RuntimeError as e:
        errors.append(f"  PowerPoint: {e}")
    # 2) WPS COM
    try:
        return _try_wps_com(src, out)
    except RuntimeError as e:
        errors.append(f"  WPS: {e}")
    # 3) LibreOffice
    try:
        return _try_libreoffice(src, out)
    except RuntimeError as e:
        errors.append(f"  LibreOffice: {e}")

    raise RuntimeError(
        "三种转换方案均不可用：\n" + "\n".join(errors) +
        "\n\n请安装任一方案后重试："
        "\n  - Microsoft PowerPoint（推荐，保真度最高）"
        "\n  - WPS Office（含演示组件）"
        "\n  - LibreOffice（免费开源，apt/brew/官方安装包均可）"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python pptx_to_pdf.py <input.pptx> [output.pdf]", file=sys.stderr)
        return 1
    src = os.path.abspath(sys.argv[1])
    if not os.path.isfile(src):
        print(f"文件不存在: {src}", file=sys.stderr)
        return 1
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        slides = export_pptx_to_pdf(src, out)
    except RuntimeError as e:
        print(f"导出失败: {e}", file=sys.stderr)
        return 1
    if not os.path.isfile(out):
        print("导出失败: 未生成输出文件", file=sys.stderr)
        return 1
    print(f"OK: {out} (slides={slides})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
