"""
PDF 解析工具

封装 pdf_parser 的批处理接口，提供 parse_pdfs() 函数。
"""

import json
import shutil
import time
from pathlib import Path
from typing import Union, Dict, Any

# 延迟导入 docling（较重的依赖）
_parse_pdf_func = None
_process_directory_func = None


def _lazy_import():
    """延迟导入原始 pdf_parser 模块。"""
    global _parse_pdf_func, _process_directory_func
    if _parse_pdf_func is not None:
        return

    # 尝试直接导入原始模块
    import sys
    original_parser_dir = str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "knowledge_db" / "tools" / "pdf_parser"
    )
    if original_parser_dir not in sys.path:
        sys.path.insert(0, original_parser_dir)

    try:
        from pdf_parser import parse_pdf as _pf, process_directory as _pd
        _parse_pdf_func = _pf
        _process_directory_func = _pd
    except Exception as e:
        print(f"⚠️  无法导入原始 pdf_parser，使用内置后备实现。错误: {e}")
        # 如果导入失败，使用内置最小实现
        _parse_pdf_func = _builtin_parse_pdf
        _process_directory_func = _builtin_process_directory


def _builtin_parse_pdf(file_path, **kwargs):
    """内置最小 PDF 解析（当原模块不可用时）。"""
    file_path = Path(file_path)
    
    # 1. 尝试 Docling
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = result.document
        markdown_text = doc.export_to_markdown()
        pages = len(doc.pages)
        title = getattr(doc, 'title', None)
    except Exception as e:
        print(f"  ⚠️ Docling 解析失败 ({e})，尝试 pypdfium2...")
        # 2. 回退到 pypdfium2
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(file_path))
            pages = len(pdf)
            text_parts = []
            for i in range(pages):
                page = pdf[i]
                text_page = page.get_textpage()
                text_parts.append(text_page.get_text_range())
            markdown_text = "\n\n".join(text_parts)
            title = file_path.stem
        except Exception as e2:
            print(f"  ❌ pypdfium2 解析也失败: {e2}")
            markdown_text = ""
            pages = 0
            title = file_path.stem

    return {
        "text": markdown_text,
        "images": [],
        "tables": [],
        "metadata": {
            "title": title or file_path.stem,
            "pages": pages,
            "source": str(file_path),
        },
    }


def _builtin_process_directory(input_dir, output_dir):
    """内置批处理（当原模块不可用时）。"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"未找到 PDF 文件: {input_path}")
        return

    existing = [d for d in output_path.iterdir() if d.is_dir() and "_paper_" in d.name]
    next_idx = len(existing) + 1

    success = fail = 0
    for i, pdf in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf.name}")
        try:
            result = _builtin_parse_pdf(pdf)
            ts = time.strftime("%Y%m%d_%H%M%S")
            folder = output_path / f"{next_idx + i - 1:02d}_paper_{ts}"
            folder.mkdir(exist_ok=True)

            (folder / f"{pdf.stem}_text.md").write_text(result["text"], encoding="utf-8")
            with open(folder / f"{pdf.stem}_parsed.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            shutil.move(str(pdf), str(folder / pdf.name))
            success += 1
        except Exception as e:
            print(f"  ❌ {e}")
            fail += 1

    print(f"\n完成: {success} 成功, {fail} 失败")


def parse_pdfs(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    解析目录下所有 PDF 文件，生成 Markdown + JSON 结构化输出。

    Args:
        input_dir: 输入目录（包含待解析的 PDF 文件）
        output_dir: 输出目录（每个 PDF 生成一个子文件夹，含 _text.md 和 _parsed.json）

    Returns:
        统计信息字典
    """
    _lazy_import()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    pdf_count = len(list(input_path.glob("*.pdf")))
    print(f"\n📄 PDF 解析")
    print(f"  输入: {input_path} ({pdf_count} 个 PDF)")
    print(f"  输出: {output_path}\n")

    if pdf_count == 0:
        print("  ⚠️ 未找到 PDF 文件")
        return {"total": 0, "success": 0, "failed": 0, "output_dir": str(output_path)}

    # 调用批处理
    _process_directory_func(str(input_path), str(output_path))

    # 统计结果
    result_folders = [d for d in output_path.iterdir() if d.is_dir() and "_paper_" in d.name]
    return {
        "total": pdf_count,
        "success": len(result_folders),
        "failed": pdf_count - len(result_folders),
        "output_dir": str(output_path),
    }


def parse_single_pdf(file_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    解析单个 PDF 文件。

    Args:
        file_path: PDF 文件路径
        output_dir: 输出目录（可选，不指定则只返回解析结果不保存文件）

    Returns:
        解析结果字典（含 text, images, tables, metadata）
    """
    _lazy_import()

    result = _parse_pdf_func(file_path)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(file_path).stem

        (out / f"{stem}_text.md").write_text(result["text"], encoding="utf-8")
        with open(out / f"{stem}_parsed.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result
