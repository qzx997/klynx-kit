"""
向量知识库创建工具

封装 ingest.py 的分块与写入逻辑，提供 create_vector_kb() 函数。
不修改原始 knowledge_db/vector_db/ingest.py。
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# 分块逻辑（与 ingest.py 保持一致，独立实现避免路径依赖）
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 800
MIN_CHUNK_CHARS = 100
OVERLAP_CHARS = 80


def _split_by_sections(text: str) -> List[Tuple[str, str]]:
    """按 Markdown 标题分割为 (section_title, section_body) 列表。"""
    pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    sections: List[Tuple[str, str]] = []
    last_end = 0
    last_title = "Abstract / Header"

    for m in pattern.finditer(text):
        body = text[last_end:m.start()].strip()
        if body:
            sections.append((last_title, body))
        last_title = m.group(2).strip()
        last_end = m.end()

    tail = text[last_end:].strip()
    if tail:
        sections.append((last_title, tail))
    return sections


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS,
                min_chars: int = MIN_CHUNK_CHARS,
                overlap: int = OVERLAP_CHARS) -> List[str]:
    """将文本按段落分块。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                while para:
                    chunks.append(para[:max_chars])
                    para = para[max_chars - overlap:] if len(para) > max_chars else ""
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # 合并过短的尾部 chunk
    merged: List[str] = []
    for c in chunks:
        if merged and len(c) < min_chars and len(merged[-1]) + len(c) + 2 <= max_chars:
            merged[-1] = f"{merged[-1]}\n\n{c}"
        else:
            merged.append(c)
    return merged


def _build_chunks_for_file(md_path: Path, doc_id: str,
                           metadata_extra: Optional[Dict] = None) -> List[Dict]:
    """读取 Markdown 文件，返回 chunk 字典列表。"""
    meta = metadata_extra or {}
    text = md_path.read_text(encoding="utf-8")
    sections = _split_by_sections(text)

    all_chunks: List[Dict] = []
    global_idx = 0

    for section_title, section_body in sections:
        chunks = _chunk_text(section_body)
        for _, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}__sec_{section_title[:30].replace(' ', '_')}__chunk_{global_idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "paper_id": doc_id,
                    "paper_title": meta.get("title", doc_id),
                    "source_file": str(md_path.name),
                    "section": section_title,
                    "chunk_index": global_idx,
                    "total_pages": meta.get("pages", 0),
                },
            })
            global_idx += 1
    return all_chunks


def _discover_documents(input_dir: Path) -> List[Dict]:
    """
    扫描输入目录，发现所有文档文件夹（含 *_text.md）。
    支持 ingest.py 的目录格式：每个子文件夹含 *_text.md 和 *_parsed.json。
    """
    docs = []
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    for entry in sorted(input_dir.iterdir()):
        if not entry.is_dir():
            continue
        md_files = list(entry.glob("*_text.md"))
        json_files = list(entry.glob("*_parsed.json"))
        if not md_files:
            continue
        docs.append({
            "doc_id": entry.name,
            "md_path": md_files[0],
            "json_path": json_files[0] if json_files else None,
        })
    return docs


def _load_metadata(json_path: Optional[Path]) -> Dict:
    """从 parsed JSON 中提取元数据。"""
    if json_path is None or not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return data.get("metadata", {})
    except Exception:
        return {}


def create_vector_kb(
    name: str,
    input_dir: str,
    output_dir: str,
    overwrite: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> str:
    """
    创建向量知识库集合。

    Args:
        name: 集合名称（如 "quantum_memory", "my_papers"）
        input_dir: 输入文件夹路径（含子文件夹，每个子文件夹有 *_text.md）
        output_dir: 输出文件夹路径（ChromaDB 存储位置，写入 chroma_store 子目录）
        overwrite: 是否覆盖已有知识库。
                   True  -> 删除旧集合并重建
                   False -> 若已存在同名 chroma_store，创建带时间后缀的新目录
        embedding_model: 嵌入模型名称或本地路径

    Returns:
        实际的 chroma_store 路径
    """
    import chromadb
    from chromadb.utils import embedding_functions

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 确定 chroma_store 路径
    chroma_dir = output_path / "chroma_store"

    if chroma_dir.exists() and not overwrite:
        # 不覆盖：创建带时间后缀的新目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        chroma_dir = output_path / f"chroma_store_{timestamp}"
        print(f"[知识库] 已有 chroma_store，创建新目录: {chroma_dir}")

    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 1. 发现文档
    print(f"\n📂 扫描输入目录: {input_path}")
    docs = _discover_documents(input_path)
    print(f"  发现 {len(docs)} 个文档文件夹\n")

    if not docs:
        print("  ⚠️ 未找到任何文档（需要子文件夹含 *_text.md 文件）")
        return str(chroma_dir)

    # 2. 构建 chunks
    all_chunks: List[Dict] = []
    for d in docs:
        metadata = _load_metadata(d["json_path"])
        chunks = _build_chunks_for_file(d["md_path"], d["doc_id"], metadata)
        all_chunks.extend(chunks)
        print(f"  📄 {d['doc_id']}: {len(chunks)} chunks "
              f"(title: {metadata.get('title', 'N/A')})")

    print(f"\n  总计: {len(all_chunks)} chunks")

    # 3. 创建 ChromaDB 客户端和集合
    print(f"\n💾 写入 ChromaDB: {chroma_dir}")
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # 配置嵌入模型
    model_path = Path(embedding_model)
    if model_path.exists():
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(model_path)
        )
    else:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

    # 处理覆盖
    if overwrite:
        try:
            client.delete_collection(name)
            print(f"  已删除旧集合 '{name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=name,
        metadata={"description": f"向量知识库: {name}"},
        embedding_function=ef
    )

    # 4. 批量写入
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"  ✅ 成功写入 {collection.count()} 条记录到集合 '{name}'")
    print(f"  📁 存储路径: {chroma_dir}")

    return str(chroma_dir)
