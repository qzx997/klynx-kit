"""
多知识库管理器

管理多个 ChromaDB 知识库路径，支持跨知识库语义检索。
替代原有的单路径 KnowledgeBaseTool。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class KBManager:
    """
    多知识库管理器
    
    通过 add_kb / remove_kb 管理多个 ChromaDB 知识库路径，
    query() 时可查询所有或指定知识库，按相似度合并结果。
    
    Usage:
        mgr = KBManager()
        mgr.add_kb("papers", "/path/to/chroma_store")
        mgr.add_kb("notes",  "/path/to/another_store")
        result = mgr.query("quantum memory", top_k=5)
    """

    def __init__(self, default_embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            default_embedding_model: 默认嵌入模型名称或本地路径
        """
        self._kbs: Dict[str, str] = {}          # name -> chroma_store_path
        self._kb_models: Dict[str, str] = {}    # name -> embedding model override
        self._clients: Dict[str, object] = {}   # name -> PersistentClient (lazy)
        self._default_model = default_embedding_model

    # -------
    # 管理 API
    # -------

    def add_kb(self, name: str, path: str,
               embedding_model: Optional[str] = None) -> None:
        """
        添加一个知识库。

        Args:
            name: 知识库名称（标识符）
            path: ChromaDB chroma_store 目录路径
            embedding_model: 该知识库使用的嵌入模型（可选，默认用全局配置）
        """
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"知识库路径不存在: {path}")
        self._kbs[name] = path
        if embedding_model:
            self._kb_models[name] = embedding_model
        # 清除旧缓存，下次查询时重建连接
        self._clients.pop(name, None)
        print(f"[KBManager] 已添加知识库 '{name}' -> {path}")

    def remove_kb(self, name: str) -> None:
        """移除一个知识库。"""
        self._kbs.pop(name, None)
        self._kb_models.pop(name, None)
        self._clients.pop(name, None)

    def list_kbs(self) -> Dict[str, str]:
        """返回所有已注册知识库 {name: path}。"""
        return dict(self._kbs)

    def has_kb(self) -> bool:
        """是否已注册任何知识库。"""
        return bool(self._kbs)

    # ----------
    # 查询 API
    # ----------

    def _get_client(self, name: str):
        """延迟初始化 ChromaDB PersistentClient。"""
        if name in self._clients:
            return self._clients[name]

        import chromadb
        path = self._kbs.get(name)
        if not path:
            raise ValueError(f"未注册的知识库: {name}")
        client = chromadb.PersistentClient(path=path)
        self._clients[name] = client
        return client

    def _find_local_model(self, kb_path: str, model_name: str) -> Optional[str]:
        """
        在知识库路径的父级目录中搜索本地嵌入模型。
        
        典型目录结构:
            knowledge_db/
              ├── all-MiniLM-L6-v2/   <-- model
              └── vector_db/
                  └── chroma_store/    <-- kb_path
        """
        search_dir = Path(kb_path)
        # 向上搜索最多 3 级父目录
        for _ in range(4):
            candidate = search_dir / model_name
            if candidate.is_dir():
                return str(candidate)
            search_dir = search_dir.parent
        return None

    def _get_embedding_function(self, kb_name: Optional[str] = None):
        """创建嵌入函数，优先使用本地模型。"""
        from chromadb.utils import embedding_functions

        # 确定模型标识
        model = (self._kb_models.get(kb_name) if kb_name else None) or self._default_model
        model_path = Path(model)

        # 1. 如果模型标识本身就是一个已存在的路径
        if model_path.is_dir():
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=str(model_path)
            )

        # 2. 在各 KB 路径的父级目录中搜索本地模型
        search_paths = [self._kbs[kb_name]] if kb_name and kb_name in self._kbs else list(self._kbs.values())
        for kb_path in search_paths:
            local = self._find_local_model(kb_path, model)
            if local:
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=local
                )

        # 3. 回退到在线下载
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model
        )

    def query(self, query: str, top_k: int = 5,
              kb_name: Optional[str] = None,
              collection_name: Optional[str] = None) -> str:
        """
        语义检索知识库。

        Args:
            query: 查询文本（自然语言）
            top_k: 返回最相似的结果数量 (1-20)
            kb_name: 指定查询某个知识库（None 则查询所有）
            collection_name: 指定集合名称（None 则使用每个知识库的第一个集合）

        Returns:
            XML 格式的检索结果
        """
        if not self._kbs:
            return "<error>未配置任何知识库。请先使用 agent.add_kb() 添加知识库路径。</error>"

        top_k = min(max(int(top_k), 1), 20)
        targets = {kb_name: self._kbs[kb_name]} if kb_name else self._kbs

        # 收集所有结果: (similarity, doc, meta, kb_name, col_name)
        all_results: List[Tuple[float, str, dict, str, str]] = []

        for name, path in targets.items():
            try:
                client = self._get_client(name)
                collections = client.list_collections()
                if not collections:
                    continue

                ef = self._get_embedding_function(kb_name=name)

                for col_meta in collections:
                    col_name = col_meta if isinstance(col_meta, str) else col_meta.name
                    if collection_name and col_name != collection_name:
                        continue

                    col = client.get_collection(name=col_name,
                                                embedding_function=ef)
                    results = col.query(query_texts=[query], n_results=top_k)

                    for doc, meta, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ):
                        score = max(0, 1 - dist)
                        all_results.append((score, doc, meta, name, col_name))

            except Exception as e:
                all_results.append((-1, f"查询 '{name}' 失败: {e}", {}, name, ""))

        # 按相似度降序排序，取 top_k
        all_results.sort(key=lambda x: x[0], reverse=True)
        top_results = all_results[:top_k]

        # 构建 XML
        xml = [f'<knowledge_results query="{self._escape_xml(query)}" '
               f'total="{len(top_results)}" kbs="{",".join(targets.keys())}">']

        for i, (score, doc, meta, kb, col) in enumerate(top_results):
            if score < 0:
                xml.append(f'  <error>{self._escape_xml(doc)}</error>')
                continue

            title = self._escape_xml(meta.get("paper_title", "N/A"))
            section = self._escape_xml(meta.get("section", "N/A"))
            source = self._escape_xml(meta.get("source_file", ""))
            content = self._escape_xml(doc)

            xml.append(f'  <result rank="{i+1}" similarity="{score:.3f}" '
                       f'kb="{kb}" collection="{col}">')
            xml.append(f'    <title>{title}</title>')
            xml.append(f'    <section>{section}</section>')
            xml.append(f'    <source>{source}</source>')
            xml.append(f'    <content>{content}</content>')
            xml.append(f'  </result>')

        xml.append('</knowledge_results>')
        return "\n".join(xml)

    def list_collections(self, kb_name: Optional[str] = None) -> str:
        """列出知识库中所有可用的集合。"""
        if not self._kbs:
            return "<error>未配置任何知识库</error>"

        targets = {kb_name: self._kbs[kb_name]} if kb_name else self._kbs
        xml = ['<knowledge_collections>']

        for name, path in targets.items():
            try:
                client = self._get_client(name)
                collections = client.list_collections()
                for col_meta in collections:
                    col_name = col_meta if isinstance(col_meta, str) else col_meta.name
                    col = client.get_collection(name=col_name)
                    xml.append(f'  <collection kb="{name}" '
                               f'name="{col_name}" count="{col.count()}" />')
            except Exception as e:
                xml.append(f'  <error kb="{name}">{str(e)}</error>')

        xml.append('</knowledge_collections>')
        return "\n".join(xml)

    @staticmethod
    def _escape_xml(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
