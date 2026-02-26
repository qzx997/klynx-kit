"""
Tavily 联网搜索工具

基于 Tavily API 提供联网搜索能力，允许 Agent 搜索最新信息。

API Key 配置方式（按优先级）：
  1. set_tavily_api("tvly-xxx")         — 代码中显式设置
  2. WebSearchTool(api_key="tvly-xxx")  — 构造时传入
  3. 环境变量 TAVILY_API_KEY
"""

import os
from typing import Optional

# 模块级 API Key 存储（由 set_tavily_api 设置）
_TAVILY_API_KEY: Optional[str] = None


def set_tavily_api(api_key: str):
    """
    设置 Tavily 搜索 API Key（全局生效）

    Usage:
        from klynx import set_tavily_api
        set_tavily_api("tvly-xxxxxxxxxxxxxxxx")

    之后创建的所有 Agent 实例将自动使用该 Key。
    """
    global _TAVILY_API_KEY
    _TAVILY_API_KEY = api_key


class WebSearchTool:
    """Tavily 联网搜索工具"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _TAVILY_API_KEY
        self._client = None
    
    def _get_client(self):
        """延迟初始化 Tavily 客户端"""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "未配置 TAVILY_API_KEY。请在 .env 文件中设置：\n"
                    "TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx"
                )
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)
        return self._client
    
    def search(self, query: str, max_results: int = 5, 
               search_depth: str = "basic",
               include_answer: bool = True) -> str:
        """
        执行联网搜索
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数 (1-10)
            search_depth: 搜索深度 ("basic" 或 "advanced")
            include_answer: 是否包含 AI 生成的摘要答案
        
        Returns:
            XML 格式的搜索结果
        """
        try:
            client = self._get_client()
            
            # 临时绕过代理（避免本地代理导致 SSL 错误）
            proxy_vars = {}
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                        "http_proxy", "https_proxy", "all_proxy"):
                if key in os.environ:
                    proxy_vars[key] = os.environ.pop(key)
            # 强制 requests 不使用任何代理（包括 Windows 系统代理）
            old_no_proxy = os.environ.get("NO_PROXY")
            os.environ["NO_PROXY"] = "*"
            
            try:
                result = client.search(
                    query=query,
                    max_results=min(int(max_results), 10),
                    search_depth=search_depth,
                    include_answer=include_answer
                )
            finally:
                # 恢复代理设置
                os.environ.update(proxy_vars)
                if old_no_proxy is not None:
                    os.environ["NO_PROXY"] = old_no_proxy
                elif "NO_PROXY" in os.environ:
                    del os.environ["NO_PROXY"]
            
            # 构建 XML 输出
            xml = [f'<search_results query="{self._escape_xml(query)}">']
            
            # AI 摘要答案
            if include_answer and result.get("answer"):
                xml.append(f'  <answer>{self._escape_xml(result["answer"])}</answer>')
            
            # 搜索结果列表
            for i, item in enumerate(result.get("results", []), 1):
                title = self._escape_xml(item.get("title", ""))
                url = self._escape_xml(item.get("url", ""))
                content = self._escape_xml(item.get("content", ""))
                score = item.get("score", 0)
                
                xml.append(f'  <result rank="{i}" score="{score:.2f}">')
                xml.append(f'    <title>{title}</title>')
                xml.append(f'    <url>{url}</url>')
                xml.append(f'    <content>{content}</content>')
                xml.append(f'  </result>')
            
            xml.append('</search_results>')
            return "\n".join(xml)
            
        except RuntimeError:
            raise
        except Exception as e:
            return f"<error>搜索失败: {str(e)}</error>"
    
    @staticmethod
    def _escape_xml(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
