"""
XML 解析工具
"""

import re
from typing import List, Dict, Any


class XMLParser:
    """XML 工具调用解析器"""
    
    @staticmethod
    def extract_thinking(content: str) -> str:
        """提取 thinking 标签内容"""
        pattern = r'<thinking>(.*?)</thinking>'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    @staticmethod
    def extract_attempt_completion(content: str) -> str:
        """提取 attempt_completion 标签内容"""
        pattern = r'<attempt_completion>(.*?)</attempt_completion>'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    @staticmethod
    def has_thinking(content: str) -> bool:
        """检查是否包含 thinking 标签"""
        return '<thinking>' in content
    
    @staticmethod
    def has_attempt_completion(content: str) -> bool:
        """检查是否包含 attempt_completion 标签"""
        return '<attempt_completion>' in content
    
    @staticmethod
    def clean_xml_tags(content: str) -> str:
        """清理 XML 标签，返回纯文本"""
        # 移除所有 XML 标签
        cleaned = re.sub(r'<[^>]+>', '', content)
        return cleaned.strip()
