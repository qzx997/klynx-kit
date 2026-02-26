"""
Klynx Agent - Token计数器
基于字符估算Token数量
"""

import re
from typing import List
from langchain_core.messages import BaseMessage


class TokenCounter:
    """Token计数器（基于字符估算）"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        估算文本的token数量
        中文字符：0.6 token/字符
        英文字符：0.3 token/字符
        """
        if not text:
            return 0
        
        # 中文字符计数
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        
        # 中文 0.6 token/字符，英文 0.3 token/字符
        tokens = chinese_chars * 0.6 + other_chars * 0.3
        return int(tokens)
    
    @staticmethod
    def count_message_tokens(messages: List[BaseMessage]) -> int:
        """计算消息列表的总token数"""
        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            total += TokenCounter.estimate_tokens(content)
        return total
