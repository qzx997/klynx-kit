"""
格式化工具
"""


def format_tool_output(tool_name: str, output: str) -> str:
    """
    格式化工具输出
    
    Args:
        tool_name: 工具名称
        output: 工具输出内容
        
    Returns:
        格式化后的字符串
    """
    lines = [
        f"┌─ 工具执行: {tool_name}",
        "│",
    ]
    
    for line in output.split('\n'):
        lines.append(f"│ {line}")
    
    lines.append("└─")
    
    return '\n'.join(lines)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    截断文本
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 后缀
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
