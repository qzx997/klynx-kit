import ast
import json
import os
import subprocess
from typing import Tuple

class SyntaxChecker:
    """
    代码语法检查器
    支持: Python, JSON, JavaScript (via node)
    """
    
    @staticmethod
    def check_file(path: str) -> str:
        """
        检查指定文件的语法
        
        Returns:
            语法检查结果消息
        """
        if not os.path.exists(path):
            return f"<error>文件不存在: {path}</error>"
            
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.py':
            return SyntaxChecker._check_python(path)
        elif ext == '.json':
            return SyntaxChecker._check_json(path)
        elif ext in ['.js', '.jsx', '.ts', '.tsx']:
            return SyntaxChecker._check_javascript(path)
        else:
            return f"<warning>暂不支持对 {ext} 文件进行语法检查</warning>"

    @staticmethod
    def _check_python(path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content, filename=path)
            return f"<success>Python 语法检查通过: {path}</success>"
        except SyntaxError as e:
            return f"<error>Python 语法错误:\n文件: {path}\n行号: {e.lineno}\n列号: {e.offset}\n错误: {e.msg}\n代码: {e.text.strip() if e.text else ''}</error>"
        except Exception as e:
            return f"<error>检查失败: {str(e)}</error>"

    @staticmethod
    def _check_json(path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                json.load(f)
            return f"<success>JSON 格式检查通过: {path}</success>"
        except json.JSONDecodeError as e:
            return f"<error>JSON 格式错误:\n文件: {path}\n行号: {e.lineno}\n列号: {e.colno}\n错误: {e.msg}</error>"
        except Exception as e:
            return f"<error>检查失败: {str(e)}</error>"

    @staticmethod
    def _check_javascript(path: str) -> str:
        # 需要系统安装 node
        try:
            result = subprocess.run(
                ['node', '--check', path],
                capture_output=True,
                text=True,
                timeout=10,
                shell=True # Windows 上可能需要 shell=True 来找到 node
            )
            
            if result.returncode == 0:
                return f"<success>JS/TS 语法检查通过: {path}</success>"
            else:
                # 尝试提取简洁错误信息
                stderr = result.stderr.strip()
                # 如果没有 stderr，看 stdout
                output = stderr if stderr else result.stdout.strip()
                return f"<error>JS/TS 语法错误:\n{output}</error>"
                
        except FileNotFoundError:
            return "<warning>未找到 node 命令，无法检查 JS 文件语法。请确保系统已安装 Node.js。</warning>"
        except subprocess.TimeoutExpired:
            return "<error>语法检查超时</error>"
        except Exception as e:
            return f"<error>执行检查失败: {str(e)}</error>"
