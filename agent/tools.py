"""
Klynx Agent - Tool Registry
实现工具注册表，包含文件操作和命令执行功能
"""

import os
import re
import subprocess
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path


class ToolRegistry:
    """工具注册表，提供文件操作和命令执行功能"""
    
    # 类级别的 working_dir
    working_dir = "."
    
    @classmethod
    def set_working_dir(cls, working_dir: str):
        """设置工作目录"""
        cls.working_dir = working_dir
    
    @classmethod
    def _resolve_path(cls, path: str) -> Path:
        """解析路径，转换为绝对路径"""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(cls.working_dir) / file_path
        return file_path.resolve()
    
    @classmethod
    def read_file(cls, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """
        读取文件内容，返回带行号的内容
        
        Args:
            path: 文件路径
            start_line: 起始行号（从1开始）
            end_line: 结束行号（包含）
            
        Returns:
            带行号的文件内容，格式: "12 | content"
        """
        try:
            file_path = cls._resolve_path(path)
            if not file_path.exists():
                return f"<error>文件不存在: {path}</error>"
            
            if not file_path.is_file():
                return f"<error>路径不是文件: {path}</error>"
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 处理行号范围
            total_lines = len(lines)
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else total_lines
            
            # 边界检查
            start = max(0, start)
            end = min(total_lines, end)
            
            # 生成带行号的输出
            result_lines = []
            for i in range(start, end):
                line_num = i + 1
                line_content = lines[i].rstrip('\n').rstrip('\r')
                result_lines.append(f"{line_num:4d} | {line_content}")
            
            return '\n'.join(result_lines)
            
        except Exception as e:
            return f"<error>读取文件失败: {str(e)}</error>"
    
    @classmethod
    def write_to_file(cls, path: str, content: str) -> str:
        """
        写入文件（全量覆盖或新建）
        
        Args:
            path: 文件路径
            content: 文件内容
            
        Returns:
            操作结果消息
        """
        try:
            file_path = cls._resolve_path(path)
            
            # 确保父目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"<success>文件已成功写入: {path}</success>"
            
        except Exception as e:
            return f"<error>写入文件失败: {str(e)}</error>"
    
    @classmethod
    def replace_in_file(cls, path: str, diff_string: str) -> str:
        """
        使用 SEARCH/REPLACE 块替换文件内容
        
        Args:
            path: 文件路径
            diff_string: 包含 SEARCH/REPLACE 块的差异字符串
                        格式:
                        <<<<<<< SEARCH
                        要搜索的内容
                        =======
                        替换后的内容
                        >>>>>>> REPLACE
                        
        Returns:
            操作结果消息
        """
        try:
            file_path = cls._resolve_path(path)
            if not file_path.exists():
                return f"<error>文件不存在: {path}</error>"
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
            
            # 解析 SEARCH/REPLACE 块
            pattern = r'<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE'
            matches = re.findall(pattern, diff_string, re.DOTALL)
            
            if not matches:
                return "<error>无法解析 SEARCH/REPLACE 块，请检查格式</error>"
            
            modified_content = original_content
            replacements_count = 0
            
            for search_block, replace_block in matches:
                # 移除末尾的换行符以匹配
                search_block = search_block.rstrip('\n')
                replace_block = replace_block.rstrip('\n')
                
                if search_block in modified_content:
                    modified_content = modified_content.replace(search_block, replace_block, 1)
                    replacements_count += 1
                else:
                    return f"<error>无法找到匹配内容: {search_block[:50]}...</error>"
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            return f"<success>成功完成 {replacements_count} 处替换</success>"
            
        except Exception as e:
            return f"<error>替换操作失败: {str(e)}</error>"
    
    @classmethod
    def execute_command(cls, command: str, timeout: int = 30, cwd: Optional[str] = None) -> str:
        """
        执行系统命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间（秒）
            cwd: 工作目录
            
        Returns:
            命令输出结果（stdout 和 stderr）
        """
        try:
            # 安全检查 - 阻止危险命令
            dangerous_patterns = [
                r'rm\s+-rf\s+/',
                r'rm\s+-rf\s+~',
                r'>:\s*/dev/',
                r'dd\s+if=.*of=/dev/',
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return f"<error>检测到危险命令，已阻止执行: {command}</error>"
            
            # 执行命令，使用工作目录
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or cls.working_dir
            )
            
            output_parts = []
            
            if result.stdout:
                output_parts.append(f"[STDOUT]\n{result.stdout}")
            
            if result.stderr:
                output_parts.append(f"[STDERR]\n{result.stderr}")
            
            if result.returncode != 0:
                output_parts.append(f"[EXIT CODE] {result.returncode}")
            
            return '\n'.join(output_parts) if output_parts else "<success>命令执行成功（无输出）</success>"
            
        except subprocess.TimeoutExpired:
            return f"<error>命令执行超时（超过 {timeout} 秒）</error>"
        except Exception as e:
            return f"<error>命令执行失败: {str(e)}</error>"
    
    @classmethod
    def create_directory(cls, path: str) -> str:
        """
        创建目录（包括父目录）
        
        Args:
            path: 目录路径
            
        Returns:
            操作结果消息
        """
        try:
            dir_path = cls._resolve_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return f"<success>目录已成功创建: {path}</success>"
        except Exception as e:
            return f"<error>创建目录失败: {str(e)}</error>"
    
    @classmethod
    def preview_file(cls, path: str, num_lines: int = 50) -> str:
        """
        预览文件内容（只读取前 N 行）
        
        Args:
            path: 文件路径
            num_lines: 预览行数（默认50行）
            
        Returns:
            带行号的文件预览内容
        """
        try:
            file_path = cls._resolve_path(path)
            if not file_path.exists():
                return f"<error>文件不存在: {path}</error>"
            
            if not file_path.is_file():
                return f"<error>路径不是文件: {path}</error>"
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            preview = lines[:num_lines]
            
            result_lines = []
            for i, line in enumerate(preview, 1):
                content = line.rstrip('\n').rstrip('\r')
                result_lines.append(f"{i:4d} | {content}")
            
            result = '\n'.join(result_lines)
            
            if total_lines > num_lines:
                result += f"\n     | ... ({total_lines - num_lines} 行未显示，共 {total_lines} 行)"
            
            return result
            
        except Exception as e:
            return f"<error>预览文件失败: {str(e)}</error>"
    
    @classmethod
    def list_directory(cls, path: str = ".", depth: int = 2) -> str:
        """
        列出目录结构
        
        Args:
            path: 目录路径
            depth: 递归深度
            
        Returns:
            目录树结构字符串
        """
        try:
            target_path = cls._resolve_path(path)
            if not target_path.exists():
                return f"<error>路径不存在: {path}</error>"
            
            if not target_path.is_dir():
                return f"<error>路径不是目录: {path}</error>"
            
            lines = []
            
            def build_tree(current_path: Path, prefix: str = "", current_depth: int = 0):
                if current_depth > depth:
                    return
                
                try:
                    entries = sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                except PermissionError:
                    lines.append(f"{prefix}[权限拒绝]")
                    return
                
                for i, entry in enumerate(entries):
                    is_last = (i == len(entries) - 1)
                    connector = "└── " if is_last else "├── "
                    
                    # 跳过隐藏文件和特定目录
                    if entry.name.startswith('.') and entry.name not in ['.gitignore', '.env.example']:
                        continue
                    
                    if entry.is_dir():
                        lines.append(f"{prefix}{connector}{entry.name}/")
                        if current_depth < depth:
                            extension = "    " if is_last else "│   "
                            build_tree(entry, prefix + extension, current_depth + 1)
                    else:
                        lines.append(f"{prefix}{connector}{entry.name}")
            
            lines.append(f"{target_path.name}/")
            build_tree(target_path)
            
            return '\n'.join(lines)
            
        except Exception as e:
            return f"<error>列出目录失败: {str(e)}</error>"
    
    @classmethod
    def search_in_files(cls, pattern: str, path: str = ".", file_pattern: str = "*",
                        case_insensitive: bool = True, is_regex: bool = False,
                        max_results: int = 50, context_lines: int = 0) -> str:
        """
        在文件中搜索匹配内容（类似 grep）
        
        Args:
            pattern: 搜索模式（文本或正则表达式）
            path: 搜索目录或文件路径（默认工作目录）
            file_pattern: 文件名匹配模式（如 *.py, *.js）
            case_insensitive: 是否忽略大小写（默认True）
            is_regex: 是否为正则表达式（默认False）
            max_results: 最大结果数（默认50）
            context_lines: 显示匹配行前后的上下文行数（默认0）
            
        Returns:
            匹配结果，格式: "文件路径:行号: 行内容"
        """
        import fnmatch
        
        try:
            search_path = cls._resolve_path(path)
            
            if not search_path.exists():
                return f"<error>路径不存在: {path}</error>"
            
            # 编译搜索模式
            flags = re.IGNORECASE if case_insensitive else 0
            if is_regex:
                try:
                    regex = re.compile(pattern, flags)
                except re.error as e:
                    return f"<error>无效的正则表达式: {e}</error>"
            else:
                regex = re.compile(re.escape(pattern), flags)
            
            results = []
            files_searched = 0
            
            # 收集要搜索的文件
            if search_path.is_file():
                files_to_search = [search_path]
            else:
                files_to_search = []
                skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vscode'}
                for root, dirs, files in os.walk(search_path):
                    dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
                    for filename in files:
                        if fnmatch.fnmatch(filename, file_pattern):
                            files_to_search.append(Path(root) / filename)
            
            for file_path in files_to_search:
                if len(results) >= max_results:
                    break
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    files_searched += 1
                    
                    for line_num, line in enumerate(lines, 1):
                        if len(results) >= max_results:
                            break
                        
                        if regex.search(line):
                            # 使用相对路径显示
                            try:
                                display_path = file_path.relative_to(search_path)
                            except ValueError:
                                display_path = file_path
                            
                            line_content = line.rstrip('\n').rstrip('\r')
                            result_entry = f"{display_path}:{line_num}: {line_content}"
                            results.append(result_entry)
                            
                            # 添加上下文行
                            if context_lines > 0:
                                for ctx_offset in range(-context_lines, context_lines + 1):
                                    if ctx_offset == 0:
                                        continue
                                    ctx_line_num = line_num + ctx_offset - 1
                                    if 0 <= ctx_line_num < len(lines):
                                        ctx_content = lines[ctx_line_num].rstrip('\n').rstrip('\r')
                                        results.append(f"  {display_path}:{ctx_line_num + 1}: {ctx_content}")
                            
                except (UnicodeDecodeError, PermissionError, OSError):
                    continue
            
            if not results:
                return f"<info>未找到匹配: '{pattern}' (搜索了 {files_searched} 个文件)</info>"
            
            header = f"找到 {len(results)} 个匹配 (搜索了 {files_searched} 个文件)"
            if len(results) >= max_results:
                header += f" [结果已截断，最多 {max_results} 条]"
            
            return header + "\n" + "\n".join(results)
            
        except Exception as e:
            return f"<error>搜索失败: {str(e)}</error>"
    
    @classmethod
    def execute(cls, tool_call: dict) -> str:
        """
        根据工具调用字典执行相应的工具
        
        Args:
            tool_call: 工具调用字典，包含 'tool' 和 'params' 键
            
        Returns:
            工具执行结果
        """
        tool_name = tool_call.get('tool')
        params = tool_call.get('params', {})
        
        tool_map = {
            'read_file': cls.read_file,
            'write_to_file': cls.write_to_file,
            'replace_in_file': cls.replace_in_file,
            'execute_command': cls.execute_command,
            'list_directory': cls.list_directory,
            'create_directory': cls.create_directory,
            'preview_file': cls.preview_file,
            'search_in_files': cls.search_in_files,
        }
        
        if tool_name not in tool_map:
            return f"<error>未知工具: {tool_name}</error>"
        
        try:
            return tool_map[tool_name](**params)
        except Exception as e:
            return f"<error>工具执行异常: {str(e)}</error>"


# ======================== Native Function Calling JSON Schemas ========================

# 标准 OpenAI Function Calling 格式的工具定义
# 当 tool_call_mode="native" 时，这些 Schema 会被传递给 LiteLLM 
TOOL_SCHEMAS = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容，支持指定行范围",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "start_line": {"type": "integer", "description": "起始行号（从1开始）"},
                    "end_line": {"type": "integer", "description": "结束行号（包含）"}
                },
                "required": ["path"]
            }
        }
    },
    "write_to_file": {
        "type": "function",
        "function": {
            "name": "write_to_file",
            "description": "将内容写入文件（覆盖或新建）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    "replace_in_file": {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "使用 SEARCH/REPLACE 块局部替换文件内容。格式：<<<<<<< SEARCH\\n要搜索的内容\\n=======\\n替换后的内容\\n>>>>>>> REPLACE",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "diff_string": {"type": "string", "description": "SEARCH/REPLACE 块差异字符串"}
                },
                "required": ["path", "diff_string"]
            }
        }
    },
    "execute_command": {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "执行系统命令（Windows 环境：dir, type, copy 等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"]
            }
        }
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出目录树结构，支持指定递归深度",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认为当前目录", "default": "."},
                    "depth": {"type": "integer", "description": "递归深度，默认 2", "default": 2}
                },
                "required": []
            }
        }
    },
    "create_directory": {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "创建目录（包括父目录）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径"}
                },
                "required": ["path"]
            }
        }
    },
    "preview_file": {
        "type": "function",
        "function": {
            "name": "preview_file",
            "description": "预览文件前 N 行内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "num_lines": {"type": "integer", "description": "预览行数，默认 50", "default": 50}
                },
                "required": ["path"]
            }
        }
    },
    "search_in_files": {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "在文件中搜索匹配内容（类似grep，支持正则）。定位代码位置后，再用 read_file 精确阅读。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "搜索模式"},
                    "path": {"type": "string", "description": "搜索路径，默认 '.'", "default": "."},
                    "file_pattern": {"type": "string", "description": "文件名模式，如 '*.py'", "default": "*"},
                    "case_insensitive": {"type": "boolean", "description": "是否忽略大小写", "default": True},
                    "is_regex": {"type": "boolean", "description": "是否为正则表达式", "default": False},
                    "max_results": {"type": "integer", "description": "最大结果数", "default": 50},
                    "context_lines": {"type": "integer", "description": "上下文行数", "default": 0}
                },
                "required": ["pattern"]
            }
        }
    },
    "create_terminal": {
        "type": "function",
        "function": {
            "name": "create_terminal",
            "description": "创建终端会话。仅用于后台批量执行非交互式命令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "终端名称"}
                },
                "required": ["name"]
            }
        }
    },
    "run_in_terminal": {
        "type": "function",
        "function": {
            "name": "run_in_terminal",
            "description": "在终端执行命令",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "终端名称"},
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["name", "command"]
            }
        }
    },
    "read_terminal": {
        "type": "function",
        "function": {
            "name": "read_terminal",
            "description": "读取终端输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "终端名称"},
                    "lines": {"type": "integer", "description": "读取行数"}
                },
                "required": ["name", "lines"]
            }
        }
    },
    "check_syntax": {
        "type": "function",
        "function": {
            "name": "check_syntax",
            "description": "检查代码语法",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }
    },
    "open_tui": {
        "type": "function",
        "function": {
            "name": "open_tui",
            "description": "启动 TUI 应用",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI 名称"},
                    "command": {"type": "string", "description": "启动命令"},
                    "rows": {"type": "integer", "description": "行数"},
                    "cols": {"type": "integer", "description": "列数"}
                },
                "required": ["name", "command"]
            }
        }
    },
    "read_tui": {
        "type": "function",
        "function": {
            "name": "read_tui",
            "description": "读取 TUI 屏幕快照",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI 名称"}
                },
                "required": ["name"]
            }
        }
    },
    "send_keys": {
        "type": "function",
        "function": {
            "name": "send_keys",
            "description": "向 TUI 发送按键",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI 名称"},
                    "keys": {"type": "string", "description": "按键序列"}
                },
                "required": ["name", "keys"]
            }
        }
    },
    "close_tui": {
        "type": "function",
        "function": {
            "name": "close_tui",
            "description": "关闭 TUI 会话",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "TUI 名称"}
                },
                "required": ["name"]
            }
        }
    },
    "activate_tui_mode": {
        "type": "function",
        "function": {
            "name": "activate_tui_mode",
            "description": "激活 TUI 交互模式：加载 TUI 操作指南到系统提示。在使用 open_tui 等 TUI 工具前必须先调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "launch_interactive_session": {
        "type": "function",
        "function": {
            "name": "launch_interactive_session",
            "description": "一键启动交互式会话（推荐）：自动激活 TUI 模式并启动应用。如需启动交互式环境（如 Python REPL, Node, Vim）或查看实时屏幕输出，请优先使用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "启动命令，如 'python', 'vim test.py'"}
                },
                "required": ["command"]
            }
        }
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索。当你需要搜索任何信息（最新动态、技术文档、API用法、新闻、事实核查等），都必须使用此工具进行搜索，严禁凭记忆编造。可设 search_depth='advanced' 进行深度搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "max_results": {"type": "integer", "description": "最大结果数", "default": 5},
                    "search_depth": {"type": "string", "description": "搜索深度: 'basic' 或 'advanced'", "default": "basic"}
                },
                "required": ["query"]
            }
        }
    },
    "query_knowledge": {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": "查询向量知识库。当用户询问专业领域知识时，优先使用此工具查询本地向量知识库获取精准的专业内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询问题"},
                    "top_k": {"type": "integer", "description": "返回结果数", "default": 5},
                    "kb_name": {"type": "string", "description": "知识库名称（可选）"}
                },
                "required": ["query"]
            }
        }
    },
    "list_knowledge_collections": {
        "type": "function",
        "function": {
            "name": "list_knowledge_collections",
            "description": "列出所有已添加的知识库及其集合",
            "parameters": {
                "type": "object",
                "properties": {
                    "kb_name": {"type": "string", "description": "知识库名称（可选）"}
                },
                "required": []
            }
        }
    },
    "browser_open": {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": "打开浏览器并访问 URL。常用于验证前端界面或获取动态网页内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要访问的 URL"}
                },
                "required": ["url"]
            }
        }
    },
    "browser_view": {
        "type": "function",
        "function": {
            "name": "browser_view",
            "description": "获取页面内容。不传 selector 则返回页面摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS 选择器（可选）"}
                },
                "required": []
            }
        }
    },
    "browser_act": {
        "type": "function",
        "function": {
            "name": "browser_act",
            "description": "操作页面元素。action 支持 click, type, hover。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "操作类型: click, type, hover"},
                    "selector": {"type": "string", "description": "CSS 选择器"},
                    "value": {"type": "string", "description": "输入值（type 时使用）"}
                },
                "required": ["action", "selector"]
            }
        }
    },
    "browser_screenshot": {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "截图并保存。返回截图路径。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "browser_console_logs": {
        "type": "function",
        "function": {
            "name": "browser_console_logs",
            "description": "获取浏览器控制台日志",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    "attempt_completion": {
        "type": "function",
        "function": {
            "name": "attempt_completion",
            "description": "当任务完成时必须调用此工具，提交最终结果。在 Native Function Calling 模式下这是唯一的任务结束方式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "任务的最终结果、回复或总结"}
                },
                "required": ["result"]
            }
        }
    },
    "update_task_state": {
        "type": "function",
        "function": {
            "name": "update_task_state",
            "description": "更新任务的全局状态。在执行中如果由于新信息需要修正总目标（overall_goal）或推进到下一个具体子任务（current_task），须调用此工具进行状态同步。",
            "parameters": {
                "type": "object",
                "properties": {
                    "overall_goal": {"type": "string", "description": "任务的总目标（可选，若不需要修改请勿传）"},
                    "current_task": {"type": "string", "description": "当前或下一步应该执行的具体操作（必传）"}
                },
                "required": ["current_task"]
            }
        }
    }
}


def get_json_schemas(tool_names: list, external_tools: dict = None) -> list:
    """
    根据已加载的工具名列表，生成 LiteLLM/OpenAI Function Calling 的 tools 数组
    
    Args:
        tool_names: 已加载的基础工具名列表
        external_tools: 外部工具 {name: description} 字典（可选）
        
    Returns:
        OpenAI tools 格式的列表
    """
    schemas = []
    for name in tool_names:
        if name in TOOL_SCHEMAS:
            schemas.append(TOOL_SCHEMAS[name])
    
    # 为外部工具（用户自定义的 callable）生成通用 schema
    if external_tools:
        for name, desc in external_tools.items():
            if name not in TOOL_SCHEMAS:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc if isinstance(desc, str) else f"外部工具: {name}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string", "description": "工具输入"}
                            },
                            "required": ["input"]
                        }
                    }
                })
    
    return schemas


class XMLParser:
    """XML 工具调用解析器"""
    
    @staticmethod
    def parse(content: str, extra_tool_names: list = None) -> List[Dict[str, Any]]:
        """
        从 LLM 回复中解析 XML 工具调用
        
        Args:
            content: LLM 的回复内容
            extra_tool_names: 额外需要解析的工具名列表（外部工具）
            
        Returns:
            工具调用列表
        """
        tool_calls = []
        
        # 解析 execute_command - 支持多种格式
        # 格式1: <execute_command>command</execute_command>
        pattern = r'<execute_command>(.*?)</execute_command>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            tool_calls.append({
                'tool': 'execute_command',
                'params': {'command': match.strip()}
            })
        
        # 格式2: <execute_command><command>...</command></execute_command> (子元素格式)
        pattern2 = r'<execute_command>\s*<command>(.*?)</command>\s*</execute_command>'
        matches2 = re.findall(pattern2, content, re.DOTALL)
        for match in matches2:
            tool_calls.append({
                'tool': 'execute_command',
                'params': {'command': match.strip()}
            })
        
        # 解析 read_file - 支持多种格式
        # 格式1: <read_file path="..." start_line="..." end_line="..."></read_file>
        pattern = r'<read_file\s+path="([^"]+)"(?:\s+start_line="(\d+)")?(?:\s+end_line="(\d+)")?>\s*</read_file>'
        matches = re.findall(pattern, content)
        for match in matches:
            path, start_line, end_line = match
            params = {'path': path}
            if start_line:
                params['start_line'] = int(start_line)
            if end_line:
                params['end_line'] = int(end_line)
            tool_calls.append({
                'tool': 'read_file',
                'params': params
            })
        
        # 格式2: <read_file><path>...</path></read_file> (子元素格式)
        pattern2 = r'<read_file>\s*<path>([^<]+)</path>(?:\s*<start_line>(\d+)</start_line>)?(?:\s*<end_line>(\d+)</end_line>)?\s*</read_file>'
        matches2 = re.findall(pattern2, content)
        for match in matches2:
            path, start_line, end_line = match
            params = {'path': path.strip()}
            if start_line:
                params['start_line'] = int(start_line)
            if end_line:
                params['end_line'] = int(end_line)
            tool_calls.append({
                'tool': 'read_file',
                'params': params
            })
        
        # 格式3: <read_file><path>...</path><lines>17-25</lines></read_file> (lines 标签格式)
        # DeepSeek 常用此格式，将 "17-25" 解析为 start_line=17, end_line=25
        pattern3 = r'<read_file>\s*<path>([^<]+)</path>\s*<lines>(\d+)(?:-(\d+))?\s*</lines>\s*</read_file>'
        matches3 = re.findall(pattern3, content)
        for match in matches3:
            path, start_line, end_line = match
            params = {'path': path.strip()}
            params['start_line'] = int(start_line)
            if end_line:
                params['end_line'] = int(end_line)
            else:
                params['end_line'] = int(start_line)  # 单行
            tool_calls.append({
                'tool': 'read_file',
                'params': params
            })
        

        # 解析 write_to_file - 支持多种格式
        # 格式1: <write_to_file path="...">content</write_to_file>
        pattern = r'<write_to_file\s+path="([^"]+)"\s*>\s*(.*?)</write_to_file>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            path, content_text = match
            tool_calls.append({
                'tool': 'write_to_file',
                'params': {
                    'path': path,
                    'content': content_text.strip()
                }
            })
        
        # 格式2: <write_to_file><path>...</path><content>...</content></write_to_file> (子元素格式)
        pattern2 = r'<write_to_file>\s*<path>([^<]+)</path>\s*<content>(.*?)</content>\s*</write_to_file>'
        matches2 = re.findall(pattern2, content, re.DOTALL)
        for match in matches2:
            path, content_text = match
            tool_calls.append({
                'tool': 'write_to_file',
                'params': {
                    'path': path.strip(),
                    'content': content_text.strip()
                }
            })
        
        # 解析 replace_in_file
        pattern = r'<replace_in_file>\s*<path>([^<]+)</path>\s*<diff>\s*(.*?)</diff>\s*</replace_in_file>'
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            path, diff = match
            tool_calls.append({
                'tool': 'replace_in_file',
                'params': {
                    'path': path.strip(),
                    'diff_string': diff.strip()
                }
            })
        
        # 解析 list_directory - 支持多种格式
        # 格式1: <list_directory path="..." depth="..." recursive="..."></list_directory>
        pattern = r'<list_directory(?:\s+path="([^"]*)")?(?:\s+depth="(\d+)")?(?:\s+recursive="(true|false)")?\s*>\s*</list_directory>'
        matches = re.findall(pattern, content)
        for match in matches:
            path, depth, recursive = match
            params = {}
            if path:
                params['path'] = path
            if depth:
                params['depth'] = int(depth)
            elif recursive and recursive.lower() == 'true':
                # 如果指定 recursive=true 但未指定 depth，默认 depth=10
                params['depth'] = 10
            tool_calls.append({
                'tool': 'list_directory',
                'params': params
            })
        
        # 格式2: <list_directory><path>...</path><recursive>...</recursive></list_directory> (子元素格式)
        pattern2 = r'<list_directory>\s*<path>([^<]+)</path>(?:\s*<depth>(\d+)</depth>)?(?:\s*<recursive>(true|false)</recursive>)?\s*</list_directory>'
        matches2 = re.findall(pattern2, content)
        for match in matches2:
            path, depth, recursive = match
            params = {'path': path.strip()}
            if depth:
                params['depth'] = int(depth)
            elif recursive and recursive.lower() == 'true':
                # 如果指定 recursive=true 但未指定 depth，默认 depth=10
                params['depth'] = 10
            tool_calls.append({
                'tool': 'list_directory',
                'params': params
            })
        
        # 解析 create_directory - 支持多种格式
        # 格式1: <create_directory path="..."></create_directory>
        pattern = r'<create_directory\s+path="([^"]+)"\s*>\s*</create_directory>'
        matches = re.findall(pattern, content)
        for match in matches:
            tool_calls.append({
                'tool': 'create_directory',
                'params': {'path': match}
            })
        
        # 格式2: <create_directory><path>...</path></create_directory> (子元素格式)
        pattern2 = r'<create_directory>\s*<path>([^<]+)</path>\s*</create_directory>'
        matches2 = re.findall(pattern2, content)
        for match in matches2:
            tool_calls.append({
                'tool': 'create_directory',
                'params': {'path': match.strip()}
            })
        
        # 解析 preview_file - 支持多种格式
        # 格式1: <preview_file path="..." num_lines="..."></preview_file>
        pattern = r'<preview_file\s+path="([^"]+)"(?:\s+num_lines="(\d+)")?\s*>\s*</preview_file>'
        matches = re.findall(pattern, content)
        for match in matches:
            path, num_lines = match
            params = {'path': path}
            if num_lines:
                params['num_lines'] = int(num_lines)
            tool_calls.append({
                'tool': 'preview_file',
                'params': params
            })
        
        # 格式2: <preview_file><path>...</path></preview_file> (子元素格式)
        pattern2 = r'<preview_file>\s*<path>([^<]+)</path>(?:\s*<num_lines>(\d+)</num_lines>)?\s*</preview_file>'
        matches2 = re.findall(pattern2, content)
        for match in matches2:
            path, num_lines = match
            params = {'path': path.strip()}
            if num_lines:
                params['num_lines'] = int(num_lines)
            tool_calls.append({
                'tool': 'preview_file',
                'params': params
            })
        
        # 解析 search_in_files
        # 格式1: <search_in_files><pattern>...</pattern><path>...</path><file_pattern>...</file_pattern></search_in_files>
        pattern_sif = r'<search_in_files>\s*<pattern>(.*?)</pattern>(?:\s*<path>([^<]*)</path>)?(?:\s*<file_pattern>([^<]*)</file_pattern>)?(?:\s*<case_insensitive>(true|false)</case_insensitive>)?(?:\s*<is_regex>(true|false)</is_regex>)?(?:\s*<context_lines>(\d+)</context_lines>)?\s*</search_in_files>'
        matches_sif = re.findall(pattern_sif, content, re.DOTALL)
        for match in matches_sif:
            search_pattern, path, file_pat, case_ins, is_regex, ctx_lines = match
            params = {'pattern': search_pattern.strip()}
            if path:
                params['path'] = path.strip()
            if file_pat:
                params['file_pattern'] = file_pat.strip()
            if case_ins:
                params['case_insensitive'] = case_ins.lower() == 'true'
            if is_regex:
                params['is_regex'] = is_regex.lower() == 'true'
            if ctx_lines:
                params['context_lines'] = int(ctx_lines)
            tool_calls.append({
                'tool': 'search_in_files',
                'params': params
            })
        
        # 通用解析：处理已注册但不在上面硬编码列表中的工具（如外部工具）
        # 格式: <tool_name><param1>value1</param1><param2>value2</param2></tool_name>
        if extra_tool_names:
            # 已经被上面的代码解析过的基础工具名
            parsed_tools = {tc['tool'] for tc in tool_calls}
            base_tool_names = {'execute_command', 'read_file', 'write_to_file', 
                            'replace_in_file', 'list_directory', 'create_directory',
                            'preview_file', 'search_in_files'}
            for tool_name in extra_tool_names:
                if tool_name in base_tool_names or tool_name in parsed_tools:
                    continue
                # 匹配 <tool_name>...</tool_name>
                generic_pattern = rf'<{re.escape(tool_name)}>(.*?)</{re.escape(tool_name)}>'
                generic_matches = re.findall(generic_pattern, content, re.DOTALL)
                for match_body in generic_matches:
                    params = {}
                    # 提取子元素 <param>value</param>
                    param_pattern = r'<([a-zA-Z_][a-zA-Z0-9_]*)>(.*?)</\1>'
                    param_matches = re.findall(param_pattern, match_body, re.DOTALL)
                    for param_name, param_value in param_matches:
                        params[param_name] = param_value.strip()
                    # 如果没有子元素，将整个内容作为单一参数
                    if not params and match_body.strip():
                        params = {'input': match_body.strip()}
                    tool_calls.append({
                        'tool': tool_name,
                        'params': params
                    })
        
        return tool_calls
    
    @staticmethod
    def has_attempt_completion(content: str) -> bool:
        """检查是否包含任务完成标记"""
        return '<attempt_completion>' in content
