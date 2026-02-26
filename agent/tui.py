"""
TUI 交互管理器

跨平台支持：
- Windows: 通过 pywinpty (ConPTY) 启动
- Unix/Linux/macOS: 通过 ptyprocess 启动
使用 pyte 解析屏幕缓冲区，提供 open_tui / read_tui / send_keys / close_tui 接口。
"""

import threading
import time
import hashlib
import os
import sys
import shlex
from typing import Dict, Optional, Callable, List

import pyte

# 按跨平台处理引入 pty 工具
if sys.platform == "win32":
    try:
        from winpty import PtyProcess
    except ImportError:
        PtyProcess = None
else:
    try:
        from ptyprocess import PtyProcessUnicode
    except ImportError:
        PtyProcessUnicode = None


# 按键映射表：友好名称 -> ANSI 转义序列
KEY_MAP = {
    # 基础键
    "Enter": "\r",
    "Return": "\r",
    "Tab": "\t",
    "Escape": "\x1b",
    "Esc": "\x1b",
    "Backspace": "\x7f",
    "Delete": "\x1b[3~",
    "Space": " ",
    
    # 方向键
    "Up": "\x1b[A",
    "Down": "\x1b[B",
    "Right": "\x1b[C",
    "Left": "\x1b[D",
    
    # 导航键
    "Home": "\x1b[H",
    "End": "\x1b[F",
    "PageUp": "\x1b[5~",
    "PageDown": "\x1b[6~",
    "Insert": "\x1b[2~",
    
    # 功能键
    "F1": "\x1bOP",
    "F2": "\x1bOQ",
    "F3": "\x1bOR",
    "F4": "\x1bOS",
    "F5": "\x1b[15~",
    "F6": "\x1b[17~",
    "F7": "\x1b[18~",
    "F8": "\x1b[19~",
    "F9": "\x1b[20~",
    "F10": "\x1b[21~",
    "F11": "\x1b[23~",
    "F12": "\x1b[24~",
    
    # Ctrl 组合键
    "Ctrl-A": "\x01",
    "Ctrl-B": "\x02",
    "Ctrl-C": "\x03",
    "Ctrl-D": "\x04",
    "Ctrl-E": "\x05",
    "Ctrl-F": "\x06",
    "Ctrl-G": "\x07",
    "Ctrl-H": "\x08",
    "Ctrl-K": "\x0b",
    "Ctrl-L": "\x0c",
    "Ctrl-N": "\x0e",
    "Ctrl-O": "\x0f",
    "Ctrl-P": "\x10",
    "Ctrl-R": "\x12",
    "Ctrl-S": "\x13",
    "Ctrl-T": "\x14",
    "Ctrl-U": "\x15",
    "Ctrl-V": "\x16",
    "Ctrl-W": "\x17",
    "Ctrl-X": "\x18",
    "Ctrl-Y": "\x19",
    "Ctrl-Z": "\x1a",
}


class TUISession:
    """单个 TUI 会话：跨平台 PTY 进程 + pyte 屏幕"""
    
    def __init__(self, name: str, command: str, rows: int = 24, cols: int = 80,
                 cwd: Optional[str] = None):
        self.name = name
        self.rows = rows
        self.cols = cols
        
        # pyte 虚拟屏幕
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)  # Stream 接收 str
        
        # 跨平台 PTY 进程启动
        if sys.platform == "win32":
            if PtyProcess is None:
                raise ImportError("未安装 pywinpty，无法在 Windows 上启动 TUI 会话。请执行 `pip install pywinpty`")
            self.pty = PtyProcess.spawn(
                command,
                dimensions=(rows, cols),
                cwd=cwd
            )
        else:
            if PtyProcessUnicode is None:
                raise ImportError("未安装 ptyprocess，无法在 Unix/Linux/macOS 上启动 TUI 会话。请执行 `pip install ptyprocess`")
            # Unix 下通常需要解析命令参数
            argv = shlex.split(command)
            self.pty = PtyProcessUnicode.spawn(
                argv,
                dimensions=(rows, cols),
                cwd=cwd or os.getcwd()
            )
        
        # 后台读取线程
        self._alive = True
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
    
    def _read_loop(self):
        """持续从 PTY 读取数据并送入 pyte 解析"""
        while self._alive:
            try:
                data = self.pty.read(4096)
                if data:
                    with self._lock:
                        self.stream.feed(data)
            except EOFError:
                self._alive = False
                break
            except Exception:
                time.sleep(0.01)
    
    def get_screen_text(self) -> str:
        """获取当前屏幕内容（纯文本行列表）"""
        with self._lock:
            lines = []
            for row in range(self.rows):
                line = self.screen.buffer[row]
                text = ""
                for col in range(self.cols):
                    char = line[col]
                    text += char.data if char.data else " "
                lines.append(text.rstrip())
            return lines
    
    def get_cursor(self):
        """获取光标位置"""
        with self._lock:
            return (self.screen.cursor.y, self.screen.cursor.x)
    
    def get_screen_hash(self) -> str:
        """获取屏幕内容哈希（用于检测变化）"""
        lines = self.get_screen_text()
        content = "\n".join(lines)
        return hashlib.md5(content.encode()).hexdigest()
    
    def write(self, data: str):
        """向 PTY 写入数据"""
        self.pty.write(data)
    
    def is_alive(self) -> bool:
        return self._alive and self.pty.isalive()
    
    def close(self):
        self._alive = False
        try:
            self.pty.terminate(force=True)
        except Exception:
            pass


class TUIManager:
    """
    TUI 会话管理器
    
    管理多个 TUI 会话，提供统一的 open/read/send/close 接口。
    """
    
    def __init__(self, default_cwd: str = ".", screen_update_callback: Callable = None):
        self.sessions: Dict[str, TUISession] = {}
        self.default_cwd = os.path.abspath(default_cwd)
        # 屏幕更新回调：callback(name, lines, cursor_row, cursor_col)
        self.screen_update_callback = screen_update_callback
    
    def open_tui(self, name: str, command: str, rows: int = 24, cols: int = 80) -> str:
        """
        启动 TUI 应用
        
        Args:
            name: 会话名称
            command: 要执行的命令 (如 "vim test.py", "python")
            rows: 终端行数
            cols: 终端列数
        """
        if name in self.sessions:
            return f"<error>TUI 会话 '{name}' 已存在</error>"
        
        try:
            session = TUISession(
                name=name,
                command=command,
                rows=rows,
                cols=cols,
                cwd=self.default_cwd
            )
            self.sessions[name] = session
            
            # 等待初始界面渲染
            time.sleep(0.5)
            
            # 触发屏幕更新回调
            self._notify_screen_update(name)
            
            return f"<success>TUI 会话 '{name}' 已启动 (命令: {command}, {cols}x{rows})</success>"
        except Exception as e:
            return f"<error>启动 TUI 失败: {str(e)}</error>"
    
    def read_tui(self, name: str) -> str:
        """
        读取 TUI 当前屏幕快照
        
        返回 XML 格式的屏幕内容，包含行文本和光标位置。
        """
        if name not in self.sessions:
            return f"<error>TUI 会话 '{name}' 不存在</error>"
        
        session = self.sessions[name]
        
        if not session.is_alive():
            del self.sessions[name]
            return f"<error>TUI 会话 '{name}' 已退出</error>"
        
        lines = session.get_screen_text()
        cursor_row, cursor_col = session.get_cursor()
        
        # 构建 XML 输出（只输出非空行，节省 token）
        xml_lines = []
        xml_lines.append(
            f'<tui_screen name="{name}" rows="{session.rows}" cols="{session.cols}" '
            f'cursor_row="{cursor_row}" cursor_col="{cursor_col}">'
        )
        
        for row_idx, line in enumerate(lines):
            if line.strip():  # 跳过空行
                xml_lines.append(f'  <line row="{row_idx}">{self._escape_xml(line)}</line>')
        
        xml_lines.append('</tui_screen>')
        return "\n".join(xml_lines)
    
    def send_keys(self, name: str, keys: str) -> str:
        """
        向 TUI 发送按键
        
        Args:
            name: 会话名称
            keys: 按键序列，特殊键用名称表示，空格分隔
                  例: "hello"          -> 输入 hello
                       "Enter"          -> 回车
                       "Up Up Down"     -> 上上下
                       "Escape :wq Enter" -> ESC :wq 回车
                       "Ctrl-C"         -> Ctrl+C
                       "hello Enter"    -> 输入 hello 并回车
        """
        if name not in self.sessions:
            return f"<error>TUI 会话 '{name}' 不存在</error>"
        
        session = self.sessions[name]
        if not session.is_alive():
            del self.sessions[name]
            return f"<error>TUI 会话 '{name}' 已退出</error>"
        
        # 解析按键序列
        data = self._parse_keys(keys)
        
        # 记录屏幕哈希（用于等待更新）
        old_hash = session.get_screen_hash()
        
        # 发送
        session.write(data)
        
        # 智能等待屏幕更新（最多 2 秒）
        self._wait_for_update(session, old_hash, timeout=2.0)
        
        # 触发屏幕更新回调
        self._notify_screen_update(name)
        
        return "<success>按键已发送</success>"
    
    def close_tui(self, name: str) -> str:
        """关闭 TUI 会话"""
        if name not in self.sessions:
            return f"<error>TUI 会话 '{name}' 不存在</error>"
        
        self.sessions[name].close()
        del self.sessions[name]
        return f"<success>TUI 会话 '{name}' 已关闭</success>"
    
    def render_to_console(self, name: str):
        """
        将 TUI 屏幕内容镜像输出到用户控制台（调试/观察用）
        
        绘制带边框的屏幕快照，让用户看到 Agent 正在操作的 TUI 界面。
        """
        if name not in self.sessions:
            return
        
        session = self.sessions[name]
        if not session.is_alive():
            return
        
        lines = session.get_screen_text()
        cursor_row, cursor_col = session.get_cursor()
        cols = session.cols
        
        # 绘制边框
        print(f"\n  ┌─ TUI: {name} {'─' * (cols - len(name) - 7)}┐")
        for row_idx, line in enumerate(lines):
            # 补齐到 cols 宽度
            padded = line.ljust(cols)[:cols]
            # 光标位置标记
            if row_idx == cursor_row:
                # 在光标位置插入标记
                chars = list(padded)
                if cursor_col < len(chars):
                    chars[cursor_col] = f"\033[7m{chars[cursor_col]}\033[0m"  # 反色显示光标
                padded = "".join(chars)
            print(f"  │{padded}│")
        print(f"  └{'─' * cols}┘")
        print(f"  光标: ({cursor_row}, {cursor_col})")
        print()
    
    def _notify_screen_update(self, name: str):
        """触发屏幕更新回调（如果已注册）"""
        if not self.screen_update_callback:
            return
        if name not in self.sessions:
            return
        session = self.sessions[name]
        if not session.is_alive():
            return
        try:
            lines = session.get_screen_text()
            cursor_row, cursor_col = session.get_cursor()
            self.screen_update_callback(name, lines, cursor_row, cursor_col)
        except Exception:
            pass  # 回调失败不影响主流程
    
    def _parse_keys(self, keys: str) -> str:
        """
        解析按键描述字符串为实际发送数据
        
        策略：按空格分割 token，每个 token：
        - 如果是 KEY_MAP 中的已知键名 -> 映射为转义序列
        - 否则 -> 作为字面文本发送
        """
        tokens = keys.split(" ")
        result = []
        
        for token in tokens:
            if token in KEY_MAP:
                result.append(KEY_MAP[token])
            else:
                # 检查 Ctrl- 前缀
                if token.startswith("Ctrl-") and token in KEY_MAP:
                    result.append(KEY_MAP[token])
                else:
                    # 字面文本
                    result.append(token)
        
        return "".join(result)
    
    def _wait_for_update(self, session: TUISession, old_hash: str, timeout: float = 2.0):
        """等待屏幕发生变化（轮询）"""
        start = time.time()
        interval = 0.05  # 50ms 轮询
        
        while time.time() - start < timeout:
            time.sleep(interval)
            new_hash = session.get_screen_hash()
            if new_hash != old_hash:
                # 屏幕已变化，再等一小段时间让渲染稳定
                time.sleep(0.1)
                return
            interval = min(interval * 1.5, 0.2)  # 指数退避
    
    @staticmethod
    def _escape_xml(text: str) -> str:
        """转义 XML 特殊字符"""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
