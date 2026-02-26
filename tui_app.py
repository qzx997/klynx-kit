"""
Klynx Agent — Textual 分屏 TUI 查看器

左侧显示 Agent 思维链（所有 print 输出），
右侧显示 Agent 正在操作的 TUI 屏幕（pyte 缓冲区实时渲染）。

用法: python tui_app.py
"""

import os
import sys
import io
import threading
from typing import Optional
from dotenv import load_dotenv

basedir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(basedir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
load_dotenv(os.path.join(basedir, ".env"))

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, RichLog, Static, Input, TextArea, Button
from textual.binding import Binding
from textual.reactive import reactive
from textual.suggester import Suggester
from textual import work
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax

from klynx.agent.graph import create_agent
from klynx.agent.context_manager import TokenCounter
from klynx.model import setup as setup_model


class ContextAwareSuggester(Suggester):
    """
    上下文感知的自动补全
    
    支持：
    - '/' 命令补全 (如 /quit, /clear)
    - '@' 文件路径补全 (如 @core/main.py)
    """
    
    def __init__(self, app_instance):
        super().__init__(use_cache=False)
        self.app_instance = app_instance
        self.commands = ["/quit", "/clear", "/context", "/context full", "/help", "/mode full", "/mode split", "/compact", "/compat"]
        
    async def get_suggestion(self, value: str) -> Optional[str]:
        if not value:
            return None
            
        # 获取最后一个 token
        tokens = value.split(" ")
        last_token = tokens[-1]
        
        # 1. 命令补全 (仅在行首)
        if len(tokens) == 1 and last_token.startswith("/"):
            for cmd in self.commands:
                if cmd.startswith(last_token) and cmd != last_token:
                    return cmd
        
        # 2. 文件路径补全 (任意位置，@开头)
        if last_token.startswith("@"):
            prefix = last_token[1:]  # 去掉 @
            try:
                # 简单的文件搜索
                folder = os.path.dirname(prefix)
                base = os.path.basename(prefix)
                
                search_dir = os.path.join(os.getcwd(), folder)
                if os.path.isdir(search_dir):
                    for name in os.listdir(search_dir):
                        if name.startswith(base):
                            # 构造建议
                            suggestion = "@" + os.path.join(folder, name).replace("\\", "/")
                            if os.path.isdir(os.path.join(search_dir, name)):
                                suggestion += "/"
                            
                            # 替换最后一个 token
                            return " ".join(tokens[:-1] + [suggestion])
            except Exception:
                pass
                
        return None

class SuggestionInput(Input):
    """
    带 Tab 补全的输入框
    """
    BINDINGS = [
        ("tab", "complete_suggestion", "补全"),
    ]
    
    
    def action_complete_suggestion(self):
        """Accept the current suggestion."""
        # accessing internal _suggestion because Textual doesn't expose it publicly yet
        suggestion = getattr(self, "_suggestion", "")
        
        if suggestion and self.cursor_at_end:
            # 直接应用补全建议 (确保总是生效)
            self.value = suggestion
            self.cursor_position = len(self.value)
        else:
            # 如果没有建议，则执行默认的 tab 行为（切换焦点）
            self.screen.focus_next()

class TUIScreenWidget(TextArea):
    """
    TUI 屏幕渲染组件 (基于 TextArea 以支持原生复制)
    
    将 pyte 屏幕缓冲区的文本行渲染为可选择的文本区域。
    使用 self.text 更新内容，支持鼠标选择复制。
    注意：暂不支持 ANSI 颜色显示 (TextArea 限制)。
    """
    
    DEFAULT_CSS = """
    TUIScreenWidget {
        border: solid $secondary;
        height: 100%;
    }
    """
    
    def __init__(self, **kwargs):
        # 强制设置为只读和无行号，模拟终端显示
        kwargs["read_only"] = True
        kwargs["show_line_numbers"] = False
        super().__init__(**kwargs)
        self.session_name = ""
        self.can_focus = True
    
    def update_screen(self, name: str, lines: list, cursor_row: int, cursor_col: int):
        """外部调用：更新屏幕内容"""
        
        # 如果用户正在选择文本（selection非空），则暂停更新，避免打断操作
        if not self.selection.is_empty:
            return

        # 简单去重
        current_state = (name, lines, cursor_row, cursor_col)
        if getattr(self, "_last_state", None) == current_state:
            return
        self._last_state = current_state
        
        self.session_name = name
        
        # 构建完整文本
        header = f" ─── {self.session_name} " + "─" * max(0, 60 - len(self.session_name))
        body = "\n".join(lines)
        footer = "─" * 65 + f"\n 光标: ({cursor_row}, {cursor_col})"
        
        full_text = f"{header}\n{body}\n{footer}"
        
        # 使用 replace 更新全部内容，尝试保持滚动位置
        # 注意：这里我们替换整个文档，并在替换后尝试同步光标
        # 如果正在滚动查看历史，可能也会有跳动，但 selection 暂停机制已经解决主要痛点
        self.replace(
            full_text, 
            start=(0, 0), 
            end=self.document.end, 
            maintain_selection_offset=False  # 不保持偏移，因为内容可能变了，光标逻辑下面单独处理
        )
        
        # 尝试同步光标位置到 TUI 的逻辑光标
        target_row = cursor_row + 1 # Header 占用 1 行
        
        try:
           # 只有当用户没有在该区域进行操作时才强制移动光标
           if self.selection.is_empty:
               self.cursor_location = (target_row, cursor_col)
               self.scroll_cursor_visible()
        except:
           pass


class StdoutRedirector(io.TextIOBase):
    """
    stdout 重定向器
    
    将 print() 输出转发到 Textual RichLog 组件。
    同时保留对原始 stdout 的写入（用于调试）。
    """
    
    def __init__(self, app: "KlynxTUIApp", original_stdout):
        self.app = app
        self.original = original_stdout
        self._buffer = ""
    
    def write(self, text: str) -> int:
        if not text:
            return 0
        
        # 缓冲行
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            try:
                self.app.call_from_thread(self._write_to_log, line)
            except Exception:
                pass  # App 还没初始化时静默
        
        return len(text)
    
    def _write_to_log(self, line: str):
        """在主线程中写入日志"""
        try:
            log_widget = self.app.query_one("#agent-log", RichLog)
            log_widget.write(line)
        except Exception:
            pass
    
    def flush(self):
        if self._buffer.strip():
            try:
                self.app.call_from_thread(self._write_to_log, self._buffer)
            except Exception:
                pass
            self._buffer = ""



class KlynxTUIApp(App):
    """
    Klynx Agent 分屏 TUI 应用
    
    布局：
    ┌───────────────────┬──────────────────┐
    │  Agent 思维链      │  TUI 屏幕        │
    │  (RichLog)        │  (TUIScreen)     │
    ├───────────────────┴──────────────────┤
    │  [轮次 N] > 输入任务                   │
    └──────────────────────────────────────┘
    """
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #main-area {
        height: 1fr;
    }
    
    #agent-log {
        width: 100%;
        height: 1fr;

        min-height: 50%;
        overflow-y: auto;
    }
    
    #active-stream-container {
        width: 100%;
        height: auto;
        max-height: 50%;
        overflow-y: auto;
        display: none;
        margin: 0;
        padding: 0;
    }
    
    #stream-reasoning {
        color: $text-muted;
        text-style: italic;
    }
    
    #stream-answer {
        color: $text;
    }

    #input-area {
        height: 3;
        margin: 0;
        padding: 0;
    }

    #task-input {
        width: 1fr;
    }
    
    #tui-screen {
        display: none;
        width: 50%;
        border: solid $secondary;
        border-title-color: $text;
        overflow-y: auto;
    }
    
    #log-container {
        width: 100%;
        height: 1fr;
        border: solid $accent;
        border-title-color: $text;
    }
    
    #main-area.split-mode #log-container {
        width: 50%;
    }
    
    #main-area.split-mode #tui-screen {
        display: block;
    }
    
    #status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """
    
    BINDINGS = [
        ("ctrl+q", "quit", "退出"),
        Binding("escape", "interrupt_agent", "停止", show=True),
        ("ctrl+l", "clear_log", "清空日志"),
        ("f2", "toggle_split", "切换分屏"),
        ("ctrl+shift+c", "copy_last", "复制"),
    ]
    
    def __init__(self, model_provider: str = "deepseek", model_name: str = "deepseek-reasoner", api_key: str = None, tavily_api_key: str = None, **model_kwargs):
        super().__init__()
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        self.tavily_api_key = tavily_api_key
        self.model_kwargs = model_kwargs
        
        import uuid
        self.thread_id = str(uuid.uuid4())[:8]
        self.agent = None
        self.accumulated_context = ""
        self.round_count = 0
        self.total_tokens = 0
        self._original_stdout = sys.stdout
        self._redirector = None
        self._last_screen_hash = {}  # {session_name: hash} 用于检测屏幕变化
        self._last_response = ""  # 存储最近一次 Agent 回复，用于复制
        self._token_buffer = ""
        self._last_response = ""  # 存储最近一次 Agent 回复，用于复制
        self._token_buffer = ""
        self._reasoning_buffer = ""
        self.is_processing = False # 用于防止双重提交

    def _format_messages(self, messages: list) -> str:
        """从消息历史构建上下文文本"""
        context = ""
        for msg in messages:
            role = "User"
            if hasattr(msg, "type"):
                if msg.type == "human":
                    role = "User"
                elif msg.type == "ai":
                    role = "Assistant"
                elif msg.type == "system":
                    role = "System"
                elif msg.type == "tool":  # LangChain tool message
                    role = "Tool"
            
            content = getattr(msg, "content", str(msg))
            context += f"\n[{role}]: {content}\n"
        return context.strip()
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-area"):
            with Vertical(id="log-container"):
                yield RichLog(id="agent-log", wrap=True, highlight=True, markup=True)
                with Vertical(id="active-stream-container"):
                    yield Static("", id="stream-reasoning")
                    yield Static("", id="stream-answer")
            yield TUIScreenWidget(id="tui-screen")
        yield Static("轮次: 0 | Token: 0 | 上下文: 0 字符", id="status-bar")
        
        # 使用自定义 Suggestor 和 Input
        with Horizontal(id="input-area"):
            yield SuggestionInput(
                placeholder="输入任务 /slash命令 或 @文件 (F2切换分屏)...", 
                id="task-input",
                suggester=ContextAwareSuggester(self)
            )
        yield Footer()
    
    
    def on_mount(self) -> None:
        """应用启动后初始化"""
        self.title = "Klynx Agent — TUI 增强版"
        
        # 设置日志区域标题
        log_container = self.query_one("#log-container", Vertical)
        log_container.border_title = "Agent 思维链"
        
        log = self.query_one("#agent-log", RichLog)
        
        tui_screen = self.query_one("#tui-screen", TUIScreenWidget)
        # 重定向 stdout
        self._redirector = StdoutRedirector(self, self._original_stdout)
        sys.stdout = self._redirector
        
        # 显示欢迎横幅
        banner = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                K L Y N X   A G E N T                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
[italic]提示: 输入 /help 查看命令，F2 切换分屏[/italic]
"""
        log.write(banner)
        
        # 初始化 Agent
        self._init_agent()
        
        # 启动 TUI 屏幕实时刷新定时器（每 200ms 轮询一次 pyte 缓冲区）
        self.set_interval(0.2, self._refresh_tui_screen)
        
        # 聚焦输入框
        self.query_one("#task-input", Input).focus()
        
    def action_toggle_split(self):
        """Action for F2 binding"""
        main_area = self.query_one("#main-area")
        if "split-mode" in main_area.classes:
            self.set_layout_mode("full")
        else:
            self.set_layout_mode("split")
            
    def action_copy_last(self):
        """Action for Ctrl+Shift+C binding"""
        log = self.query_one("#agent-log", RichLog)
        if self._last_response:
            self.copy_to_clipboard(self._last_response)
            log.write("[系统] 已复制最近回复到剪贴板")
        else:
            log.write("[系统] 暂无可复制的回复")
            
    def action_interrupt_agent(self):
        """Action for Ctrl+C binding"""
        if self.is_processing and self.agent:
            log = self.query_one("#agent-log", RichLog)
            log.write("[系统] 正在中断 Agent 执行...")
            # 触发打断标志（需要在 graph.py 中支持挂起机制）
            if hasattr(self.agent, "_cancel_event"):
                self.agent._cancel_event.set()
        else:
            # 如果没在处理任务，按两次 Ctrl+C 或提示确认退出？这里为了安全只提示
            self.query_one("#agent-log", RichLog).write("[系统] 当前无活动任务。输入 /quit 退出应用。")
            
    def set_layout_mode(self, mode: str) -> str:
        """
        设置界面布局模式
        
        Args:
            mode: 'full' (全屏日志) 或 'split' (分屏显示 TUI)
            
        Returns:
            操作结果消息
        """
        main_area = self.query_one("#main-area")
        
        if mode == "split":
            main_area.add_class("split-mode")
            return "已切换到分屏模式 (日志 + TUI 屏幕)"
        elif mode == "full":
            main_area.remove_class("split-mode")
            return "已切换到全屏日志模式"
        else:
            return f"未知布局模式: {mode} (支持: full, split)"
    
    def _init_agent(self):
        """初始化 Agent（在主线程完成，因为很快）"""
        log = self.query_one("#agent-log", RichLog)
        log.write("[系统] 正在初始化 Agent...")

        
        working_dir = os.getcwd()
        
        try:
            if self.tavily_api_key:
                from .agent.web_search import set_tavily_api
                set_tavily_api(self.tavily_api_key)
                
            model = setup_model(
                self.model_provider, 
                self.model_name, 
                api_key=self.api_key, 
                **self.model_kwargs
            )
            self.agent = create_agent(
                working_dir=working_dir,
                model=model,
                max_iterations=50,
                memory_dir=None,
                load_project_docs=False
            )
            self.agent.add_tools("all")
            
            # 注册 TUI 布局控制工具
            def set_tui_layout(mode: str = None, layout: str = None, input: str = None, **kwargs) -> str:
                # 兼容性处理：模型可能会幻觉出各种奇怪的参数名
                # 优先使用 standard 参数 'mode'
                target_mode = mode
                
                # 如果 mode 为空，尝试从 input 或 layout 中解析
                if not target_mode:
                    content = input or layout or ""
                    if "split" in content.lower():
                        target_mode = "split"
                    elif "full" in content.lower():
                        target_mode = "full"
                
                # 默认值
                if not target_mode:
                    return "错误：请提供 mode 参数，例如 <mode>split</mode>"
                    
                return self.call_from_thread(self.set_layout_mode, target_mode)
                
            self.agent.add_tools(set_tui_layout, "控制 TUI 界面布局。必须使用 XML 格式：<set_tui_layout><mode>split</mode></set_tui_layout> (开启分屏) 或 <set_tui_layout><mode>full</mode></set_tui_layout> (全屏日志)。严禁使用 JSON 或其他格式。")
            
            # 设置 TUI 屏幕更新回调
            if hasattr(self.agent, 'tui_manager'):
                self.agent.tui_manager.screen_update_callback = self._on_tui_screen_update
            
            log.write(f"[系统] Agent 就绪 (工作目录: {working_dir})")
            log.write("[系统] 输入任务开始交互")
        except Exception as e:
            log.write(f"[错误] Agent 初始化失败: {e}")
    
    def _on_tui_screen_update(self, name: str, lines: list, cursor_row: int, cursor_col: int):
        """TUI 屏幕更新回调 — 从 worker 线程调用"""
        try:
            self.call_from_thread(
                self._update_tui_widget, name, lines, cursor_row, cursor_col
            )
        except Exception:
            pass
    
    def _update_tui_widget(self, name: str, lines: list, cursor_row: int, cursor_col: int):
        """在主线程中更新 TUI 屏幕组件"""
        tui_widget = self.query_one("#tui-screen", TUIScreenWidget)
        tui_widget.update_screen(name, lines, cursor_row, cursor_col)
    
    def _refresh_tui_screen(self):
        """
        定时轮询 TUI 屏幕（每 200ms）
        
        检查所有活跃 TUI 会话的 pyte 缓冲区，
        如果屏幕内容发生变化则刷新右侧面板。
        """
        if not self.agent or not hasattr(self.agent, 'tui_manager'):
            return
        
        tm = self.agent.tui_manager
        tui_widget = self.query_one("#tui-screen", TUIScreenWidget)
        
        # 检测已关闭的会话：如果当前显示的会话已不在 sessions 中，清空右侧面板
        if tui_widget.session_name and tui_widget.session_name not in tm.sessions:
            tui_widget.session_name = ""
            tui_widget.screen_lines = []
            tui_widget.cursor_pos = (0, 0)
            self._last_screen_hash.pop(tui_widget.session_name, None)
            return
        
        for name, session in list(tm.sessions.items()):
            if not session.is_alive():
                continue
            try:
                current_hash = session.get_screen_hash()
                if current_hash != self._last_screen_hash.get(name):
                    self._last_screen_hash[name] = current_hash
                    lines = session.get_screen_text()
                    cursor_row, cursor_col = session.get_cursor()
                    tui_widget.update_screen(name, lines, cursor_row, cursor_col)
            except Exception:
                pass
    
    def _update_status(self):
        """更新底部状态栏"""
        ctx_len = len(self.accumulated_context)
        status = self.query_one("#status-bar", Static)
        status.update(
            f"轮次: {self.round_count} | Token: {self.total_tokens} | 上下文: {ctx_len} 字符"
        )
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """用户提交任务"""
        task = event.value.strip()
        event.input.clear()
        
        if not task:
            return
        
        log = self.query_one("#agent-log", RichLog)
        
        # 内置命令
        if task.lower() in ("exit", "quit", "退出", "q"):
            self.exit()
            return
        
        if task.lower() in ("clear", "清空"):
            self.accumulated_context = ""
            self.round_count = 0
            
    def on_input_submitted(self, event: Input.Submitted):
        """处理用户输入"""
        task = event.value.strip()
        if not task:
            return
            
        event.input.value = ""
        
        # 处理 Slash 命令
        if task.startswith("/"):
            parts = task.split(" ")
            cmd = parts[0].lower()
            args = parts[1:]
            
            log = self.query_one("#agent-log", RichLog)
            
            if cmd == "/quit":
                self.exit()
                return
            elif cmd == "/clear":
                log.clear()
                import uuid
                self.thread_id = str(uuid.uuid4())[:8]
                log.write(f"[系统] 日志及上下文已清空，开启新会话 (thread: {self.thread_id})")
                self.round_count = 0
                return
            elif cmd.startswith("/context"):
                is_full = "full" in task.lower()
                state_values = self.agent.get_context(thread_id=self.thread_id)
                
                if state_values:
                    msgs = state_values.get("messages", [])
                    total_chars = sum(len(m.content) for m in msgs if hasattr(m, "content") and isinstance(m.content, str))
                    estimated_tokens = total_chars // 2
                    goal = state_values.get("overall_goal", "")
                    
                    info = f"Thread ID: {self.thread_id}\n"
                    info += f"历史消息: {len(msgs)} 条\n"
                    info += f"估算用量: ~{estimated_tokens} Tokens\n"
                    info += f"当前目标: {goal if goal else '(无)'}\n"
                    
                    log.write(Panel(info, title=f"当前上下文状态", border_style="blue"))
                    
                    if is_full and msgs:
                        full_content = ""
                        show_thinking = state_values.get("thinking_context", True)
                        
                        for i, r in enumerate(msgs):
                            role = r.__class__.__name__.replace("Message", "")
                            
                            # 尝试提取深度思考过程 (如果存在且允许显示)
                            reasoning = ""
                            if show_thinking and hasattr(r, "additional_kwargs") and r.additional_kwargs:
                                reasoning = r.additional_kwargs.get("reasoning_content", "")
                            
                            content_str = str(r.content)
                            if reasoning:
                                content_str = f"<think>\n{reasoning}\n</think>\n{content_str}"
                                
                            # 如果全量太长也做一定的保护
                            if len(content_str) > 5000:
                                content_str = content_str[:5000] + "\n... (截断)"
                            full_content += f"[{i+1}] [{role}]:\n{content_str}\n{'-'*40}\n"
                        log.write(Panel(full_content, title="全部历史消息流水", border_style="green"))
                    elif msgs:
                        recent = msgs[-2:] # 只看最后两条
                        preview = ""
                        show_thinking = state_values.get("thinking_context", True)
                        
                        for r in recent:
                            role = r.__class__.__name__.replace("Message", "")
                            
                            reasoning = ""
                            if show_thinking and hasattr(r, "additional_kwargs") and r.additional_kwargs:
                                reasoning = r.additional_kwargs.get("reasoning_content", "")
                                
                            content_str = str(r.content)
                            if reasoning:
                                content_str = f"<think>...</think> {content_str}"
                                
                            content_preview = str(content_str)[:100].replace("\n", " ") + "..."
                            preview += f"[{role}]: {content_preview}\n"
                        preview += "\n(提示: 输入 '/context full' 查看所有对话详细记录)"
                        log.write(Panel(preview, title="最近消息预览", border_style="dim"))
                        
                else:
                    log.write(Panel("（空，没有任何历史记录）", title="当前上下文状态", border_style="blue"))
                return
            elif cmd == "/copy":
                if self._last_response:
                    self.copy_to_clipboard(self._last_response)
                    log.write("[系统] 已复制最近回复到剪贴板")
                else:
                    log.write("[系统] 暂无可复制的回复")
                return
            elif cmd == "/help":
                help_text = """
[bold]Klynx TUI 帮助[/bold]
-----------------
[cyan]/quit[/cyan]         退出程序
[cyan]/clear[/cyan]        开启新会话（清理当前所有记忆上下文）
[cyan]/copy[/cyan]         复制最近一次Agent回复到剪贴板
[cyan]/context[/cyan]      查看当前累积上下文的短摘要和 Token 用量
[cyan]/context full[/cyan] 打印当前记忆体中完整的所有历史消息对话
[cyan]/compact[/cyan]      手动压缩上下文摘要
[cyan]/mode split[/cyan]   切换到分屏模式
[cyan]/mode full[/cyan]    切换到全屏日志模式
[cyan]F2[/cyan]            快捷切换分屏
[cyan]Ctrl+Shift+C[/cyan]  复制最近回复到剪贴板
[cyan]Shift+鼠标拖选[/cyan]  终端原生选择文字并复制
[cyan]@[/cyan]             输入 @ 触发文件路径补全
"""
                log.write(help_text)
                return
            elif cmd == "/mode":
                if args and args[0] in ["split", "full"]:
                    msg = self.set_layout_mode(args[0])
                    log.write(f"[系统] {msg}")
                else:
                    log.write("[错误] 用法: /mode [split|full]")
                return
            elif cmd == "/compact" or cmd == "/compat":
                if getattr(self, "is_processing", False):
                     log.write("[系统] 任务执行中，无法压缩上下文")
                     return
                self.is_processing = True
                self._run_compact()
                return
            else:
                log.write(f"[错误] 未知命令: {cmd}，输入 /help 查看帮助")
                return

        # 异步执行 Agent
        if getattr(self, "is_processing", False):
            log = self.query_one("#agent-log", RichLog)
            log.write(Text("[系统] 上个任务正在执行中，请稍候...", style="yellow"))
            return

        self.is_processing = True

        # 更新状态栏
        self.round_count += 1
        self.query_one("#status-bar").update(f"轮次: {self.round_count} | 正在执行...")
        
        # 显示用户输入
        log = self.query_one("#agent-log", RichLog)
        log.write(f"\n[bold green]USER (Round {self.round_count})>[/bold green] {task}")
        
        # 处理 @ 文件引用
        # 查找所有 @ 开头的文件路径，并将其转换为绝对路径提示，附加在 Prompt 后
        # 这样 Agent 就能明确知道用户指的是哪个文件，而不需要 list_directory
        import re
        
        file_refs = []
        
        def replace_file_ref(match):
            ref = match.group(1)
            # 尝试解析路径
            # 1. 尝试直接路径
            if os.path.exists(ref):
                abs_path = os.path.abspath(ref)
                file_refs.append(f"File '{ref}' is located at: {abs_path}")
                return ref # 去掉 @
                
            # 2. 尝试去掉末尾标点 (如 @file.py.)
            if ref and ref[-1] in ",.?!;:'\"":
                clean_ref = ref[:-1]
                punctuation = ref[-1]
                if os.path.exists(clean_ref):
                     abs_path = os.path.abspath(clean_ref)
                     file_refs.append(f"File '{clean_ref}' is located at: {abs_path}")
                     return clean_ref + punctuation
            
            # 3. 尝试相对于当前目录
            # (ContextAwareSuggester 也是相对于 os.getcwd())
            # 如果上面 check 失败，可能是不完整路径? 
            # 暂时只处理存在的文件
            
            return match.group(0) # 保持原样

        # 匹配 @filename (允许路径字符)
        pattern = r'@([a-zA-Z0-9_./\\-]+)'
        task = re.sub(pattern, replace_file_ref, task)
        
        if file_refs:
            task += "\n\n[System Note: User mentioned files]\n" + "\n".join(file_refs)
            log.write(Text(f"[系统] 已解析文件引用: {', '.join(file_refs)}", style="dim"))

        self._run_agent(task)
    
    @work(thread=True, exclusive=True)
    def _run_agent(self, task: str) -> None:
        """在后台线程执行 Agent（避免阻塞 UI）"""
        try:
            # 重置流式状态标志
            self._has_streamed_tokens = False
            self._has_streamed_reasoning = False
            self._token_buffer = ""
            self._reasoning_buffer = ""
            self._last_response = ""  # 清空上一轮回答，准备接收新回答
            
            # 清理可能残留的中断标志
            if hasattr(self.agent, "_cancel_event"):
                self.agent._cancel_event.clear()
            
            # 流式执行 Agent
            for event in self.agent.invoke(
                task=task,
                thread_id=self.thread_id, 
                thinking_context=True
            ):
                self.call_from_thread(self._handle_event, event)
                
                if event.get("type") == "done":
                    # 最后一个事件包含最终结果
                    result = event
                    
                    # 获取最新的内部状态
                    state_values = self.agent.get_context(thread_id=self.thread_id)
                    
                    # 更新粗略的字符串上下文（主要供之前可能用到的旧接口/统计使用）
                    messages = state_values.get("messages", [])
                    if messages:
                        self.accumulated_context = self._format_messages(messages)
                    else:
                        self.accumulated_context = ""
                        
                    round_tokens = result.get("total_tokens", 0)
                    self.total_tokens += round_tokens
                    
                    # 在主线程更新 UI
                    self.call_from_thread(self._on_agent_done, result, state_values, round_tokens)
                    break # 结束事件后终止循环，防止重复处理
            
        except Exception as e:
            self.call_from_thread(self._on_agent_error, str(e))
            import traceback
            traceback.print_exc()

    def _handle_event(self, event: dict):
        """处理 Agent 事件流"""
        log = self.query_one("#agent-log", RichLog)
        stream_container = self.query_one("#active-stream-container")
        stream_reasoning = self.query_one("#stream-reasoning", Static)
        stream_answer = self.query_one("#stream-answer", Static)
        
        typ = event.get("type")
        content = event.get("content", "")
        
        # 内部保证缓冲区初始化
        if not hasattr(self, "_token_buffer"):
            self._token_buffer = ""
        if not hasattr(self, "_reasoning_buffer"):
            self._reasoning_buffer = ""
            
        def flush_active_streams():
            """强制定稿并将当前流式内容写入主要日志，清空缓冲区"""
            has_flushed = False
            if hasattr(self, "_reasoning_buffer") and self._reasoning_buffer:
                log.write(Text(self._reasoning_buffer, style="dim italic"))
                self._reasoning_buffer = ""
                has_flushed = True
            if hasattr(self, "_token_buffer") and self._token_buffer:
                log.write(self._token_buffer)
                self._token_buffer = ""
                has_flushed = True
                
            if has_flushed:
                stream_reasoning.update("")
                stream_answer.update("")
                stream_container.styles.display = "none"

        if typ == "token":
            self._has_streamed_tokens = True
            
            # 累积到完整回复中 (用于复制)
            self._last_response += content
            self._token_buffer += content
            
            stream_container.styles.display = "block"
            stream_answer.update(self._token_buffer)
            # 让流式容器滚动到底部
            try:
                stream_container.scroll_end(animate=False)
            except Exception:
                pass
                
        elif typ == "reasoning_token":
            self._has_streamed_reasoning = True
            
            # 累积到完整回复中
            self._reasoning_buffer += content
            
            stream_container.styles.display = "block"
            stream_reasoning.update(self._reasoning_buffer)
            try:
                stream_container.scroll_end(animate=False)
            except Exception:
                pass
                
        elif typ == "done":
            # 结束时刷新残留 buffer
            flush_active_streams()
                
        elif typ in ["info", "warning", "error", "tool_exec", "tool_result", "tool_calls", "answer", "reasoning"]:
            # 对于非流式事件，先刷新 buffer，保证顺序
            flush_active_streams()
            
            # 忽略 answer/reasoning 的整块输出（因为我们已经流式输出了）
            # 除非是 answer 且我们没收到 token?
            if typ == "answer":
                # 如果有已显示的 token (通过 buffer 检测不太准，因为 buffer可能被清空)
                # 我们添加一个标志位 self._has_streamed_tokens
                if not getattr(self, "_has_streamed_tokens", False):
                     log.write(content)
            elif typ == "reasoning":
                if not getattr(self, "_has_streamed_reasoning", False):
                     log.write(Text(content, style="dim italic"))
            else:
                # 其他事件正常显示
                style = None
                prefix = ""
                if typ == "error":
                    style = "bold red"
                    prefix = "[错误] "
                elif typ == "warning":
                    style = "yellow"
                    prefix = "[警告] "
                elif typ == "tool_exec":
                    style = "blue"
                    prefix = "[执行] "
                elif typ == "tool_calls":
                    style = "magenta"
                    prefix = "[触发] "
                elif typ == "tool_result":
                    style = "green"
                    prefix = "[结果] "
                    # 限制过长的结果输出
                    lines = content.split('\n')
                    if len(lines) > 10:
                        content = "\n".join(lines[:10]) + f"\n... (剩余 {len(lines)-10} 行)"
                else:
                    prefix = ""
                
                if prefix:
                    log.write(Text(f"{prefix}{content}", style=style))
                else:
                    log.write(content)

    def _on_agent_done(self, result: dict, state_values: dict, round_tokens: int):
        """Agent 执行完成"""
        log = self.query_one("#agent-log", RichLog)

        # 刷新所有 buffer
        if hasattr(self, "_token_buffer") and self._token_buffer:
            log.write(self._token_buffer)
            self._token_buffer = ""
        if hasattr(self, "_reasoning_buffer") and self._reasoning_buffer:
            log.write(Text(self._reasoning_buffer, style="dim italic"))
            self._reasoning_buffer = ""
            
        # 清除活动流式 UI显示
        try:
            self.query_one("#stream-reasoning", Static).update("")
            self.query_one("#stream-answer", Static).update("")
            self.query_one("#active-stream-container").styles.display = "none"
        except:
            pass
            
        log.write(f"\n{'─' * 50}")
        log.write(f"[第 {self.round_count} 轮完成]")
        
        if state_values.get("summary_content"):
            log.write(f"\n{state_values['summary_content']}")
        
        # Token metrics
        prompt_tokens = result.get('prompt_tokens', 0)
        max_tokens = 128000
        
        # Try to calculate actual context text length footprint instead of prompt tokens exactly if prompt_tokens not passed.
        # But actually graph.py still returns `prompt_tokens` in the event payload.
        
        usage_pct = (prompt_tokens / max_tokens * 100) if max_tokens > 0 else 0.0
        
        log.write(f"迭代: {result.get('iteration_count', 0)} | "
                  f"完成: {'是' if result.get('task_completed') else '否'} | "
                  f"本轮 Token: {round_tokens} | 累计: {self.total_tokens} | "
                  f"上下文用量: {prompt_tokens}/{max_tokens} ({usage_pct:.1f}%)")
        log.write(f"{'─' * 50}\n")
        
        log.write(f"{'─' * 50}\n")
        
        self.is_processing = False
        self._update_status()
    
    def _on_agent_error(self, error_msg: str):
        """Agent 执行出错"""
        self.is_processing = False
        log = self.query_one("#agent-log", RichLog)
        log.write(f"[错误] {error_msg}")
    
    def action_clear_log(self):
        """Ctrl+L 清空日志"""
        log = self.query_one("#agent-log", RichLog)
        log.clear()
    
    def on_unmount(self) -> None:
        """退出时恢复 stdout"""
        sys.stdout = self._original_stdout



    @work(thread=True)
    def _run_compact(self) -> None:
        """异步执行上下文压缩"""
        log = self.query_one("#agent-log", RichLog)
        
        try:
            log.write(Text("[系统] 正在压缩上下文，请稍候...", style="yellow"))
            status_msg, summary = self.agent.compact_context(self.thread_id)
            
            # 排空可能在 compact 过程中压入的事件缓冲
            while self.agent._event_buffer:
                event = self.agent._event_buffer.pop(0)
                self.call_from_thread(self._handle_event, event)
                
            log.write(Text(f"[系统] {status_msg}", style="green"))
            
            # 同步压缩后的摘要到 TUI 的 accumulated_context
            if summary:
                old_tokens = TokenCounter.estimate_tokens(self.accumulated_context) if self.accumulated_context else 0
                self.accumulated_context = f"<context_summary>\n{summary}\n</context_summary>"
                new_tokens = TokenCounter.estimate_tokens(self.accumulated_context)
                log.write(Text(f"[系统] 上下文已更新: {old_tokens} -> {new_tokens} tokens", style="cyan"))
        except Exception as e:
            log.write(f"[错误] 上下文压缩失败: {e}")
        finally:
            self.is_processing = False
            self.query_one("#status-bar").update(f"轮次: {self.round_count} | 就绪")


if __name__ == "__main__":
    app = KlynxTUIApp()
    app.run()
