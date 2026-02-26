"""
Klynx Agent - LangGraph StateGraph
实现 OODA 循环的核心图结构
"""

import re
import os
from typing import TypedDict, List, Annotated, Dict, Any, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

from klynx.agent.tools import ToolRegistry, XMLParser, get_json_schemas
from klynx.agent.terminal import TerminalManager
from klynx.agent.syntax import SyntaxChecker
from klynx.agent.tui import TUIManager
from klynx.agent.web_search import WebSearchTool
from klynx.agent.browser import BrowserManager
from klynx.agent.mcp_manager import MCPManager
from klynx.agent.context_manager import TokenCounter
from klynx.model.adapter import LiteLLMResponse
from klynx.kbase.kb_manager import KBManager

class AgentState(TypedDict):
    """全局状态"""
    # 消息历史（LangGraph 标准 reducer，支持追加）
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 环境快照 - 每次 LLM 运行前动态更新
    env_snapshot: str
    
    # 待执行的工具调用
    pending_tool_calls: List[Dict[str, Any]]
    
    # 迭代计数器
    iteration_count: int
    
    # 工作目录
    working_dir: str
    
    # 任务完成状态
    task_completed: bool
    
    # 最后动作类型: "think" | "observe" | "act"
    last_action: str
    
    # 连续空回复计数
    empty_response_count: int
    
    # Token 使用统计
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    
    # 总任务目标（由 classify 节点从用户输入生成，全程不变）
    overall_goal: str
    
    # 当前任务目标（由 classify 生成初始值，agent 节点可通过 <task_goal> 更新）
    current_task: str
    
    # 上下文摘要（当历史过长时）
    context_summary: str
    
    # 最大上下文长度（token数）
    max_context_tokens: int
    
    # 进度摘要 - 记录已完成的关键操作
    progress_summary: str
    
    # 任务类型: "ask" | "thinking"
    task_type: str
    
    # 上下文变量 - 存储历史对话和文档读取的内容
    context: str
    
    # 用户原始输入
    user_input: str
    
    # DeepSeek 思考内容 - 存储模型的 reasoning_content
    reasoning_content: str
    
    # 上下文是否已被总结
    context_summarized: bool
    
    # KLYNX.md 文档内容（由 load_context 节点填充）
    klynx_docs: str
    
    # 项目规则（由 init 节点从 .klynx/.rules 加载）
    project_rules: str
    
    # summary 节点的总结内容
    summary_content: str

    # TUI 交互指南是否已加载（由 activate_tui_mode 工具触发）
    tui_guide_loaded: bool
    
    # 是否将思考内容添加到上下文 (由 invoke 参数控制)
    thinking_context: bool




class KlynxAgent:
    """Klynx Agent 主类"""
    
    # 基础工具注册表：工具名 -> 使用说明
    BASE_TOOLS = {
        "read_file": "读取文件内容，支持指定行范围：read_file(path: str, start_line: int=None, end_line: int=None)。可以使用 <lines>10-20</lines> 替代 start_line/end_line 参数。",
        "write_to_file": "将内容写入文件（覆盖或新建）：write_to_file(path: str, content: str)",
        "replace_in_file": "使用 SEARCH/REPLACE 块局部替换文件内容：replace_in_file(path: str, diff: str)。请严格使用标准的上下文diff格式（带 - 和 + 前缀）或 SEARCH/REPLACE 块格式提供 diff。",
        "execute_command": "执行系统命令（Windows 环境：dir, type, copy 等）：execute_command(command: str)",
        "list_directory": "列出目录树结构，支持指定递归深度：list_directory(path: str='.', depth: int=2)",
        "create_directory": "创建目录（包括父目录）：create_directory(path: str)",
        "preview_file": "预览文件前 N 行内容：preview_file(path: str, num_lines: int=50)",
        "search_in_files": "在文件中搜索匹配内容（类似grep，支持正则）：search_in_files(pattern: str, path: str='.', file_pattern: str='*', case_insensitive: bool=True, is_regex: bool=False, max_results: int=50, context_lines: int=0)。定位代码位置后，再用 read_file 精确阅读。",
        "create_terminal": "创建终端会话：create_terminal(name: str)。【重要】仅用于后台批量执行非交互式命令。",
        "run_in_terminal": "在终端执行命令：run_in_terminal(name: str, command: str)",
        "read_terminal": "读取终端输出：read_terminal(name: str, lines: int)",
        "check_syntax": "检查代码语法：check_syntax(path: str)",
        "open_tui": "启动 TUI 应用：open_tui(name: str, command: str, rows: int, cols: int)",
        "read_tui": "读取 TUI 屏幕快照：read_tui(name: str)",
        "send_keys": "向 TUI 发送按键：send_keys(name: str, keys: str)",
        "close_tui": "关闭 TUI 会话：close_tui(name: str)",
        "activate_tui_mode": "激活 TUI 交互模式：加载 TUI 操作指南到系统提示。在使用 open_tui 等 TUI 工具前必须先调用此工具。",
        "launch_interactive_session": "一键启动交互式会话（推荐）：自动激活 TUI 模式并启动应用。参数：command (如 'python', 'vim test.py')。【重要】如需启动交互式环境（如 Python REPL, Node, Vim）或查看实时屏幕输出，请优先使用本工具。",
        "web_search": "联网搜索：web_search(query: str, max_results: int=5, search_depth: str='basic')。【重要】当你需要搜索任何信息（最新动态、技术文档、API用法、新闻、事实核查等），都必须使用此工具进行搜索，严禁凭记忆编造。可设 search_depth='advanced' 进行深度搜索。",
        "query_knowledge": "查询向量知识库：query_knowledge(query: str, top_k: int=5, kb_name: str=None)。【重要】当用户询问专业领域知识时，优先使用此工具查询本地向量知识库获取精准的专业内容。可用 list_knowledge_collections 查看可用集合。",
        "list_knowledge_collections": "列出所有已添加的知识库及其集合：list_knowledge_collections(kb_name: str=None)",
        "browser_open": "打开浏览器并访问 URL：browser_open(url: str)。【浏览器调试】常用于验证前端界面或获取动态网页内容。后续配合 browser_view 查看内容、browser_act 操作元素、browser_screenshot 截图。",
        "browser_view": "获取页面内容：browser_view(selector: str=None)。不传 selector 则返回页面摘要。",
        "browser_act": "操作页面元素：browser_act(action: str, selector: str, value: str=None)。如果不确定 action，就使用 click。action 支持 click, type, hover。",
        "browser_screenshot": "截图并保存：browser_screenshot()。返回截图路径。",
        "browser_console_logs": "获取浏览器控制台日志：browser_console_logs()"
    }
    
    # 系统提示词（指令部分）
    SYSTEM_PROMPT = """<system_instructions>
  <role>Klynx</role>
  
  <workflow>
    <step number="1">
      <name>判断任务目标是否需要更新</name>
      <details>
        - 如果工具执行结果揭示了新信息，导致任务目标需要调整
        - 调用 `update_task_state` 工具函数同步更新你的最新动作或总目标
      </details>
    </step>
    <step number="2">
      <name>分析和决策</name>
      <details>
        - 分析上一步工具执行的结果
        - 判断任务是否已完成
        - 如果未完成，确定下一步需要执行的操作
      </details>
    </step>
    <step number="3">
      <name>执行操作</name>
      <details>
        - 如果需要使用工具，使用 XML 格式调用工具（XML 模式）或直接调用工具函数（Native 模式）
        - 如果任务完成，使用 XML `&lt;attempt_completion&gt;` 标签提供结果（XML模式），或直接调用 `attempt_completion` 函数（Native模式）
      </details>
    </step>
    <step number="4">
      <name>错误反思</name>
      <details>
        - 如果上一步工具返回了错误（&lt;error&gt; 标签），你必须先在回复中使用 &lt;reflection&gt; 标签分析失败原因
        - 例如：&lt;reflection&gt;我刚才提供的 SEARCH 块缩进不一致&lt;/reflection&gt;
        - 禁止不经反思就重试相同操作
      </details>
    </step>
  </workflow>

  <instructions>
    阅读markdown文件时，优先匹配并读取其标题和总行数和总字数等信息，再判断是否读取完整的文档。
    当你不理解用户提问或者任务含义时，请询问用户，不允许私自随便操作。
    当任务目标明确时，请直奔任务目标。
    当你完成操作后，请检查是否有遗漏的部分。
  </instructions>


</system_instructions>"""
    
    def _get_system_prompt(self) -> str:
        """
        生成完整的系统提示词（基于已加载的工具和配置动态生成）
        
        Returns:
            完整的系统提示词字符串
        """
        prompt_parts = [self.SYSTEM_PROMPT]
        
        # OS 系统提示
        os_lower = self.os_name.lower()
        if os_lower in ["linux", "macos", "mac"]:
            sys_msg = f"当前环境是 {self.os_name} 系统，不要使用 Windows 命令。阅读代码文件时，优先使用 Linux/Unix 命令进行搜索（例如 grep ）。"
        else:
            sys_msg = f"当前环境是 {self.os_name} 系统，不要使用 Linux/Unix 命令。阅读代码文件时，优先使用 windows 命令进行搜索（例如 findstr ）。"
        prompt_parts.append(f"<system_note>\n  <note>{sys_msg}</note>\n</system_note>")
        
        # 工具使用提示（仅在有工具加载时）
        if self._tool_prompts_cache:
            prompt_parts.append(self._tool_prompts_cache)
        
        # 记忆系统提示（仅在有 memory_dir 时）
        if self.memory_dir:
            memory_path = os.path.join(self.memory_dir, ".klynx", ".memory")
            rel_path = os.path.relpath(memory_path, self.working_dir) if self.working_dir else ".klynx/.memory"
            prompt_parts.append(f"""
<memory_system>
  <description>你拥有一个长期记忆系统，存储在 {rel_path} 文件中（XML格式）。</description>
  <usage>
    <item>当你需要查看历史记忆时，使用 read_file 读取 {rel_path}</item>
    <item>当你获取到重要的用户信息、项目上下文、设计决策等中长期有用的信息时，主动保存到 {rel_path}</item>
    <item>随着项目变化，及时更新 .memory 中的过时信息</item>
    <item>使用 replace_in_file 或 write_to_file 更新记忆内容</item>
  </usage>
  <format>
    XML格式，示例：
    &lt;memory&gt;
      &lt;entry key="user_preferences"&gt;内容&lt;/entry&gt;
      &lt;entry key="project_architecture"&gt;内容&lt;/entry&gt;
    &lt;/memory&gt;
  </format>
</memory_system>""")
        
        # TUI 交互指南（仅在 activate_tui_mode 工具被调用后注入）
        if getattr(self, '_tui_guide_loaded', False):
            prompt_parts.append(self._get_tui_guide())
        
        return "\n".join(prompt_parts)
    
    def _get_tui_guide(self) -> str:
        """返回 TUI 交互指南（由 activate_tui_mode 触发注入 system prompt）"""
        return """
<tui_interaction_guide>
  <description>你已激活 TUI 交互模式。以下是使用 TUI 工具的完整指南。</description>

  <workflow>
    <step number="1">
      <name>启动 TUI 应用</name>
      <details>
        使用 open_tui 启动目标应用：
        &lt;open_tui&gt;&lt;name&gt;会话名&lt;/name&gt;&lt;command&gt;应用命令&lt;/command&gt;&lt;/open_tui&gt;
        示例：&lt;open_tui&gt;&lt;name&gt;editor&lt;/name&gt;&lt;command&gt;vim test.py&lt;/command&gt;&lt;/open_tui&gt;
      </details>
    </step>
    <step number="2">
      <name>读取屏幕状态</name>
      <details>
        在每次操作后，使用 read_tui 查看当前屏幕：
        &lt;read_tui&gt;&lt;name&gt;editor&lt;/name&gt;&lt;/read_tui&gt;
        返回 XML 格式屏幕快照，包含每行文本和光标位置。
        **重要：每次 send_keys 后必须 read_tui 确认结果！**
      </details>
    </step>
    <step number="3">
      <name>发送按键</name>
      <details>
        使用 send_keys 发送输入。按键用空格分隔，支持特殊键：
        &lt;send_keys&gt;&lt;name&gt;editor&lt;/name&gt;&lt;keys&gt;按键序列&lt;/keys&gt;&lt;/send_keys&gt;
      </details>
    </step>
    <step number="4">
      <name>关闭会话</name>
      <details>完成后用 close_tui 关闭会话。</details>
    </step>
  </workflow>

  <key_reference>
    <category name="基础键">
      Enter, Tab, Escape (Esc), Backspace, Delete, Space
    </category>
    <category name="方向键">
      Up, Down, Left, Right, Home, End, PageUp, PageDown
    </category>
    <category name="Ctrl 组合">
      Ctrl-C (中断), Ctrl-D (EOF), Ctrl-Z (挂起), Ctrl-S (保存),
      Ctrl-A, Ctrl-E, Ctrl-K, Ctrl-W, Ctrl-U, Ctrl-R 等
    </category>
    <category name="功能键">
      F1 ~ F12
    </category>
  </key_reference>

  <send_keys_examples>
    <example desc="在 vim 中输入文本">i hello world Escape</example>
    <example desc="vim 保存退出">Escape :wq Enter</example>
    <example desc="vim 不保存退出">Escape :q! Enter</example>
    <example desc="Python REPL 计算">1+1 Enter</example>
    <example desc="向上翻页">PageUp</example>
    <example desc="发送中断信号">Ctrl-C</example>
    <example desc="连续方向键">Up Up Down Enter</example>
    <example desc="输入带空格文本（每个词是独立token）">hello world Enter</example>
  </send_keys_examples>

  <best_practices>
    <rule>每次 send_keys 后立即 read_tui 确认操作结果</rule>
    <rule>操作 vim 时先确认当前模式（NORMAL/INSERT/VISUAL），再决定按键</rule>
    <rule>屏幕底部状态栏通常包含模式、光标位置等关键信息</rule>
    <rule>如果屏幕没有预期变化，可能需要等待后再次 read_tui</rule>
    <rule>完成操作后始终 close_tui，释放资源</rule>
    <rule>对于 send_keys 中的普通文本，每个空格分隔的词会被直接拼接发送</rule>
  </best_practices>
</tui_interaction_guide>"""
    
    # 上下文总结阈值（90%）
    CONTEXT_SUMMARIZE_THRESHOLD = 0.9
    
    def __init__(self, working_dir: str = ".", model=None, max_iterations: int = 1000,
                 memory_dir: str = "", load_project_docs: bool = True,
                 os_name: str = "windows", browser_headless: bool = False,
                 tool_call_mode: str = "native"):
        """
        初始化 Klynx Agent
        
        Args:
            working_dir: 工作目录
            model: LangChain 模型实例（需设置 max_context_tokens 属性）
            max_iterations: 最大迭代次数，防止无限循环
            memory_dir: Agent 记忆目录路径（.klynx/.rules/.memory 的存储位置），为空则不加载记忆
            load_project_docs: 是否递归加载工作目录下的 KLYNX.md 项目文档，默认 True
            os_name: 操作系统名称，用于给 Agent 提供环境提示 (例如: windows, linux, macos)
            browser_headless: 浏览器工具是否运行在无头模式，默认为 False (显示界面)
            tool_call_mode: 工具调用模式，"native" 使用 LiteLLM 原生 Function Calling，
                           "xml" 使用传统 XML 文本解析（兼容小模型/不支持 FC 的模型）
        """
        self.working_dir = os.path.abspath(working_dir)
        self.model = model
        self.max_iterations = max_iterations
        self.memory_dir = os.path.abspath(memory_dir) if memory_dir else ""
        self.load_project_docs = load_project_docs
        self.os_name = os_name
        self.tool_call_mode = tool_call_mode  # "native" | "xml"
        
        # 自动回退：如果模型明确是 o1 或 reasoner，强制使用 xml 模式（因不支持 Native FC）
        if self.model and hasattr(self.model, "model"):
            model_name_lower = self.model.model.lower()
            if "reasoner" in model_name_lower or "o1" in model_name_lower:
                self.tool_call_mode = "xml"
        
        # 终端管理器
        self.terminal_manager = TerminalManager(working_dir)
        # TUI 管理器
        self.tui_manager = TUIManager(working_dir)
        # 联网搜索工具
        self.web_search_tool = WebSearchTool()
        # 知识库查询工具
        self.kb_manager = KBManager()
        # 浏览器管理工具
        self.browser_manager = BrowserManager(headless=browser_headless)
        # 从模型对象读取上下文窗口大小，默认 128k
        self.max_context_tokens = getattr(model, 'max_context_tokens', 128000)
        
        # MCP (Model Context Protocol) 管理器
        self.mcp_manager = MCPManager()
        
        # 流式输出回调函数
        self.streaming_callback = None
        
        # 初始化组件
        self.xml_parser = XMLParser()
        
        # 工具系统：已加载的工具（名称 -> 说明）和外部工具函数（名称 -> callable）
        self.tools: Dict[str, str] = {}
        self.external_tool_funcs: Dict[str, callable] = {}
        
        # 预加载的工具提示词缓存
        self._tool_prompts_cache: str = ""
        
        # Native Function Calling 用的 JSON Schema 缓存
        self._json_schemas: list = []
        
        # 回滚目标配置（存储 checkpoint config，供 stream 使用）
        self._rollback_config = None
        
        # 事件缓冲区（流式输出用）
        self._event_buffer: List[Dict[str, Any]] = []
        
        # 设置工具的工作目录
        ToolRegistry.set_working_dir(self.working_dir)
        
        # 构建图
        self.workflow = self._build_graph()
        # 使用 MemorySaver 作为 checkpointer 来保存状态
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _refresh_tool_prompts(self):
        """刷新工具提示词缓存 & Native Function Calling Schema 缓存"""
        if not self.tools:
            self._tool_prompts_cache = ""
            self._json_schemas = []
            return
        
        # 始终刷新 Native FC Schema 缓存（即使当前模式是 xml，也提前准备好以支持动态切换）
        external_descs = {name: desc for name, desc in self.tools.items() if name in self.external_tool_funcs}
        self._json_schemas = get_json_schemas(list(self.tools.keys()), external_tools=external_descs)
        
        if self.tool_call_mode == "native":
            # Native 模式：工具定义已通过 API tools 参数传递，prompt 中只做简略提示
            lines = []
            lines.append("\n<tool_usage_guide>")
            lines.append("  <description>你可以直接调用工具函数来完成任务。工具列表已通过 API 提供。当任务完成时，请**必须**调用 `attempt_completion` 工具函数并传递 `result` 参数来结束任务。</description>")
            lines.append("  <note>在调用原生工具的同时，你可以并且应该继续在文本回复中输出 `<thinking>` 或 `<reflection>` 等辅助 XML 标签。若需更新目标，请调用 `update_task_state` 工具。</note>")
            lines.append("</tool_usage_guide>")
            self._tool_prompts_cache = "\n".join(lines)
        else:
            # XML 模式：保持原来的详细工具列表
            lines = []
            lines.append("\n<tool_usage_guide>")
            lines.append("  <description>你可以使用以下工具来完成任务。使用 XML 格式调用工具。</description>")
            lines.append("  <available_tools>")
            for name, desc in self.tools.items():
                lines.append(f'    <tool name="{name}">{desc}</tool>')
            lines.append("  </available_tools>")
            lines.append("</tool_usage_guide>")
            lines.append("<completion_format>")
            lines.append("  &lt;attempt_completion&gt;&lt;result&gt;任务结果说明&lt;/result&gt;&lt;/attempt_completion&gt;")
            lines.append("</completion_format>")
            self._tool_prompts_cache = "\n".join(lines)
        
    def add_tools(self, *args):
        """
        加载工具到 Agent
        
        支持三种调用方式：
        1. agent.add_tools("all")              - 加载所有基础工具
        2. agent.add_tools("read_file", "write_to_file")  - 加载指定基础工具
        3. agent.add_tools(my_func, "工具说明")  - 加载外部函数作为工具
        
        可多次调用以组合加载：
            agent.add_tools("read_file", "list_directory")
            agent.add_tools(my_analyzer, "分析代码结构并返回报告")
        """
        if not args:
            raise ValueError("add_tools 需要至少一个参数")
        
        # 模式1: add_tools("all") - 加载所有基础工具
        if len(args) == 1 and args[0] == "all":
            self.tools.update(self.BASE_TOOLS)
            self._emit("info", f"[工具] 已加载全部 {len(self.BASE_TOOLS)} 个基础工具")
            self._refresh_tool_prompts()
            return self
        
        # 模式3: add_tools(callable, "说明") - 加载外部函数
        if len(args) == 2 and callable(args[0]) and isinstance(args[1], str):
            func, description = args
            tool_name = func.__name__
            self.tools[tool_name] = description
            self.external_tool_funcs[tool_name] = func
            self._emit("info", f"[工具] 已加载外部工具: {tool_name} - {description}")
            self._refresh_tool_prompts()
            return self
        
        # 模式2: add_tools("read_file", "write_to_file", ...) - 加载指定基础工具
        for name in args:
            if not isinstance(name, str):
                raise ValueError(f"无效参数: {name}，期望工具名称字符串或 (callable, str)")
            if name not in self.BASE_TOOLS:
                raise ValueError(f"未知基础工具: {name}，可用工具: {', '.join(self.BASE_TOOLS.keys())}")
            self.tools[name] = self.BASE_TOOLS[name]
        self._emit("info", f"[工具] 已加载 {len(args)} 个基础工具: {', '.join(args)}")
        self._refresh_tool_prompts()
        return self
    
    def add_mcp(self, mcp_json_path: str):
        """
        加载外部 MCP (Model Context Protocol) Server，
        并将它们暴露出的 Tools 挂载到 Agent 中。
        """
        self._emit("info", f"[MCP] 加载 MCP 配置文件: {mcp_json_path}")
        mcp_tools = self.mcp_manager.load_from_json(mcp_json_path)
        if mcp_tools:
            # MCP tools are returned as LangChain StructuredTools
            for tool in mcp_tools:
                self.add_tools(tool, tool.description)
            self._emit("info", f"[MCP] 成功挂载 {len(mcp_tools)} 个 MCP 工具")
        else:
            self._emit("warning", f"[MCP] 未从 {mcp_json_path} 加载到任何工具")
        return self
    
    def add_kb(self, name: str, path: str, embedding_model: str = None):
        """
        添加一个向量知识库到 Agent
        
        Args:
            name: 知识库名称（标识符，如 'my_papers'）
            path: ChromaDB chroma_store 目录路径
            embedding_model: 嵌入模型名称或本地路径（可选）
        
        Usage:
            agent.add_kb('papers', '/path/to/chroma_store')
            agent.add_kb('notes',  '/path/to/another_store')
        """
        self.kb_manager.add_kb(name, path, embedding_model)
        # 自动加载知识库工具（如果尚未加载）
        if "query_knowledge" not in self.tools:
            self.tools["query_knowledge"] = self.BASE_TOOLS["query_knowledge"]
            self.tools["list_knowledge_collections"] = self.BASE_TOOLS["list_knowledge_collections"]
            self._refresh_tool_prompts()
        return self
    
    def _get_tools_prompt(self) -> str:
        """
        生成工具列表的 prompt 片段（含名称和说明）
        
        Returns:
            XML 格式的工具列表字符串
        """
        if not self.tools:
            return "无可用工具"
        
        lines = []
        for name, desc in self.tools.items():
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)
    
    def _init_klynx_dir(self):
        """
        初始化 .klynx 配置目录
        
        在 memory_dir 下创建 .klynx 文件夹，并生成模板 .rules 和 .memory 文件。
        如果 memory_dir 为空则跳过。
        """
        if not self.memory_dir:
            return
        
        klynx_dir = os.path.join(self.memory_dir, ".klynx")
        
        if not os.path.isdir(klynx_dir):
            os.makedirs(klynx_dir, exist_ok=True)
            self._emit("info", f"[配置] 已创建 .klynx/ 目录: {klynx_dir}")
        
        # 初始化 .rules 文件
        rules_path = os.path.join(klynx_dir, ".rules")
        if not os.path.isfile(rules_path):
            template = '<rules>\n</rules>'
            with open(rules_path, 'w', encoding='utf-8') as f:
                f.write(template)
            self._emit("info", "[配置] 已创建 .klynx/.rules 模板")
        
        # 初始化 .memory 文件
        memory_path = os.path.join(klynx_dir, ".memory")
        if not os.path.isfile(memory_path):
            template = '<memory>\n</memory>'
            with open(memory_path, 'w', encoding='utf-8') as f:
                f.write(template)
            self._emit("info", "[配置] 已创建 .klynx/.memory 模板")
    
    def _load_rules(self) -> str:
        """
        加载 .klynx/.rules 文件内容（从 memory_dir 读取）
        
        Returns:
            .rules 文件的内容字符串
        """
        if not self.memory_dir:
            return ""
        
        rules_path = os.path.join(self.memory_dir, ".klynx", ".rules")
        
        if not os.path.isfile(rules_path):
            self._emit("info", "[配置] .klynx/.rules 文件不存在")
            return ""
        
        try:
            with open(rules_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            if content:
                self._emit("info", f"[配置] 已加载 .klynx/.rules ({len(content)} 字符)")
                return content
            return ""
        except Exception as e:
            self._emit("error", f"[配置] 读取 .klynx/.rules 失败: {e}")
            return ""
    
    def _find_klynx_docs(self) -> str:
        """
        递归搜索工作目录下的所有 KLYNX.md 文件
        
        KLYNX.md 描述对应文件夹的项目内容，在 classify 节点后、agent 节点前加载。
        
        Returns:
            所有 KLYNX.md 的路径和内容（XML格式）
        """
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 
                     '.idea', '.vscode', '.klynx'}
        docs_parts = []
        
        for root, dirs, files in os.walk(self.working_dir):
            # 跳过忽略目录
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            
            if 'KLYNX.md' in files:
                filepath = os.path.join(root, 'KLYNX.md')
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                    if content:
                        rel_dir = os.path.relpath(root, self.working_dir)
                        if rel_dir == '.':
                            rel_dir = '(根目录)'
                        docs_parts.append(
                            f'  <doc path="{self._escape_xml(rel_dir)}">\n'
                            f'    {self._escape_xml(content)}\n'
                            f'  </doc>'
                        )
                        self._emit("info", f"[配置] 已加载 {os.path.relpath(filepath, self.working_dir)}")
                except Exception as e:
                    self._emit("error", f"[配置] 读取 {filepath} 失败: {e}")
        
        if not docs_parts:
            return ""
        
        return '<project_docs description="KLYNX.md 描述了对应文件夹下的项目内容">\n' + '\n'.join(docs_parts) + '\n</project_docs>'
    
    def _init_node(self, state: AgentState) -> Dict[str, Any]:
        """
        初始化节点 - 在每次运行最开始（分类节点之前）执行
        
        职责：
        1. 如果 memory_dir 非空，检查/创建 .klynx 文件夹及 .rules/.memory 文件
        2. 加载 .klynx/.rules 内容到 state
        """
        if self.memory_dir:
            self._emit("info", f"[初始化] 检查 .klynx 配置 ({self.memory_dir})...")
            self._init_klynx_dir()
            rules_content = self._load_rules()
        else:
            self._emit("info", "[初始化] 未设置 memory_dir，跳过 .klynx 配置加载")
            rules_content = ""
        
        return {
            "project_rules": rules_content
        }
    
    def _get_env_snapshot(self) -> str:
        """
        获取环境快照（XML格式）
        """
        # 生成文件树
        file_tree = self._generate_file_tree(self.working_dir, depth=2)
        
        snapshot = f"""<file_tree>
{file_tree}
</file_tree>"""
        return snapshot
    
    def _generate_file_tree(self, path: str, depth: int = 2, prefix: str = "") -> str:
        """生成文件树结构"""
        if depth < 0:
            return ""
        
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return f"{prefix}[无权限访问]"
        
        lines = []
        # 过滤隐藏文件和常见忽略目录
        ignore_patterns = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vscode'}
        entries = [e for e in entries if e not in ignore_patterns and not e.startswith('.')]
        
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            entry_path = os.path.join(path, entry)
            
            if os.path.isdir(entry_path):
                lines.append(f"{prefix}{connector}{entry}/")
                if depth > 0:
                    extension = "    " if is_last else "│   "
                    subtree = self._generate_file_tree(entry_path, depth - 1, prefix + extension)
                    if subtree:
                        lines.append(subtree)
            else:
                lines.append(f"{prefix}{connector}{entry}")
        
        return "\n".join(lines)
    
    def _parse_model_response(self, response) -> Dict[str, str]:
        """
        解析模型响应 - 支持 DeepSeek 思考模式
        
        DeepSeek 思考模式返回两个字段：
        - reasoning_content: 思维链内容（思考过程）
        - content: 最终回答内容
        
        Args:
            response: 模型返回的响应对象
            
        Returns:
            包含 reasoning_content 和 content 的字典
        """
        result = {
            "reasoning_content": "",
            "content": ""
        }
        
        # 获取消息对象
        message = response
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
        
        # 提取 reasoning_content（DeepSeek 思考模式特有）
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            result["reasoning_content"] = message.reasoning_content
        elif hasattr(message, 'additional_kwargs'):
            # 备选：从 additional_kwargs 中获取
            result["reasoning_content"] = message.additional_kwargs.get('reasoning_content', '')
        
        # 提取 content
        if hasattr(message, 'content'):
            result["content"] = message.content or ""
        
        return result
    
    def _extract_reasoning_content(self, response) -> str:
        """
        提取 DeepSeek 模型的思考内容 (reasoning_content)
        
        支持多种获取方式：
        1. response.reasoning_content (直接属性 - LiteLLMResponse)
        2. response.additional_kwargs['reasoning_content']
        3. response.response_metadata['reasoning_content']
        
        Args:
            response: 模型响应对象
            
        Returns:
            思考内容字符串
        """
        # 方式1: 直接属性 (LiteLLMResponse)
        if hasattr(response, 'reasoning_content') and response.reasoning_content:
            return response.reasoning_content
        
        # 方式2: additional_kwargs
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            rc = response.additional_kwargs.get('reasoning_content', '')
            if rc:
                return rc
        
        # 方式3: response_metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            rc = response.response_metadata.get('reasoning_content', '')
            if rc:
                return rc
        
        return ""
    
    def _build_context(self, state: AgentState, include_history: bool = True) -> str:
        """
        构建XML格式的上下文变量（完整注入，不截断）
        
        上下文包含：
        - 历史对话（可选）
        - 文档读取内容
        - 任务目标（如果已生成）
        - 环境信息
        
        只有当上下文为空时，才添加系统身份介绍
        
        Args:
            state: 当前状态
            include_history: 是否包含历史对话
            
        Returns:
            构建好的XML上下文字符串
        """
        # Token统计
        token_stats = {
            "system_identity": 0,
            "task_goal": 0,
            "environment": 0,
            "working_directory": 0,
            "progress_summary": 0,
            "conversation_history": 0,
            "context_summary": 0,
            "document_content": 0
        }
        
        # 判断上下文是否为空（没有任何实质内容）
        existing_context = state.get("context", "")
        current_task = state.get("current_task", "")
        progress = state.get("progress_summary", "")
        has_content = bool(existing_context.strip() or current_task.strip() or progress.strip())
        
        # 构建XML上下文
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>\n<context>']
        
        # 总任务目标
        overall_goal = state.get("overall_goal", "")
        if overall_goal:
            escaped_goal = self._escape_xml(overall_goal)
            goal_xml = f"""  <overall_goal>
    {escaped_goal}
  </overall_goal>"""
            xml_parts.append(goal_xml)
        
        # 当前任务目标
        if current_task:
            escaped_task = self._escape_xml(current_task)
            task_xml = f"""  <current_task>
    {escaped_task}
  </current_task>"""
            xml_parts.append(task_xml)
            token_stats["task_goal"] = TokenCounter.estimate_tokens(task_xml)
        
        # 环境信息
        env_snapshot = state.get("env_snapshot", "")
        if env_snapshot:
            escaped_env = self._escape_xml(env_snapshot)
            env_xml = f"""  <environment>
    {escaped_env}
  </environment>"""
            xml_parts.append(env_xml)
            token_stats["environment"] = TokenCounter.estimate_tokens(env_xml)
        
        # 工作目录
        escaped_workdir = self._escape_xml(str(self.working_dir))
        workdir_xml = f"""  <working_directory>
    {escaped_workdir}
  </working_directory>"""
        xml_parts.append(workdir_xml)
        token_stats["working_directory"] = TokenCounter.estimate_tokens(workdir_xml)
        
        # 进度摘要
        if progress:
            escaped_progress = self._escape_xml(progress)
            progress_xml = f"""  <progress_summary>
    {escaped_progress}
  </progress_summary>"""
            xml_parts.append(progress_xml)
            token_stats["progress_summary"] = TokenCounter.estimate_tokens(progress_xml)
        
        # 对话历史（如果有总结，使用总结；否则使用完整历史）
        context_summary = state.get("context_summary", "")
        if include_history:
            if context_summary:
                # 使用已总结的上下文
                summary_xml = f"""  <context_summary>
    {self._escape_xml(context_summary)}
  </context_summary>"""
                xml_parts.append(summary_xml)
                token_stats["context_summary"] = TokenCounter.estimate_tokens(summary_xml)
                self._emit("info", "  [上下文] 使用已总结的历史")
            else:
                # 使用完整对话历史
                history_msgs = state.get("messages", [])
                if history_msgs:
                    history_xml = self._format_conversation_history_xml(history_msgs)
                    if history_xml:
                        full_history_xml = f"""  <conversation_history>
{history_xml}
  </conversation_history>"""
                        xml_parts.append(full_history_xml)
                        token_stats["conversation_history"] = TokenCounter.estimate_tokens(full_history_xml)
        
        # 已有上下文（文档读取内容等）
        if existing_context:
            escaped_context = self._escape_xml(existing_context)
            doc_xml = f"""  <document_content>
    {escaped_context}
  </document_content>"""
            xml_parts.append(doc_xml)
            token_stats["document_content"] = TokenCounter.estimate_tokens(doc_xml)
        
        xml_parts.append("</context>")
        context_xml = "\n".join(xml_parts)
        
        # 计算总token并显示统计
        total_tokens = sum(token_stats.values())
        usage_pct = (total_tokens / self.max_context_tokens) * 100
        
        # 打印上下文用量统计
        self._emit("context_stats", f"[上下文统计] Token用量: {total_tokens:,} / {self.max_context_tokens:,} ({usage_pct:.1f}%)")
        
        # 显示各部分用量（仅显示非零部分）
        non_zero_stats = {k: v for k, v in token_stats.items() if v > 0}
        if non_zero_stats:
            stats_str = " | ".join([f"{k}: {v:,}" for k, v in non_zero_stats.items()])
            self._emit("context_stats", f"  细分: {stats_str}")
        
        # 警告：接近上限
        if usage_pct > 80:
            self._emit("warning", "⚠️ 警告: 上下文用量超过80%，建议清理历史对话")
        
        return context_xml
    
    def _escape_xml(self, text: str) -> str:
        """转义XML特殊字符"""
        if not text:
            return ""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        return text
    
    def _format_conversation_history_xml(self, messages: List[BaseMessage]) -> str:
        """
        格式化对话历史为XML格式
        
        对AI消息进行压缩：只保留工具调用（包含操作细节）和完成标记，
        去掉Agent的分析/思考文本，避免Agent重新读取自己的分析并误判为用户请求。
        对用户/工具结果消息则完整保留。
        
        Args:
            messages: 消息列表
            
        Returns:
            XML格式的对话历史字符串
        """
        if not messages:
            return ""
        
        # 工具调用相关的XML标签（需要保留的）
        tool_tags = [
            'read_file', 'write_to_file', 'execute_command', 
            'list_directory', 'replace_in_file', 'create_directory',
            'preview_file', 'search_in_files', 'attempt_completion'
        ]
        
        entries = []
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                # 用户消息和工具结果：过滤无意义的消息
                content = msg.content if hasattr(msg, 'content') else str(msg)
                stripped = content.strip()
                # 过滤仅包含“继续”、“好”、“ok”等无信息量的消息
                if stripped.lower() in ('', '继续', '好', '好的', 'ok', 'yes', '易', '嗯', '确认', 'continue', 'go', 'y'):
                    continue
                escaped_content = self._escape_xml(stripped)
                entries.append(f'    <message index="{i}" role="user">\n      {escaped_content}\n    </message>')
            elif isinstance(msg, AIMessage):
                # AI消息：只提取工具调用部分，去掉分析/思考文本
                content = msg.content if hasattr(msg, 'content') else str(msg)
                compressed = self._compress_ai_message(content, tool_tags)
                if compressed:
                    escaped_content = self._escape_xml(compressed.strip())
                    entries.append(f'    <message index="{i}" role="assistant" compressed="true">\n      {escaped_content}\n    </message>')
            else:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                escaped_content = self._escape_xml(content.strip())
                entries.append(f'    <message index="{i}" role="system">\n      {escaped_content}\n    </message>')
        
        return "\n".join(entries)
    
    def _compress_ai_message(self, content: str, tool_tags: list) -> str:
        """
        压缩AI消息：保留工具调用XML标签及其内容，去掉分析/思考文本。
        
        保留的内容：工具调用（包括写入的代码、修改的内容等）和attempt_completion。
        去掉的内容：<step>、<thinking>、<task_goal>及其他纯文本分析。
        
        Args:
            content: AI消息的完整内容
            tool_tags: 需要保留的工具标签列表
            
        Returns:
            压缩后的内容字符串
        """
        preserved_parts = []
        
        for tag in tool_tags:
            # 匹配完整的工具调用标签（包括子元素格式和属性格式）
            # 子元素格式: <tag>...</tag>
            pattern = rf'<{tag}[\s>].*?</{tag}>'
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                preserved_parts.append(match)
        
        if not preserved_parts:
            return ""
        
        return "[Agent操作记录]\n" + "\n".join(preserved_parts)
    
    def _format_conversation_history(self, messages: List[BaseMessage]) -> str:
        """
        格式化对话历史（完整注入，不截断）
        
        Args:
            messages: 消息列表
            
        Returns:
            格式化的对话历史字符串
        """
        history_parts = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                history_parts.append(f"[User/Tool Result]: {content}")
            elif isinstance(msg, AIMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                history_parts.append(f"[Assistant]: {content}")
        
        return "\n".join(history_parts)
    
    def _quick_summarize_messages(self, messages: list, max_entries: int = 5) -> str:
        """
        快速本地总结消息历史（不调用LLM）
        
        提取最近的用户输入和AI回复摘要，用于新一轮对话的上下文继承。
        避免将完整的旧消息注入prompt导致Agent误解为当前任务。
        
        Args:
            messages: 消息列表
            max_entries: 最多提取的消息对数
            
        Returns:
            简短的对话摘要字符串
        """
        if not messages:
            return ""
        
        summary_parts = []
        # 只取最近的消息
        recent = messages[-max_entries * 2:] if len(messages) > max_entries * 2 else messages
        
        for msg in recent:
            if isinstance(msg, HumanMessage):
                content = getattr(msg, 'content', str(msg))
                # 截断长内容
                snippet = content[:150] + "..." if len(content) > 150 else content
                summary_parts.append(f"用户: {snippet}")
            elif isinstance(msg, AIMessage):
                content = getattr(msg, 'content', str(msg))
                # AI回复只取前100字符
                snippet = content[:100] + "..." if len(content) > 100 else content
                summary_parts.append(f"助手: {snippet}")
        
        if not summary_parts:
            return ""
        
        return "[之前的对话摘要]\n" + "\n".join(summary_parts)
    
    def _summarize_context(self, state: AgentState) -> Dict[str, Any]:
        """
        上下文总结节点 - 当上下文接近爆满时调用LLM总结对话历史
        
        保留：当前任务目标、正在进行的操作
        删除：已完成任务的详细内容
        
        注意：不直接修改 messages（因为使用 reducer），而是将总结存入 context_summary
        
        Returns:
            更新后的状态，包含总结
        """
        current_task = state.get("current_task", "")
        messages = state.get("messages", [])
        
        if not messages or not self.model:
            return {}
        
        self._emit("info", f"[上下文总结] 对话历史过长（{len(messages)} 条消息），正在总结...")
        
        # 构建总结提示词
        history_text = self._format_conversation_history(messages)
        
        summarize_prompt = f"""请总结以下对话历史，生成一个简洁的摘要。

<current_task>
{current_task}
</current_task>

<conversation_history>
{history_text}
</conversation_history>

<requirements>
1. 保留当前任务的关键信息和进度
2. 保留最近执行的操作和结果
3. 删除已完成子任务的详细内容
4. 保留任何重要的发现或错误信息
5. 摘要应该让LLM能够继续执行当前任务
6. 输出结构化的摘要，包含：任务进度、已完成操作、待处理事项
</requirements>

请直接输出总结内容，不需要额外标签。"""
        
        try:
            response = self.model.invoke([HumanMessage(content=summarize_prompt)])
            summary = response.content.strip()
            
            # Token 用量对比
            if hasattr(response, 'usage') and response.usage:
                actual_usage = response.usage
                self._emit("token_usage", f"  [总结Token用量] Prompt: {actual_usage['prompt_tokens']:,} | Completion: {actual_usage['completion_tokens']:,}")
            
            # 计算压缩效果
            original_tokens = TokenCounter.count_message_tokens(messages)
            summary_tokens = TokenCounter.estimate_tokens(summary)
            self._emit("info", f"[上下文总结] 压缩效果: {original_tokens:,} tokens -> {summary_tokens:,} tokens ({(summary_tokens/original_tokens*100):.1f}%)")
            
            # 将总结存入 context_summary，并使用 RemoveMessage 删除所有旧消息
            # LangGraph 的 operator.add reducer 需要 RemoveMessage 对象才能真正删除
            delete_messages = [RemoveMessage(id=msg.id) for msg in messages if hasattr(msg, "id") and msg.id]
            
            return {
                "context_summary": summary,
                "context_summarized": True,
                "messages": delete_messages
            }
            
        except Exception as e:
            self._emit("error", f"[上下文总结] 总结失败: {e}")
            return {}
    
    def _check_context_overflow(self, state: AgentState) -> bool:
        """
        检查上下文是否需要总结
        
        Returns:
            True 如果需要总结，False 否则
        """
        messages = state.get("messages", [])
        if not messages:
            return False
        
        # 计算当前对话历史的 token 数
        history_tokens = TokenCounter.count_message_tokens(messages)
        threshold = int(self.max_context_tokens * self.CONTEXT_SUMMARIZE_THRESHOLD)
        
        return history_tokens > threshold
    
    def _classify_task_node(self, state: AgentState) -> Dict[str, Any]:
        """
        任务分类与回答节点 - 由LLM直接回答用户问题并决定下一步路由
        
        职责:
        1. 直接回答用户的问题
        2. 判断是否需要进入Agent工具执行流程
        3. 给出下一步建议
        
        路由结果:
        - end: 简单问题已直接回答，无需工具，结束
        - agent: 需要使用工具执行的任务，进入Agent流程
        """
        user_input = state.get("user_input", "")
        
        self._emit("info", "[分类与回答] 由LLM分析并回答用户问题...")
        
        if self.model is None:
            self._emit("info", "[分类与回答] 未设置模型，默认进入Agent流程")
            return {"task_type": "agent"}
        
        # 获取已有的对话历史（优先使用 messages，其次使用 context_summary）
        history_msgs = state.get("messages", [])
        context_summary = state.get("context_summary", "")
        history_text = ""
        if history_msgs:
            history_text = self._format_conversation_history(history_msgs)
        elif context_summary:
            # 新一轮对话：旧消息已被总结为 context_summary
            history_text = context_summary
        
        # 构建提示词：包含对话历史，要求LLM回答问题并决定路由
        history_section = ""
        if history_text:
            history_section = f"""
以下是之前的对话历史：
{history_text}

"""
        
        # 获取 .rules 规则
        project_rules = state.get("project_rules", "")
        rules_section = ""
        if project_rules:
            rules_section = f"\n<project_rules>\n{project_rules}\n</project_rules>\n"
        
        # 生成已加载工具列表提示
        tools_hint = ""
        if self.tools:
            tool_names = ", ".join(self.tools.keys())
            tools_hint = f"\n你可以使用的工具包括: {tool_names}\n"
        
        classify_prompt = f"""你是 Klynx Agent，一个智能编程助手。
{rules_section}{history_section}{tools_hint}用户说: {user_input}

请你完成以下任务：

1. **直接回答用户的问题**：结合对话历史，给出清晰、有帮助的回复。

2. **判断下一步路由**：在回答末尾，使用以下标签标记你的路由决策：
   - [end]: 如果这是一个简单的问候、闲聊、常识性问答等，你已经完整回答，不需要使用任何工具。
   - [agent]: 如果用户的请求涉及以下任何操作：查看文件、执行命令、编写代码、联网搜索、启动交互式环境等，需要进入Agent执行流程。

3. **如果路由为 [agent]**：请额外输出以下两个标签：
   - <overall_goal>从用户需求中提炼的总任务目标（简明扼要，一句话概括）</overall_goal>
   - <current_task>当前应该执行的第一步具体操作</current_task>

【重要】当用户提到"搜索"、"查找最新"、"联网"等关键词时，必须路由到 [agent]，让工具 web_search 执行实际搜索，严禁自行编造搜索结果。

注意：路由标签必须放在回答的最后一行，格式为 [end] 或 [agent]。"""
        
        try:
            messages = [HumanMessage(content=classify_prompt)]
            
            # 流式调用模型
            has_stream = hasattr(self.model, 'stream') and callable(self.model.stream)
            full_content = ""
            full_reasoning = ""
            
            if has_stream:
                for chunk in self.model.stream(messages):
                    content_delta = chunk.get("content", "")
                    reasoning_delta = chunk.get("reasoning_content", "")
                    if reasoning_delta:
                        full_reasoning += reasoning_delta
                        self._emit("reasoning_token", reasoning_delta)
                    if content_delta:
                        full_content += content_delta
                        self._emit("token", content_delta)
            else:
                response = self.model.invoke(messages)
                full_content = response.content.strip()
                full_reasoning = self._extract_reasoning_content(response) or ""
                if full_reasoning:
                    self._emit("reasoning", full_reasoning[:500] + "..." if len(full_reasoning) > 500 else full_reasoning)
                self._emit("answer", full_content)
            
            content = full_content.strip()
            
            # 解析路由决策
            route_match = re.search(r'\[(end|agent)\]', content, re.IGNORECASE)
            
            if route_match:
                task_type = route_match.group(1).lower()
                self._emit("routing", f"[路由决策] {task_type} (由LLM判断)")
            else:
                # 降级策略：如果没有显式标签，检查是否有工具调用意图
                if self.xml_parser.parse(content):
                    task_type = "agent"
                    self._emit("routing", "[路由决策] agent (检测到工具调用)")
                else:
                    task_type = "end"
                    self._emit("routing", "[路由决策] end (无工具调用，默认结束)")
            
            # 将回答转换为 AIMessage
            kwargs = {}
            if full_reasoning:
                kwargs["reasoning_content"] = full_reasoning
            ai_message = AIMessage(content=content, additional_kwargs=kwargs)
            
            # 提取 overall_goal 和 current_task（仅 agent 路由时）
            result = {
                "messages": [ai_message],
                "task_type": task_type,
                "iteration_count": 1
            }
            
            if task_type == "agent":
                goal_match = re.search(r'<overall_goal>(.*?)</overall_goal>', content, re.DOTALL)
                task_match = re.search(r'<current_task>(.*?)</current_task>', content, re.DOTALL)
                
                if goal_match:
                    extracted_goal = goal_match.group(1).strip()
                    self._emit("info", f"[分类] 总目标: {extracted_goal}")
                    result["overall_goal"] = extracted_goal
                else:
                    stripped_input = user_input.strip()
                    # 降级：如果用户明确输入了新任务且非无意义的短语，才作为新的总目标
                    if stripped_input and stripped_input.lower() not in ('', '继续', '好', '好的', 'ok', 'yes', '易', '嗯', '确认', 'continue', 'go', 'y'):
                        result["overall_goal"] = stripped_input
                
                if task_match:
                    extracted_task = task_match.group(1).strip()
                    self._emit("info", f"[分类] 当前任务: {extracted_task}")
                    result["current_task"] = extracted_task
            
            return result
                    
        except Exception as e:
            self._emit("error", f"[分类与回答] LLM调用失败: {e}")
            self._emit("info", "[分类与回答] 默认进入Agent流程")
            return {"task_type": "agent"}
    
    def _route_after_classify(self, state: AgentState) -> str:
        """
        分类后的路由决策
        - end: 已直接回答，结束
        - load_context: 需要工具，先加载项目上下文再进入Agent
        """
        task_type = state.get("task_type", "agent")
        
        if task_type == "end":
            self._emit("routing", "[路由] 简单问题已回答，结束执行")
            return "end"
        else:
            self._emit("routing", "[路由] 需要工具执行，加载项目上下文后进入Agent")
            return "load_context"
    
    def _direct_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        直接回答节点 - 简单问题不需要工具
        """
        if self.model is None:
            raise ValueError("未设置模型，无法执行推理")
        
        task = state.get("current_task", "")
        self._emit("info", "[直接回答] 处理简单问题...")
        
        # 构建简单的提示词
        messages = [
            HumanMessage(content=f"""你是 Klynx Agent，一个智能编程助手。

用户说: {task}

请直接回复用户，不需要使用任何工具。""")
        ]
        
        try:
            # 流式调用模型
            has_stream = hasattr(self.model, 'stream') and callable(self.model.stream)
            full_content = ""
            full_reasoning = ""
            
            if has_stream:
                for chunk in self.model.stream(messages):
                    content_delta = chunk.get("content", "")
                    reasoning_delta = chunk.get("reasoning_content", "")
                    if reasoning_delta:
                        full_reasoning += reasoning_delta
                        self._emit("reasoning_token", reasoning_delta)
                    if content_delta:
                        full_content += content_delta
                        self._emit("token", content_delta)
            else:
                response = self.model.invoke(messages)
                full_content = response.content or ""
                full_reasoning = self._extract_reasoning_content(response) or ""
                if full_reasoning:
                    self._emit("reasoning", full_reasoning[:1000] + "..." if len(full_reasoning) > 1000 else full_reasoning)
                self._emit("answer", full_content)
            
            content = full_content
            kwargs = {}
            if full_reasoning:
                kwargs["reasoning_content"] = full_reasoning
            ai_message = AIMessage(content=content, additional_kwargs=kwargs)
            
            return {
                "messages": [ai_message],
                "task_completed": True,
                "iteration_count": 1
            }
        except Exception as e:
            self._emit("error", f"[错误] {e}")
            return {
                "task_completed": True,
                "iteration_count": 1
            }
    
    def _load_context_node(self, state: AgentState) -> Dict[str, Any]:
        """
        加载项目上下文节点 - 在 classify 后、agent 前执行
        
        递归搜索工作目录下所有 KLYNX.md 文件，将路径和内容加载到 state 中。
        可通过 load_project_docs=False 跳过。
        """
        if not self.load_project_docs:
            self._emit("info", "[加载上下文] 已禁用项目文档加载，跳过 KLYNX.md 搜索")
            return {"klynx_docs": ""}
        
        self._emit("info", "[加载上下文] 搜索 KLYNX.md 文件...")
        
        klynx_docs = self._find_klynx_docs()
        
        if klynx_docs:
            doc_count = klynx_docs.count('<doc ')
            self._emit("info", f"[加载上下文] 找到 {doc_count} 个 KLYNX.md 文件")
        else:
            self._emit("info", "[加载上下文] 未找到 KLYNX.md 文件")
        
        return {
            "klynx_docs": klynx_docs
        }
    
    def _summary_node(self, state: AgentState) -> Dict[str, Any]:
        """
        总结节点 - 在任务结束前对所有已完成的工作进行总结
        
        职责：
        1. 总结修改了哪些内容
        2. 完成了哪些工作
        3. 还有哪些地方存在问题
        """
        self._emit("info", "[总结] 正在生成工作总结...")
        
        progress = state.get("progress_summary", "")
        user_input = state.get("user_input", "")
        current_task = state.get("current_task", "")
        project_rules = state.get("project_rules", "")
        klynx_docs = state.get("klynx_docs", "")
        
        # 获取对话历史摘要
        messages = state.get("messages", [])
        history_text = self._format_conversation_history(messages[-20:])  # 最近20条
        
        summary_prompt = f"""你是 Klynx Agent。任务已经执行完毕，请对本次工作做一个完整的总结。

用户原始请求: {user_input}

任务目标: {current_task}

执行进度记录:
{progress}

最近的对话历史:
{history_text}

请用以下格式输出总结：

## 工作总结

### 完成的工作
- 列出所有已完成的工作项

### 修改的文件
- 列出所有被修改/创建/删除的文件及修改内容

### 存在的问题
- 如果有未解决的问题或需要注意的事项，在此列出
- 如果没有问题，写"无"
"""
        
        summary_content = ""
        if self.model:
            try:
                has_stream = hasattr(self.model, 'stream') and callable(self.model.stream)
                if has_stream:
                    for chunk in self.model.stream([HumanMessage(content=summary_prompt)]):
                        reasoning_part = chunk.get("reasoning_content", "")
                        content_part = chunk.get("content", "")
                        
                        if reasoning_part:
                            self._emit("reasoning_token", reasoning_part)
                        
                        if content_part:
                            summary_content += content_part
                            self._emit("token", content_part)
                            
                    summary_content = summary_content.strip()
                else:
                    response = self.model.invoke([HumanMessage(content=summary_prompt)])
                    summary_content = response.content.strip()
                    
                    # 提取并显示思考内容
                    reasoning = self._extract_reasoning_content(response)
                    if reasoning:
                        self._emit("reasoning", f"{reasoning[:300]}..." if len(reasoning) > 300 else reasoning)
                
            except Exception as e:
                self._emit("error", f"[总结] LLM调用失败: {e}")
                # 降级：使用进度记录生成简单总结
                summary_content = f"## 工作总结\n\n### 执行记录\n{progress}\n\n### 存在的问题\n总结生成失败: {e}"
        else:
            summary_content = f"## 工作总结\n\n### 执行记录\n{progress}"
        
        self._emit("summary", summary_content)
        
        return {
            "messages": [AIMessage(content=summary_content)],
            "task_completed": True,
            "summary_content": summary_content
        }
    
    def _observe_env_node(self, state: AgentState) -> Dict[str, Any]:
        """
        观察环境节点 - 获取环境信息后再进入思考
        """
        self._emit("info", "[观察环境] 获取工作目录信息...")
        
        # 获取环境快照
        env_snapshot = self._get_env_snapshot()
        
        # 打印简要信息
        tree_match = re.search(r'<file_tree>(.*?)</file_tree>', env_snapshot, re.DOTALL)
        if tree_match:
            tree_content = tree_match.group(1).strip()
            lines = tree_content.split('\n')[:]
            self._emit("info", f"[目录结构] {len(lines)} 项目录/文件")
            for line in lines[:5]:
                self._emit("info", f"  {line}")
            if len(lines) > 5:
                self._emit("info", f"  ... (还有 {len(lines)-5} 项)")
        
        return {
            "env_snapshot": env_snapshot,
            "last_action": "observe"
        }
    
    def _model_inference_node(self, state: AgentState) -> Dict[str, Any]:
        """
        模型推理节点 - Agent思维环路中的思考
        
        职责：
        1. 根据上一步工具执行的反馈结果，做出新的思考和判断
        2. 判断任务目标是否需要更新
        3. 使用 context 变量构建提示词
        """
    def _model_inference_node(self, state: AgentState) -> Dict[str, Any]:
        """
        模型推理节点 - Agent思维环路中的思考
        
        职责：
        1. 根据上一步工具执行的反馈结果，做出新的思考和判断
        2. 判断任务目标是否需要更新
        3. 使用 context 变量构建提示词
        """
        if self.model is None:
            raise ValueError("未设置模型，无法执行推理")
        
        iteration = state.get("iteration_count", 0) + 1
        self._emit("iteration", f"[迭代 {iteration}]")
        
        # 检查上下文是否需要总结
        if self._check_context_overflow(state):
            summarize_result = self._summarize_context(state)
            if summarize_result:
                # 更新 state 中的 messages
                state = {**state, **summarize_result}
        
        # 获取结构化目标
        overall_goal = state.get("overall_goal", "") or state.get("user_input", "")
        current_task = state.get("current_task", "")
        
        # 构建上下文（包含历史对话）
        context = self._build_context(state, include_history=True)
        
        # 构建 .rules 规则部分（从 state 中读取，由 init 节点加载，始终保留，不压缩）
        project_rules = state.get("project_rules", "")
        rules_xml = ""
        if project_rules:
            rules_xml = f"""\n<project_rules>
{self._escape_xml(project_rules)}
</project_rules>\n"""
        
        # 构建 KLYNX.md 文档部分（从 state 中读取，由 load_context 节点填充）
        klynx_docs = state.get("klynx_docs", "")
        docs_xml = ""
        if klynx_docs:
            docs_xml = f"\n{klynx_docs}\n"
        
        # 构建当前任务描述
        current_task_desc = f"\n  <current_task>{self._escape_xml(current_task)}</current_task>" if current_task else ""
        
        prompt = f"""{self._get_system_prompt()}
{rules_xml}{docs_xml}
{context}

<iteration_status>
  <current_iteration>{iteration}</current_iteration>
  <system_note>
    你在第 {iteration} 次迭代。
    请根据上方对话历史中的工具执行结果判断当前进度，继续执行当前任务。
    用户没有新的输入，不要将历史记录或工具输出解读为新的用户请求。
    如果 current_task 已完成，使用 &lt;task_goal&gt; 标签更新下一步任务，或使用 attempt_completion 完成。
  </system_note>
</iteration_status>

<task_context>
  <overall_goal>{self._escape_xml(overall_goal)}</overall_goal>{current_task_desc}
  <available_tools>
{self._get_tools_prompt()}
  </available_tools>
</task_context>"""
        
        messages = [HumanMessage(content=prompt)]
        
        # 调用模型
        # 根据 tool_call_mode 决定是否传递 tools 参数
        stream_tools = self._json_schemas if (self.tool_call_mode == "native" and self._json_schemas) else None
        
        try:
            # 使用流式调用
            stream = self.model.stream(messages, tools=stream_tools)
            
            full_content = ""
            full_reasoning = ""
            usage_info = {}
            native_tool_calls = []  # 原生 FC 返回的工具调用
            
            for chunk in stream:
                content_delta = chunk.get("content", "")
                reasoning_delta = chunk.get("reasoning_content", "")
                usage_delta = chunk.get("usage", {})
                
                # 原生 Function Calling: 流结束时的 tool_calls 组装结果
                if "tool_calls" in chunk:
                    native_tool_calls = chunk["tool_calls"]
                
                if usage_delta:
                    usage_info = usage_delta
                
                if content_delta:
                    full_content += content_delta
                    self._emit("token", content_delta)
                    if self.streaming_callback:
                        self.streaming_callback({"type": "token", "content": content_delta})
                    
                if reasoning_delta:
                    full_reasoning += reasoning_delta
                    self._emit("reasoning_token", reasoning_delta)
                    if self.streaming_callback:
                        self.streaming_callback({"type": "reasoning_token", "content": reasoning_delta})
            
            # 构造完整响应对象，保持原有逻辑兼容性
            response = LiteLLMResponse(content=full_content, reasoning_content=full_reasoning)
            if usage_info:
                response.usage = usage_info
            if native_tool_calls:
                response.tool_calls = native_tool_calls
            
        except Exception as e:
            self._emit("error", f"[Error] 模型调用失败: {e}")
            raise
        
        content = response.content
        reasoning_content = response.reasoning_content
        
        if reasoning_content:
            pass 
        
        # 修复: DeepSeek 有时将工具调用 XML 放在 reasoning_content 而非 content 中
        # 当 content 为空但 reasoning_content 包含工具调用XML时，使用 reasoning_content 作为 content
        if (not content or len(content.strip()) == 0) and reasoning_content:
            # 检查 reasoning_content 中是否有工具调用或 attempt_completion
            if self.xml_parser.parse(reasoning_content, extra_tool_names=list(self.tools.keys())) or self.xml_parser.has_attempt_completion(reasoning_content):
                self._emit("info", "[修复] 从 reasoning_content 中提取工具调用")
                content = reasoning_content
        
        # 如果开启了 thinking_context，将思考内容添加到 content 中
        if state.get("thinking_context", False) and reasoning_content:
            content = f"<thinking>\n{reasoning_content}\n</thinking>\n\n{content}"

        # 打印回答内容
        if content:
            self._emit("answer", content)
        
        # 提取任务目标更新（如果有）
        task_goal_match = re.search(r'<task_goal>(.*?)</task_goal>', content, re.DOTALL)
        if task_goal_match:
            current_task = task_goal_match.group(1).strip()
            self._emit("info", f"[任务目标更新] {current_task[:100]}..." if len(current_task) > 100 else f"[任务目标更新] {current_task}")
        
        # 提取并打印思考内容
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
        if thinking_match:
            thinking_text = thinking_match.group(1).strip()
            self._emit("info", f"[Thinking] {thinking_text[:200]}..." if len(thinking_text) > 200 else f"[Thinking] {thinking_text}")
        
        # ============ 双模式提取工具调用 ============
        tool_calls = []
        if self.tool_call_mode == "native" and response.tool_calls:
            # Native Function Calling 模式：直接使用 API 返回的 tool_calls
            tool_calls = response.tool_calls
            tools_str = ", ".join([tc.get('tool', 'unknown') for tc in tool_calls])
            self._emit("tool_calls", f"[Native Function Calling] 触发了系统工具: {tools_str}")
        else:
            # XML 回退模式（或 native 模式下模型选择了纯文本回复）：使用 XML 解析
            tool_calls = self.xml_parser.parse(content, extra_tool_names=list(self.tools.keys()))
            if tool_calls:
                tools_str = ", ".join([tc.get('tool', 'unknown') for tc in tool_calls])
                self._emit("tool_calls", f"[XML 格式] 触发了系统工具: {tools_str}")
        
        # 检查任务是否完成
        task_completed = self.xml_parser.has_attempt_completion(content)
        if task_completed:
            self._emit("complete", "[任务完成]")
        
        # 处理空回复（Native FC 模式下，content 为空但有 tool_calls 是正常行为，不算空回复）
        empty_count = state.get("empty_response_count", 0)
        has_native_tool_calls = bool(tool_calls and self.tool_call_mode == "native")
        if (not content or len(content.strip()) == 0) and not has_native_tool_calls:
            empty_count += 1
            self._emit("warning", f"[Warning] 空回复 ({empty_count}/3)")
            if empty_count >= 3:
                kwargs = {}
                if reasoning_content:
                    kwargs["reasoning_content"] = reasoning_content
                ai_message = AIMessage(content=content, additional_kwargs=kwargs) if not isinstance(response, AIMessage) else response
                return {
                    "messages": [ai_message],
                    "iteration_count": iteration,
                    "task_completed": True,
                    "empty_response_count": empty_count,
                    "current_task": current_task
                }
        else:
            empty_count = 0
        
        # 将响应转换为 AIMessage（使用可能已修正的 content）
        kwargs = {}
        if reasoning_content:
            kwargs["reasoning_content"] = reasoning_content
        # Native FC 模式：将 tool_calls 存入 additional_kwargs，供 _tool_parser_node 读取
        if tool_calls and self.tool_call_mode == "native":
            kwargs["tool_calls"] = tool_calls
        ai_message = AIMessage(content=content, additional_kwargs=kwargs)
        
        # 更新 Token 用量
        prev_total = state.get("total_tokens", 0)
        prev_completion = state.get("completion_tokens", 0)
        
        curr_prompt = 0
        curr_completion = 0
        curr_total = 0
        
        if hasattr(response, 'usage') and response.usage:
            curr_prompt = response.usage.get('prompt_tokens', 0)
            curr_completion = response.usage.get('completion_tokens', 0)
            curr_total = response.usage.get('total_tokens', 0)

        # If usage is missing or total_tokens is 0, use fallback
        if curr_total == 0:
            curr_completion = TokenCounter.estimate_tokens(content)
            curr_prompt = TokenCounter.estimate_tokens(prompt)
            curr_total = curr_prompt + curr_completion

        return {
            "messages": [ai_message],
            "iteration_count": iteration,
            "task_completed": task_completed,
            "empty_response_count": empty_count,
            "current_task": current_task,
            "total_tokens": prev_total + curr_total,
            "prompt_tokens": curr_prompt,
            "completion_tokens": prev_completion + curr_completion
        }
    
    def _feedback_node(self, state: AgentState) -> Dict[str, Any]:
        """
        反馈节点 - 评估工具执行结果并判断任务是否完成
        """
        self._emit("info", "[反馈节点] 评估工具执行结果...")
        
        # 获取最新的消息（包含工具执行结果）
        messages = state.get("messages", [])
        if not messages:
            self._emit("info", "[反馈节点] 无消息可评估")
            return {"task_completed": False, "last_action": "think_more"}
        
        # 获取最新的工具执行结果
        last_message = messages[-1]
        content = getattr(last_message, 'content', '') if hasattr(last_message, 'content') else str(last_message)
        
        # 简化输出
        self._emit("info", f"[反馈节点] 工具执行结果: {content[:100]}..." if len(content) > 100 else f"[反馈节点] 工具执行结果: {content}")
        
        # 任务完成只能由 LLM 明确标记（使用 <attempt_completion> 标签）
        # 不使用粗糙的关键词匹配，避免误判
        task_completed = state.get("task_completed", False)
        
        if task_completed:
            self._emit("info", "[反馈节点] 任务已完成，结束执行")
            return {"task_completed": True, "last_action": "complete"}
        else:
            self._emit("info", "[反馈节点] 任务未完成，回到思考节点")
            return {"task_completed": False, "last_action": "think_more"}
    
    def _tool_parser_node(self, state: AgentState) -> Dict[str, Any]:
        """
        工具解析节点 - 从 LLM 回复中提取工具调用
        支持双模式：Native Function Calling 和 XML 解析
        """
        messages = state.get("messages", [])
        if not messages:
            self._emit("info", "[工具解析] 无消息可解析")
            return {"pending_tool_calls": []}
        
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            self._emit("info", "[工具解析] 最后一条消息不是 AI 消息")
            return {"pending_tool_calls": []}
        
        tool_calls = []
        
        # 双模式解析
        if self.tool_call_mode == "native":
            # Native 模式：优先从 AIMessage 的 additional_kwargs 中读取原生 tool_calls
            native_tcs = last_message.additional_kwargs.get("tool_calls", [])
            if native_tcs:
                tool_calls = native_tcs
            else:
                # Native 模式回退：模型可能以纯文本方式返回了 XML 工具调用
                tool_calls = self.xml_parser.parse(last_message.content, extra_tool_names=list(self.tools.keys()))
        else:
            # XML 模式：使用传统 XML 解析
            tool_calls = self.xml_parser.parse(last_message.content, extra_tool_names=list(self.tools.keys()))
        
        if tool_calls:
            pass
        else:
            self._emit("info", "[无工具调用]")
        
        return {"pending_tool_calls": tool_calls}
    
    def _extract_paper_summary(self, content: str, file_path: str) -> str:
        """从论文内容中提取摘要信息"""
        lines = content.split('\n')
        title = ""
        abstract = ""
        
        # 提取标题（通常在前几行的##开头）
        for line in lines[:30]:
            if line.strip().startswith('##') and 'abstract' not in line.lower():
                title = line.replace('#', '').strip()
                if len(title) > 10:  # 确保是有效标题
                    break
        
        # 提取摘要（寻找Abstract后的内容或DOI前的长段落）
        in_abstract = False
        for i, line in enumerate(lines):
            if 'abstract' in line.lower() or i < 20:  # 摘要通常在开头
                if len(line.strip()) > 200:  # 找到长段落作为摘要
                    abstract = line.strip()[:500]
                    break
            if 'DOI:' in line and abstract:
                break
        
        if not abstract:
            # 如果没找到，取前20行中最长的段落
            for line in lines[:20]:
                if len(line.strip()) > len(abstract):
                    abstract = line.strip()[:500]
        
        return f"标题: {title[:100]}\n摘要: {abstract[:300]}..."
    
    def _tool_executor_node(self, state: AgentState) -> Dict[str, Any]:
        """
        工具执行节点 - 执行工具调用
        
        该节点负责执行 LLM 决定的工具调用，并返回执行结果。
        所有工具的选择和使用决策都由 LLM 做出，本节点只负责执行。
        """
        tool_calls = state.get("pending_tool_calls", [])
        
        # 如果没有待执行的工具，直接返回
        if not tool_calls:
            return {
                "messages": [],
                "pending_tool_calls": [],
                "last_action": "act"
            }
        
        results = []
        executed_tools = []
        
        # 本次节点执行中可能更新的状态
        new_overall_goal = None
        new_current_task = None
        
        # 依次执行每个工具调用
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.get('tool', 'unknown')
            params = tool_call.get('params', {})
            
            # 打印工具执行信息
            self._emit("tool_exec", f"[工具 {i}/{len(tool_calls)}] {tool_name}")
            self._print_tool_params(tool_name, params)
            
            try:
                # ====== 任务完成检测（处理 Native 模式下的 attempt_completion 工具） ======
                if tool_name == "attempt_completion":
                    output = params.get("result", "操作完成")
                    self._emit("complete", f"[任务完成] {output}")
                    return {
                        "messages": [],
                        "pending_tool_calls": [],
                        "last_action": "complete",
                        "task_completed": True
                    }
                
                # ====== 任务状态更新（处理 Native 模式下的 update_task_state 工具） ======
                elif tool_name == "update_task_state":
                    new_task_val = params.get("current_task")
                    new_goal_val = params.get("overall_goal")
                    outputs = []
                    if new_goal_val:
                        new_overall_goal = new_goal_val
                        outputs.append(f"总目标已更新为: {new_goal_val}")
                        self._emit("info", f"[任务目标更新/总目标] {new_goal_val}")
                    if new_task_val:
                        new_current_task = new_task_val
                        outputs.append(f"当前动作已更新为: {new_task_val}")
                        self._emit("info", f"[任务目标更新/下一步] {new_task_val}")
                    
                    output = "任务状态已成功同步。 " + " ; ".join(outputs)
                    # 这个工具只是一个状态同步助手，它不会完成整个系统流，继续留在工具循环里
                    
                # 执行工具：优先检查外部工具，然后检查新基础工具，最后使用 ToolRegistry
                elif tool_name in self.external_tool_funcs:
                    output = str(self.external_tool_funcs[tool_name](**params))
                # 终端工具
                elif tool_name == "create_terminal":
                    output = self.terminal_manager.create_terminal(params.get("name"), params.get("cwd"))
                elif tool_name == "run_in_terminal":
                    output = self.terminal_manager.run_command(params.get("name"), params.get("command"))
                elif tool_name == "read_terminal":
                    lines = params.get("lines", 50)
                    output = self.terminal_manager.read_terminal(params.get("name"), int(lines) if lines else 50)
                # 语法检查工具
                elif tool_name == "check_syntax":
                    output = SyntaxChecker.check_file(params.get("path"))
                # TUI 工具
                elif tool_name == "open_tui":
                    rows = int(params.get("rows", 24))
                    cols = int(params.get("cols", 80))
                    output = self.tui_manager.open_tui(params.get("name"), params.get("command"), rows, cols)
                    self.tui_manager.render_to_console(params.get("name"))
                elif tool_name == "read_tui":
                    output = self.tui_manager.read_tui(params.get("name"))
                    self.tui_manager.render_to_console(params.get("name"))
                elif tool_name == "send_keys":
                    output = self.tui_manager.send_keys(params.get("name"), params.get("keys"))
                    self.tui_manager.render_to_console(params.get("name"))
                elif tool_name == "close_tui":
                    output = self.tui_manager.close_tui(params.get("name"))
                # TUI 模式激活工具
                elif tool_name == "activate_tui_mode":
                    self._tui_guide_loaded = True
                    output = "<success>TUI 交互模式已激活。TUI 操作指南已加载到系统提示中。\n你现在可以使用 open_tui / read_tui / send_keys / close_tui 工具进行 TUI 交互。</success>"
                # TUI 一键启动工具
                elif tool_name == "launch_interactive_session":
                    self._tui_guide_loaded = True
                    # 默认使用 command 名称作为会话名（去除参数）
                    command = params.get("command", "cmd")
                    name = command.split()[0].replace(".exe", "")
                    # 避免名称重复
                    if name in self.tui_manager.sessions:
                        import uuid
                        name = f"{name}_{str(uuid.uuid4())[:4]}"
                    
                    output = self.tui_manager.open_tui(name, command, 24, 80)
                    self.tui_manager.render_to_console(name)
                    
                    # 追加指南提示
                    output += "\n<system>TUI 模式已自动激活。请遵循 TUI 操作指南（read_tui -> send_keys -> read_tui）。</system>"
                # 联网搜索工具
                elif tool_name == "web_search":
                    output = self.web_search_tool.search(
                        query=params.get("query", ""),
                        max_results=int(params.get("max_results", 5)),
                        search_depth=params.get("search_depth", "basic")
                    )
                # 知识库查询工具
                elif tool_name == "query_knowledge":
                    output = self.kb_manager.query(
                        query=params.get("query", ""),
                        top_k=int(params.get("top_k", 5)),
                        kb_name=params.get("kb_name", None)
                    )
                elif tool_name == "list_knowledge_collections":
                    output = self.kb_manager.list_collections(
                        kb_name=params.get("kb_name", None)
                    )
                # 浏览器工具
                elif tool_name == "browser_open":
                    output = self.browser_manager.goto(params.get("url"))
                elif tool_name == "browser_view":
                    output = self.browser_manager.get_content(params.get("selector"))
                elif tool_name == "browser_act":
                    output = self.browser_manager.act(
                        params.get("action"), 
                        params.get("selector"), 
                        params.get("value")
                    )
                elif tool_name == "browser_screenshot":
                    output = self.browser_manager.screenshot()
                elif tool_name == "browser_console_logs":
                    output = self.browser_manager.get_console_logs()
                # 原有基础工具
                else:
                    output = ToolRegistry.execute(tool_call)
                
                # 为特定工具失败注入 Actionable Hint（建设性错误反馈）
                if "<error>" in output:
                    if tool_name == "replace_in_file":
                        output += "\n<hint>1. 请先用 read_file 读取目标文件的精确行内容。2. 确保 SEARCH 块的缩进与原文件完全一致（包括空格/制表符）。3. 若多次失败，考虑用 write_to_file 重写整个文件。</hint>"
                    elif tool_name == "write_to_file":
                        output += "\n<hint>检查文件路径是否正确、目标目录是否存在。</hint>"
                
                # 打印执行结果（限制输出长度）
                self._print_tool_output(output)
                
                # 构建结果消息
                result_content = f"<tool_result tool=\"{tool_name}\">\n{output}\n</tool_result>"
                results.append(HumanMessage(content=result_content))
                
                # 记录已执行的工具
                executed_tools.append({
                    "tool": tool_name,
                    "params": params,
                    "success": True
                })
                
            except Exception as e:
                # 工具执行失败，记录错误信息
                error_msg = f"工具执行失败: {str(e)}"
                self._emit("error", f"  错误: {error_msg}")
                
                result_content = f"<tool_result tool=\"{tool_name}\" status=\"error\">\n{error_msg}\n</tool_result>"
                results.append(HumanMessage(content=result_content))
                
                executed_tools.append({
                    "tool": tool_name,
                    "params": params,
                    "success": False,
                    "error": str(e)
                })
        
        self._emit("info", f"[工具执行完成] 共 {len(tool_calls)} 个工具")
        
        # 更新进度摘要
        progress = state.get("progress_summary", "")
        for tool_info in executed_tools:
            tool_name = tool_info["tool"]
            params = tool_info["params"]
            status = "成功" if tool_info["success"] else "失败"
            
            entry = self._format_progress_entry(tool_name, params, status)
            if entry:
                progress += entry + "\n"
        
        result = {
            "messages": results,
            "pending_tool_calls": [],
            "last_action": "act",
            "empty_response_count": 0,
            "progress_summary": progress
        }
        
        # 将被 update_task_state 更新的临时变量存入最终长短期图状态
        if new_overall_goal is not None:
            result["overall_goal"] = new_overall_goal
        if new_current_task is not None:
            result["current_task"] = new_current_task
        
        # 如果 TUI 模式被激活，更新状态
        if getattr(self, '_tui_guide_loaded', False):
            result["tui_guide_loaded"] = True
        
        return result
    
    def _print_tool_params(self, tool_name: str, params: Dict[str, Any]) -> None:
        """打印工具参数信息"""
        if tool_name == 'execute_command':
            self._emit("tool_exec", f"  命令: {params.get('command', '')}")
        elif tool_name == 'read_file':
            self._emit("tool_exec", f"  文件: {params.get('path', '')}")
            if params.get('start_line'):
                self._emit("tool_exec", f"  行范围: {params.get('start_line')}-{params.get('end_line', 'end')}")
        elif tool_name == 'write_to_file':
            self._emit("tool_exec", f"  文件: {params.get('path', '')}")
            content = params.get('content', '')
            self._emit("tool_exec", f"  内容长度: {len(content)} 字符")
        elif tool_name == 'replace_in_file':
            self._emit("tool_exec", f"  文件: {params.get('path', '')}")
        elif tool_name == 'list_directory':
            self._emit("tool_exec", f"  目录: {params.get('path', '.')}")
            self._emit("tool_exec", f"  深度: {params.get('depth', 2)}")
        elif tool_name == 'create_directory':
            self._emit("tool_exec", f"  目录: {params.get('path', '')}")
        elif tool_name == 'preview_file':
            self._emit("tool_exec", f"  文件: {params.get('path', '')}")
            self._emit("tool_exec", f"  预览行数: {params.get('num_lines', 50)}")
        elif tool_name == 'search_in_files':
            self._emit("tool_exec", f"  搜索: {params.get('pattern', '')}")
            self._emit("tool_exec", f"  路径: {params.get('path', '.')}")
            if params.get('file_pattern', '*') != '*':
                self._emit("tool_exec", f"  文件类型: {params.get('file_pattern')}")
        else:
            # 其他工具，打印所有参数
            for key, value in params.items():
                self._emit("tool_exec", f"  {key}: {value}")
    
    def _print_tool_output(self, output: str, max_lines: int = 20) -> None:
        """打印工具输出（限制行数）"""
        lines = output.split('\n')
        result_text = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            result_text += f"\n    ... (还有 {len(lines) - max_lines} 行)"
        self._emit("tool_result", result_text)
    
    def _format_progress_entry(self, tool_name: str, params: Dict[str, Any], status: str) -> str:
        """格式化进度记录条目"""
        if tool_name == 'list_directory':
            path = params.get('path', '.')
            return f"- 已列出目录: {path} [{status}]"
        elif tool_name == 'read_file':
            path = params.get('path', '')
            return f"- 已读取文件: {path} [{status}]"
        elif tool_name == 'write_to_file':
            path = params.get('path', '')
            return f"- 已写入文件: {path} [{status}]"
        elif tool_name == 'preview_file':
            path = params.get('path', '')
            return f"- 已预览文件: {path} [{status}]"
        elif tool_name == 'execute_command':
            cmd = params.get('command', '')[:50]
            return f"- 已执行命令: {cmd} [{status}]"
        elif tool_name == 'replace_in_file':
            path = params.get('path', '')
            return f"- 已替换文件内容: {path} [{status}]"
        elif tool_name == 'create_directory':
            path = params.get('path', '')
            return f"- 已创建目录: {path} [{status}]"
        elif tool_name == 'search_in_files':
            pattern = params.get('pattern', '')[:50]
            path = params.get('path', '.')
            return f"- 已搜索: '{pattern}' in {path} [{status}]"
        else:
            return f"- 已执行 {tool_name} [{status}]"
    
    def _should_continue(self, state: AgentState) -> str:
        """
        路由决策 - 让LLM决定下一步流向
        
        Returns:
            "tools" - 有工具调用需要执行
            "agent" - 回到思考
            "ask" - 直接回答并结束
            "end" - 任务完成或达到最大迭代次数
        """
        iteration = state.get("iteration_count", 0)
        
        # 检查是否达到最大迭代次数
        if iteration >= self.max_iterations:
            self._emit("routing", f"[路由] 达到最大迭代次数 ({self.max_iterations})，强制结束执行")
            self._emit("warning", "[提示] 任务可能未完成，请检查任务复杂度或增加 max_iterations")
            return "end"
        
        # 检查任务是否已完成（由LLM明确标记）
        if state.get("task_completed", False):
            self._emit("routing", "[路由] 任务已完成，结束执行")
            return "end"
        
        # 让LLM决定下一步行动（通过last_action状态）
        last_action = state.get("last_action", "")
        pending_tools = state.get("pending_tool_calls", [])
        
        # 优先检查是否有工具待执行（工具执行优先级最高）
        if pending_tools:
            self._emit("routing", f"[路由] 有 {len(pending_tools)} 个工具待执行")
            return "tools"
        
        if last_action == "use_tool":
            # LLM决定使用工具但没有解析到工具
            self._emit("routing", "[路由] LLM决定使用工具但没有解析到工具，回到思考")
            return "agent"
        elif last_action == "think_more":
            # LLM决定继续思考
            self._emit("routing", "[路由] LLM决定继续思考")
            return "agent"
        elif last_action == "ask":
            # LLM决定直接回答
            self._emit("routing", "[路由] LLM决定直接回答并结束")
            return "ask"
        elif last_action == "complete":
            # LLM决定任务完成
            self._emit("routing", "[路由] LLM决定任务完成")
            return "end"
        else:
            # 默认行为：继续思考
            self._emit("routing", "[路由] 默认路由：无工具调用，继续思考")
            return "agent"

    def _route_after_agent(self, state: AgentState) -> str:
        """
        Agent 思考后的路由决策
        如果检测到工具调用 -> parser
        否则（直接回答） -> end
        
        支持双模式：Native FC 检查 additional_kwargs["tool_calls"]，XML 检查文本内容
        """
        messages = state.get("messages", [])
        if not messages:
            return "end"
        
        last_message = messages[-1]
        content = last_message.content
        
        # Native FC 模式：先检查 additional_kwargs 中的原生 tool_calls
        if self.tool_call_mode == "native":
            native_tcs = last_message.additional_kwargs.get("tool_calls", [])
            if native_tcs:
                self._emit("routing", "[路由] 检测到原生工具调用，进入解析")
                return "parser"
        
        # XML 解析（同时作为 native 模式的回退）
        if self.xml_parser.parse(content, extra_tool_names=list(self.tools.keys())):
            self._emit("routing", "[路由] 检测到工具调用，进入解析")
            return "parser"
        
        # 检查是否有默认路由逻辑（无工具调用时）
        if state.get("task_completed", False):
            self._emit("routing", "[路由] 任务已完成，结束执行")
            return "end"
        
        self._emit("routing", "[路由] 默认路由：无工具调用，继续思考")
        return "end"
    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图
        新流程：分类 -> (直接回答 | 观察环境 | 思考) -> 解析 -> 执行 -> 回到思考 -> 完成
        """
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("init", self._init_node)                    # 初始化（加载 .klynx 配置）
        workflow.add_node("classify", self._classify_task_node)      # 任务分类
        workflow.add_node("load_context", self._load_context_node)   # 加载 KLYNX.md
        workflow.add_node("ask", self._direct_answer_node)           # 直接回答
        workflow.add_node("observe", self._observe_env_node)         # 观察环境
        workflow.add_node("agent", self._model_inference_node)       # 思考/推理
        workflow.add_node("parser", self._tool_parser_node)          # 工具解析
        workflow.add_node("tools", self._tool_executor_node)         # 工具执行
        workflow.add_node("feedback", self._feedback_node)           # 反馈评估
        workflow.add_node("summary", self._summary_node)             # 工作总结
        
        # 设置入口点：先初始化
        workflow.set_entry_point("init")
        
        # 初始化后进入分类
        workflow.add_edge("init", "classify")
        
        # 分类后的条件路由
        workflow.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "end": END,                     # 已直接回答，结束
                "load_context": "load_context"   # 需要工具，先加载上下文
            }
        )
        
        # 加载上下文后进入 Agent
        workflow.add_edge("load_context", "agent")

        # 思考后解析工具
        workflow.add_edge("agent", "parser")
        
        # 解析后的条件路由
        workflow.add_conditional_edges(
            "parser",
            self._should_continue,
            {
                "tools": "tools",         # 执行工具
                "agent": "agent",         # 回到思考
                "ask": "ask",             # 直接回答并结束
                "end": "summary"          # 结束前先总结
            }
        )
        
        # 工具执行后进入反馈节点
        workflow.add_edge("tools", "feedback")
        
        # 反馈节点的条件路由
        workflow.add_conditional_edges(
            "feedback",
            self._should_continue,
            {
                "agent": "agent",         # 任务未完成，回到思考
                "end": "summary"          # 任务完成，先总结再结束
            }
        )
        
        # 总结后结束
        workflow.add_edge("summary", END)
        
        return workflow
    
    def _emit(self, event_type: str, content: str, **kwargs):
        """将事件存入缓冲区（替代 print）"""
        self._event_buffer.append({"type": event_type, "content": content, **kwargs})

    def ask(self, message: str, system_prompt: str = None, thread_id: str = "default"):
        """
        简单问答接口 —— 直接向 LLM 发送消息并流式返回回答。
        
        与 invoke 解耦（不消耗迭代、不触发工具），但会从 MemorySaver
        中读取过去的对话历史，保持上下文连续性。回答将不会存入记忆，
        以免污染主执行流（除非是 invoke 内部调用的分类或总结）。
        
        适用场景：
        - 只需要 AI 回答问题，不需要执行任何工具
        - 作为子任务的轻量级 LLM 调用
        - 生成摘要、翻译、评审等纯文本任务
        
        Args:
            message:       用户消息（问题或指令）
            system_prompt: 可选的系统提示词，默认使用 "你是 Klynx Agent，一个智能助手。"
            thread_id:     会话标识，用于读取历史上下文
        
        Yields:
            事件字典，包含 type 和 content 字段：
            - type="reasoning_token": 思考过程的流式 token
            - type="token":           回答的流式 token
            - type="done":            完成事件，包含完整的 answer 和 reasoning
        """
        if self.model is None:
            yield {"type": "error", "content": "未设置模型，无法执行问答"}
            return
        
        default_system = "你是 Klynx Agent，一个智能编程助手。请直接回答用户问题。"
        sys_prompt = system_prompt or default_system
        
        # 尝试从 MemorySaver 获取对话历史
        history_msgs = []
        if thread_id:
            config = {"configurable": {"thread_id": thread_id}}
            current_state = self.app.get_state(config)
            if current_state and current_state.values:
                history_msgs = current_state.values.get("messages", [])
        
        # 组装消息列表
        messages = [SystemMessage(content=sys_prompt)]
        
        # 将历史记录添加进去（仅为 AI 提供上下文，不写回数据库）
        if history_msgs:
            # 取最后 20 条消息作为简要上下文以节省 token
            for msg in history_msgs[-20:]:
                # 屏蔽工具调用的消息以防基础 chat 产生幻觉工具
                if isinstance(msg, (HumanMessage, AIMessage)):
                    messages.append(msg)
                    
        messages.append(HumanMessage(content=message))
        
        # 尝试流式调用
        has_stream = hasattr(self.model, 'stream') and callable(self.model.stream)
        
        full_content = ""
        full_reasoning = ""
        total_tokens = 0
        
        try:
            if has_stream:
                # 流式输出
                for chunk in self.model.stream(messages):
                    reasoning_part = chunk.get("reasoning_content", "")
                    content_part = chunk.get("content", "")
                    
                    if reasoning_part:
                        full_reasoning += reasoning_part
                        yield {"type": "reasoning_token", "content": reasoning_part}
                    
                    if content_part:
                        full_content += content_part
                        yield {"type": "token", "content": content_part}
            else:
                # 非流式 fallback
                response = self.model.invoke(messages)
                full_content = response.content or ""
                full_reasoning = self._extract_reasoning_content(response) or ""
                
                # 提取 token 用量
                usage = getattr(response, 'usage', None)
                if usage and isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens", 0)
                
                if full_reasoning:
                    yield {"type": "reasoning", "content": full_reasoning}
                if full_content:
                    yield {"type": "answer", "content": full_content}
        
        except Exception as e:
            yield {"type": "error", "content": f"ask 调用失败: {str(e)}"}
            return
        
        # 发送完成事件
        yield {
            "type": "done",
            "content": "",
            "answer": full_content,
            "reasoning": full_reasoning,
            "total_tokens": total_tokens,
        }


    def invoke(self, task: str, thread_id: str = "default", thinking_context: bool = False):
        """
        执行用户任务，流式返回事件（生成器）
        
        Yields:
            事件字典，包含 type 和 content 字段：
            - type: 事件类型（iteration, reasoning, answer, tool_exec, tool_result, routing, info, warning, error, ...）
            - content: 事件内容
            
            最后一个事件 type="done"，包含最终结果字段。
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 2000
        }
        
        # 构建消息列表
        messages = [HumanMessage(content=task)]
        
        # initial_state 只包含需要在这轮覆盖或更新的值
        # overall_goal, current_task, context_summary, progress_summary, klynx_docs, project_rules 等
        # 需要保留上轮对话的状态，所以不能放在这里（否则会被空字符串覆盖）
        initial_state = {
            "messages": messages,
            "pending_tool_calls": [],
            "iteration_count": 0,
            "working_dir": self.working_dir,
            "task_completed": False,
            "last_action": "",
            "empty_response_count": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_context_tokens": self.max_context_tokens,
            "task_type": "",
            "user_input": task,
            "summary_content": "",
            "thinking_context": thinking_context
        }
        
        # 清空缓冲区
        self._event_buffer.clear()
        
        # 内部取消信号量
        if not hasattr(self, "_cancel_event"):
            import threading
            self._cancel_event = threading.Event()
        else:
            self._cancel_event.clear()
        
        # 使用后台线程执行图，主线程实时 drain 缓冲区实现逐 token 流式输出
        import threading
        import time
        
        graph_done = threading.Event()
        graph_error = [None]  # 用列表存以便线程内修改
        
        def _run_graph():
            try:
                for node_event in self.app.stream(initial_state, config=config):
                    if getattr(self, "_cancel_event", None) and self._cancel_event.is_set():
                        self._emit("warning", "[System] 任务已被用户中断(Ctrl+C)。")
                        break
            except Exception as e:
                import traceback
                traceback.print_exc()
                graph_error[0] = e
            finally:
                graph_done.set()
        
        thread = threading.Thread(target=_run_graph, daemon=True)
        thread.start()
        
        # 主线程持续 drain 缓冲区，直到图执行完毕且缓冲区为空
        while not graph_done.is_set() or self._event_buffer:
            if getattr(self, "_cancel_event", None) and self._cancel_event.is_set() and not graph_done.is_set():
                break
                
            if self._event_buffer:
                yield self._event_buffer.pop(0)
            else:
                time.sleep(0.01)  # 避免忙等待
        
        # 报告图执行中的异常
        if graph_error[0]:
            self._emit("error", f"[System Error] {str(graph_error[0])}")
        
        # drain 最终残余
        while self._event_buffer:
            yield self._event_buffer.pop(0)
            
        if getattr(self, "_cancel_event", None) and self._cancel_event.is_set():
            yield {
                "type": "warning",
                "content": "\n[系统] 已终止该轮次操作"
            }
            yield {
                "type": "done",
                "content": "已取消",
                "iteration_count": 0,
                "task_completed": False,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            return
        
        # 获取最终状态并 yield done 事件
        final_state = self.app.get_state(config)
        values = final_state.values if final_state else {}
        
        yield {
            "type": "done",
            "content": "",
            "iteration_count": values.get("iteration_count", 0),
            "task_completed": values.get("task_completed", False),
            "total_tokens": values.get("total_tokens", 0),
            "prompt_tokens": values.get("prompt_tokens", 0),
            "completion_tokens": values.get("completion_tokens", 0),
        }
    


    def compact_context(self, thread_id: str = "default") -> tuple:
        """
        手动触发上下文压缩
        
        Returns:
            (status_message: str, summary_text: str or None)
        """
        config = {"configurable": {"thread_id": thread_id}}
        current_state = self.app.get_state(config)
        
        if not current_state or not current_state.values:
            return ("无有效上下文", None)
            
        state = current_state.values
        messages = state.get("messages", [])
        
        if not messages:
            return ("消息历史为空，无需压缩", None)
            
        # 强制执行总结
        try:
            self._emit("info", "[手动压缩] 开始压缩上下文...")
            result = self._summarize_context(state)
            
            if result:
                # 更新状态
                self.app.update_state(config, result)
                summary = result.get("context_summary", "")
                return (f"上下文压缩完成。摘要长度: {len(summary)} 字符", summary)
            else:
                return ("上下文压缩失败或无需压缩", None)
        except Exception as e:
            return (f"压缩过程出错: {e}", None)

    def get_context(self, thread_id: str = "default") -> dict:
        """
        获取指定会话的当前完整状态上下文
        
        封装了向 LangGraph MemorySaver 查询特定 thread_id 最新状态的逻辑，
        方便用户直接调用来获取对话历史、Token用量等数据。
        
        Args:
            thread_id: 会话线程 ID
            
        Returns:
            包含当前状态所有键值对的字典。如果会话不存在，返回空字典。
            常用键包括: "messages", "overall_goal", "total_tokens", "iteration_count" 等。
        """
        config = {"configurable": {"thread_id": thread_id}}
        current_state = self.app.get_state(config)
        return current_state.values if current_state else {}

    def get_history(self, thread_id: str = "default", limit: int = 20) -> list:
        """
        获取操作流历史记录
        
        从检查点历史中提取Agent的操作流（工具调用、用户输入），
        以便用户了解Agent做了什么并选择回滚目标。
        
        Args:
            thread_id: 会话线程 ID
            limit: 返回的最大历史记录数
            
        Returns:
            操作流列表，每项包含 index, checkpoint_id, action_summary 等信息
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        history = []
        prev_msg_count = 0
        
        try:
            snapshots = list(self.app.get_state_history(config))
            
            for idx, state_snapshot in enumerate(snapshots):
                if idx >= limit:
                    break
                
                values = state_snapshot.values
                messages = values.get("messages", [])
                checkpoint_id = state_snapshot.config["configurable"].get("checkpoint_id", "")
                next_nodes = list(state_snapshot.next) if state_snapshot.next else []
                node_name = next_nodes[0] if next_nodes else "(结束)"
                
                # 提取本步骤新增的操作摘要
                action_summary = self._extract_action_summary(messages, prev_msg_count)
                prev_msg_count = len(messages)
                
                history.append({
                    "index": idx,
                    "checkpoint_id": checkpoint_id,
                    "node": node_name,
                    "message_count": len(messages),
                    "iteration": values.get("iteration_count", 0),
                    "action": action_summary,
                    "task_completed": values.get("task_completed", False),
                    "progress": values.get("progress_summary", "")  
                })
        except Exception as e:
            self._emit("error", f"[错误] 获取历史记录失败: {e}")
        
        # 反转顺序：从旧到新显示（get_state_history 返回最新在前）
        history.reverse()
        # 重新编号
        for i, item in enumerate(history):
            item["display_index"] = i
        
        return history
    
    def _extract_action_summary(self, messages: list, prev_count: int) -> str:
        """
        从新增的消息中提取操作摘要
        
        Args:
            messages: 当前所有消息
            prev_count: 上一个检查点的消息数
            
        Returns:
            操作摘要字符串
        """
        if not messages:
            return "初始化"
        
        # 获取新增的消息
        new_messages = messages[prev_count:] if prev_count < len(messages) else messages[-1:]
        
        actions = []
        for msg in new_messages:
            content = getattr(msg, 'content', str(msg))
            
            if isinstance(msg, HumanMessage):
                # 检查是否是工具结果
                if '<tool_result' in content:
                    import re
                    tool_match = re.search(r'tool="(\w+)"', content)
                    if tool_match:
                        tool_name = tool_match.group(1)
                        # 提取关键信息
                        if 'success' in content.lower():
                            actions.append(f"✓ {tool_name}")
                        elif 'error' in content.lower():
                            actions.append(f"✗ {tool_name}")
                        else:
                            actions.append(f"→ {tool_name}")
                elif content.startswith('[---'):
                    actions.append("─── 新一轮对话 ───")
                else:
                    # 用户输入
                    snippet = content[:60] + "..." if len(content) > 60 else content
                    actions.append(f"用户: {snippet}")
            elif isinstance(msg, AIMessage):
                # 从AI消息中提取工具调用
                import re
                tool_tags = ['read_file', 'write_to_file', 'execute_command', 
                            'search_in_files', 'replace_in_file', 'list_directory',
                            'create_directory', 'attempt_completion']
                found_tools = []
                for tag in tool_tags:
                    if f'<{tag}' in content or f'<{tag}>' in content:
                        found_tools.append(tag)
                
                if found_tools:
                    actions.append(f"Agent调用: {', '.join(found_tools)}")
                elif '<attempt_completion>' in content:
                    actions.append("任务完成")
        
        return " | ".join(actions) if actions else "状态更新"
    
    def rollback(self, thread_id: str = "default", target_index: int = None) -> bool:
        """
        回滚到指定的检查点
        
        通过存储目标检查点的 config，在下次 stream 调用时恢复到该状态。
        不使用 update_state（会创建新检查点导致重复）。
        
        Args:
            thread_id: 会话线程 ID
            target_index: 目标检查点在 get_state_history 中的索引
            
        Returns:
            是否成功设置回滚
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            snapshots = list(self.app.get_state_history(config))
            
            if not snapshots:
                self._emit("info", "[回滚] 暂无历史记录")
                return False
            
            if target_index is None:
                # 默认回退1步
                target_index = 1
            
            if target_index >= len(snapshots):
                self._emit("error", f"[回滚] 索引 {target_index} 超出历史范围 (共 {len(snapshots)} 条)")
                return False
            
            target_state = snapshots[target_index]
            target_values = target_state.values
            
            # 存储回滚配置，供下次 stream 使用
            self._rollback_config = target_state.config
            
            # 打印回滚信息
            msgs = target_values.get("messages", [])
            progress = target_values.get("progress_summary", "")
            
            self._emit("info", f"[回滚] 已设置回滚到检查点 (index={target_index})")
            self._emit("info", f"  消息数: {len(msgs)}")
            self._emit("info", f"  迭代: {target_values.get('iteration_count', 0)}")
            if progress:
                # 显示最后几行进度
                progress_lines = progress.strip().split('\n')
                for line in progress_lines[-3:]:
                    self._emit("info", f"  {line}")
            
            return True
            
        except Exception as e:
            self._emit("error", f"[回滚] 回滚失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_terminal_agent_stream(self, task: str, thread_id: str) -> dict:
        """运行 agent，捕捉流式输出并在控制台打印，返回最终的执行结果"""
        from klynx.agent.package import run_terminal_agent_stream
        return run_terminal_agent_stream(self, task, thread_id)
        
    def run_terminal_ask_stream(self, message: str, system_prompt: str = None, thread_id: str = "default") -> str:
        """运行 agent.ask()，流式打印回答，返回完整回答文本"""
        from klynx.agent.package import run_terminal_ask_stream
        return run_terminal_ask_stream(self, message, system_prompt, thread_id)


import platform
def create_agent(working_dir: str = ".", model=None, max_iterations: int = 1000,
                 memory_dir: str = "", load_project_docs: bool = True,
                 os_name: str = platform.system(), browser_headless: bool = False) -> KlynxAgent:
    """
    创建 Klynx Agent 实例
    
    Args:
        working_dir: 工作目录
        model: LangChain 模型实例（需设置 max_context_tokens 属性）
        max_iterations: 最大迭代次数
        memory_dir: Agent 记忆目录路径，为空则不加载 .klynx/.rules/.memory
        load_project_docs: 是否递归加载 KLYNX.md 项目文档，默认 True
        os_name: 操作系统名称，默认为当前检测到的系统类型 (如 'Windows', 'Linux', 'Darwin')
        browser_headless: 浏览器工具是否运行在无头模式，默认为 False (显示界面)
        
    Returns:
        KlynxAgent 实例
    """
    return KlynxAgent(
        working_dir=working_dir,
        model=model,
        max_iterations=max_iterations,
        memory_dir=memory_dir,
        load_project_docs=load_project_docs,
        os_name=os_name,
        browser_headless=browser_headless
    )
