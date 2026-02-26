import subprocess
import threading
import queue
import time
import os
from typing import Dict, Optional, List

class TerminalManager:
    """
    终端会话管理器 (Windows cmd.exe)
    
    管理多个后台 Shell 进程，支持:
    - 创建/销毁会话
    - 发送命令
    - 实时读取输出
    """
    
    def __init__(self, default_cwd: str = "."):
        self.sessions: Dict[str, subprocess.Popen] = {}
        self.output_queues: Dict[str, queue.Queue] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.default_cwd = os.path.abspath(default_cwd)
        self.lock = threading.Lock()

    def create_terminal(self, name: str, cwd: Optional[str] = None) -> str:
        with self.lock:
            if name in self.sessions:
                return f"<error>终端会话 '{name}' 已存在</error>"
            
            cwd = os.path.abspath(cwd) if cwd else self.default_cwd
            if not os.path.exists(cwd):
                return f"<error>目录不存在: {cwd}</error>"

            try:
                # 使用 shell=True 启动 cmd
                process = subprocess.Popen(
                    "cmd.exe",
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=cwd,
                    text=True,
                    bufsize=0, # 无缓冲
                    shell=False, # 使用 shell=False 直接启动 cmd.exe 程序，避免中间 shell
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                
                self.sessions[name] = process
                self.output_queues[name] = queue.Queue()
                
                # 启动后台线程
                thread = threading.Thread(target=self._read_output_loop, args=(name, process), daemon=True)
                thread.start()
                self.threads[name] = thread
                
                return f"<success>终端会话 '{name}' 已创建 (CWD: {cwd})</success>"
            except Exception as e:
                return f"<error>无法启动终端: {str(e)}</error>"

    def _read_output_loop(self, name: str, process: subprocess.Popen):
        """后台线程：持续读取输出到队列"""
        try:
            # 初始提示符可能需要时间
            time.sleep(0.1)
            while True:
                # 检查进程是否结束
                if process.poll() is not None:
                    break
                    
                # 逐字符读取以实时获取，避免行缓冲导致的卡顿
                # 但 Python 的 text mode read(1) 可能会阻塞，这里使用 readline 配合无缓冲
                line = process.stdout.readline()
                if line:
                    if name in self.output_queues:
                        self.output_queues[name].put(line)
                else:
                    # 没有读到数据，可能进程快结束了，稍微等待
                    time.sleep(0.05)
        except Exception:
            pass
        finally:
            if name in self.output_queues:
                self.output_queues[name].put(f"\n[终端 '{name}' 已退出]\n")

    def run_command(self, name: str, command: str) -> str:
        """发送命令到终端，不等待结果（结果通过 read_terminal 获取）"""
        with self.lock:
            if name not in self.sessions:
                return f"<error>终端会话 '{name}' 不存在</error>"
            
            process = self.sessions[name]
            if process.poll() is not None:
                del self.sessions[name]
                return f"<error>终端会话 '{name}' 已终止</error>"
            
            try:
                # 确保有换行符
                cmd_str = command.strip() + "\n"
                process.stdin.write(cmd_str)
                process.stdin.flush()
                return "<success>命令已发送</success>"
            except Exception as e:
                return f"<error>发送命令失败: {str(e)}</error>"

    def read_terminal(self, name: str, lines: int = 50) -> str:
        """读取累积的输出"""
        if name not in self.output_queues:
            return f"<error>终端会话 '{name}' 不存在</error>"
        
        q = self.output_queues[name]
        output_lines = []
        
        # 读取队列中所有现有内容
        while not q.empty():
            try:
                output_lines.append(q.get_nowait())
            except queue.Empty:
                break
        
        if not output_lines:
            return "(无新输出)"
            
        # 如果需要保留历史，这里应该设计为 buffer。
        # 此处简化：每次 read_terminal 会清空队列（消费模式），还是保留（View模式）？
        # 考虑到 agent 交互，通常是“消费模式”，读了就认为 agent 看到了。
        # 如果需要保留，应该在类里维护一个 list buffer。
        # 修正：改为维护 list buffer 支持回看，队列仅用于线程传输
        
        # 重新设计：_read_output_loop 写入 list，read_terminal 读取 list 切片
        # 这里改用消费模式更简单：Agent 读一次清空一次新内容，符合“读取最新输出”的语义
        
        full_output = "".join(output_lines)
        return full_output

    def close_terminal(self, name: str):
        with self.lock:
            if name in self.sessions:
                try:
                    self.sessions[name].terminate()
                except:
                    pass
                del self.sessions[name]
            if name in self.output_queues:
                del self.output_queues[name]
            if name in self.threads:
                del self.threads[name]
