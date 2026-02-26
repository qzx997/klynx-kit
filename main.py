"""Klynx Agent - 主入口文件
基于 LangGraph 实现的 Coding Agent
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
# 加载环境变量
basedir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(basedir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

load_dotenv(os.path.join(basedir, ".env"))

from klynx import create_agent, setup_model, list_models


def print_banner():
    """打印欢迎横幅"""
    print("")
    print("=" * 60)
    print("  Klynx Agent")
    print("  基于 LangGraph 的智能编程助手")
    print("=" * 60)
    print("")


def execute_task():
    """
    直接执行任务模式
    
    Args:
        working_dir: 工作目录
        task: 任务描述
        max_iterations: 最大迭代次数
        max_context_tokens: 最大上下文token数
    
    Returns:
        执行结果字典
    """
    # 工作目录
    # working_dir = r"E:\BaiduSyncdisk\research\Programming_Development\prodev\pycli_kit-main\knowledge_db"
    working_dir = r"E:\BaiduSyncdisk\research\Programming_Development\dev_notebook"
    # 任务
    task = "查看当前项目内容"
    # task = "你的训练数据截至日期和上下文长度分别是多少"
    
    print("=" * 60)
    print("Klynx Agent 测试")
    print("=" * 60)
    print(f"工作目录: {working_dir}")
    print(f"任务: {task}")
    print("=" * 60)
    
    # 设置模型
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = setup_model("deepseek", "deepseek-reasoner", api_key)
    
    agent = create_agent(
        working_dir=working_dir,
        model=model,
        max_iterations=50,
        memory_dir=working_dir,  # 记忆目录与工作目录相同
        load_project_docs=True
    )
    agent.add_tools("all")
    
    
    # 执行
    try:
        result = agent.invoke(task)
        
        print("\n" + "=" * 60)
        print("[执行完成]")
        print("=" * 60)
        
        if result.get("summary_content"):
            print(f"\n{result['summary_content']}")
        
        print(f"\n迭代次数: {result.get('iteration_count', 0)}")
        print(f"任务完成: {'是' if result.get('task_completed') else '否'}")
        print(f"Token消耗: {result.get('total_tokens', 0)}")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


def interactive_invoke_mode():
    """
    基于 invoke 的交互式模式
    
    每轮对话通过 invoke 执行任务，将返回的 context 重新注入下一轮调用，
    实现多轮对话的上下文连续性。
    
    与 interactive_task_mode 的区别：
    - interactive_task_mode: 使用 stream + checkpointer 自动管理状态
    - interactive_invoke_mode: 使用 invoke + 显式 context 传递，更轻量
    """
    
    # 工作目录
    working_dir = os.getcwd()
    
    print("=" * 60)
    print("Klynx Agent — Invoke 交互模式")
    print("=" * 60)
    print(f"工作目录: {working_dir}")
    print("=" * 60)
    
    # 设置模型
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = setup_model("deepseek", "deepseek-reasoner", api_key)
    
    agent = create_agent(
        working_dir=working_dir,
        model=model,
        max_iterations=50,
        memory_dir=None,
        load_project_docs=False
    )
    agent.add_tools("all")
    
    # 加载 MCP (Model Context Protocol) 插件 (如果存在)
    mcp_config_path = os.path.join(basedir, "mcp_servers.json")
    if os.path.exists(mcp_config_path):
        agent.add_mcp(mcp_config_path)
    
    print("\n" + "=" * 60)
    print("[Invoke 交互模式] 输入任务让 Agent 执行")
    print("提示: 输入 'exit' 或 'quit' 退出程序")
    print("      输入 'clear' 或 '清空' 清空上下文")
    print("      输入 'context' 查看当前上下文长度")
    print("=" * 60 + "\n")
    
    # 上下文累积器：每轮 invoke 返回的 context 注入下一轮
    import uuid
    thread_id = str(uuid.uuid4())[:8]
    round_count = 0
    total_tokens = 0
    
    while True:
        try:
            # 获取用户输入
            prompt_hint = f"[轮次 {round_count + 1}]"
            task = input(f"{prompt_hint} > ").strip()
            
            # 检查退出命令
            if task.lower() in ("exit", "quit", "退出", "q"):
                print("\n[系统] 再见！")
                print(f"[统计] 共 {round_count} 轮对话，累计 Token: {total_tokens}")
                break
            
            # 空输入跳过
            if not task:
                continue
            
            # 清空上下文（使用全新 thread_id）
            if task.lower() in ("clear", "清空"):
                thread_id = str(uuid.uuid4())[:8]
                round_count = 0
                total_tokens = 0
                print(f"[系统] 上下文已清空，开始新对话 (thread: {thread_id})\n")
                continue
            
            # 查看当前上下文占用
            if task.lower() in ("context", "上下文", "context full"):
                is_full = "full" in task.lower()
                state_values = agent.get_context(thread_id)
                if state_values:
                    msgs = state_values.get("messages", [])
                    total_chars = sum(len(m.content) for m in msgs if hasattr(m, "content") and isinstance(m.content, str))
                    estimated_tokens = total_chars // 2
                    goal = state_values.get("overall_goal", "")
                    
                    print(f"\n[上下文] Thread: {thread_id}")
                    print(f"  - 历史消息: {len(msgs)} 条")
                    print(f"  - 估算用量: ~{estimated_tokens} Tokens")
                    print(f"  - 当前目标: {goal if goal else '(无)'}\n")
                    
                    if is_full and msgs:
                        print(f"\n[{Fore.GREEN}全部历史消息流水{Style.RESET_ALL}]")
                        print("-" * 50)
                        show_thinking = state_values.get("thinking_context", True)
                        
                        for i, r in enumerate(msgs):
                            role = r.__class__.__name__.replace("Message", "")
                            
                            reasoning = ""
                            if show_thinking and hasattr(r, "additional_kwargs") and r.additional_kwargs:
                                reasoning = r.additional_kwargs.get("reasoning_content", "")
                                
                            content_str = str(r.content)
                            if reasoning:
                                content_str = f"<think>\n{reasoning}\n</think>\n{content_str}"
                                
                            print(f"[{i+1}] [{role}]:\n{content_str}")
                            print("-" * 50)          
                    elif msgs: # This block was modified to only show the hint if not full
                        print("  (提示: 输入 'context full' 显示所有消息的完整内容)")
                else:
                    print(f"\n[上下文] Thread: {thread_id} (当前为空)\n")
                continue
            
            print("\n" + "-" * 60)
            print(f"[Agent 开始执行...]")
            print("-" * 60 + "\n")
            
            # 流式消费 invoke 事件
            result = {}
            # 标记是否已经打印过标题
            has_printed_reasoning_header = False
            has_printed_answer_header = False
            # 标记是否通过流式输出过内容
            has_streamed_reasoning = False
            has_streamed_answer = False

            for event in agent.invoke(task=task, thread_id=thread_id):
                etype = event.get("type", "")
                content = event.get("content", "")
                
                if etype == "done":
                    # 最终事件，提取状态
                    result = event
                elif etype == "iteration":
                    print(f"\n{content}")
                    
                elif etype == "reasoning_token":
                    if not has_printed_reasoning_header:
                        print("\n[思考过程]", end="\n", flush=True)
                        has_printed_reasoning_header = True
                    print(content, end="", flush=True)
                    has_streamed_reasoning = True
                    
                elif etype == "token":
                    if not has_printed_answer_header:
                        print("\n[回答]", end="\n", flush=True)
                        has_printed_answer_header = True
                    print(content, end="", flush=True)
                    has_streamed_answer = True

                elif etype == "reasoning":
                    if not has_streamed_reasoning:
                        print(f"\n[思考过程]\n{content}")
                elif etype == "answer":
                    if not has_streamed_answer:
                        print(f"\n[回答]\n{content}")
                        
                elif etype == "summary":
                    print(f"\n[总结]\n{content}")
                elif etype == "tool_exec":
                    print(content)
                elif etype == "tool_result":
                    print(f"  结果:\n{content}")
                elif etype == "tool_calls":
                    print(content)
                elif etype == "token_usage":
                    print(content)
                elif etype == "context_stats":
                    print(content)
                elif etype == "routing":
                    print(content)
                elif etype == "complete":
                    print(f"\n{content}")
                elif etype == "warning":
                    print(f"⚠ {content}")
                elif etype == "error":
                    print(f"✗ {content}")
                elif etype == "info":
                    print(content)
                else:
                    # 未知类型，直接输出
                    if content:
                        print(content)
            
            # 轮次统计
            round_count += 1
            round_tokens = result.get("total_tokens", 0)
            total_tokens += round_tokens
            
            print("\n" + "-" * 60)
            print(f"[第 {round_count} 轮完成]")
            print("-" * 60)
            
            if result.get("summary_content"):
                print(f"\n{result['summary_content']}")
            
            print(f"\n迭代次数: {result.get('iteration_count', 0)}")
            print(f"任务完成: {'是' if result.get('task_completed') else '否'}")
            print(f"本轮Token: {round_tokens} | 累计Token: {total_tokens}")
            
            print("\n")
            
        except KeyboardInterrupt:
            print(f"\n\n[系统] 用户中断，退出程序")
            print(f"[统计] 共 {round_count} 轮对话，累计 Token: {total_tokens}")
            break
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            print("\n")




def main():  



    print_banner()
    interactive_invoke_mode()
    # execute_task()
    







if __name__ == "__main__":
    main()
