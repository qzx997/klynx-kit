# 封装给外部调用的原生终端流式打印接口

def run_terminal_agent_stream(agent, task: str, thread_id: str) -> dict:
    """辅助函数：运行 agent，捕捉流式输出并在控制台打印，返回最终的执行结果
    包含完整的事件解析与用量统计
    """
    print("\n" + "=" * 60)
    print(f"[Agent 开始执行] (Thread: {thread_id})")
    print("-" * 60)
    
    result = {}
    has_printed_reasoning_header = False
    has_printed_answer_header = False
    has_streamed_reasoning = False
    has_streamed_answer = False

    try:
        for event in agent.invoke(task=task, thread_id=thread_id):
            etype = event.get("type", "")
            content = event.get("content", "")
            
            if etype == "done":
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
                pass
                
    except KeyboardInterrupt:
        print("\n\n[系统] Agent 被手动中断。")
    except Exception as e:
         print(f"\n❌ 执行发生异常: {e}")
         
    print("\n" + "-" * 60)
    print(f"[Agent 执行完毕]")
    
    # 打印统计信息
    round_tokens = result.get('total_tokens', 0)
    task_completed = result.get('task_completed', False)
    print(f"本轮 Token 消耗: {round_tokens}")
    print(f"任务是否完成: {'是' if task_completed else '否'}")
    print("=" * 60 + "\n")
    
    return result


def run_terminal_ask_stream(agent, message: str, system_prompt: str = None, thread_id: str = "default") -> str:
    """辅助函数：运行 agent.ask()，流式打印回答，返回完整回答文本"""
    has_printed_reasoning_header = False
    has_printed_answer_header = False
    full_answer = ""
    
    for event in agent.ask(message=message, system_prompt=system_prompt, thread_id=thread_id):
        etype = event.get("type", "")
        content = event.get("content", "")
        
        if etype == "reasoning_token":
            if not has_printed_reasoning_header:
                print("\n[思考]", end="\n", flush=True)
                has_printed_reasoning_header = True
            print(content, end="", flush=True)
        elif etype == "token":
            if not has_printed_answer_header:
                print("\n[回答]", end="\n", flush=True)
                has_printed_answer_header = True
            print(content, end="", flush=True)
        elif etype == "reasoning":
            if not has_printed_reasoning_header:
                print(f"\n[思考]\n{content}")
        elif etype == "answer":
            if not has_printed_answer_header:
                print(f"\n[回答]\n{content}")
        elif etype == "error":
            print(f"\n❌ {content}")
        elif etype == "done":
            full_answer = event.get("answer", "")
    
    print()
    return full_answer
