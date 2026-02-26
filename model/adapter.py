"""
模型适配器 - 统一模型调用接口

提供与 KlynxAgent 兼容的 .invoke(messages) 接口。
使用 LiteLLM 作为统一标准接口，完美支持全球各大模型厂商 (OpenAI, Anthropic, Google, DeepSeek 等)。
通过显式传递 httpx.Client 避开局域网代理限制。
"""

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import List, Optional
import litellm

# 禁用从远端请求价格表导致的 SSL 请求超时警告
litellm.suppress_debug_info = True
class LiteLLMResponse:
    """
    响应包装类
    兼容 graph.py 中对 response.content / response.reasoning_content / response.usage 的访问
    同时支持原生 Function Calling 返回的 tool_calls
    """
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content
        self.additional_kwargs = {}
        self.response_metadata = {}
        self.usage = None  # 由调用方填充
        self.tool_calls = []  # 原生 Function Calling 返回的工具调用列表


class LiteLLMChat:
    """
    模型适配器
    
    使用 LiteLLM 标准化接口调用各模型厂商，
    对外暴露与原来相同的 .invoke(messages) 和 .stream(messages) 接口。
    支持可选的原生 Function Calling (tools 参数)。
    """
    
    def __init__(self, model: str, api_key: str = None, api_base: str = None,
                 temperature: float = 0.1, **kwargs):
        """
        Args:
            model: 模型标识符 (如 "deepseek/deepseek-reasoner", "gpt-4o")
                   支持 litellm provider/model 格式
            api_key: API Key
            api_base: 自定义 API 端点 (可选)
            temperature: 生成温度
            **kwargs: 额外参数
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.extra_kwargs = kwargs
        self.max_context_tokens = 128000
        
        # 创建客户端（禁用代理，避免 SSL 错误）
        self.http_client = httpx.Client(trust_env=False)
    
    def invoke(self, messages, tools: Optional[list] = None):
        """
        调用模型，返回 LiteLLMResponse
        
        Args:
            messages: LangChain BaseMessage 列表 或 OpenAI 格式字典列表
            tools: 可选，OpenAI 格式的 tools JSON Schema 列表（用于原生 Function Calling）
            
        Returns:
            LiteLLMResponse 对象
        """
        # 转换消息格式
        openai_messages = self._convert_messages(messages)
        
        # 构建调用参数
        call_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "api_key": self.api_key,
        }
        
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
            
        # 部分思考模型对 temperature 有严格限制
        if not any(k in self.model for k in ["reasoner", "o1", "kimi-k2.5"]):
            call_kwargs["temperature"] = self.temperature
        
        # 原生 Function Calling: 注入 tools
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"
        
        # 合并额外参数
        call_kwargs.update(self.extra_kwargs)
        
        # 调用 API
        response = litellm.completion(**call_kwargs)
        
        # 解析响应
        message = response.choices[0].message
        content = message.content or ""
        # 兼容各大厂商对 reasoning_content 的非标准输出名称
        reasoning_content = getattr(message, 'reasoning_content', '') or ""
        
        # 构建响应对象
        resp = LiteLLMResponse(content=content, reasoning_content=reasoning_content)
        
        # 解析原生 tool_calls（如果有）
        if hasattr(message, 'tool_calls') and message.tool_calls:
            resp.tool_calls = self._parse_native_tool_calls(message.tool_calls)
        
        # 填充 usage
        usage = response.usage
        if usage:
            resp.usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        
        return resp
    
    def stream(self, messages, tools: Optional[list] = None):
        """
        流式调用模型
        
        Args:
            messages: LangChain 消息列表
            tools: 可选，OpenAI 格式的 tools JSON Schema 列表（用于原生 Function Calling）
            
        Yields:
            chunks with 'content', 'reasoning_content', 'usage', and 'tool_calls'
        """
        openai_messages = self._convert_messages(messages)
        
        call_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "api_key": self.api_key,
        }
        
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
            
        if not any(k in self.model for k in ["reasoner", "o1", "kimi-k2.5"]):
            call_kwargs["temperature"] = self.temperature
        
        # 原生 Function Calling: 注入 tools
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"
            
        call_kwargs.update(self.extra_kwargs)
            
        try:
            response = litellm.completion(**call_kwargs)
            
            # 用于拼接流式 tool_calls 片段
            tool_call_accumulators = {}
            
            for chunk in response:
                # 处理 Usage 信息 (通常在最后一个 chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    yield {
                        "usage": {
                            "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(chunk.usage, "completion_tokens", 0),
                            "total_tokens": getattr(chunk.usage, "total_tokens", 0)
                        }
                    }
                
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                content = delta.content or ""
                reasoning_content = getattr(delta, 'reasoning_content', '') or ""
                
                # 拼接流式 tool_calls 片段
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {
                                "id": "",
                                "function": {"name": "", "arguments": ""}
                            }
                        acc = tool_call_accumulators[idx]
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                acc["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                acc["function"]["arguments"] += tc_delta.function.arguments
                
                if content or reasoning_content or (hasattr(delta, 'tool_calls') and delta.tool_calls):
                    yield {
                        "content": content,
                        "reasoning_content": reasoning_content
                    }
            
            # 流结束后，如果有拼接好的 tool_calls，一次性 yield 出去
            if tool_call_accumulators:
                assembled = []
                for idx in sorted(tool_call_accumulators.keys()):
                    acc = tool_call_accumulators[idx]
                    assembled.append(self._parse_single_tool_call(acc))
                yield {"tool_calls": assembled}
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def _parse_native_tool_calls(self, raw_tool_calls) -> list:
        """
        将 LiteLLM 返回的原生 tool_calls 对象列表转换为统一的字典格式
        
        Returns:
            [{"tool": "read_file", "params": {"path": "..."}}, ...]
        """
        import json
        results = []
        for tc in raw_tool_calls:
            func = tc.function
            name = func.name
            try:
                args = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
            except json.JSONDecodeError:
                args = {"raw_arguments": func.arguments}
            results.append({
                "tool": name,
                "params": args
            })
        return results
    
    def _parse_single_tool_call(self, acc: dict) -> dict:
        """
        解析单个拼接完成的 tool_call 累积器为统一格式
        """
        import json
        name = acc["function"]["name"]
        raw_args = acc["function"]["arguments"]
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"raw_arguments": raw_args}
        return {
            "tool": name,
            "params": args
        }
    
    def _convert_messages(self, messages):
        """将 LangChain 消息列表转换为 OpenAI 格式"""
        openai_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                openai_messages.append(msg)
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            else:
                role = getattr(msg, 'type', 'user')
                openai_messages.append({"role": role, "content": str(msg.content)})
        return openai_messages

    def __repr__(self):
        return f"LiteLLMChat(model={self.model!r}, max_context_tokens={self.max_context_tokens})"


# 为了兼容历史代码依赖，提供原名的别名，建议未来直接使用 LiteLLMChat
DeepSeekReasonerChat = LiteLLMChat
DeepSeekReasonerResponse = LiteLLMResponse
