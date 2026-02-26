"""
模型注册表 - 预定义模型配置 + setup_model() 工厂函数

通过 LiteLLM 统一桥接国内外各大云商模型，并管理 API Key 的获取。
"""

import os
from .adapter import LiteLLMChat

# 模型别名映射至真正的 litellm provider/model
# 这里定义了用户常用输入对应的标准路径，如果用户输入不在本表中，则按原样传给 litellm 匹配
MODEL_REGISTRY = {
    # ---------------- DeepSeek ----------------
    "deepseek-reasoner": {
        "model": "deepseek/deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY",
        "description": "DeepSeek Reasoner (深度思考模型)",
    },
    "deepseek-chat": {
        "model": "deepseek/deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "description": "DeepSeek Chat (通用对话模型)",
    },
    
    # ---------------- OpenAI (GPT-5/O series) ----------------
    "gpt-5.3": {
        "model": "openai/gpt-5.3",
        "env_key": "OPENAI_API_KEY",
        "description": "GPT-5.3 (OpenAI 2026旗舰巨兽)",
    },
    "gpt-5.2": {
        "model": "openai/gpt-5.2",
        "env_key": "OPENAI_API_KEY",
        "description": "GPT-5.2 (前沿多模态)",
    },
    "o3-mini": {
        "model": "openai/o3-mini",
        "env_key": "OPENAI_API_KEY",
        "description": "O3-mini (经济极速推理)",
    },
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "description": "GPT-4o",
    },
    "o1-preview": {
        "model": "openai/o1-preview",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI O1 Preview",
    },
    
    # ---------------- Anthropic (Claude 4) ----------------
    "claude-4.6-sonnet": {
        "model": "anthropic/claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Claude 4.6 Sonnet (最新极速全能旗舰)",
    },
    "claude-4.5-opus": {
        "model": "anthropic/claude-opus-4-5-20251101",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Claude 4.5 Opus (超级复杂多步推理)",
    },
    "claude-3.5-sonnet": {
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Claude 3.5 Sonnet",
    },

    # ---------------- Google (Gemini 3) ----------------
    "gemini-3.1-pro": {
        "model": "gemini/gemini-3.1-pro",
        "env_key": "GEMINI_API_KEY",
        "description": "Gemini 3.1 Pro (超长上下文终极统御版)",
    },
    "gemini-2.5-flash": {
        "model": "gemini/gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "description": "Gemini 2.5 Flash (疾速响应轻量)",
    },
    "gemini-1.5-pro": {
        "model": "gemini/gemini-1.5-pro",
        "env_key": "GEMINI_API_KEY",
        "description": "Gemini 1.5 Pro",
    },

    # ---------------- 智谱 (GLM) ----------------
    "glm-5": {
        "model": "openai/glm-5",
        "env_key": "GLM_API_KEY",
        "description": "GLM-5 (智谱AI 744B参数最新超前基座)",
        "default_kwargs": {
            "api_base": "https://open.bigmodel.cn/api/paas/v4/"
        }
    },
    "glm-4-plus": {
        "model": "openai/glm-4-plus",
        "env_key": "GLM_API_KEY",
        "description": "GLM-4 Plus",
        "default_kwargs": {
            "api_base": "https://open.bigmodel.cn/api/paas/v4/"
        }
    },
    "glm-4-long": {
        "model": "openai/glm-4-long",
        "env_key": "GLM_API_KEY",
        "description": "GLM-4 Long",
        "default_kwargs": {
            "api_base": "https://open.bigmodel.cn/api/paas/v4/"
        }
    },
    "glm-4-flash": {
        "model": "openai/glm-4-flash",
        "env_key": "GLM_API_KEY",
        "description": "GLM-4 Flash (极速响应免费版)",
        "default_kwargs": {
            "api_base": "https://open.bigmodel.cn/api/paas/v4/"
        }
    },
    
    # ---------------- Moonshot (Kimi k2.5) ----------------
    "kimi-k2.5": {
        "model": "openai/kimi-k2.5",
        "env_key": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
        "description": "Kimi-k2.5 (Moonshot AI 最新最强长文本)",
        "default_kwargs": {
            "api_base": "https://api.moonshot.cn/v1",
            "extra_body": {"enable_thinking": True}
        }
    },
    "kimi": {
        "model": "openai/moonshot-v1-auto",
        "env_key": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
        "description": "Kimi (Moonshot 经典版)",
        "default_kwargs": {
            "api_base": "https://api.moonshot.cn/v1"
        }
    },

    # ---------------- Minimax (海螺 M2.5) ----------------
    "minimax-m2.5": {
        "model": "minimax/abab-m2.5",
        "env_key": "MINIMAX_API_KEY",
        "description": "Minimax Abab M2.5 (前沿生产力框架)",
    },
    "minimax-text-01": {
        "model": "minimax/abab6.5g-chat",
        "env_key": "MINIMAX_API_KEY",
        "description": "Minimax Abab6.5G",
    },

    # ---------------- Qwen (通义千问 动态版) ----------------
    "qwen-max-latest": {
        "model": "dashscope/qwen-max-latest",
        "env_key": "DASHSCOPE_API_KEY",
        "description": "通义千问 Max (动态绑定阿里最新顶级模型)",
    },
    "qwen-plus-latest": {
        "model": "dashscope/qwen-plus-latest",
        "env_key": "DASHSCOPE_API_KEY",
        "description": "通义千问 Plus (动态绑定最新极速版)",
    },
}

def setup(provider: str, model_name: str = None, api_key: str = None, **kwargs) -> LiteLLMChat:
    """
    根据提供商和模型名称创建模型实例。
    
    支持调用方式:
      1. setup("deepseek", "deepseek-reasoner")  (推荐，新版规范)
      2. setup("claude-3.5-sonnet")              (向后兼容：仅提供第一参数作为简称别名)
      3. setup("openai", "gpt-4o", "sk-xxx...")  (显式指定 key)
      
    Args:
        provider: 模型服务商（如 "openai", "deepseek", "anthropic", "gemini"，或者直接提供模型别名）
        model_name: 具体的模型名。可缺省（当 provider 参数传的是 Klynx 注册表里预定义的别名时）。
        api_key: API 服务密钥。如果为空，则根据注册表推断所需的环境变量去 os.environ 读取。
        **kwargs: 其他模型生成参数（如 temperature，api_base 等），直接透传给 liteLLM
        
    Returns:
        配置完成的 LiteLLMChat 实例
        
    Raises:
        ValueError: 当 API Key 缺失或无法加载底层模型配置时
    """
    
    # 解析 provider/model 的逻辑
    if model_name is None:
        # 如果只传了一个参数，视为别名 (例如 setup("gpt-4o"))
        alias = provider
        if alias in MODEL_REGISTRY:
            config = MODEL_REGISTRY[alias]
            actual_model_string = config["model"]
            desc = config["description"]
            env_keys = config.get("env_key")
            default_kwargs = config.get("default_kwargs", {})
        else:
            actual_model_string = alias
            desc = f"Custom Model: {alias}"
            env_keys = None
            default_kwargs = {}
    else:
        # 如果传了两个参数 (例如 setup("deepseek", "deepseek-reasoner"))
        # 先看 model_name 是否在别名表里，如果在，直接用它的 model 字符串
        if model_name in MODEL_REGISTRY:
            config = MODEL_REGISTRY[model_name]
            actual_model_string = config["model"]
            desc = config["description"]
            env_keys = config.get("env_key")
            default_kwargs = config.get("default_kwargs", {})
        else:
            # 不在别名表，直接拼接 provider/model_name 喂给 litellm
            actual_model_string = f"{provider}/{model_name}"
            desc = f"Custom Model: {actual_model_string}"
            env_keys = None
            default_kwargs = {}

    # 强制要求用户显式传入 API Key
    if not api_key:
        raise ValueError(
            f"未找到 {desc} 的 API Key。\\n"
            f"请在代码中通过 api_key 参数显式传入：例如 setup_model('提供商', '模型名', 'sk-xxx')"
        )
    final_api_key = api_key

    # 将默认参数配置与用户传入的 kwargs 合并（用户优先级更高）
    final_kwargs = default_kwargs.copy()
    final_kwargs.update(kwargs)

    # 处理旧版本预设上下文参数或者额外的参数
    # 如果没显式指定 max_context_tokens，给一个兜底经验值（因为现在主要支持动态检索而不强依赖截断了）
    max_tokens = final_kwargs.pop("max_context_tokens", 128000)
    api_base = final_kwargs.pop("api_base", None)
    temperature = final_kwargs.pop("temperature", 0.1)
    
    print(f"[Model] {desc} (liteLLM route: {actual_model_string})")
    
    model = LiteLLMChat(
        model=actual_model_string,
        api_key=final_api_key,
        api_base=api_base,
        temperature=temperature,
        **final_kwargs
    )
    model.max_context_tokens = max_tokens
    
    return model


def list_models():
    """列出所有可用模型别名配置"""
    print("\n预定义模型别名 (可用作 setup 的 model_name):")
    print("-" * 75)
    for name, config in MODEL_REGISTRY.items():
        env_keys = config.get("env_key", "")
        if isinstance(env_keys, list):
            env_str = " / ".join(env_keys)
        else:
            env_str = env_keys or "N/A"
        
        # 检查 key 是否存在
        has_key = False
        if isinstance(config.get("env_key"), list):
            has_key = any(os.getenv(k) for k in config["env_key"])
        elif config.get("env_key"):
            has_key = bool(os.getenv(config["env_key"]))
        
        status = "✓" if has_key else "✗"
        print(f"  {status} {name:20s} | {config['description']:35s} | Key: {env_str}")
    print("-" * 75)
    print("注：您也可以使用 setup(\"provider\", \"model_name\") 的形式直接调用列表中未枚举的其它任意模型。")
