"""
Klynx Model 模块

统一的模型管理接口，基于 LiteLLM 实现多模型支持。

Usage:
    from model import setup, list_models
    # Initialize a model
    import os
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = setup("deepseek", "deepseek-reasoner", api_key)

    # Call the model
    # model = setup("gpt-4o")
    
    list_models()  # 查看所有可用模型
"""

from .registry import setup, list_models, MODEL_REGISTRY
from .adapter import LiteLLMChat, LiteLLMResponse

__all__ = ["setup", "list_models", "MODEL_REGISTRY", "LiteLLMChat", "LiteLLMResponse"]
