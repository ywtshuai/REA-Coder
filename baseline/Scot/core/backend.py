# ============================================================
# 任务 1: 劫持 LLM 调用，使用 generate_code.py 中的 LLMClient
# ============================================================

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 延迟初始化全局 LLM 客户端，避免循环导入
_GLOBAL_LLM = None


def _get_llm():
    """延迟导入和初始化 LLM 客户端"""
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        from core.generate_code import build_llm

        # 根据当前环境中的 MODEL_C 选择与 scot_baseline 一致的配置
        model_name = os.environ.get("MODEL_C", "deepseek-chat")

        # gpt-5-mini-2025-08-07：使用 Responses API，给足 max_tokens，并开启 minimal reasoning
        if model_name == "gpt-5-mini-2025-08-07":
            _GLOBAL_LLM = build_llm(
                "MODEL_C",
                temperature=0,
                max_tokens=60000,
                reasoning={"effort": "minimal"},
            )
        # Gemini 3 Flash：通过 Chat Completions API，使用更大的 max_tokens
        elif model_name == "gemini-3-flash-preview":
            _GLOBAL_LLM = build_llm(
                "MODEL_C",
                temperature=0,
                max_tokens=60000,
            )
        # 其他模型：统一提升上下文长度，避免复杂题目代码被截断
        else:
            _GLOBAL_LLM = build_llm(
                "MODEL_C",
                temperature=0,
                max_tokens=8192,
            )
    return _GLOBAL_LLM


def call_chatgpt(prompt, model='gpt-3.5-turbo', stop=None, temperature=0., top_p=0.95,
                 max_tokens=128, echo=False, majority_at=None):
    """
    重写的 call_chatgpt 函数，使用 DeepSeek LLMClient 替代 OpenAI。
    忽略 model 参数，强制使用 _GLOBAL_LLM。
    支持 majority_at 参数（多次采样）。
    """
    llm = _get_llm()  # 使用延迟加载的 LLM 客户端
    num_completions = majority_at if majority_at is not None else 1
    completions = []
    
    for i in range(num_completions):
        try:
            # 针对 Gemini 3 Flash 进行特殊消息格式处理（与 SCoT 保持一致）
            model_name = os.environ.get("MODEL_C", "").strip()
            messages = prompt

            if model_name == "gemini-3-flash-preview":
                # 将历史对话消息压缩为一个文本，包在 Gemini 原生 request_body 中
                if isinstance(messages, list):
                    input_text = "\n\n".join(
                        f"{m.get('role', '')}: {m.get('content', '')}"
                        for m in messages
                    )
                else:
                    input_text = str(messages)

                request_body = {
                    "contents": [
                        {
                            "parts": [
                                {"text": input_text}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "thinkingConfig": {
                            "thinkingBudget": 0
                        },
                        "thinking_level": "minimal"
                    }
                }
                gemini_messages = [
                    {
                        "role": "user",
                        "content": json.dumps(request_body, ensure_ascii=False)
                    }
                ]

                response = llm.chat(
                    messages=gemini_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # 标准 OpenAI 兼容 messages
                response = llm.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            completions.append(response)
        except Exception as e:
            print(f"❌ LLM 调用失败 (第 {i+1}/{num_completions} 次): {e}")
            # 如果失败，添加空字符串作为占位
            completions.append("")
    
    return completions[:num_completions]