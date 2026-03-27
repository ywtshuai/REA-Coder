import os
from openai import OpenAI
import time

class Model:
    def __init__(self, model, temperature):
        self.model_name = model
        self.client = self.model_setup()
        self.total_tokens = 0
        if temperature is not None:
            self.temperature = temperature

    def model_setup(self):
        # 优先使用环境变量配置（兼容 specfix_baseline 的 qwen/deepseek）
        base_url = os.environ.get("MODEL_API_BASE_URL")
        api_key_env = os.environ.get("MODEL_API_KEY_ENV")
        if base_url and api_key_env:
            api_key = os.environ.get(api_key_env, "")
            if not api_key:
                api_key = os.environ.get("LLM_API_KEY", "")
            return OpenAI(api_key=api_key, base_url=base_url)

        api_key = os.environ.get("LLM_API_KEY", "")
        if "qwen" in self.model_name.lower():
            base = os.environ.get("MODEL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            key = os.environ.get("DASHSCOPE_API_KEY", api_key)
            return OpenAI(api_key=key, base_url=base)
        if "deepseek" in self.model_name.lower():
            base = os.environ.get("MODEL_API_BASE_URL", "https://api.deepseek.com/v1")
            key = os.environ.get("DEEPSEEK_API_KEY", api_key)
            return OpenAI(api_key=key, base_url=base)
        if "gpt" in self.model_name.lower() or "o1" in self.model_name or "o3" in self.model_name:
            return OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
        if "llama" in self.model_name.lower():
            return OpenAI(api_key=api_key, base_url=os.environ.get("MODEL_API_BASE_URL", ""))
        raise ValueError(f"Invalid model: {self.model_name}. Supported: qwen, deepseek, gpt, llama.")

    def get_response_sample(self, instruction, prompt, n=20, use_model_settings=None):
        """生成 n 个样本。DeepSeek/Qwen 等 API 仅支持 n=1，会循环调用 n 次。"""
        responses = []
        use_temp = use_model_settings is not None
        kwargs = {"temperature": self.temperature} if use_temp else {}

        try:
            for _ in range(n):
                if self.model_name == "gpt-5-mini-2025-08-07":
                    chat_completion = self.client.responses.create(
                        model=self.model_name,
                        input=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": prompt}
                        ],
                        reasoning={"effort": "minimal"}
                    )
                    c = getattr(chat_completion, "output_text", None)
                    if c:
                        responses.append(c)
                elif self.model_name == "gemini-3-flash-preview":
                    import json
                    request_body = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": f"{instruction}\n\n{prompt}"}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "thinkingConfig": {"thinkingBudget": 0},
                            "thinking_level": "minimal"
                        }
                    }
                    chat_completion = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": json.dumps(request_body, ensure_ascii=False)}],
                        model=self.model_name,
                        **kwargs,
                    )
                    if chat_completion.choices:
                        c = chat_completion.choices[0].message.content
                        if c:
                            responses.append(c)
                else:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model_name,
                        **kwargs,
                    )
                    if chat_completion.choices:
                        c = chat_completion.choices[0].message.content
                        if c:
                            responses.append(c)
                if hasattr(chat_completion, "usage") and chat_completion.usage:
                    self.total_tokens += getattr(chat_completion.usage, "total_tokens", 0) or 0
        except Exception as e:
            print('[ERROR]', e)
            time.sleep(5)
        return responses

    def get_response(self, instruction, prompt, use_model_settings=None):
        try:
            kwargs = {}
            if use_model_settings is not None:
                kwargs = {"temperature": 0, "top_p": 0.95, "frequency_penalty": 0}

            if self.model_name == "gpt-5-mini-2025-08-07":
                chat_completion = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}
                    ],
                    reasoning={"effort": "minimal"}
                )
                response = getattr(chat_completion, "output_text", None)
            elif self.model_name == "gemini-3-flash-preview":
                import json
                request_body = {
                    "contents": [
                        {
                            "parts": [
                                {"text": f"{instruction}\n\n{prompt}"}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "thinkingConfig": {"thinkingBudget": 0},
                        "thinking_level": "minimal"
                    }
                }
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": json.dumps(request_body, ensure_ascii=False)}],
                    model=self.model_name,
                    **kwargs,
                )
                response = chat_completion.choices[0].message.content
            else:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    **kwargs,
                )
                response = chat_completion.choices[0].message.content

            if hasattr(chat_completion, "usage") and chat_completion.usage:
                self.total_tokens += getattr(chat_completion.usage, "total_tokens", 0) or 0
            if response:
                return response
            else:
                return ""
        except Exception as e:
            print('[ERROR]', e)
            time.sleep(5)
