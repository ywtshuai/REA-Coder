# llm/client.py
import os, time, json, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class LLMConfig:
    provider: str
    base_url: str
    api_key_env: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 1400  # 旧名，内部会映射到 max_output_tokens / max_tokens
    timeout_sec: int = 2000
    max_retries: int = 5
    api_mode: str = "auto"  # "auto" | "chat" | "responses"

    # ===== Gemini wrapper options =====
    # 当 model 以 "gemini-" 开头时，将 messages 打包成 Gemini 的 request_body JSON
    gemini_enable_wrapper: bool = True
    # thinkingBudget=0 => 思考最低
    gemini_thinking_budget: Optional[int] = 0
    # 你也可以扩展更多 generationConfig 字段（如有需要）
    gemini_extra_generation_config: Optional[Dict[str, Any]] = None
    gemini_thinking_level = "minimal"


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = os.getenv(cfg.api_key_env, "")
        if cfg.provider != "openai_compatible":
            raise ValueError(f"Unsupported provider: {cfg.provider}")
        if not self.api_key:
            raise RuntimeError(
                f"Missing API key env var: {cfg.api_key_env}. "
                f"Please export {cfg.api_key_env}=..."
            )

    def _is_gpt5_family(self) -> bool:
        """
        GPT-5 系列常见命名：
        - gpt-5, gpt-5-mini, gpt-5-nano
        - gpt-5-xxxx-xx-xx
        - gpt-5.1 / gpt-5.2 等
        - gpt-5-chat-latest / gpt-5.2-codex 等
        """
        m = (self.cfg.model or "").lower().strip()
        return m.startswith("gpt-5") or m.startswith("gpt5") or ("gpt-5" in m) or ("gpt5" in m)

    def _is_gemini_family(self) -> bool:
        m = (self.cfg.model or "").lower().strip()
        return m.startswith("gemini-")

    def _should_use_responses(self) -> bool:
        """
        需求：只要是 gpt-5 系列模型，就强制走 Responses。
        其他模型：遵循 api_mode 或原有 auto 逻辑。
        兼容：Gemini 聚合网关通常只支持 /chat/completions + JSON 字符串包装，因此默认不走 responses。
        """
        if self.cfg.api_mode == "responses":
            # 用户强制走 responses
            return True
        if self.cfg.api_mode == "chat":
            return False

        # auto
        if self._is_gemini_family():
            return False

        if self._is_gpt5_family():
            return True

        m = (self.cfg.model or "").lower()
        return m.startswith("o3-pro")

    @staticmethod
    def _is_retryable_status(code: int) -> bool:
        return code in (408, 409, 425, 429, 500, 502, 503, 504)

    @staticmethod
    def _format_messages_plain(messages: List[Dict[str, str]]) -> str:
        """
        将多轮 messages 合并成一个文本块，喂给 Gemini 的 contents.parts[0].text
        """
        chunks: List[str] = []
        for msg in messages:
            role = (msg.get("role") or "user").strip()
            content = msg.get("content") or ""
            # 保守做法：保留 role 以免丢信息
            chunks.append(f"{role}:\n{content}".strip())
        return "\n\n".join(chunks).strip()

    def _wrap_messages_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        按示例的“Gemini 3 Pro 请求体结构”，将其 JSON 字符串塞进 chat.completions 的 user.content
        """
        merged_text = self._format_messages_plain(messages)

        generation_config: Dict[str, Any] = {}
        # thinkingBudget
        if self.cfg.gemini_thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": int(self.cfg.gemini_thinking_budget)}
        if self.cfg.gemini_thinking_level is not None:
            generation_config["thinking_level"] = self.cfg.gemini_thinking_level

        # 可选追加字段（如 topK/topP/stopSequences 等）
        if isinstance(self.cfg.gemini_extra_generation_config, dict):
            # 允许用户覆盖/扩展 generationConfig
            generation_config.update(self.cfg.gemini_extra_generation_config)

        request_body: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": merged_text}
                    ]
                }
            ]
        }

        # 只有在确实有 generationConfig 时才放进去
        if generation_config:
            request_body["generationConfig"] = generation_config

        return [
            {
                "role": "user",
                "content": json.dumps(request_body, ensure_ascii=False)
            }
        ]

    def _build_request(
        self,
        base: str,
        use_responses: bool,
        is_gpt5: bool,
        is_gemini: bool,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        mt: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        - GPT-5：使用 /responses，并设置 reasoning.effort="minimal"
        - Gemini：默认走 /chat/completions，但 messages 会被包装成 Gemini JSON，并注入 thinkingBudget
        - 非 GPT-5：遵循原有逻辑
        """
        # Gemini：强制用 chat.completions（除非用户显式 api_mode=responses）
        if is_gemini and self.cfg.gemini_enable_wrapper and not use_responses:
            wrapped_messages = self._wrap_messages_for_gemini(messages)
            url = base + "/chat/completions"

            payload: Dict[str, Any] = {
                "model": self.cfg.model,
                "messages": wrapped_messages,
                # 这些字段按示例保留在“外层”（聚合网关通常就是这样解析的）
                "max_tokens": mt,
                "temperature": self.cfg.temperature if temperature is None else temperature,
            }
            return url, payload

        # 原 responses 逻辑
        if use_responses:
            url = base + "/responses"
            payload: Dict[str, Any] = {
                "model": self.cfg.model,
                "input": messages,
                "max_output_tokens": mt,
            }

            if is_gpt5:
                payload["reasoning"] = {"effort": "minimal"}
            else:
                payload["temperature"] = self.cfg.temperature if temperature is None else temperature

            return url, payload

        # 原 chat.completions 逻辑
        url = base + "/chat/completions"
        payload: Dict[str, Any] = {"model": self.cfg.model, "messages": messages}

        if is_gpt5:
            
            payload["max_output_tokens"] = mt
        else:
            payload["max_tokens"] = mt
            payload["temperature"] = self.cfg.temperature if temperature is None else temperature

        return url, payload

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        base = self.cfg.base_url.rstrip("/")
        use_responses = self._should_use_responses()
        is_gpt5 = self._is_gpt5_family()
        is_gemini = self._is_gemini_family()

        mt = self.cfg.max_tokens if max_tokens is None else max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        max_attempts = max(1, int(self.cfg.max_retries))

        for attempt in range(max_attempts):
            url, payload = self._build_request(
                base=base,
                use_responses=use_responses,
                is_gpt5=is_gpt5,
                is_gemini=is_gemini,
                messages=messages,
                temperature=temperature,
                mt=mt,
            )

            try:
                r = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False),
                    timeout=self.cfg.timeout_sec,
                )

                if r.status_code < 400:
                    data = r.json()

                    if isinstance(data, dict) and data.get("_token_limit_exceeded"):
                        print(
                            "Warning: Token limit exceeded, returning empty response: "
                            f"{data.get('error_message', '')}"
                        )
                        return ""

                    if use_responses:
                        if isinstance(data, dict) and isinstance(data.get("output_text"), str):
                            return data["output_text"]

                        out_items = (data.get("output", []) or []) if isinstance(data, dict) else []
                        chunks: List[str] = []
                        for item in out_items:
                            for c in (item.get("content", []) or []):
                                if c.get("type") == "output_text" and "text" in c:
                                    chunks.append(c["text"])
                        return "".join(chunks).strip()

                    # chat.completions 提取
                    try:
                        return data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError) as e:
                        print(f"Warning: Unexpected response format, returning empty: {e}")
                        return ""

                if r.status_code == 400:
                    try:
                        error_data = r.json()
                        error_msg = error_data.get("error", {}).get("message", "")
                        if "maximum context length" in error_msg or "tokens" in error_msg:
                            print(f"Warning: Token limit exceeded, returning empty response: {error_msg}")
                            return ""
                    except Exception:
                        pass
                    raise RuntimeError(f"HTTP 400: {r.text[:500]}")

                if self._is_retryable_status(r.status_code):
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                    if attempt < max_attempts - 1:
                        sleep_s = min(2 ** attempt, 30)
                        print(
                            f"Warning: retryable error ({r.status_code}). "
                            f"Sleep {sleep_s}s then retry ({attempt+1}/{max_attempts-1})"
                        )
                        time.sleep(sleep_s)
                        continue
                    raise last_err

                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if attempt < max_attempts - 1:
                    sleep_s = min(2 ** attempt, 30)
                    print(
                        f"Warning: network error {type(e).__name__}. "
                        f"Sleep {sleep_s}s then retry ({attempt+1}/{max_attempts-1})"
                    )
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"LLM call failed after retries: {last_err}") from e

            except Exception as e:
                last_err = e
                msg = str(e).lower()
                retryable_by_text = any(
                    k in msg for k in ["rate limit", "too many requests", "overloaded", "temporarily", "timeout"]
                )
                if retryable_by_text and attempt < max_attempts - 1:
                    sleep_s = min(2 ** attempt, 30)
                    print(
                        f"Warning: retryable exception. Sleep {sleep_s}s then retry "
                        f"({attempt+1}/{max_attempts-1}): {e}"
                    )
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(f"LLM call failed after retries: {last_err}") from e

        raise RuntimeError(f"LLM call failed after retries: {last_err}")
