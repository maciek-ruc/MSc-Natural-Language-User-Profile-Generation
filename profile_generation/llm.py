from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import time
from typing import Any

import requests


@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.2
    timeout: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    env_paths = [REPO_ROOT / ".env"]
    try:
        load_dotenv = importlib.import_module("dotenv").load_dotenv
        for env_path in env_paths:
            if env_path.is_file():
                load_dotenv(env_path)
        load_dotenv()
        return
    except Exception:
        pass

    for env_path in env_paths:
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def pick_first_nonempty(*values: str | None) -> str:
    for value in values:
        candidate = (value or "").strip()
        if candidate:
            return candidate
    return ""


def resolve_api_key(endpoint_hint: str = "") -> str:
    endpoint = endpoint_hint.lower()
    if "api.openai.com" in endpoint:
        return pick_first_nonempty(
            os.getenv("OPENAI_API_KEY"),
            os.getenv("LLM_API_KEY"),
            os.getenv("OLLAMA_API_KEY"),
        )
    return pick_first_nonempty(
        os.getenv("OLLAMA_API_KEY"),
        os.getenv("LLM_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    )


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _url_candidates(self) -> list[str]:
        base = self.config.base_url.rstrip("/")
        return [
            f"{base}/chat/completions",
            f"{base}/v1/chat/completions",
            f"{base}/api/chat",
        ]

    def chat_text(self, prompt: str, system_prompt: str = "", max_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt.strip():
            messages.insert(0, {"role": "system", "content": system_prompt})

        last_error: Exception | None = None
        for url in self._url_candidates():
            for attempt in range(self.config.max_retries):
                try:
                    payload: dict[str, Any]
                    if url.endswith("/api/chat"):
                        payload = {
                            "model": self.config.model,
                            "messages": messages,
                            "stream": False,
                            "options": {
                                "temperature": self.config.temperature,
                                "num_predict": int(max_tokens),
                            },
                        }
                    else:
                        payload = {
                            "model": self.config.model,
                            "messages": messages,
                            "temperature": self.config.temperature,
                            "max_tokens": int(max_tokens),
                        }
                    response = requests.post(
                        url,
                        json=payload,
                        headers=self._headers(),
                        timeout=self.config.timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if url.endswith("/api/chat"):
                        return str(data.get("message", {}).get("content") or "").strip()
                    return str(data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                except Exception as exc:
                    last_error = exc
                    if attempt + 1 < self.config.max_retries:
                        time.sleep(self.config.retry_backoff_seconds)

        raise RuntimeError(f"LLM request failed: {last_error}")


def config_from_env(model_env: str, default_model: str) -> LLMConfig:
    load_project_env()
    is_judge = "JUDGE" in model_env.upper()
    if is_judge:
        base_url = pick_first_nonempty(
            os.getenv("LLM_JUDGE_API_BASE"),
            os.getenv("JUDGE_API_BASE"),
            os.getenv("LLM_API_BASE"),
            os.getenv("LLM_BASE_URL"),
            os.getenv("OLLAMA_HOST"),
            os.getenv("OLLAMA_API_URL"),
            os.getenv("LLM_API_URL"),
        )
    else:
        base_url = pick_first_nonempty(
            os.getenv("LLM_GEN_BASE_URL"),
            os.getenv("LLM_BASE_URL"),
            os.getenv("OLLAMA_HOST"),
            os.getenv("OLLAMA_API_URL"),
            os.getenv("LLM_API_URL"),
        )
    if not base_url:
        raise ValueError("Missing LLM endpoint in env (.env, LLM_BASE_URL, OLLAMA_HOST, or related judge/gen variables)")
    return LLMConfig(
        model=os.getenv(model_env) or default_model,
        base_url=base_url,
        api_key=resolve_api_key(base_url),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )
