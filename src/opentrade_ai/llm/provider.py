import json
import logging
import time

import requests

from opentrade_ai.config import LLMConfig

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2


class LLMProvider:
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if self.config.provider == "ollama":
            call_fn = self._call_ollama
        elif self.config.provider == "openai":
            call_fn = self._call_openai
        elif self.config.provider == "lmstudio":
            call_fn = self._call_lmstudio
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

        last_err: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return call_fn(prompt, system_prompt)
            except (ConnectionError, TimeoutError, RuntimeError) as exc:
                last_err = exc
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "LLM call attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt, _MAX_RETRIES, exc, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
        raise last_err  # type: ignore[misc]

    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        url = f"{self.config.ollama_base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.ollama_base_url}. "
                "Make sure Ollama is running: https://ollama.com"
            )
        except requests.Timeout:
            raise TimeoutError("Ollama request timed out after 120 seconds")
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Ollama returned HTTP error {getattr(e.response, 'status_code', '?')}: "
                f"{getattr(e.response, 'text', str(e))}"
            )

    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        resp = requests.post(url, headers=headers, json=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_lmstudio(self, prompt: str, system_prompt: str) -> str:
        base_url = self.config.lmstudio_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                raise RuntimeError(f"Unexpected LM Studio response: {data}")
            return choices[0]["message"]["content"]
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {base_url}. "
                "Make sure LM Studio is running with a model loaded."
            )
        except requests.Timeout:
            raise TimeoutError("LM Studio request timed out after 120 seconds")
        except requests.HTTPError as e:
            raise RuntimeError(
                f"LM Studio returned HTTP error {getattr(e.response, 'status_code', '?')}: "
                f"{getattr(e.response, 'text', str(e))}"
            )

    def is_available(self) -> bool:
        if self.config.provider == "ollama":
            try:
                resp = requests.get(
                    f"{self.config.ollama_base_url}/api/tags", timeout=5
                )
                return resp.status_code == 200
            except (requests.ConnectionError, requests.Timeout):
                return False
        elif self.config.provider == "openai":
            return bool(self.config.openai_api_key)
        elif self.config.provider == "lmstudio":
            try:
                base_url = self.config.lmstudio_base_url.rstrip("/")
                resp = requests.get(f"{base_url}/models", timeout=5)
                return resp.status_code == 200
            except (requests.ConnectionError, requests.Timeout):
                return False
        return False
