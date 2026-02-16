from unittest.mock import MagicMock, patch

import pytest
import requests

from opentrade_ai.config import LLMConfig
from opentrade_ai.llm.provider import LLMProvider


class TestLLMProvider:
    def test_init(self):
        config = LLMConfig()
        provider = LLMProvider(config)
        assert provider.config.provider == "ollama"

    def test_generate_unsupported_provider(self):
        config = LLMConfig(provider="unknown")
        provider = LLMProvider(config)
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            provider.generate("test prompt")

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_ollama_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Test response from LLM"}
        mock_post.return_value = mock_resp

        config = LLMConfig(provider="ollama", model="llama3")
        provider = LLMProvider(config)
        result = provider.generate("Analyze AAPL", "You are an analyst")

        assert result == "Test response from LLM"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["model"] == "llama3"
        assert call_kwargs[1]["json"]["prompt"] == "Analyze AAPL"

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_ollama_connection_error(self, mock_post):
        mock_post.side_effect = requests.ConnectionError()
        config = LLMConfig(provider="ollama")
        provider = LLMProvider(config)
        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            provider.generate("test")

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_ollama_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout()
        config = LLMConfig(provider="ollama")
        provider = LLMProvider(config)
        with pytest.raises(TimeoutError, match="timed out"):
            provider.generate("test")

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_openai_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "OpenAI response"}}]
        }
        mock_post.return_value = mock_resp

        config = LLMConfig(provider="openai", model="gpt-4o-mini", openai_api_key="test-key")
        provider = LLMProvider(config)
        result = provider.generate("test", "system")
        assert result == "OpenAI response"

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_lmstudio_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "LM Studio response"}}]
        }
        mock_post.return_value = mock_resp

        config = LLMConfig(provider="lmstudio", model="local-model")
        provider = LLMProvider(config)
        result = provider.generate("test", "system")
        assert result == "LM Studio response"
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:1234/v1/chat/completions"

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_lmstudio_connection_error(self, mock_post):
        mock_post.side_effect = requests.ConnectionError()
        config = LLMConfig(provider="lmstudio")
        provider = LLMProvider(config)
        with pytest.raises(ConnectionError, match="Cannot connect to LM Studio"):
            provider.generate("test")

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_lmstudio_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout()
        config = LLMConfig(provider="lmstudio")
        provider = LLMProvider(config)
        with pytest.raises(TimeoutError, match="LM Studio request timed out"):
            provider.generate("test")

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_lmstudio_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = requests.HTTPError(
            response=mock_resp
        )
        mock_resp.text = '{"error": "model not found"}'
        mock_post.return_value = mock_resp
        config = LLMConfig(provider="lmstudio", model="bad-model")
        provider = LLMProvider(config)
        with pytest.raises(RuntimeError, match="LM Studio returned HTTP error"):
            provider.generate("test")

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_lmstudio_invalid_json(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"unexpected": "format"}
        mock_post.return_value = mock_resp
        config = LLMConfig(provider="lmstudio")
        provider = LLMProvider(config)
        with pytest.raises(RuntimeError, match="Unexpected LM Studio response"):
            provider.generate("test")

    @patch("opentrade_ai.llm.provider.requests.post")
    def test_call_ollama_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = requests.HTTPError(
            response=mock_resp
        )
        mock_resp.text = "model not found"
        mock_post.return_value = mock_resp
        config = LLMConfig(provider="ollama", model="bad-model")
        provider = LLMProvider(config)
        with pytest.raises(RuntimeError, match="Ollama returned HTTP error"):
            provider.generate("test")


class TestLLMAvailability:
    @patch("opentrade_ai.llm.provider.requests.get")
    def test_ollama_available(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        config = LLMConfig(provider="ollama")
        provider = LLMProvider(config)
        assert provider.is_available() is True

    @patch("opentrade_ai.llm.provider.requests.get")
    def test_ollama_unavailable(self, mock_get):
        mock_get.side_effect = requests.ConnectionError()
        config = LLMConfig(provider="ollama")
        provider = LLMProvider(config)
        assert provider.is_available() is False

    def test_openai_available_with_key(self):
        config = LLMConfig(provider="openai", openai_api_key="test-key")
        provider = LLMProvider(config)
        assert provider.is_available() is True

    def test_openai_unavailable_without_key(self):
        config = LLMConfig(provider="openai", openai_api_key="")
        provider = LLMProvider(config)
        assert provider.is_available() is False

    @patch("opentrade_ai.llm.provider.requests.get")
    def test_lmstudio_available(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        config = LLMConfig(provider="lmstudio")
        provider = LLMProvider(config)
        assert provider.is_available() is True

    @patch("opentrade_ai.llm.provider.requests.get")
    def test_lmstudio_unavailable(self, mock_get):
        mock_get.side_effect = requests.ConnectionError()
        config = LLMConfig(provider="lmstudio")
        provider = LLMProvider(config)
        assert provider.is_available() is False

    def test_unknown_provider_unavailable(self):
        config = LLMConfig(provider="unknown")
        provider = LLMProvider(config)
        assert provider.is_available() is False
