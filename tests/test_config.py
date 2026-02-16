import os
from unittest.mock import patch

from opentrade_ai.config import AppConfig, LLMConfig, TradingConfig


class TestLLMConfig:
    def test_default_values(self):
        config = LLMConfig()
        assert config.provider == "ollama"
        assert config.model == "llama3"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.temperature == 0.3

    def test_from_env(self):
        env_vars = {
            "LLM_PROVIDER": "openai",
            "OLLAMA_MODEL": "mistral",
            "OLLAMA_BASE_URL": "http://custom:11434",
            "OPENAI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = LLMConfig.from_env()
            assert config.provider == "openai"
            assert config.model == "mistral"
            assert config.ollama_base_url == "http://custom:11434"
            assert config.openai_api_key == "test-key"

    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig.from_env()
            assert config.provider == "ollama"
            assert config.model == "llama3"


class TestTradingConfig:
    def test_default_values(self):
        config = TradingConfig()
        assert config.max_debate_rounds == 2
        assert config.risk_tolerance == "moderate"
        assert config.analysis_period_days == 90
        assert len(config.tickers) > 0

    def test_validate_valid_config(self):
        config = TradingConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_debate_rounds(self):
        config = TradingConfig(max_debate_rounds=0)
        errors = config.validate()
        assert any("max_debate_rounds" in e for e in errors)

    def test_validate_invalid_risk_tolerance(self):
        config = TradingConfig(risk_tolerance="extreme")
        errors = config.validate()
        assert any("risk_tolerance" in e for e in errors)

    def test_validate_invalid_analysis_period(self):
        config = TradingConfig(analysis_period_days=3)
        errors = config.validate()
        assert any("analysis_period_days" in e for e in errors)

    def test_validate_empty_tickers(self):
        config = TradingConfig(tickers=[])
        errors = config.validate()
        assert any("tickers" in e for e in errors)

    def test_validate_multiple_errors(self):
        config = TradingConfig(
            max_debate_rounds=0, risk_tolerance="yolo", analysis_period_days=1, tickers=[]
        )
        errors = config.validate()
        assert len(errors) == 4


class TestAppConfig:
    def test_default_creation(self):
        config = AppConfig()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.trading, TradingConfig)

    def test_from_env(self):
        config = AppConfig.from_env()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.trading, TradingConfig)

    def test_validate_delegates_to_trading(self):
        config = AppConfig(trading=TradingConfig(max_debate_rounds=0))
        errors = config.validate()
        assert len(errors) > 0
